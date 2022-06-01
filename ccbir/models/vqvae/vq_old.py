from functools import cached_property, partial
from itertools import starmap
from turtle import forward
from typing import List
from einops import rearrange, reduce
from more_itertools import unzip
from torch.cuda.amp import autocast
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch.vector_quantize_pytorch import (
    CosineSimCodebook,
    l2norm,
    ema_inplace,
    gumbel_sample,
)
from ccbir.util import apply

# TODO: use consistent naming for encoding throughout codebase
# right now it's called embed_ind, latent, discrete_latent etc.


class Codebook(CosineSimCodebook):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.8,
        eps=0.00001,
        threshold_ema_dead_code=2,
        use_ddp=False,
        learnable_codebook=False,
        sample_codebook_temp=0,
        inject_noise=False,
        inject_noise_sigma=1.0,
    ):
        super().__init__(
            dim,
            codebook_size,
            kmeans_init,
            kmeans_iters,
            decay,
            eps,
            threshold_ema_dead_code,
            use_ddp,
            learnable_codebook,
            sample_codebook_temp,
        )
        self.inject_noise = inject_noise
        self.inject_noise_sigma = inject_noise_sigma

    def _inject_noise(self, embed_ind: Tensor) -> Tensor:
        noise = torch.round(
            self.inject_noise_sigma
            * torch.randn(embed_ind.shape, device=embed_ind.device)
        )
        return ((embed_ind + noise) % self.codebook_size).type(embed_ind.dtype)

    @autocast(enabled=False)
    def forward(self, x):
        """This method is heavily based on
        https://github.com/lucidrains/vector-quantize-pytorch"""
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        flatten = l2norm(flatten)

        self.init_embed_(flatten)

        embed = (
            self.embed if not self.learnable_codebook else
            self.embed.detach()
        )
        embed = l2norm(embed)

        dist = flatten @ embed.t()

        # TODO: refactor following logic
        embed_ind = gumbel_sample(
            dist,
            dim=-1,
            temperature=self.sample_codebook_temp,
        )

        if self.inject_noise and self.training:
            embed_ind_noisy = self._inject_noise(embed_ind)
            embed_onehot_ema = (
                F.one_hot(embed_ind_noisy, self.codebook_size).type(dtype)
            )

            embed_ind = embed_ind.view(*shape[:-1])
            embed_ind_output = embed_ind_noisy.view(*shape[:-1])
            quantize_commit_loss = F.embedding(embed_ind, self.embed)
            quantize_output = F.embedding(embed_ind_output, self.embed)
        else:
            embed_onehot_ema = (
                F.one_hot(embed_ind, self.codebook_size).type(dtype)
            )
            embed_ind_output = embed_ind.view(*shape[:-1])
            quantize_commit_loss = F.embedding(embed_ind_output, self.embed)
            quantize_output = quantize_commit_loss

        if self.training:
            bins = embed_onehot_ema.sum(0)
            self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = flatten.t() @ embed_onehot_ema
            self.all_reduce_fn(embed_sum)

            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(zero_mask[..., None], embed,
                                           embed_normalized)
            ema_inplace(self.embed, embed_normalized, self.decay)
            self.expire_codes_(x)

        return quantize_output, quantize_commit_loss, embed_ind_output


class VectorQuantizer(VectorQuantize):
    def __init__(
        self,
        dim,
        codebook_size,
        n_embed=None,
        codebook_dim=None,
        decay=0.99,
        eps=0.00001,
        kmeans_init=False,
        kmeans_iters=10,
        use_cosine_sim=False,
        threshold_ema_dead_code=0,
        channel_last=True,
        accept_image_fmap=False,
        commitment_weight=None,
        commitment=1,
        orthogonal_reg_weight=0,
        orthogonal_reg_active_codes_only=False,
        orthogonal_reg_max_codes=None,
        sample_codebook_temp=0,
        sync_codebook=False,
        inject_noise=False,
        inject_noise_sigma=1.0,
    ):
        super().__init__(
            dim,
            codebook_size,
            n_embed,
            codebook_dim,
            decay,
            eps,
            kmeans_init,
            kmeans_iters,
            use_cosine_sim,
            threshold_ema_dead_code,
            channel_last,
            accept_image_fmap,
            commitment_weight,
            commitment,
            orthogonal_reg_weight,
            orthogonal_reg_active_codes_only,
            orthogonal_reg_max_codes,
            sample_codebook_temp,
            sync_codebook,
        )
        self._codebook = Codebook(
            dim=dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=sync_codebook,
            learnable_codebook=False,
            sample_codebook_temp=0,
            inject_noise=inject_noise,
            inject_noise_sigma=inject_noise_sigma,
        )

    def forward(self, x):
        """This method is heavily based on
        https://github.com/lucidrains/vector-quantize-pytorch"""
        shape, device, codebook_size = x.shape, x.device, self.codebook_size

        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        x = self.project_in(x)

        quantize_output, quantize_commit_loss, embed_ind = self._codebook(x)

        if self.training:
            quantize_output = x + (quantize_output - x).detach()
            if self._codebook.inject_noise:
                quantize_commit_loss = x + (quantize_commit_loss - x).detach()
            else:
                quantize_commit_loss = quantize_output
        
        loss = torch.tensor([0.], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize_commit_loss.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize_output = self.project_out(quantize_output)

        if need_transpose:
            quantize_output = rearrange(quantize_output, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize_output = rearrange(
                quantize_output, 'b (h w) c -> b c h w', h=height, w=width)
            embed_ind = rearrange(
                embed_ind, 'b (h w) -> b h w', h=height, w=width)

        return quantize_output, embed_ind, loss

    def quantize_encoding(
        self,
        encoding: Tensor
    ) -> Tensor:
        if self.accept_image_fmap:
            assert encoding.dim() == 3
            encoding_idxs = rearrange(encoding, 'b h w -> b (h w)')
        else:
            assert encoding.dim() == 2
            encoding_idxs = encoding

        quantized = F.embedding(encoding_idxs, self._codebook.embed)

        if self.accept_image_fmap:
            _b, h, w = encoding.shape
            quantized = rearrange(quantized, 'b (h w) c -> b c h w', h=h, w=w)

        return quantized

    @cached_property
    def _distance_matrix(self) -> Tensor:
        codewords = self.codebook.unsqueeze(0)
        print('Pre-computing codebook distances')
        distances = torch.cdist(codewords, codewords, p=2).squeeze(0)
        print('Precomputed codebook distances')

        return distances

    def latents_distance(
        self,
        latent1: Tensor,  # B x *
        latent2: Tensor,  # B x *
    ) -> Tensor:  # B
        """Distance between discrete latents (of size B x *) computed as the MSE
        between the corresponding quantized tensors (of size B x * x D)"""
        assert latent1.shape == latent2.shape
        flatten = partial(rearrange, pattern='b ... -> b (...)')
        idxs1 = flatten(latent1)
        idxs2 = flatten(latent2)
        distances = self._distance_matrix[idxs1, idxs2]
        distance = reduce(distances, 'b n -> b', 'mean')

        return distance


class ProductQuantizer(nn.Module):
    def __init__(
        self,
        num_quantizers: int,
        **kwargs,
    ):
        super().__init__()
        self.vqs = nn.ModuleList([
            VectorQuantizer(**kwargs) for _ in range(num_quantizers)
        ])
        self.num_quantizers = num_quantizers

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] % self.num_quantizers == 0
        x_chunks = torch.chunk(x, len(self.vqs), dim=1)
        z_q_chunks, z_chunks, vq_losses = (
            unzip(starmap(apply, zip(self.vqs, x_chunks)))
        )
        quantized = torch.cat(list(z_q_chunks), dim=1)
        encoding = torch.cat(list(z_chunks), dim=1)
        vq_loss = sum(vq_losses)

        return quantized, encoding, vq_loss

    def quantize_encoding(
        self,
        encoding: Tensor
    ) -> Tensor:
        encoding_chunks = torch.chunk(encoding, len(self.vqs), dim=1)
        quantized_chunks = [
            vq.quantize_encoding(encoding_chunk)
            for vq, encoding_chunk in
            zip(self.vqs, encoding_chunks)
        ]
        quantized = torch.cat(quantized_chunks, dim=1)

        return quantized

    def latents_distance(
        self,
        latent1: Tensor,  # B x *
        latent2: Tensor,  # B x *
    ) -> Tensor:  # B
        latent1_chunks = torch.chunk(latent1, len(self.vqs), dim=1)
        latent2_chunks = torch.chunk(latent2, len(self.vqs), dim=1)
        distance = sum(
            vq.latents_distance(l1_chunk, l2_chunk)
            for vq, l1_chunk, l2_chunk in
            zip(self.vqs, latent1_chunks, latent2_chunks)
        )

        return distance
