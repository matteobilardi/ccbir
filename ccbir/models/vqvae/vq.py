from functools import cached_property, partial
from itertools import starmap
from pprint import pprint
from turtle import forward
from typing import Dict, List, Union
from einops import rearrange, reduce
from more_itertools import unzip
from torch.cuda.amp import autocast
from torch import BoolTensor, Tensor, nn
import torch
import torch.nn.functional as F
from toolz import assoc, merge, merge_with
from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch.vector_quantize_pytorch import (
    CosineSimCodebook,
    EuclideanCodebook,
    l2norm,
    ema_inplace,
    gumbel_sample,
    laplace_smoothing
)
from ccbir.util import apply

# TODO: use consistent naming for encoding throughout codebase
# right now it's called embed_ind, latent, discrete_latent etc.


class BasicCodebook(EuclideanCodebook):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.99,
        eps=0.00001,
        threshold_ema_dead_code=0,
        use_ddp=False,
        learnable_codebook=False,
        sample_codebook_temp=0,
        inject_noise=False,
        inject_noise_sigma=1.0,
        inject_noise_rings=1,
        wrap_ring_noise=True,
        noise_type='ring',
        return_negative_sample=False,
        clean_cluster_size_update=True,
        limit_per_ring_expiry=None,
    ):
        super().__init__(
            dim=dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            eps=eps,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_ddp=use_ddp,
            learnable_codebook=learnable_codebook,
            sample_codebook_temp=sample_codebook_temp,
        )

        assert codebook_size % inject_noise_rings == 0
        assert 1 <= inject_noise_rings <= codebook_size
        self.inject_noise = inject_noise
        self.inject_noise_sigma = inject_noise_sigma
        self.inject_noise_rings = inject_noise_rings
        self.return_negative_sample = return_negative_sample
        self.clean_cluster_size_update = clean_cluster_size_update
        self.register_buffer('clean_cluster_size', torch.zeros(codebook_size))

        self.noise_type = noise_type
        self.num_rings = inject_noise_rings
        self.ring_size = codebook_size // inject_noise_rings
        self.wrap_ring_noise = wrap_ring_noise

        self.register_buffer(
            name='ring_start_idx',
            tensor=(
                (self.ring_size * torch.arange(inject_noise_rings))
                .repeat_interleave(self.ring_size)
            ),
        )
        self.limit_per_ring_expiry = limit_per_ring_expiry

    def _inject_noise(self, embed_ind: Tensor) -> Tensor:
        if self.num_rings == 1:
            return self._inject_noise_default(embed_ind)
        else:
            return self._inject_noise_multiring(embed_ind)

    def _inject_noise_multiring(self, embed_ind: Tensor) -> Tensor:
        noise = torch.round(
            self.inject_noise_sigma
            * torch.randn(embed_ind.shape, device=embed_ind.device)
        )
        ring_start_idx = self.ring_start_idx[embed_ind]
        ring_relative_idx = embed_ind - ring_start_idx
        if self.noise_type == 'ring':
            max_shift = 1 + self.ring_size // 2
            noise = noise.fmod(max_shift)
            next_ring_relative_idx = (
                (ring_relative_idx + noise) % self.ring_size
            )
        elif self.noise_type == 'line':
            right_noise = noise.abs() % (self.ring_size - ring_relative_idx)
            next_ring_relative_idx = ring_relative_idx + right_noise
        else:
            raise RuntimeError(f'Unsupported {self.noise_type=}')

        next_absolute_idx = ring_start_idx + next_ring_relative_idx
        next_absolute_idx = next_absolute_idx.type(embed_ind.dtype)

        return next_absolute_idx

    def _inject_noise_default(self, embed_ind: Tensor) -> Tensor:
        noise = torch.round(
            self.inject_noise_sigma
            * torch.randn(embed_ind.shape, device=embed_ind.device)
        )
        return ((embed_ind + noise) % self.codebook_size).type(embed_ind.dtype)

    @autocast(enabled=False)
    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = self.embed.t()

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_idxs = gumbel_sample(
            dist, dim=-1, temperature=self.sample_codebook_temp)

        output = dict()
        if self.inject_noise and self.training:
            embed_idxs_noisy = self._inject_noise(embed_idxs)
            #embed_onehot_ema_clean = (
            #    F.one_hot(embed_idxs, self.codebook_size).type(dtype)
            #)
            embed_onehot_ema = (
                F.one_hot(embed_idxs_noisy, self.codebook_size).type(dtype)
            )

            embed_idxs = embed_idxs.view(*shape[:-1])
            embed_idxs_output = embed_idxs_noisy.view(*shape[:-1])

            z_q = F.embedding(embed_idxs, self.embed)
            z_q_noisy = F.embedding(embed_idxs_output, self.embed)
            output = merge(
                output,
                dict(
                    quantized_for_commit_loss=z_q,
                    quantized_output=z_q_noisy,
                    embedding_indices=embed_idxs_output,
                ),
            )
        else:
            embed_onehot_ema = (
                F.one_hot(embed_idxs, self.codebook_size).type(dtype)
            )
            embed_onehot_ema_clean = embed_onehot_ema
            embed_idxs_output = embed_idxs.view(*shape[:-1])
            z_q = F.embedding(embed_idxs_output, self.embed)
            output = merge(
                output,
                dict(
                    quantized_for_commit_loss=z_q,
                    quantized_output=z_q,
                    embedding_indices=embed_idxs_output,
                )
            )

        if self.training:
            cluster_size = embed_onehot_ema.sum(0)
            self.all_reduce_fn(cluster_size)

            ema_inplace(self.cluster_size, cluster_size, self.decay)

            embed_sum = flatten.t() @ embed_onehot_ema
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(
                self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        return output

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        cluster_size = (
            self.clean_cluster_size if self.clean_cluster_size_update else
            self.cluster_size
        )

        if self.limit_per_ring_expiry is None:
            expired_codes = cluster_size < self.threshold_ema_dead_code
            if not torch.any(expired_codes):
                return
        else:
            limit = self.limit_per_ring_expiry
            candidate_codes = cluster_size < self.threshold_ema_dead_code
            if not torch.any(candidate_codes):
                return

            candidates_codes_per_rng = candidate_codes.chunk(self.num_rings)

            def limit_expired(codes_mask: Tensor) -> Tensor:
                candidate_codes_idxs = codes_mask.nonzero().flatten()
                rand_idxs = torch.randperm(len(candidate_codes_idxs))[:limit]
                expired_codes_idxs = candidate_codes_idxs[rand_idxs]
                expired_codes_mask = torch.tensor(
                    data=False,
                    device=codes_mask.device,
                ).expand_as(codes_mask).clone()
                expired_codes_mask[expired_codes_idxs] = True
                return expired_codes_mask

            expired_codes = torch.cat(list(
                map(limit_expired, candidates_codes_per_rng)
            ))

        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)


class Codebook(CosineSimCodebook):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init=False,
        kmeans_iters=10,
        decay=0.99,
        eps=0.00001,
        threshold_ema_dead_code=0,
        use_ddp=False,
        learnable_codebook=False,
        sample_codebook_temp=0,
        inject_noise=False,
        inject_noise_sigma=1.0,
        inject_noise_rings=1,
        wrap_ring_noise=True,
        noise_type='ring',
        return_negative_sample=False,
        clean_cluster_size_update=True,
        limit_per_ring_expiry=None,
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
        assert codebook_size % inject_noise_rings == 0
        assert 1 <= inject_noise_rings <= codebook_size
        self.inject_noise = inject_noise
        self.inject_noise_sigma = inject_noise_sigma
        self.inject_noise_rings = inject_noise_rings
        self.return_negative_sample = return_negative_sample
        self.clean_cluster_size_update = clean_cluster_size_update
        self.register_buffer('clean_cluster_size', torch.zeros(codebook_size))

        self.noise_type = noise_type
        self.num_rings = inject_noise_rings
        self.ring_size = codebook_size // inject_noise_rings
        self.wrap_ring_noise = wrap_ring_noise

        self.register_buffer(
            name='ring_start_idx',
            tensor=(
                (self.ring_size * torch.arange(inject_noise_rings))
                .repeat_interleave(self.ring_size)
            ),
        )
        self.limit_per_ring_expiry = limit_per_ring_expiry

    def _inject_noise(self, embed_ind: Tensor) -> Tensor:
        if self.num_rings == 1:
            return self._inject_noise_default(embed_ind)
        else:
            return self._inject_noise_multiring(embed_ind)

    def _inject_noise_multiring(self, embed_ind: Tensor) -> Tensor:
        noise = torch.round(
            self.inject_noise_sigma
            * torch.randn(embed_ind.shape, device=embed_ind.device)
        )
        ring_start_idx = self.ring_start_idx[embed_ind]
        ring_relative_idx = embed_ind - ring_start_idx
        if self.noise_type == 'ring':
            max_shift = 1 + self.ring_size // 2
            noise = noise.fmod(max_shift)
            next_ring_relative_idx = (
                (ring_relative_idx + noise) % self.ring_size
            )
        elif self.noise_type == 'line':
            right_noise = noise.abs() % (self.ring_size - ring_relative_idx)
            next_ring_relative_idx = ring_relative_idx + right_noise
        else:
            raise RuntimeError(f'Unsupported {self.noise_type=}')

        next_absolute_idx = ring_start_idx + next_ring_relative_idx
        next_absolute_idx = next_absolute_idx.type(embed_ind.dtype)

        return next_absolute_idx

    def _inject_noise_default(self, embed_ind: Tensor) -> Tensor:
        noise = torch.round(
            self.inject_noise_sigma
            * torch.randn(embed_ind.shape, device=embed_ind.device)
        )
        return ((embed_ind + noise) % self.codebook_size).type(embed_ind.dtype)

    @autocast(enabled=False)
    def forward(self, x) -> Dict:
        """This method is adapted from
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

        embed_idxs = gumbel_sample(
            dist,
            dim=-1,
            temperature=self.sample_codebook_temp,
        )

        output = dict()

        if self.training and self.return_negative_sample:
            random_shift = torch.randint_like(
                input=embed_idxs,
                low=0,
                high=self.codebook_size - 1,
            )
            rand_negative_idxs = (
                (embed_idxs + random_shift) % self.codebook_size
            )
            rand_negative_idxs = rand_negative_idxs.view(*shape[:-1])
            quantized_negative_sample = F.embedding(
                rand_negative_idxs,
                self.embed
            )
            output = merge(
                output,
                dict(quantized_negative_sample=quantized_negative_sample)
            )

        if self.inject_noise and self.training:
            embed_idxs_noisy = self._inject_noise(embed_idxs)
            embed_onehot_ema_clean = (
                F.one_hot(embed_idxs, self.codebook_size).type(dtype)
            )
            embed_onehot_ema = (
                F.one_hot(embed_idxs_noisy, self.codebook_size).type(dtype)
            )

            embed_idxs = embed_idxs.view(*shape[:-1])
            embed_idxs_output = embed_idxs_noisy.view(*shape[:-1])

            z_q = F.embedding(embed_idxs, self.embed)
            z_q_noisy = F.embedding(embed_idxs_output, self.embed)
            output = merge(
                output,
                dict(
                    quantized_for_commit_loss=z_q,
                    quantized_output=z_q_noisy,
                    embedding_indices=embed_idxs_output,
                ),
            )
        else:
            embed_onehot_ema = (
                F.one_hot(embed_idxs, self.codebook_size).type(dtype)
            )
            embed_onehot_ema_clean = embed_onehot_ema
            embed_idxs_output = embed_idxs.view(*shape[:-1])
            z_q = F.embedding(embed_idxs_output, self.embed)
            output = merge(
                output,
                dict(
                    quantized_for_commit_loss=z_q,
                    quantized_output=z_q,
                    embedding_indices=embed_idxs_output,
                )
            )

        if self.training:
            bins = embed_onehot_ema.sum(0)
            self.all_reduce_fn(bins)
            ema_inplace(self.cluster_size, bins, self.decay)
            if self.clean_cluster_size_update:
                clean_bins = embed_onehot_ema_clean.sum(0)
                self.all_reduce_fn(clean_bins)
                ema_inplace(
                    self.clean_cluster_size,
                    clean_bins,
                    self.decay,
                )

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

        return output

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        cluster_size = (
            self.clean_cluster_size if self.clean_cluster_size_update else
            self.cluster_size
        )

        if self.limit_per_ring_expiry is None:
            expired_codes = cluster_size < self.threshold_ema_dead_code
            if not torch.any(expired_codes):
                return
        else:
            limit = self.limit_per_ring_expiry
            candidate_codes = cluster_size < self.threshold_ema_dead_code
            if not torch.any(candidate_codes):
                return

            candidates_codes_per_rng = candidate_codes.chunk(self.num_rings)

            def limit_expired(codes_mask: Tensor) -> Tensor:
                candidate_codes_idxs = codes_mask.nonzero().flatten()
                rand_idxs = torch.randperm(len(candidate_codes_idxs))[:limit]
                expired_codes_idxs = candidate_codes_idxs[rand_idxs]
                expired_codes_mask = torch.tensor(
                    data=False,
                    device=codes_mask.device,
                ).expand_as(codes_mask).clone()
                expired_codes_mask[expired_codes_idxs] = True
                return expired_codes_mask

            expired_codes = torch.cat(list(
                map(limit_expired, candidates_codes_per_rng)
            ))

        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)


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
        inject_noise_sigma=0,
        inject_noise_rings=1,
        wrap_ring_noise=True,
        noise_type='ring',
        negative_sample_weight=0.0,
        clean_cluster_size_update=True,
        limit_per_ring_expiry=None,
        l2_norm_loss=False,
    ):
        assert negative_sample_weight >= 0
        assert inject_noise_sigma >= 0

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
        if l2_norm_loss:
            assert use_cosine_sim
        self.l2_norm_loss = l2_norm_loss
        mk_codebook = Codebook if use_cosine_sim else BasicCodebook
        self._codebook = mk_codebook(
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
            return_negative_sample=negative_sample_weight > 0,
            clean_cluster_size_update=clean_cluster_size_update,
            inject_noise_rings=inject_noise_rings,
            wrap_ring_noise=wrap_ring_noise,
            noise_type=noise_type,
            limit_per_ring_expiry=limit_per_ring_expiry,
        )
        self.inject_noise = inject_noise
        self.negative_sample_weight = negative_sample_weight

    def forward(self, x):
        """This method is adapted from
        https://github.com/lucidrains/vector-quantize-pytorch"""
        shape, device, codebook_size = x.shape, x.device, self.codebook_size

        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        x = self.project_in(x)

        if self.l2_norm_loss:
            x = l2norm(x)

        codebook_output = self._codebook(x)
        quantized_output = codebook_output['quantized_output']
        quantized_commit_loss = codebook_output['quantized_for_commit_loss']
        embed_ind = codebook_output['embedding_indices']

        if self.training:
            quantized_output = x + (quantized_output - x).detach()
            if self.inject_noise:
                quantized_commit_loss = (
                    x + (quantized_commit_loss - x).detach()
                )
            else:
                # optimization for when discrete noise is not applied
                quantized_commit_loss = quantized_output

        loss = torch.tensor([0.], device=device, requires_grad=self.training)
        metrics = dict()

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantized_commit_loss.detach(), x)
                loss = loss + commit_loss * self.commitment_weight
                metrics = merge(metrics, dict(commit_loss=commit_loss.item()))

            if self.negative_sample_weight > 0:
                z_q_negative = codebook_output['quantized_negative_sample']
                neg_sample_loss = 1 / F.mse_loss(z_q_negative.detach(), x)
                loss = loss + neg_sample_loss * self.negative_sample_weight
                metrics = merge(
                    metrics,
                    dict(negative_sample_loss=neg_sample_loss.item()),
                )

        quantized_output = self.project_out(quantized_output)

        if need_transpose:
            quantized_output = rearrange(quantized_output, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantized_output = rearrange(
                quantized_output, 'b (h w) c -> b c h w', h=height, w=width)
            embed_ind = rearrange(
                embed_ind, 'b (h w) -> b h w', h=height, w=width)

        return quantized_output, embed_ind, loss, metrics

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

    @ cached_property
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
        inject_noise_rings: Union[int, List[int]] = 1,
        **kwargs,
    ):
        super().__init__()
        if isinstance(inject_noise_rings, int):
            noise_rings_per_quantizer = [inject_noise_rings] * num_quantizers
        elif isinstance(inject_noise_rings, list):
            assert len(inject_noise_rings) == num_quantizers
            noise_rings_per_quantizer = inject_noise_rings
        else:
            raise RuntimeError(f'Unsupported {type(inject_noise_rings)=}')

        self.vqs = nn.ModuleList([
            VectorQuantizer(inject_noise_rings=num_rings, **kwargs)
            for num_rings in noise_rings_per_quantizer
        ])
        self.num_quantizers = num_quantizers

        log_hyperparams = merge(
            kwargs,
            dict(
                num_quantizers=num_quantizers,
                inject_noise_rings=inject_noise_rings
            )
        )
        pprint(log_hyperparams)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] % self.num_quantizers == 0
        x_chunks = torch.chunk(x, len(self.vqs), dim=1)
        z_q_chunks, z_chunks, vq_losses, metrics = (
            unzip(starmap(apply, zip(self.vqs, x_chunks)))
        )
        quantized = torch.cat(list(z_q_chunks), dim=1)
        encoding = torch.cat(list(z_chunks), dim=1)

        def mean(xs):
            xs_ = list(xs)
            return sum(xs_) / len(xs_)
        vq_loss = mean(vq_losses)
        metrics = merge_with(mean, metrics)
        #vq_loss = sum(vq_losses)
        #metrics = merge_with(sum, metrics)

        return quantized, encoding, vq_loss, metrics

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
