from einops import rearrange, reduce
from ccbir.arch import PreActResBlock
from ccbir.util import ActivationFunc, activation_layer_ctor
from functools import cached_property, partial
import pytorch_lightning as pl
from torch import LongTensor, Tensor, nn
from typing import Callable, Literal
import torch
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from toolz import keymap


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
        sync_codebook=False
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

    def quantize_encoding(
        self,
        encoding: Tensor
    ) -> Tensor:
        _b, h, w = encoding.shape
        encoding_idxs = rearrange(encoding, 'b h w -> b (h w)')
        quantized = F.embedding(encoding_idxs, self._codebook.embed)
        if self.accept_image_fmap:
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


class VQVAEComponent(nn.Module):

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        codebook_size: int,
        commitment_cost: float,
        activation: Callable[..., nn.Module],
        width_scale: int,
    ):
        super().__init__()
        C = in_channels
        W = width_scale
        L = latent_dim

        resblocks = partial(
            PreActResBlock.multi_block,
            activation=activation,
            use_se=True,
        )

        # overwrite encoder/decoder from parent class
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 32, 4, 2, bias=False),
            nn.BatchNorm2d(32),
            activation(),
            nn.Conv2d(32, 64, 3, 1, bias=False),
            nn.BatchNorm2d(64),
            activation(),
            nn.Conv2d(64, 64, 4, 1),
            resblocks(4, 64, 64, activation=activation),
            nn.Conv2d(64, L, 1),
        )
        self.vq = VectorQuantizer(
            dim=latent_dim,
            codebook_size=codebook_size,
            use_cosine_sim=True,
            accept_image_fmap=True,
            decay=0.99,
            commitment_weight=commitment_cost,
            threshold_ema_dead_code=2,
            # sample_codebook_temp=0.1,
        )
        self.decoder = nn.Sequential(
            # scales 8x8 to 32x32
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Conv2d(L, 64, 3, 1, bias=False),
            nn.BatchNorm2d(64),
            activation(),
            nn.Conv2d(64, 64, 3, 1, bias=False),
            resblocks(4, 64, 64, activation=activation),
            nn.Conv2d(64, C, 1),
            nn.Tanh()
        )

    def encode(self, x: Tensor):
        z_e = self.encoder(x)
        _z_q, e, _loss = self.vq(z_e)
        return e

    def decode(self, discrete_latent: Tensor):
        z_q = self.vq.quantize_encoding(discrete_latent)
        x_recon = self.decoder(z_q)
        return x_recon

    def forward(self, x: Tensor):
        z_e = self.encoder(x)
        z_q, e, vq_loss = self.vq(z_e)
        x_recon = self.decoder(z_q)

        return dict(
            encoder_output=z_e,
            discrete_latent=e,
            decoder_input=z_q,
            recon=x_recon,
            vq_loss=vq_loss,
        )


class VQVAE(pl.LightningModule):

    LatentType = Literal['encoder_output', 'discrete', 'decoder_input']

    def __init__(
        self,
        in_channels: int = 1,
        codebook_size: int = 256,         # K
        latent_dim: int = 8,  # 8,  # 16         # dimension of z
        commit_loss_weight: float = 0.25,  # 1.0,  # beta
        lr: float = 5e-4,
        activation: ActivationFunc = 'mish',
        width_scale: int = 16,  # 8,  # influences width of convolutional layers
        vector_quantizer_strength: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.commit_loss_weight = commit_loss_weight
        self.lr = lr
        self.activation = activation
        self.width_scale = width_scale
        self.vector_quantizer_strength = vector_quantizer_strength
        self.save_hyperparameters()

        activation_ctor = partial(
            activation_layer_ctor(activation),
            inplace=True,
        )

        self.model = VQVAEComponent(
            in_channels=in_channels,
            latent_dim=latent_dim,
            codebook_size=codebook_size,
            activation=activation_ctor,
            width_scale=width_scale,
            commitment_cost=commit_loss_weight,
        )

    @property
    def vq(self) -> VectorQuantizer:
        return self.model.vq

    def forward(self, x):
        output = self.model.forward(x)
        x_recon = output['recon']
        z_e = output['encoder_output']
        z_q = output['decoder_input']
        return x_recon, z_e, z_q

    def encode(self, x):
        """Discrete latent embedding e_x"""
        return self.model.encode(x)

    def encoder_net(self, x):  # TODO: remove
        """(continuous) encoder network output z_e_x i.e. before generating
        discrete embedding via the codebook"""
        return self.model.encoder(x)

    def decode(self, e_x):
        return self.model.decode(e_x)

    # TODO: write equivalent decoding method
    def embed(
        self,
        x: Tensor,
        latent_type: LatentType,
    ) -> Tensor:
        if latent_type == 'encoder_output':
            z_e = self.model.encoder(x)
            return z_e
        elif latent_type == 'discrete':
            e_x = self.encode(x)
            return e_x
        elif latent_type == 'decoder_input':
            output = self.model.forward(x)
            return output['decoder_input']
        else:
            raise ValueError(f'Invalid {latent_type=}')

    def _step(self, x):
        output = self.model(x)
        x_recon = output['recon']
        vq_loss = output['vq_loss']

        recon_loss = F.mse_loss(x_recon, x)

        loss = recon_loss + self.vector_quantizer_strength * vq_loss

        # not reporting loss_commit since it's identical to loss_vq
        metrics = {
            'loss': loss,
            'loss_recon': recon_loss,
            'loss_vq': vq_loss,
        }

        return loss, metrics

    def training_step(self, batch, _batch_idx):
        loss, metrics = self._step(batch)
        train_metrics = keymap('train/'.__add__, metrics)
        self.log_dict(train_metrics)
        return loss

    def validation_step(self, batch, _batch_idx):
        _loss, metrics = self._step(batch)
        self.log_dict({f"val_{k}": v for k, v in metrics.items()})
        return metrics

    def test_step(self, batch, _batch_idx):
        _loss, metrics = self._step(batch)
        self.log_dict({f"test_{k}": v for k, v in metrics.items()})
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
