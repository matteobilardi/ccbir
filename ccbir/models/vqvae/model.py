from einops.layers.torch import Rearrange
from ccbir.arch import PreActResBlock
from ccbir.util import ActivationFunc, activation_layer_ctor
from ccbir.models.vqvae.vq import ProductQuantizer, VectorQuantizer
from functools import cached_property, partial
import pytorch_lightning as pl
from torch import LongTensor, Tensor, nn
from typing import Callable, Literal, Union
import torch
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from toolz import keymap


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
        D = latent_dim

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
            nn.Conv2d(64, D, 1),
            Rearrange('b d h w -> b (h w) d'),
        )
        self.vq = ProductQuantizer(
            num_quantizers=8,
            dim=latent_dim,
            codebook_size=codebook_size,
            use_cosine_sim=True,
            accept_image_fmap=False,
            decay=0.99,
            commitment_weight=commitment_cost,
            threshold_ema_dead_code=0.05 * 8,
            inject_noise=True,
            inject_noise_sigma=1.0,
            inject_noise_rings=1,
            # negative_sample_weight=0.0,
            clean_cluster_size_update=True,
            wrap_ring_noise=True,
            noise_type='ring',
            limit_per_ring_expiry=None,
            # sample_codebook_temp=0.1,
        )
        self.decoder = nn.Sequential(
            Rearrange('b (h w) d -> b d h w', h=8, w=8),
            # scales 8x8 to 32x32
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Conv2d(D, 64, 3, 1, bias=False),
            nn.BatchNorm2d(64),
            activation(),
            nn.Conv2d(64, 64, 3, 1, bias=False),
            resblocks(4, 64, 64, activation=activation),
            nn.Conv2d(64, C, 1),
            nn.Tanh()
        )

    def encode(self, x: Tensor):
        z_e = self.encoder(x)
        _z_q, e, _loss, _metrics = self.vq(z_e)
        return e

    def decode(self, discrete_latent: Tensor):
        z_q = self.vq.quantize_encoding(discrete_latent)
        x_recon = self.decoder(z_q)
        return x_recon

    def forward(self, x: Tensor):
        z_e = self.encoder(x)
        z_q, e, vq_loss, vq_metrics = self.vq(z_e)
        x_recon = self.decoder(z_q)

        return dict(
            encoder_output=z_e,
            discrete_latent=e,
            decoder_input=z_q,
            recon=x_recon,
            vq_loss=vq_loss,
            vq_metrics=vq_metrics,
        )


class VQVAE(pl.LightningModule):

    LatentType = Literal['encoder_output', 'discrete', 'decoder_input']

    def __init__(
        self,
        in_channels: int = 1,
        codebook_size: int = 256,         # K
        latent_dim: int = 8,  # 8,  # 16         # dimension of z
        commit_loss_weight: float = 8.0,  # 1.0,  # beta
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
        vq_metrics = output['vq_metrics']

        recon_loss = F.mse_loss(x_recon, x)

        loss = recon_loss + self.vector_quantizer_strength * vq_loss
        metrics = {
            'loss': loss,
            'loss_recon': recon_loss,
            'loss_vq': vq_loss,
            **vq_metrics,
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
