from ccbir.pytorch_vqvae.modules import (
    ResBlock,
    VectorQuantizedVAE,
    weights_init,
)
from ccbir.util import ActivationFunc, activation_layer_ctor, maybe_unbatched_apply
from functools import partial
import pytorch_lightning as pl
from torch import Tensor, nn
from typing import Callable, Literal
import torch
import torch.nn.functional as F


class VQVAEComponent(VectorQuantizedVAE):
    """Wrapper of VQ-VAE in https://github.com/ritheshkumar95/pytorch-vqvae
    to allow for easy modification of high-level architecture"""

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        codebook_size: int,
        activation: Callable[..., nn.Module],
        width_scale: int,
    ):
        super().__init__(in_channels, latent_dim, codebook_size)
        # overwrite encoder/decoder from parent class
        self.encoder = self._mk_encoder(
            in_channels, latent_dim, activation, width_scale
        )
        self.decoder = self._mk_decoder(
            in_channels, latent_dim, activation, width_scale
        )

        self.apply(weights_init)

    @classmethod
    def _mk_encoder(cls, in_channels, latent_dim, activation, width_scale):
        C = in_channels
        W = width_scale
        L = latent_dim
        return nn.Sequential(
            nn.Conv2d(C, 4 * W * L, 4, 2),
            nn.BatchNorm2d(4 * W * L),
            activation(),
            nn.Conv2d(4 * W * L, 2 * W * L, 3, 1),
            nn.BatchNorm2d(2 * W * L),
            activation(),
            nn.Conv2d(2 * W * L, L, 4, 1),
            ResBlock(L, activation=activation),
            ResBlock(L, activation=activation),
        )

    @classmethod
    def _mk_decoder(cls, in_channels, latent_dim, activation, width_scale):
        C = in_channels
        W = width_scale
        L = latent_dim
        return nn.Sequential(
            ResBlock(L, activation=activation),
            ResBlock(L, activation=activation),
            activation(),
            nn.ConvTranspose2d(L, 2 * W * L, 4, 1),
            nn.BatchNorm2d(2 * W * L),
            activation(),
            nn.ConvTranspose2d(2 * W * L, 4 * W * L, 3, 1),
            nn.BatchNorm2d(4 * W * L),
            activation(),
            nn.ConvTranspose2d(4 * W * L, C, 4, 2),
            nn.Tanh()
        )


class VQVAE(pl.LightningModule):

    LatentType = Literal['encoder_output', 'discrete', 'decoder_input']

    def __init__(
        self,
        in_channels: int = 1,
        codebook_size: int = 1024,         # K
        latent_dim: int = 2,  # 8,  # 16         # dimension of z
        commit_loss_weight: float = 0.25,  # 1.0,  # beta
        lr: float = 2e-4,
        activation: ActivationFunc = 'mish',
        width_scale: int = 16  # 8,  # influences width of convolutional layers
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.commit_loss_weight = commit_loss_weight
        self.lr = lr
        self.activation = activation
        self.width_scale = width_scale
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
        )

    def forward(self, x):
        return self.model.forward(x)

    def encode(self, x):
        """Discrete latent embedding e_x"""
        return self.model.encode(x)

    def encoder_net(self, x):
        """(continuous) encoder network output z_e_x i.e. before generating
        discrete embedding via the codebook"""
        return self.model.encoder(x)

    def decode(self, e_x):
        return self.model.decode(e_x)

    def embed(
        self,
        x: Tensor,
        latent_type: LatentType,
    ) -> Tensor:
        if latent_type == 'encoder_output':
            z_e_x = self.model.encoder(x)
            return z_e_x
        elif latent_type == 'discrete':
            e_x = self.encode(x)
            return e_x
        elif latent_type == 'decoder_input':
            _x_hat, _z_e_x, z_q_x = self(x)
            return z_q_x
        else:
            raise ValueError(f'Invalid {latent_type=}')

    def _step(self, x):
        x_tilde, z_e_x, z_q_x = self.model(x)

        # reconstruction loss
        loss_recon = F.mse_loss(x_tilde, x)
        # vector quantization loss
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # commitment loss
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recon + loss_vq + self.commit_loss_weight * loss_commit

        # not reporting loss_commit since it's identical to loss_vq
        metrics = {
            'loss': loss,
            'loss_recon': loss_recon,
            'loss_vq': loss_vq
        }

        return loss, metrics

    def training_step(self, batch, _batch_idx):
        loss, _metrics = self._step(batch)
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
