from ccbir.arch import PreActResBlock
from ccbir.util import ActivationFunc, activation_layer_ctor
from functools import partial
import pytorch_lightning as pl
from torch import Tensor, nn
from typing import Callable, Literal
import torch
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Class adapted from https://github.com/zalandoresearch/pytorch-vq-vae"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
    ):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            num_embeddings=self._num_embeddings,
            embedding_dim=self._embedding_dim,
        )
        self._embedding.weight.data.uniform_(
            (-1.0 / self._num_embeddings),
            (1.0 / self._num_embeddings),
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            - 2 * flat_input @ self._embedding.weight.t()
            + torch.sum(self._embedding.weight ** 2, dim=1)
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            size=(encoding_indices.shape[0], self._num_embeddings),
            device=inputs.device,
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = (encodings @ self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()

        """
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            - torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )
        """

        discrete_latent = encoding_indices.view(*input_shape[:-1])
        # convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return discrete_latent, quantized, loss

    def quantize_encoding(
        self,
        encoding: Tensor
    ) -> Tensor:
        e_flat = encoding.flatten(start_dim=1)
        encodings = torch.zeros(
            size=(e_flat.shape[0], self._num_embeddings),
            device=encoding.device,
        )
        encodings.scatter_(1, e_flat, 1)
        z_q_flat = (encodings @ self._embedding.weight)
        z_q = z_q_flat.view((*encoding.shape, -1))
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


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
            nn.Conv2d(C, 4 * W * L, 4, 2, bias=False),
            nn.BatchNorm2d(4 * W * L),
            activation(),
            nn.Conv2d(4 * W * L, 2 * W * L, 3, 1, bias=False),
            nn.BatchNorm2d(2 * W * L),
            activation(),
            nn.Conv2d(2 * W * L, L, 4, 1),
            resblocks(2, L, L, activation=activation),
        )
        self.vq = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = nn.Sequential(
            resblocks(2, L, L, activation=activation),
            activation(),
            nn.ConvTranspose2d(L, 2 * W * L, 4, 1, bias=False),
            nn.BatchNorm2d(2 * W * L),
            activation(),
            nn.ConvTranspose2d(2 * W * L, 4 * W * L, 3, 1, bias=False),
            nn.BatchNorm2d(4 * W * L),
            activation(),
            nn.ConvTranspose2d(4 * W * L, C, 4, 2),
            nn.Tanh()
        )

    def encode(self, x: Tensor):
        z_e = self.encoder(x)
        e, _z_q = self.vq(z_e)
        return e

    def decode(self, discrete_latent: Tensor):
        z_q = self.vq.quantize_encoding(discrete_latent)
        x_recon = self.decoder(z_q)
        return x_recon

    def forward(self, x: Tensor):
        z_e = self.encoder(x)
        e, z_q, vq_loss = self.vq(z_e)
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
        codebook_size: int = 1024,         # K
        latent_dim: int = 2,  # 8,  # 16         # dimension of z
        commit_loss_weight: float = 0.25,  # 1.0,  # beta
        lr: float = 2e-4,
        activation: ActivationFunc = 'mish',
        width_scale: int = 16,  # 8,  # influences width of convolutional layers
        vector_quantizer_strength: float = 0.5,
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
            z_e = self.model.encoder(x)
            return z_e
        elif latent_type == 'discrete':
            e_x = self.encode(x)
            return e_x
        elif latent_type == 'decoder_input':
            output = self(x)
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
