from ccbir.configuration import config
config.pythonpath_fix()
from ccbir.data.dataset import InterleaveDataset
from ccbir.data.morphomnist.datamodule import MorphoMNISTDataModule
from ccbir.data.morphomnist.dataset import FracturedMorphoMNIST, LocalPerturbationsMorphoMNIST, MorphoMNIST, SwollenMorphoMNIST
from ccbir.pytorch_vqvae.modules import ResBlock, VectorQuantizedVAE, weights_init
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, Literal, Optional, Type
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


class VQVAEComponent(VectorQuantizedVAE):
    """Wrapper of VQ-VAE in https://github.com/ritheshkumar95/pytorch-vqvae
    to allow for easy modification of high-level architecture"""

    def __init__(self, in_channels, latent_dim, codebook_size):
        super().__init__(in_channels, latent_dim, codebook_size)
        # overwrite encoder/decoder from parent class
        self.encoder = self._mk_encoder(in_channels, latent_dim)
        self.decoder = self._mk_decoder(in_channels, latent_dim)

        self.apply(weights_init)

    @classmethod
    def _mk_activation(cls):
        return nn.ReLU(inplace=True)

    @classmethod
    def _mk_encoder(cls, in_channels, latent_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels, 4 * latent_dim, 4, 2),
            nn.BatchNorm2d(4 * latent_dim),
            cls._mk_activation(),
            nn.Conv2d(4 * latent_dim, 2 * latent_dim, 3, 1),
            nn.BatchNorm2d(2 * latent_dim),
            cls._mk_activation(),
            nn.Conv2d(2 * latent_dim, latent_dim, 4, 1),
            ResBlock(latent_dim, activation=cls._mk_activation),
            ResBlock(latent_dim, activation=cls._mk_activation),
        )

    @classmethod
    def _mk_decoder(cls, in_channels, latent_dim):
        return nn.Sequential(
            ResBlock(latent_dim, activation=cls._mk_activation),
            ResBlock(latent_dim, activation=cls._mk_activation),
            cls._mk_activation(),
            nn.ConvTranspose2d(latent_dim, 2 * latent_dim, 4, 1),
            nn.BatchNorm2d(2 * latent_dim),
            cls._mk_activation(),
            nn.ConvTranspose2d(2 * latent_dim, 4 * latent_dim, 3, 1),
            nn.BatchNorm2d(4 * latent_dim),
            cls._mk_activation(),
            nn.ConvTranspose2d(4 * latent_dim, in_channels, 4, 2),
            nn.Tanh()
        )


class VQVAE(pl.LightningModule):

    def __init__(
        self,
        in_channels: int = 1,
        codebook_size: int = 1024,         # K
        latent_dim: int = 8,  # 16         # dimension of z
        commit_loss_weight: float = 0.25,  # 1.0,  # beta
        lr: float = 2e-4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.commit_loss_weight = commit_loss_weight
        self.lr = lr
        self.save_hyperparameters()
        self.model = VQVAEComponent(
            in_channels=in_channels,
            latent_dim=latent_dim,
            codebook_size=codebook_size,
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
        latent_type: Literal['encoder_output', 'discrete', 'decoder_input'],
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


class VQVAEDataset(InterleaveDataset):

    def __init__(
        self,
        train: bool,
        transform=None,
    ) -> None:
        kwargs = dict(
            train=train,
            transform=transform,
        )

        super().__init__(datasets=[
            SwollenMorphoMNIST(**kwargs),
            FracturedMorphoMNIST(**kwargs),
        ])


class VQVAEMorphoMNISTDataModule(MorphoMNISTDataModule):
    def __init__(
        self,
        *,
        dataset_type: Type[MorphoMNIST] = VQVAEDataset,
        batch_size: int = 64,
        pin_memory: bool = True,
    ):
        super().__init__(
            dataset_ctor=dataset_type,
            batch_size=batch_size,
            pin_memory=pin_memory,
            transform=transforms.Compose([
                transforms.Lambda(lambda item: item['image']),
                # enforce range [-1, 1] in line with tanh NN output
                # see https://discuss.pytorch.org/t/understanding-transform-normalize/21730/2
                transforms.Normalize(mean=0.5, std=0.5)
            ]),
        )


def main():
    from pytorch_lightning.utilities.cli import LightningCLI
    from pytorch_lightning.callbacks import ModelCheckpoint

    cli = LightningCLI(
        VQVAE,
        VQVAEMorphoMNISTDataModule,
        save_config_overwrite=True,
        run=False,  # deactivate automatic fitting
        trainer_defaults=dict(
            callbacks=[
                ModelCheckpoint(
                    monitor='val_loss',
                    # dirpath=str(config.checkpoints_path_for_model(
                    #    model_type=VQVAE
                    # )),
                    filename='vqvae-morphomnist-{epoch:03d}-{val_loss:.7f}',
                    save_top_k=3,
                    save_last=True,
                )
            ],
            max_epochs=1000,
            gpus=1,
        ),
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, ckpt_path="best", datamodule=cli.datamodule)


if __name__ == '__main__':
    main()
