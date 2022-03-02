from ccbir.configuration import config
config.pythonpath_fix()
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from torchvision import transforms
from ccbir.pytorch_vqvae.modules import VectorQuantizedVAE
from ccbir.data.morphomnist.datamodule import MorphoMNISTDataModule
from ccbir.data.morphomnist.dataset import LocalPerturbationsMorphoMNIST, MorphoMNIST
from typing import Optional, Type


class VQVAE(pl.LightningModule):
    """Wrapper of VQ-VAE in https://github.com/ritheshkumar95/pytorch-vqvae
    Hyperparameters as in https://github.com/praeclarumjj3/VQ-VAE-on-MNIST
    """

    def __init__(
        self,
        in_channels: int = 1,
        codebook_size: int = 512,          # K
        latent_dim: int = 16,             # dimension of z
        commit_loss_weight: float = 1.0,  # beta
        lr: float = 2e-4,
    ):
        super().__init__()
        self.model = VectorQuantizedVAE(
            in_channels, latent_dim, K=codebook_size
        )
        self.commit_loss_weight = commit_loss_weight
        self.lr = lr

    def forward(self, x):
        return self.model.forward(x)

    def encode(self, x):
        """Returns discrete latent embedding e_x"""
        return self.model.encode(x)

    def encoder_net(self, x):
        """Returns (continuous) encoder network output z_e_x i.e. before 
        generating discrete embedding via the codebook"""
        return self.model.encoder(x)

    def decode(self, latents):
        return self.model.decode(latents)

    def _prep_batch(self, batch):
        x = batch['image']
        return x

    def _step(self, batch):
        x = self._prep_batch(batch)
        x_tilde, z_e_x, z_q_x = self.model(x)

        # reconstruction loss
        loss_recon = F.mse_loss(x_tilde, x)
        # vector quantization loss
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # commitment loss
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recon + loss_vq + self.commit_loss_weight * loss_commit

        # not reporting loss_vq since it's identical to loss_commit
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


class VQVAEMorphoMNISTDataModule(MorphoMNISTDataModule):
    def __init__(
        self,
        *,
        dataset_type: Type[MorphoMNIST] = LocalPerturbationsMorphoMNIST,
        train_batch_size: int = 64,
        test_batch_size: int = 64,
        pin_memory: bool = True
    ):
        super().__init__(
            dataset_ctor=dataset_type,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            pin_memory=pin_memory,
            transform=transforms.Compose([
                transforms.ToTensor(),
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
                )
            ],
            max_epochs=100,
            gpus=1,
        ),
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, ckpt_path="best", datamodule=cli.datamodule)


if __name__ == '__main__':
    main()
