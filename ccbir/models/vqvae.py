import pytorch_lightning as pl
import torch
from ccbir.pytorch_vqvae.modules import VectorQuantizedVAE
import torch.nn.functional as F


class VQVAE(pl.LightningModule):
    """Wrapper of VQ-VAE in https://github.com/ritheshkumar95/pytorch-vqvae
    Hyperparameters as in https://github.com/praeclarumjj3/VQ-VAE-on-MNIST
    """

    def __init__(
        self,
        in_channels: int = 1,
        codebook_size: int = 32,          # K
        latent_dim: int = 256,            # dimension of z
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


if __name__ == '__main__':
    # TODO train
    ...
