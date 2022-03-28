from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from typing import Callable, Dict, Literal, Mapping, Optional, Tuple
from torchvision import transforms
from torch import Tensor, embedding, nn
from functools import partial
from ccbir.models.vqvae import VQVAE
from ccbir.models.model import load_best_model
from ccbir.data.morphomnist.dataset import (
    FracturedMorphoMNIST,
    PlainMorphoMNIST,
    SwollenMorphoMNIST
)
from ccbir.data.morphomnist.datamodule import MorphoMNISTDataModule
from ccbir.data.dataset import CombinedDataset
from ccbir.configuration import config
config.pythonpath_fix()


class SimpleDeepTwinNetComponent(nn.Module):
    """Twin network for DAG: X -> Y <- Z that allows sampling from
    P(Y, Y* | X, X*, Z)"""

    def __init__(
        self,
        treatment_dim: int,
        confounders_dim: int,
        outcome_noise_dim: int,
        outcome_shape: torch.Size,
        noise_inject_mode: Literal['concat', 'add', 'multiply'] = 'concat',
    ):
        super().__init__()

        # TODO: For clarity consider not using lazy layers once the architecture
        # is stabilised

        def Activation():
            return nn.SiLU(inplace=True)
            #return nn.LeakyReLU(inplace=True)

        # NOTE: this architecture output needs to have the same shape as the
        # vq-vae encoder output
        base_dim = outcome_shape[0]

        """
        self.net = nn.Sequential(
            nn.LazyLinear(base_dim * 8),
            Activation(),
            nn.Unflatten(1, (-1, 1, 1)),
            nn.LazyConvTranspose2d(base_dim * 4, 3, 1),
            # nn.LazyBatchNorm2d(),
            Activation(),
            nn.LazyConvTranspose2d(base_dim * 2, 3, 1),
            # nn.LazyBatchNorm2d(),
            Activation(),
            nn.LazyConvTranspose2d(base_dim, 3, 1),
        )
        """

        # conv transpose without batchnorm
        """
        self.net = nn.Sequential(
            nn.Unflatten(1, (-1, 1, 1)),
            nn.LazyConvTranspose2d(128, 2),
            Activation(),
            nn.LazyConvTranspose2d(128, 2),
            Activation(),
            nn.LazyConvTranspose2d(64, 2),
            Activation(),
            nn.LazyConvTranspose2d(64, 2),
            Activation(),
            nn.LazyConvTranspose2d(32, 2),
            Activation(),
            nn.LazyConvTranspose2d(16, 2),
        )
        """

        """
        # a very deep convtranspose net without batchnorm (better val loss than previous)
        self.net = nn.Sequential(
            nn.Unflatten(1, (-1, 1, 1)),
            nn.LazyConvTranspose2d(128, 2),
            Activation(),
            nn.LazyConvTranspose2d(128, 2),
            Activation(),
            nn.LazyConvTranspose2d(64, 2),
            Activation(),
            nn.LazyConvTranspose2d(64, 2),
            Activation(),
            nn.LazyConvTranspose2d(32, 2),
            Activation(),
            nn.LazyConvTranspose2d(16, 2),
            Activation(),
            nn.LazyConv2d(16, 3, padding=1),
            Activation(),
            nn.LazyConv2d(16, 3, padding=1),
            Activation(),
            nn.LazyConv2d(16, 3, padding=1),
        )
        """

        """
        self.net = nn.Sequential(
            nn.Unflatten(1, (-1, 1, 1)),
            nn.LazyConvTranspose2d(128, 3),
            nn.LazyBatchNorm2d(),
            Activation(),
            nn.LazyConvTranspose2d(128, 3),
            nn.LazyBatchNorm2d(),
            Activation(),
            nn.LazyConvTranspose2d(128, 3),
            nn.LazyBatchNorm2d(),
            Activation(),
            nn.LazyConvTranspose2d(128, 3),
            nn.LazyBatchNorm2d(),
            Activation(),
            nn.LazyConv2d(64, 3),
            nn.LazyBatchNorm2d(),
            Activation(),
            nn.LazyConv2d(16, 1),
            nn.LazyBatchNorm2d(),
            Activation(),
            nn.LazyConv2d(16, 1),
        )
        """
        
        # best so far
        """
        self.net = nn.Sequential(
            nn.Unflatten(1, (-1, 1, 1)),
            nn.LazyConvTranspose2d(128, 3),
            Activation(),
            nn.LazyConvTranspose2d(128, 3),
            Activation(),
            nn.LazyConvTranspose2d(128, 3),
            Activation(),
            nn.LazyConvTranspose2d(64, 3),
            Activation(),
            nn.LazyConv2d(64, 3),
            Activation(),
            nn.LazyConv2d(16, 1),
            Activation(),
            nn.LazyConv2d(4, 1),
        )
        """

        # a fully linear attempt
        self.net = nn.Sequential(
            nn.LazyLinear(1024),
            Activation(),
            nn.LazyLinear(1024),
            Activation(),
            nn.LazyLinear(1024),
            Activation(),
            nn.LazyLinear(512),
            Activation(),
            nn.LazyLinear(196),
            nn.Unflatten(1, (4, 7, 7))
        )

        self.data_dim = treatment_dim + confounders_dim

        if noise_inject_mode == 'concat':
            self.input_dim = self.data_dim * 2
            self.inject_noise = (
                lambda data, noise: torch.concat((data, noise), dim=1)
            )
        elif noise_inject_mode == 'add':
            self.input_dim = self.data_dim
            self.inject_noise = torch.add
        elif noise_inject_mode == 'multiply':
            self.input_dim = self.data_dim
            self.inject_noise = torch.mul
        else:
            raise TypeError(f"Invalid {noise_inject_mode=}")

        # NOTE: to inject noise via multiplication/addition, we currently force
        # the reparametrised noise to have the same number of dimensions as the
        # data (i.e. [treatement, confounders]) regardless of the value of
        # outcome_noise_dim, which is just the size of the input to the
        # reparametrize_noise_net. For pure convenience, currentll this is the
        # case even when we inject noise by concatenation
        self.reparametrize_noise_net = nn.Sequential(
            # For now just a linear layer, but might need something with more
            # representation power
            nn.LazyLinear(self.data_dim),
        )

        # force lazy init
        dummy_noise_output = self.reparametrize_noise_net(
            torch.rand(1, outcome_noise_dim)
        )
        dummy_output = self.net(torch.rand(1, self.input_dim))

        assert dummy_noise_output.shape[1] == self.data_dim
        assert dummy_output.shape[1:] == outcome_shape

    def forward(
        self,
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        outcome_noise: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        factual_data = torch.cat(
            (factual_treatment, confounders),
            dim=1
        )
        counterfactual_data = torch.cat(
            (counterfactual_treatment, confounders),
            dim=1,
        )

        noise = self.reparametrize_noise_net(outcome_noise)
        factual_input = self.inject_noise(factual_data, noise)
        counterfactual_input = self.inject_noise(counterfactual_data, noise)

        # for now, just have a single network so as to implicitly enforce
        # weight sharing between the factual and counterfactual branches of the
        # twin network
        factual_outcome = self.net(factual_input)
        counterfactual_outcome = self.net(counterfactual_input)

        return factual_outcome, counterfactual_outcome


class SimpleDeepTwinNet(pl.LightningModule):
    def __init__(
        self,
        treatment_dim: int,
        confounders_dim: int,
        outcome_noise_dim: int,
        outcome_size: torch.Size,
        lr: float,
    ):
        super().__init__()
        self.twin_net = SimpleDeepTwinNetComponent(
            treatment_dim=treatment_dim,
            confounders_dim=confounders_dim,
            outcome_noise_dim=outcome_noise_dim,
            outcome_shape=outcome_size,
        )
        self.outcome_noise_dim = outcome_noise_dim
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        return self.twin_net(**x)

    def _step(self, batch):
        X, y, _ = batch

        factual_outcome_hat, counterfactual_outcome_hat = self(X)

        factual_outcome = y['factual_outcome']
        counterfactual_outcome = y['counterfactual_outcome']

        # NOTE: loss is calculated on the continous encoder outputs
        # before vector quantization into a discrete representation
        factual_loss = F.mse_loss(factual_outcome_hat, factual_outcome)
        counterfactual_loss = F.mse_loss(
            counterfactual_outcome_hat,
            counterfactual_outcome,
        )

        loss = factual_loss + counterfactual_loss

        return loss, dict(
            loss=loss,
            factual_loss=factual_loss.item(),
            counterfactual_loss=counterfactual_loss.item(),
        )

    def training_step(self, batch, _batch_idx):
        loss, _metrics = self._step(batch)
        return loss

    def validation_step(self, batch, _batch_idx):
        _loss, metrics = self._step(batch)
        self.log_dict({f"val_{k}": v for k, v in metrics.items()})

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.twin_net.parameters(), lr=self.lr)


class PSFTwinNetDataset(Dataset):
    # NOTE: For now relying only binary treatments: either swelling or
    # fracture but might want to expand in the future
    treatments = ["swell", "fracture"]
    _treatement_to_index = {t: idx for idx, t in enumerate(treatments)}

    # TODO: probably better move metrics and labels somewhere more appropriate
    metrics = ['area', 'length', 'thickness', 'slant', 'width', 'height']
    labels = list(range(10))
    outcome_noise_dim: int = 32

    def __init__(
        self,
        *,
        embed_image: Callable[[Tensor], Tensor],
        train: bool,
        transform=None,
        normalize_metrics: bool = True,
    ):
        super().__init__()
        self.embed_image = embed_image

        kwargs = dict(
            train=train,
            transform=transform,
            normalize_metrics=normalize_metrics
        )
        self.psf_dataset = CombinedDataset(dict(
            plain=PlainMorphoMNIST(**kwargs),
            swollen=SwollenMorphoMNIST(**kwargs),
            fractured=FracturedMorphoMNIST(**kwargs),
        ))

        self.outcome_noise = torch.randn(
            (len(self.psf_dataset), self.outcome_noise_dim),
        )

    def treatment_to_index(self, treatment: str) -> int:
        try:
            return self._treatement_to_index[treatment]
        except KeyError:
            raise ValueError(
                f"{treatment=} is not a valid treatment: must be one of "
                f"{', '.join(self.treatments)}"
            )

    def one_hot_treatment(self, treatment: str) -> Tensor:
        return F.one_hot(
            torch.as_tensor(self.treatment_to_index(treatment)),
            num_classes=len(self.treatments),
        ).float()

    def one_hot_label(self, label: Tensor) -> Tensor:
        return F.one_hot(
            label.long(),
            num_classes=len(self.labels)
        ).float()

    def metrics_vector(self, metrics: Dict[str, Tensor]) -> Tensor:
        # ensure consitent order of metrics
        sorted_metrics = torch.tensor([
            metrics[metric] for metric in sorted(metrics.keys())
        ])

        return sorted_metrics

    def __len__(self):
        return len(self.psf_dataset)

    def __getitem__(self, index):
        psf_item = self.psf_dataset[index]
        outcome_noise = self.outcome_noise[index]

        swell = self.one_hot_treatment('swell')
        fracture = self.one_hot_treatment('fracture')
        label = self.one_hot_label(psf_item['plain']['label'])
        metrics = self.metrics_vector(psf_item['plain']['metrics'])
        label_and_metrics = torch.cat((label, metrics))

        x = dict(
            factual_treatment=swell,
            counterfactual_treatment=fracture,
            confounders=label_and_metrics,
            outcome_noise=outcome_noise,
        )

        swollen_z = self.embed_image(psf_item['swollen']['image'])
        fractured_z = self.embed_image(psf_item['fractured']['image'])

        y = dict(
            factual_outcome=swollen_z,
            counterfactual_outcome=fractured_z,
        )

        # adding psf_item for ease of debugging but not necessary for training
        return x, y, psf_item


class PSFTwinNet(SimpleDeepTwinNet):
    def __init__(
        self,
        outcome_size: torch.Size,
        lr: float = 0.001  # 0.0005,
    ):
        super().__init__(
            outcome_size=outcome_size,
            treatment_dim=len(PSFTwinNetDataset.treatments),
            confounders_dim=(
                len(PSFTwinNetDataset.labels) + len(PSFTwinNetDataset.metrics)
            ),
            outcome_noise_dim=PSFTwinNetDataset.outcome_noise_dim,
            lr=lr,
        )
        self.save_hyperparameters()


class PSFTwinNetDataModule(MorphoMNISTDataModule):
    # TODO: better class name
    def __init__(
        self,
        *,
        embed_image: Callable[[Tensor], Tensor],
        train_batch_size: int = 64,
        test_batch_size: int = 64,
    ):

        super().__init__(
            dataset_ctor=partial(
                PSFTwinNetDataset,
                embed_image=embed_image
            ),
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            pin_memory=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]),
        )

# TODO: find better place for this funcion
def vqvae_embed_image(vqvae: VQVAE, image: Tensor):
    with torch.no_grad():
        # the vqvae encoder expects a 4 dimensional tensor to support
        # batching hence unsqueeze
        return vqvae.encoder_net(image.unsqueeze(0)).squeeze(0)


def main():
    from pytorch_lightning.utilities.cli import LightningCLI
    from pytorch_lightning.callbacks import ModelCheckpoint

    vqvae = load_best_model(VQVAE)
    embedding_size = vqvae.encoder_net(torch.rand((64, 1, 28, 28))).shape[1:]
    print(f"{embedding_size}")

    # FIXME: initialsiation, traning, saving and storing a bit hacky currently
    # Functions signatures needed by LightningCLI so can't use functools.partial
    def model() -> PSFTwinNet:
        return PSFTwinNet(outcome_size=embedding_size)

    def datamodule() -> PSFTwinNetDataModule:
        return PSFTwinNetDataModule(
            embed_image=partial(vqvae_embed_image, vqvae)
        )

    cli = LightningCLI(
        model_class=model,
        datamodule_class=datamodule,
        save_config_overwrite=True,
        run=False,  # deactivate automatic fitting
        trainer_defaults=dict(
            callbacks=[
                ModelCheckpoint(
                    monitor='val_loss',
                    filename='twinnet-{epoch:03d}-{val_loss:.7f}',
                    save_top_k=3,
                    save_last=True,
                ),
            ],
            max_epochs=2000,
            gpus=1,
        ),
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
