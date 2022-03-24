from ccbir.configuration import config
config.pythonpath_fix()
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


class SimpleDeepTwinNetComponent(nn.Module):
    """Twin network for DAG: X -> Y <- Z that allows sampling from
    P(Y, Y* | X, X*, Z)"""

    def __init__(
        self,
        treatment_dim: int,
        confounders_dim: int,
        outcome_noise_dim: int,
        outcome_size: torch.Size,
    ):
        super().__init__()
        input_dim = treatment_dim + confounders_dim + outcome_noise_dim
        def Activation(): return nn.LeakyReLU(inplace=True)
        """
        # NOTE: for now using simple fully connected architecture but
        # convolutions might end up being necessary
        base_dim =
        self.network = nn.Sequential(
            nn.LazyLinear(input_dim, 256),
            nn.LazyLinear(256, 512),
            nn.LazyLinear(512, outcome_size.numel())
            nn.Unflatten(outcome_size.numel(), outcome_size),
            nn.BatchNorm2d(),
        )
        """

        # NOTE: this architecture output need to have the same shape as the the
        # vq-vae encoder output
        base_dim = outcome_size[0]
        self.network = nn.Sequential(
            nn.LazyLinear(base_dim * 8),
            Activation(),
            nn.Unflatten(1, (-1, 1, 1)),
            nn.LazyConvTranspose2d(base_dim * 4, 3, 1),
            nn.LazyBatchNorm2d(),
            Activation(),
            nn.LazyConvTranspose2d(base_dim * 2, 3, 1),
            nn.LazyBatchNorm2d(),
            Activation(),
            nn.LazyConvTranspose2d(base_dim, 3, 1),
        )

        # force lazy init
        dummy_output = self.network(torch.rand(64, input_dim))
        assert dummy_output.shape[1:] == outcome_size

    def forward(
        self,
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        outcome_noise: Tensor
    ) -> Tuple[Tensor, Tensor]:

        factual_input = torch.cat(
            (factual_treatment, confounders, outcome_noise),
            dim=1,
        )

        counterfactual_input = torch.cat(
            (counterfactual_treatment, confounders, outcome_noise),
            dim=1,
        )

        # NOTE: for now, just have a single network i.e. assume weight sharing
        # between the factual and counterfactual branches of the twin network
        factual_outcome = self.network(factual_input)
        counterfactual_outcome = self.network(counterfactual_input)

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
            outcome_size=outcome_size,
        )
        self.outcome_noise_dim = outcome_noise_dim
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        return self.twin_net(**x)

    def _step(self, batch):
        X, y = batch

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


class PSFTwinNetDataset:
    # NOTE: For now relying on binary treatments: either swelling or
    # fracture
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
    ):
        self.embed_image = embed_image
        self.psf_dataset = CombinedDataset(dict(
            plain=PlainMorphoMNIST(train=train, transform=transform),
            swollen=SwollenMorphoMNIST(train=train, transform=transform),
            fractured=FracturedMorphoMNIST(train=train, transform=transform),
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
        # NOTE: for now, no metric normalisation is occurring
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

        return x, y


class PSFTwinNet(SimpleDeepTwinNet):
    def __init__(
        self,
        outcome_size: torch.Size,
        lr: float = 0.001,
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


def vqvae_embed_image(vqvae: VQVAE, image: Tensor):
    # TODO: find better place for this funcion
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
                ),
            ],
            max_epochs=2,
            gpus=1,
        ),
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    main()

"""
class DeepTwinNet(BaseTwinNet):
    def __init__(self):
        super().__init__()

    def forward(self, observed_factual_intensity, intervention_counterfactual_intensity):
        noise_thickness = sample_gaussian()
        sampled_factual_thickness = f_thickness(
            observed_factual_intensity, noise_thickness)
        sampled_counterfactual_thickness = f_thickness(
            do_counterfactual_intensity, noise_thickness)

        noise_image = sample_gaussian())
        sampled_factual_image = f_image(
            observed_factual_intensity, sampled_factual_thickness, noise_image)
        sampled_counterfactual_image = f_image(
            intervention_counterfactual_intensity, sampled_counterfactual_thickness, noise_image)


        output = {
            'factual': [observed_factual_intensity, sampled_factual_thickness, sampled_factual_image]
            'counterfactual': [intervention_counterfactual_intensity, sampled_counterfactual_thickness, sampled_counterfactual_image]
        }

        return output

    def sample(self, observed_intensity, observed_thickness, observed_image, intervention_intensity):
        while True:
            output = self.forward(observed_intensity, intervention_intensity)
            _, sampled_thickness, sampled_image =  output['factual']
            if sampled_thickness.isclose(observed_thickness) and sampled_image.isclose(observed_image):
                _, _, counterfactual_image = output['counterfactual']
                return counterfactual_image"
"""
