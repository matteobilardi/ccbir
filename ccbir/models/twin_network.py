from more_itertools import all_equal, first
from sklearn import metrics
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from typing import Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union
from torchvision import transforms
from torch import Tensor, embedding, nn
from functools import partial, reduce
from ccbir.pytorch_vqvae.modules import ResBlock
from ccbir.tranforms import DictTransform
from ccbir.util import ActivationFunc, activation_layer_ctor, maybe_unbatched_apply, tune_lr
from ccbir.models.vqvae import VQVAE
from ccbir.models.model import load_best_model
from ccbir.data.morphomnist.dataset import (
    FracturedMorphoMNIST,
    PlainMorphoMNIST,
    SwollenMorphoMNIST
)
from ccbir.data.morphomnist.datamodule import MorphoMNISTDataModule
from ccbir.data.dataset import ZipDataset
from torch.distributions import Normal


class SimpleDeepTwinNetComponent(nn.Module):
    """Twin network for DAG: X -> Y <- Z that allows sampling from
    P(Y, Y* | X, X*, Z)"""

    def __init__(
        self,
        treatment_dim: int,
        confounders_dim: int,
        outcome_noise_dim: int,
        outcome_shape: torch.Size,
        noise_inject_mode: Literal['concat', 'add', 'multiply'],
        use_combine_net: bool,
        weight_sharing: bool,
        activation: Callable[..., nn.Module],
        batch_norm: bool,
    ):
        super().__init__()
        self.use_combine_net = use_combine_net
        self.data_dim = treatment_dim + confounders_dim
        self.weight_sharing = weight_sharing

        if noise_inject_mode == 'concat':
            self.input_dim = self.data_dim * 2
            self.inject_noise = (
                lambda data, noise: torch.cat((data, noise), dim=-1)
            )
        elif noise_inject_mode == 'add':
            self.input_dim = self.data_dim
            self.inject_noise = torch.add
        elif noise_inject_mode == 'multiply':
            self.input_dim = self.data_dim
            self.inject_noise = torch.mul
        else:
            raise ValueError(f"Invalid {noise_inject_mode=}")

        # TODO: For clarity consider not using lazy layers once the architecture
        # is stabilised
        def linear_layer(out_features: int, batch_norm: bool = batch_norm):
            layers = []
            layers.append(nn.LazyLinear(out_features=out_features))
            if batch_norm:
                layers.append(nn.LazyBatchNorm1d())

            return nn.Sequential(*layers)

        self.combine_treatment_confounders_net = nn.Sequential(
            linear_layer(self.data_dim),
            activation(),
            linear_layer(self.data_dim),
        ) if use_combine_net else None

        # to inject noise via multiplication/addition, we currently force
        # the reparametrised noise to have the same number of dimensions as the
        # data (i.e. [treatement, confounders]) regardless of the value of
        # outcome_noise_dim, which is just the size of the input to the
        # reparametrize_noise_net. For pure convenience, currently this is the
        # case even when we inject noise by concatenation
        self.reparametrize_noise_net = nn.Sequential(
            linear_layer(self.data_dim),
            activation(),
            linear_layer(self.data_dim),
        )

        # NOTE: so far best architecture: obtained  MSE ~0.075, without
        # weight sharing, beating previous best (version 75)
        def make_branch_():
            return nn.Sequential(
                linear_layer(1024),
                activation(),
                linear_layer(1024),
                activation(),
                linear_layer(1024),
                activation(),
                linear_layer(256),
                activation(),
                linear_layer(128),
                nn.Unflatten(1, outcome_shape)
            )

        def make_branch():
            return nn.Sequential(
                nn.Unflatten(1, (-1, 1, 1)),
                nn.LazyConvTranspose2d(128, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConvTranspose2d(128, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConvTranspose2d(128, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConvTranspose2d(128, 3),
                ResBlock(128, activation),
                ResBlock(128, activation),
                ResBlock(128, activation),
                nn.LazyConv2d(2, 2),
            )

        self.factual_branch = make_branch()
        self.counterfactual_branch = (
            self.factual_branch if weight_sharing else make_branch()
        )

        # force lazy init
        dummy_X_Z_output = self.combine_treatment_confounders_net(
            torch.rand(64, self.data_dim)
        )
        dummy_noise_output = self.reparametrize_noise_net(
            torch.rand(64, outcome_noise_dim)
        )
        dummy_output_fact = self.factual_branch(torch.rand(64, self.input_dim))
        dummy_output_counterfact = (
            self.counterfactual_branch(torch.rand(64, self.input_dim))
        )

        assert dummy_X_Z_output.shape[1] == self.data_dim
        assert dummy_noise_output.shape[1] == self.data_dim
        assert dummy_output_fact.shape[1:] == outcome_shape
        assert dummy_output_counterfact.shape[1:] == outcome_shape

    def combine_treatment_confounders(
        self,
        treatment: Tensor,
        confounders: Tensor,
    ) -> Tensor:
        combined = torch.cat((treatment, confounders), dim=-1)
        if self.use_combine_net:
            combined = self.combine_treatment_confounders_net(combined)

        return combined

    def forward(
        self,
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        outcome_noise: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        factual_data = self.combine_treatment_confounders(
            factual_treatment,
            confounders,
        )

        counterfactual_data = self.combine_treatment_confounders(
            counterfactual_treatment,
            confounders
        )

        noise = self.reparametrize_noise_net(outcome_noise)

        factual_input = self.inject_noise(factual_data, noise)
        counterfactual_input = self.inject_noise(counterfactual_data, noise)

        # factual_branch and counterfactual_branch are the same network
        # when weight sharing is on
        factual_outcome = self.factual_branch(factual_input)
        counterfactual_outcome = (
            self.counterfactual_branch(counterfactual_input)
        )

        return factual_outcome, counterfactual_outcome


class TwinNet(pl.LightningModule):
    def __init__(
        self,
        treatment_dim: int,
        confounders_dim: int,
        outcome_noise_dim: int,
        outcome_size: torch.Size,
        lr: float,
        noise_inject_mode: Literal['concat', 'add', 'multiply'] = 'multiply',
        use_combine_net: bool = True,
        weight_sharing: bool = False,
        activation: ActivationFunc = 'mish',
        batch_norm: bool = False,
    ):
        super().__init__()
        self.twin_net = SimpleDeepTwinNetComponent(
            treatment_dim=treatment_dim,
            confounders_dim=confounders_dim,
            outcome_noise_dim=outcome_noise_dim,
            outcome_shape=outcome_size,
            noise_inject_mode=noise_inject_mode,
            use_combine_net=use_combine_net,
            weight_sharing=weight_sharing,
            activation=activation_layer_ctor(activation),
            batch_norm=batch_norm,
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

        factual_loss = F.mse_loss(factual_outcome_hat, factual_outcome)
        counterfactual_loss = F.mse_loss(
            counterfactual_outcome_hat,
            counterfactual_outcome,
        )

        loss = factual_loss + counterfactual_loss

        # FIXME: this helps pytorch-lighnting with ambiguos batch sizes
        batch_size = y['factual_outcome'].shape[0]
        self.log('batch_size_fix', batch_size, batch_size=batch_size)

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
    pert_types = ['swelling', 'fracture']
    max_num_pert_args: int = ...  # TODO
    max_num_pert_locations: int = 3
    _pert_type_to_index = {t: idx for idx, t in enumerate(pert_types)}

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
        self.psf_dataset = ZipDataset(dict(
            plain=PlainMorphoMNIST(**kwargs),
            swollen=SwollenMorphoMNIST(**kwargs),
            fractured=FracturedMorphoMNIST(**kwargs),
        ))

        self.outcome_noise = self.sample_outcome_noise(
            sample_shape=(len(self.psf_dataset), self.outcome_noise_dim),
        )

    @classmethod
    def treatment_dim(cls) -> int:
        max_pert_coords = 2 * cls.max_num_pert_locations
        return len(cls.pert_types) + max_pert_coords

    @classmethod
    def confounders_dim(cls) -> int:
        return len(cls.labels) + len(cls.metrics)

    @classmethod
    def sample_outcome_noise(
        cls,
        sample_shape: torch.Size,
        scale: float = 0.25,
    ) -> Tensor:
        # TODO: consider moving to DataModule
        return (Normal(0, scale).sample(sample_shape) % 1) + 1

    @classmethod
    def pert_type_to_index(cls, pert_type: str) -> int:
        try:
            return cls._pert_type_to_index[pert_type]
        except KeyError:
            raise ValueError(
                f"{pert_type=} is not a supported perturbation type: must be "
                f"one of {', '.join(cls.pert_types)}"
            )

    @classmethod
    def one_hot_pert_type(cls, pert_type: Union[str, Sequence[str]]) -> Tensor:
        if isinstance(pert_type, str):
            index = cls.pert_type_to_index(pert_type)
        else:
            index = list(map(cls.pert_type_to_index, pert_type))

        return F.one_hot(
            torch.as_tensor(index),
            num_classes=len(cls.pert_types),
        ).float()

    @classmethod
    def one_hot_label(cls, label: Tensor) -> Tensor:
        return F.one_hot(label.long(), num_classes=len(cls.labels)).float()

    @classmethod
    def _to_vector(cls, elems: Sequence[Tensor]) -> Tensor:
        """Concatanates Tensor elems into a single vector. Elements in elems
        must either be all scalar tensors, or all batched scalar tensors (i.e.
        vectors) with the same size, in which case the resulting tensor is a
        batch of vectors."""

        assert len(elems) > 0
        assert all_equal(map(Tensor.size, elems))
        first = elems[0]
        dim = first.dim()
        assert dim == 0 or dim == 1

        elems_ = [e.unsqueeze(-1) for e in elems]
        return torch.cat(elems_, dim=-1)

    @classmethod
    def perturbation_vector(
        cls,
        # TODO: consider changing the type column to something else throughout
        # the pipeline to avoid conflict with python's built-in
        type: Union[str, Sequence[str]],
        args: Dict[str, Tensor],
        locations: Dict[int, Dict[Literal['x', 'y'], Tensor]],
    ):

        one_hot_type = cls.one_hot_pert_type(type)

        # NOTE: ignored for now
        sorted_args = (
            args[arg] for arg in sorted(args)
        )

        sorted_locations_coords = [
            locations[loc_idx][pos_coord]
            for loc_idx in sorted(locations)
            for pos_coord in sorted(locations[loc_idx])
        ]

        # enforce an exact number of perturbation locations across
        # perturbation types so that the input tensors to the network have
        # the same shape (even for perturbations that don't have a relevant
        # location
        coords_to_pad = (
            2 * cls.max_num_pert_locations - len(sorted_locations_coords)
        )

        if coords_to_pad < 0:
            raise RuntimeError(
                f"Found number of locations {len(locations)} in perturbation "
                f"higher than expected maximum {cls.max_num_per_locations}."
                f"Check the data or update {cls}.max_num_per_locations."
            )
        elif coords_to_pad > 0:
            dummy_location = torch.tensor(-1.0)
            is_batch = not isinstance(type, str)
            if is_batch:
                batch_size = len(type)
                dummy_location = dummy_location.expand(batch_size)

            for _ in range(coords_to_pad):
                sorted_locations_coords.append(dummy_location)

        locations_coors_vect = cls._to_vector(sorted_locations_coords)

        return torch.cat(
            tensors=(
                one_hot_type,
                # NOTE: currently perturbations args are the same for a given
                # perturbation type so no point in including them in the
                # treatment vector
                # *sorted_args,
                locations_coors_vect,
            ),
            dim=-1,
        )

    @classmethod
    def metrics_vector(cls, metrics: Dict[str, Tensor]) -> Tensor:
        # ensure consitent order
        return cls._to_vector([
            metrics[metric_name]
            for metric_name in sorted(metrics)
        ])

    def __len__(self):
        return len(self.psf_dataset)

    def __getitem__(self, index):
        psf_item = self.psf_dataset[index]
        outcome_noise = self.outcome_noise[index]

        swelling_data = psf_item['swollen']['perturbation_data']
        fracture_data = psf_item['fractured']['perturbation_data']

        swelling = self.perturbation_vector(**swelling_data)
        fracture = self.perturbation_vector(**fracture_data)
        label = self.one_hot_label(psf_item['plain']['label'])
        metrics = self.metrics_vector(psf_item['plain']['metrics'])
        label_and_metrics = torch.cat((label, metrics), dim=-1)

        x = dict(
            factual_treatment=swelling,
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


class PSFTwinNet(TwinNet):
    def __init__(
        self,
        outcome_size: torch.Size,
        # 0.0005,  # 1.7013748158991985e-06 # 4.4157044735331275e-05
        lr: float = 0.0005  # 0.0005  # 1.7013748158991985e-06
    ):
        super().__init__(
            outcome_size=outcome_size,
            treatment_dim=PSFTwinNetDataset.treatment_dim(),
            confounders_dim=PSFTwinNetDataset.confounders_dim(),
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
        batch_size: int = 64,
        pin_memory: bool = True,
    ):

        super().__init__(
            dataset_ctor=partial(
                PSFTwinNetDataset,
                embed_image=embed_image
            ),
            batch_size=batch_size,
            pin_memory=pin_memory,
            transform=DictTransform(
                key='image',
                transform_value=transforms.Normalize(mean=0.5, std=0.5),
            ),
        )


# TODO: find better place for this funcion
def vqvae_embed_image(vqvae: VQVAE, image: Tensor):
    embed = partial(vqvae.embed, latent_type='decoder_input')

    with torch.no_grad():
        return maybe_unbatched_apply(embed, image)


def main():
    from pytorch_lightning.utilities.cli import LightningCLI
    from pytorch_lightning.callbacks import ModelCheckpoint

    vqvae = load_best_model(VQVAE)
    embedding_size = vqvae.encoder_net(torch.rand((64, 1, 28, 28))).shape[1:]
    print(f"{embedding_size}")

    # FIXME: initialsiation, traning, saving and storing a bit hacky currently
    # Functions signatures needed by LightningCLI so can't use functools.partial
    def Model() -> PSFTwinNet:
        return PSFTwinNet(outcome_size=embedding_size)

    def Datamodule() -> PSFTwinNetDataModule:
        return PSFTwinNetDataModule(
            embed_image=partial(vqvae_embed_image, vqvae)
        )

    cli = LightningCLI(
        model_class=Model,
        datamodule_class=Datamodule,
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

    """
    lr_finder = cli.trainer.tuner.lr_find(
        model=cli.model,
        datamodule=cli.datamodule,
        max_lr=0.01,
        num_training=10000,
        early_stop_threshold=6.0,
    )

    tune_lr(lr_finder, cli.model)
    """

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
