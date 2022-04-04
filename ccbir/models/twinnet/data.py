from ccbir.tranforms import DictTransform
from ccbir.data.dataset import ZipDataset
from ccbir.data.morphomnist.datamodule import MorphoMNISTDataModule
from ccbir.data.morphomnist.dataset import FracturedMorphoMNIST, PlainMorphoMNIST, SwollenMorphoMNIST
from functools import partial
from more_itertools import all_equal
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, Dict, Literal, Sequence, Union
import toolz.curried as C
import torch
import torch.nn.functional as F


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
