from ccbir.configuration import config
from ccbir.data.util import BatchDict, random_split_repeated
from ccbir.data.morphomnist.datamodule import MorphoMNISTDataModule
from ccbir.data.morphomnist.dataset import FracturedMorphoMNIST, PlainMorphoMNIST, SwollenMorphoMNIST
from functools import partial
from more_itertools import all_equal
from torch import Tensor, default_generator
from torch.distributions import Normal
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from typing import Callable, Dict, Generator, Literal, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
import diskcache


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
        repeats: int = 1,
        transform: Callable[[BatchDict], BatchDict] = None,
        normalize_metrics: bool = True,
        normalize_perturbation_data: bool = True,
        shared_cache: Optional[diskcache.Index] = None,
    ):
        super().__init__()
        self.embed_image = embed_image
        self.train = train
        self.transform = transform
        self.normalize_metrics = normalize_metrics
        self.normalize_perturbation_data = normalize_perturbation_data
        self.repeats = repeats

        # NOTE: *should* be free from race conditions
        # as the DataModule runs prepare_setup in a single process,
        # after which everyone should see the database in the shared cache
        cache_key = 'train' if train else 'test'
        if shared_cache is None:
            dataset = self._generate_dataset(repeats)
        elif cache_key in shared_cache:
            print('Loading dataset from cache')
            dataset = shared_cache[cache_key]
        else:
            print('Dataset not found in cache, generating it...')
            shared_cache[cache_key] = dataset = self._generate_dataset(repeats)

        # psf_items and outcome_noise are not really necessary but
        # are kept as attributes for ease of debugging
        # the outcome noise should be the same once the dataset is
        # generated (we don't want to resample during training)
        self.psf_items, self.outcome_noise, self.items = dataset

    def get_items(self) -> BatchDict:
        return self.items

    def __len__(self):
        return len(self.psf_items)

    def __getitem__(self, index):
        item = self.items[index]
        return item['x'], item['y']

    def train_val_random_split(
        self,
        train_ratio: float,
        generator: Generator = default_generator,
    ) -> Tuple[Subset, Subset]:
        assert self.train
        return random_split_repeated(
            dataset=self,
            train_ratio=train_ratio,
            repeats=self.repeats,
            generator=generator
        )

    # TODO: make classmethod
    def _generate_dataset(
        self,
        repeats: int,
    ) -> Tuple[BatchDict, Tensor, BatchDict]:
        kwargs = dict(
            train=self.train,
            transform=self.transform,
            normalize_metrics=self.normalize_metrics,
        )

        plain = PlainMorphoMNIST(**kwargs).get_items().ncycles(repeats)
        perturbed_kwargs = {
            **kwargs,
            'normalize_perturbation_data': self.normalize_perturbation_data,
            'repeats': repeats,
        }
        swollen = SwollenMorphoMNIST(**perturbed_kwargs).get_items()
        fractured = FracturedMorphoMNIST(**perturbed_kwargs).get_items()

        psf_items = BatchDict.zip(dict(
            plain=plain,
            swollen=swollen,
            fractured=fractured,
        ))

        psf_items_d = psf_items.dict()

        outcome_noise = self.sample_outcome_noise(
            sample_shape=(len(psf_items), self.outcome_noise_dim),
        )

        swelling_data = psf_items_d['swollen']['perturbation_data']
        fracture_data = psf_items_d['fractured']['perturbation_data']

        swelling = self.perturbation_vector(**swelling_data)
        fracture = self.perturbation_vector(**fracture_data)
        label = self.one_hot_label(psf_items_d['plain']['label'])
        metrics = self.metrics_vector(psf_items_d['plain']['metrics'])
        label_and_metrics = torch.cat((label, metrics), dim=-1)

        x = dict(
            factual_treatment=swelling,
            counterfactual_treatment=fracture,
            confounders=label_and_metrics,
            outcome_noise=outcome_noise,
        )

        swollen_z = self.embed_image(psf_items_d['swollen']['image'])
        fractured_z = self.embed_image(psf_items_d['fractured']['image'])

        y = dict(
            factual_outcome=swollen_z,
            counterfactual_outcome=fractured_z,
        )

        items = BatchDict(dict(x=x, y=y))
        dataset = psf_items, outcome_noise, items

        return dataset

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
        # NOTE: this should be exact comparison despite float
        if scale == 0:
            return torch.ones(sample_shape)
        else:
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
        assert dim == 0 or dim == 1, dim

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

        locations_coords_vect = cls._to_vector(sorted_locations_coords)

        return torch.cat(
            tensors=(
                one_hot_type,
                # NOTE: currently perturbations args are the same for a given
                # perturbation type so no point in including them in the
                # treatment vector
                # *sorted_args,
                locations_coords_vect,
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


class PSFTwinNetDataModule(MorphoMNISTDataModule):
    # TODO: better class name
    def __init__(
        self,
        *,
        embed_image: Callable[[Tensor], Tensor],
        batch_size: int = 64,
        pin_memory: bool = True,
        **kwargs,
    ):
        self.shared_cache = diskcache.Index(
            str(config.temporary_data_path / self.__class__.__name__.lower())
        )

        super().__init__(
            dataset_ctor=partial(
                PSFTwinNetDataset,
                embed_image=embed_image,
                shared_cache=self.shared_cache,
            ),
            batch_size=batch_size,
            pin_memory=pin_memory,
            transform=partial(
                BatchDict.map_feature,
                keys=['image'],
                func=transforms.Normalize(mean=0.5, std=0.5),
            ),
            **kwargs,
        )
