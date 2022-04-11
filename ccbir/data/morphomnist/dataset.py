from datetime import datetime
from email.generator import Generator
from multiprocessing.sharedctypes import Value
import time
from typing_extensions import Self

from typer import Option
from ccbir.data.util import BatchDict, random_split_repeated, standard_scaler_for, tensor_from_numpy_image, default_convert_series
from ccbir.configuration import config
from ccbir.data.morphomnist.perturb import perturb_image
from ccbir.util import DictOfFunctions, dict_funcs_apply, leaves_map, reset_random_seed, star_apply, tensor_ncycles
from deepscm.morphomnist import perturb
from deepscm.morphomnist.perturb import Fracture, Perturbation, Swelling
from deepscm.datasets.morphomnist import (
    MorphoMNISTLike,
    load_morphomnist_like,
    save_morphomnist_like,
)
from torch.utils.data import Subset, random_split
from itertools import repeat
from more_itertools import ncycles, unzip
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from toolz import assoc_in, assoc, valmap, keymap, compose, identity
import toolz.curried as C
from torch import Tensor, default_generator
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import default_convert
import tqdm
from torch.utils.data import Dataset, ConcatDataset


class MorphoMNIST(Dataset):

    def __init__(
        self,
        *,
        data_dir: str,
        train: bool,
        transform: Callable[[BatchDict], BatchDict] = None,
        normalize_metrics: bool = False,
        binarize: bool = False,
        to_tensor: bool = True,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train = train

        images, labels, metrics_df = load_morphomnist_like(
            root_dir=data_dir,
            train=train,
            columns=None  # include all metrics available
        )
        assert len(images) == len(labels) and len(images) == len(metrics_df)

        self.images = images.copy()

        if to_tensor:
            self.images = tensor_from_numpy_image(
                image=self.images,
                has_channel_dim=False
            )

        if binarize:
            if not to_tensor:
                raise ValueError(
                    f"{to_tensor=} but must be True when binarize is True"
                )

            # it's important to stochastically binarize the dataset only once
            # (i.e. not in __getitem__) otherwise repeated sampling is akin
            # to introducing regularisation to the dataset itself and would
            # cause an unfair evaluation of performance. Hence, this can't
            # be done by passing a stochastic transform.
            # Also, its seems that deterministic binarization like thresholding
            # should also be avoided for fair comparisons.
            # See https://twitter.com/poolio/status/1001535260008443904?s=20&t=JeEO0hZkserQwfCnR7sYSg
            self.images = torch.bernoulli(self.images)

        # copy labels so that torch doesn't complain about non-writable
        # underlying numpy array
        self.labels = torch.from_numpy(labels.copy())

        # enforce float32 metrics (instead of float64 of loaded numpy array) in
        # line with default pytorch tensors
        self.metrics = {
            metric_name: torch.as_tensor(
                metric_values.values,
                dtype=torch.float32,
            )
            for metric_name, metric_values in metrics_df.items()
        }

        if normalize_metrics:
            self.metrics = self._normalize_metrics(self.metrics)

        items = BatchDict(dict(
            image=self.images,
            label=self.labels,
            metrics=self.metrics,
        ))

        self.items = items if transform is None else transform(items)
        assert isinstance(items, BatchDict), type(items)

    def train_val_random_split(
        self,
        train_ratio: float,
        generator: Optional[Generator] = default_generator,
    ) -> Tuple[Subset, Subset]:
        assert self.train
        return random_split_repeated(
            dataset=self,
            train_ration=train_ratio,
            repeats=1,
            generator=generator
        )

    # TODO: formalize the notion of an in memory dataset a bit more

    def get_items(self) -> BatchDict:
        return self.items

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index):
        return self.items[index]

    # TODO: make most of the following classmethods as they are just helpers

    def _raw_training_metrics(self):
        """Metrics from the training dataset without any normalisation"""
        return self.__class__(
            data_dir=self.data_dir,
            normalize_metrics=False,
            train=True,
        ).metrics

    def _normalize_metrics(
        self,
        metrics: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        fittable_data = self._raw_training_metrics()
        scalers = leaves_map(standard_scaler_for, fittable_data)
        normalized_metrics = dict_funcs_apply(scalers, metrics)
        return normalized_metrics

    @classmethod
    def path_prefix(cls, train: bool) -> str:
        return 'train' if train else 't10k'

    @classmethod
    def get_images_path(cls, data_path: Path, train: bool):
        return data_path / f"{cls.path_prefix(train)}-images-idx3-ubyte.gz"

    @classmethod
    def get_labels_path(cls, data_path: Path, train: bool):
        return data_path / f"{cls.path_prefix(train)}-labels-idx1-ubyte.gz"

    @classmethod
    def get_morpho_path(cls, data_path: Path, train: bool):
        return data_path / f"{cls.path_prefix(train)}-morpho.csv"


class PerturbedMorphoMNIST(MorphoMNIST):
    def __init__(
        self,
        train: bool,
        normalize_perturbation_data: bool = False,
        perturbation_type: Optional[Type[Perturbation]] = None,
        perturbation_args: Optional[Dict] = None,
        transform: Optional[Callable[[BatchDict], BatchDict]] = None,
        data_dir: Optional[str] = None,
        original_mnist_data_dir: Optional[str] = None,
        repeats: int = 1,
        **kwargs,
    ):

        if data_dir is None:
            assert issubclass(perturbation_type, Perturbation)
            folder_name = perturbation_type.__name__.lower()

            assert repeats > 0, repeats
            if repeats > 1:
                folder_name = f"{folder_name}_repeats_{repeats}"

            data_path = config.project_data_path / folder_name
        else:
            data_path = Path(data_dir)

        images_path = self.get_images_path(data_path, train)
        if not images_path.exists():
            print(f"{images_path} not found, generating perturbed dataset...")
            if (perturbation_type is None or perturbation_args is None):
                raise ValueError(
                    'must specify perturbation type and args when dataset does'
                    'not exits'
                )

            original_data_path = (
                Path(original_mnist_data_dir) if original_mnist_data_dir else
                config.original_mnist_data_path
            )

            self._generate_dataset(
                original_data_path=original_data_path,
                output_path=data_path,
                train=train,
                perturbation_type=perturbation_type,
                perturbation_args=perturbation_args,
                repeats=repeats,
            )

        super().__init__(
            data_dir=str(data_path),
            train=train,
            **kwargs,
        )
        self.repeats = repeats
        self.data_path = data_path
        # self.perturbation_type = perturbation_type
        # self.perturbation_args = perturbation_args
        self.perturbation_data = self._load_perturbation_data(data_path, train)
        if normalize_perturbation_data:
            self.perturbation_data = self._normalize_perturbation_data(
                self.perturbation_data
            )

        self.items = self.items.set_feature(
            keys=['perturbation_data'],
            value=self.perturbation_data,
        )

        self.items = self.items if transform is None else transform(self.items)

    def train_val_random_split(
        self,
        train_ratio: float,
        generator: Generator = default_generator,
    ) -> Tuple[Subset, Subset]:
        assert self.train
        return random_split_repeated(
            dataset=self,
            train_ration=train_ratio,
            repeats=self.repeats,
            generator=generator
        )

    @classmethod
    def get_perturbations_data_path(cls, data_path: Path, train: bool) -> Path:
        return data_path / f"{cls.path_prefix(train)}-perturbations-data.csv"

    @classmethod
    def _load_perturbation_data(
        cls,
        data_path: Path,
        train: bool,
    ) -> Dict:
        df = pd.read_csv(
            cls.get_perturbations_data_path(data_path, train),
            index_col='index',
        )

        df.columns = df.columns.str.split('.', expand=True)
        df = df.rename(columns={np.NaN: ""})

        def parse_key(k: str) -> Union[str, int]:
            if k.isnumeric():
                assert float(k).is_integer()
                return int(k)
            else:
                return k

        def valid_key(k: str) -> bool:
            assert isinstance(k, str), type(k)
            return k != ""

        data = dict()
        for col_keys, col_values in df.items():
            keys = map(parse_key, filter(valid_key, col_keys))
            values = default_convert_series(col_values)
            data = assoc_in(data, keys, values)

        return data

    def _raw_training_perturbation_data(self) -> Dict:
        return self.__class__(
            train=True,
            data_dir=str(self.data_path),
            normalize_perturbation_data=False,
        ).perturbation_data

    def _normalize_perturbation_data(self, perturbation_data: Dict) -> Dict:
        fittable_data = self._raw_training_perturbation_data()
        scalers = leaves_map(standard_scaler_for, fittable_data)
        normalized_data = dict_funcs_apply(scalers, perturbation_data)
        return normalized_data

    @classmethod
    def _generate_dataset(
        cls,
        original_data_path: Path,
        output_path: Path,
        train: bool,
        perturbation_type: Type[Perturbation],
        perturbation_args: Dict[str, Any],
        repeats: int,
    ):
        output_path.mkdir(parents=True, exist_ok=True)
        original_dataset = MorphoMNISTLike(
            str(original_data_path),
            train=train
        )
        original_images = original_dataset.images
        original_num_imgs = len(original_images)

        # NOTE: these should be made BatchDict to support arbitrary
        # perturbations and perturbations' args in a single dataset
        # but since we don't need that for now, just repeating the perturbation
        perturbations_type = repeat(perturbation_type, original_num_imgs)
        perturbations_args = repeat(perturbation_args, original_num_imgs)

        # handle potential dataset augmentation by repeating the perturbations
        perturbations_to_perform = ncycles(
            iterable=zip(
                original_images,
                perturbations_type,
                perturbations_args,
            ),
            n=repeats,
        )
        labels = tensor_ncycles(original_dataset.labels, n=repeats)

        with multiprocessing.Pool() as pool:
            perturbed_images_gen = pool.imap(
                star_apply(perturb_image),
                perturbations_to_perform,
                chunksize=128,
            )

            # dispaly progress bar (technically not up to date because generator
            # blocks for correct result order)
            perturbed_images_gen = tqdm.tqdm(
                iterable=perturbed_images_gen,
                total=original_num_imgs * repeats,
                unit='img',
                ascii=True,
            )
            perturbed_images, perturbations_data = unzip(perturbed_images_gen)
            perturbed_images = list(perturbed_images)
            perturbations_data = list(perturbations_data)

        perturbed_images = np.stack(perturbed_images)
        perturbations_data = pd.json_normalize(
            data=perturbations_data,
            # perturbations may have different number of locations, hence
            # do not enforce that all locations always appear
            errors='ignore',
        )

        # TODO? For now no need to compute metrics for perturbed images
        # but might become necessary in the future
        dummy_empty_metrics = pd.DataFrame(index=range(len(perturbed_images)))
        save_morphomnist_like(
            images=perturbed_images,
            labels=labels,
            metrics=dummy_empty_metrics,
            root_dir=str(output_path),
            train=train,
        )

        perturbations_data.to_csv(
            path_or_buf=cls.get_perturbations_data_path(output_path, train),
            index_label='index'
        )


class PlainMorphoMNIST(MorphoMNIST):
    def __init__(
        self,
        data_dir: str = str(config.plain_morphomnist_data_path),
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, **kwargs)


class SwollenMorphoMNIST(PerturbedMorphoMNIST):
    def __init__(
        self,
        *,
        perturbation_args=dict(
            strength=3.0,
            radius=7.0,
        ),
        **kwargs,
    ):
        super().__init__(
            perturbation_type=Swelling,
            perturbation_args=perturbation_args,
            **kwargs,
        )


class FracturedMorphoMNIST(PerturbedMorphoMNIST):
    def __init__(
        self,
        *,
        perturbation_args=dict(
            thickness=1.5,
            prune=2.0,
            num_frac=3,
        ),
        **kwargs
    ):
        super().__init__(
            perturbation_type=Fracture,
            perturbation_args=perturbation_args,
            **kwargs,
        )


class LocalPerturbationsMorphoMNIST(MorphoMNIST):
    # NOTE: for now just use the pre-built local perturbations dataset from
    # MorphoMNIST repo i.e. randomly interleaved plain images + swelling +
    # fractures
    def __init__(
        self,
        data_dir: str = str(config.local_morphomnist_data_path),
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, **kwargs)
