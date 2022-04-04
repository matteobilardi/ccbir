
from ccbir.configuration import config
from ccbir.data.morphomnist.perturb import (
    FractureArgsSampler,
    PerturbationArgsSampler,
    SwellingArgsSampler,
    perturb_image,
)
from ccbir.tranforms import from_numpy_image
from ccbir.util import leavesmap, star_apply
from deepscm.morphomnist.perturb import Fracture, Perturbation, Swelling
from deepscm.datasets.morphomnist import (
    MorphoMNISTLike,
    load_morphomnist_like,
    save_morphomnist_like,
)
from itertools import repeat
from more_itertools import unzip
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from toolz import assoc_in, assoc, valmap
import toolz.curried as C
from torch import Tensor
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import default_convert
import torchvision
import tqdm


class MorphoMNIST(torchvision.datasets.VisionDataset):
    # NOTE: for now just wrapper and adaptation of deepscm MorphoMNISTLike for
    # consistency with torchvision and pytorch

    def __init__(
        self,
        *,
        data_dir: str,
        train: bool,
        transform=None,
        normalize_metrics: bool = False,
        binarize: bool = False,
        to_tensor: bool = True,
    ):
        super().__init__(root=data_dir, transform=transform)

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
            self.images = from_numpy_image(self.images, has_channel_dim=False)

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

    def _raw_training_metrics(self):
        """Metrics from the training dataset without any normalisation"""

        if self.train:
            return self.metrics
        else:
            return self.__class__(data_dir=self.data_dir, train=True).metrics

    def _load_scalers(self, recompute=True) -> Dict[str, StandardScaler]:
        # TODO: consider removing saving to file entirely
        scalers_file = Path(self.root) / 'metrics_scalers.pkl'

        # load from disk scalers for each metric if they exits, otherwise fit
        # new scalers and save them to disk for future use
        if scalers_file.exists() and not recompute:
            with open(scalers_file, 'rb') as f:
                scaler_for_metric = pickle.load(f)

            for scaler in scaler_for_metric.values():
                check_is_fitted(scaler)
        else:
            # never fit a scaler on the test data!
            fittable_metrics = self._raw_training_metrics()

            scaler_for_metric = {}
            for metric, metric_values in fittable_metrics.items():
                scaler = StandardScaler()
                # standard scaler requires 2d array hence view
                scaler.fit(metric_values.view(-1, 1).numpy())
                scaler_for_metric[metric] = scaler

            scalers_file.write_bytes(pickle.dumps(scaler_for_metric))

        return scaler_for_metric

    def _normalize_metrics(
        self,
        metrics: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        scaler_for_metric = self._load_scalers()
        normalized_metrics = {
            metric: torch.from_numpy(
                scaler_for_metric[metric].transform(
                    metric_values.view(-1, 1).numpy()
                ).flatten()
            )
            for metric, metric_values in metrics.items()
        }

        return normalized_metrics

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index):
        item = dict(
            image=self.images[index],
            label=self.labels[index],
            metrics=valmap(C.get(index), self.metrics),
        )

        if self.transform is not None:
            item = self.transform(item)

        return item

    @staticmethod
    def get_images_path(data_path: Path, train: bool):
        prefix = "train" if train else "t10k"
        return data_path / f"{prefix}-images-idx3-ubyte.gz"

    @staticmethod
    def get_labels_path(data_path: Path, train: bool):
        prefix = "train" if train else "t10k"
        return data_path / f"{prefix}-labels-idx1-ubyte.gz"

    @staticmethod
    def get_morpho_path(data_path: Path, train: bool):
        prefix = "train" if train else "t10k"
        return data_path / f"{prefix}-morpho.csv"


class PerturbedMorphoMNISTGenerator:
    """ Generate MorphoMNIST dataset after perturbations of a (single) given
    perturbation_type initilised with arguments obtained by sampling from the
    given perturbation_sampler and applied to the original MNIST dataset. """

    def __init__(
        self,
        perturbation_type: Type[Perturbation],
        perturbation_args_sampler: PerturbationArgsSampler,
        original_data_dir: str = str(config.original_mnist_data_path),
    ):

        if not issubclass(
            perturbation_args_sampler.perturbation_type,
            perturbation_type
        ):
            raise TypeError(
                f"{perturbation_args_sampler.perturbation_type=} is not a "
                f"subclass of {perturbation_type=}"
            )

        self.perturbation_type = perturbation_type
        self.perturbation_args_sampler = perturbation_args_sampler
        self.original_data_dir = original_data_dir

    def generate_dataset(
        self,
        out_dir: str,
        *,
        train: bool,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        original_dataset = MorphoMNISTLike(self.original_data_dir, train=train)
        original_images = original_dataset.images
        num_images = len(original_images)
        perturbation_types = repeat(self.perturbation_type, num_images)
        perturbations_args = (
            self.perturbation_args_sampler
            .sample_args(num_images)
            .to_dict(orient='records')
        )

        with multiprocessing.Pool() as pool:
            perturbed_images_gen = pool.imap(
                star_apply(perturb_image),
                zip(original_images, perturbation_types, perturbations_args),
                chunksize=128,
            )

            # dispaly progress bar (technically not up to date because generator
            # blocks for correct result order)
            perturbed_images_gen = tqdm.tqdm(
                iterable=perturbed_images_gen,
                total=num_images,
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

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        perturbations_data_path = (
            PerturbedMorphoMNIST.get_perturbations_data_path(out_path, train)
        )

        # TODO? For now no need to compute metrics for perturbed images
        # but might become necessary in the future
        dummy_empty_metrics = pd.DataFrame(index=range(len(perturbed_images)))
        save_morphomnist_like(
            images=perturbed_images,
            labels=original_dataset.labels,
            metrics=dummy_empty_metrics,
            root_dir=out_dir,
            train=train,
        )

        perturbations_data.to_csv(perturbations_data_path, index_label='index')

        return perturbed_images, perturbations_data


class PerturbedMorphoMNIST(MorphoMNIST):
    """ MorphoMNIST dataset on which perturbations of the single
    perturbation_type have been performed.

    If the database doesn't already exist at the given data_dir, a dataset
    generator must be given.
    """

    def __init__(
        self,
        perturbation_type: Type[Perturbation],
        *,
        train: bool,
        data_dir: Optional[str] = None,
        generator: Optional[PerturbedMorphoMNISTGenerator] = None,
        overwrite: bool = False,
        transform: Callable = None,
        **kwargs,
    ):
        self.perturbation_type = perturbation_type
        self.train = train

        if data_dir is None:
            self.data_path = config.project_data_path / self.perturbation_name
        else:
            self.data_path = Path(data_dir).resolve(strict=True)

        images_path = self.get_images_path(self.data_path, train)
        perturbations_data_path = (
            self.get_perturbations_data_path(self.data_path, train)
        )

        # Load dataset or generate it if it does not exist
        exists = images_path.exists() and perturbations_data_path.exists()
        if exists and not overwrite:
            perturbations_df = (
                pd.read_csv(perturbations_data_path, index_col='index')
            )
        elif generator is None:
            raise ValueError(
                f"generator cannot be None when the dataset does not exist"
                f"already in {data_dir=} or when overwrite=True"
            )
        else:
            dataset_type = 'train' if train else 'test'
            print(
                f"Generating {dataset_type} dataset for perturbation "
                f"{perturbation_type}"
            )

            _images, perturbations_df = generator.generate_dataset(
                out_dir=str(self.data_path),
                train=train,
            )

        self.perturbations_data, self._perturbations_df = (
            self._cleanup_perturbation_data(perturbations_df)
        )

        super().__init__(data_dir=str(self.data_path), train=train, **kwargs)
        # FIXME: hacky overwritten transform fix
        self.transform = transform

    @classmethod
    def _cleanup_perturbation_data(
        cls,
        raw_df: pd.DataFrame
    ) -> Tuple[Dict, pd.DataFrame]:
        # unflatten and cleanup perturbations data
        df = raw_df
        df.columns = df.columns.str.split('.', expand=True)
        df = df.rename(columns={np.NaN: ""})

        data = dict()
        for col_keys, col_values in df.items():
            valid_keys = []
            for k in col_keys:
                assert isinstance(k, str), type(k)
                if k == "":
                    continue
                else:
                    if k.isnumeric():
                        assert float(k).is_integer()
                        valid_keys.append(int(k))
                    else:
                        valid_keys.append(k)
            assert len(valid_keys) > 0

            values = col_values.to_numpy()
            if np.issubdtype(values.dtype, np.floating):
                default_float_dtype = torch.get_default_dtype()
                values = torch.as_tensor(values, dtype=default_float_dtype)
            else:
                values = default_convert(values)

            data = assoc_in(data, valid_keys, values)

        return data, df

    @staticmethod
    def get_perturbations_data_path(data_path: Path, train: bool) -> Path:
        prefix = "train" if train else "t10k"
        return data_path / f"{prefix}-perturbations-data.csv"

    @property
    def perturbation_name(self) -> str:
        return self.perturbation_type.__name__.lower()

    def __getitem__(self, index):
        # FIXME: very hacky way to fix overwritten transform attribute: probably
        # should prefer composition over inheritance
        transform = self.transform
        self.transform = None
        item = super().__getitem__(index)
        self.transform = transform

        perturbation_data = leavesmap(C.get(index), self.perturbations_data)

        item = assoc(item, 'perturbation_data', perturbation_data)

        if self.transform is not None:
            item = self.transform(item)

        return item


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
        generator=PerturbedMorphoMNISTGenerator(
            Swelling, SwellingArgsSampler()
        ),
        **kwargs,
    ):
        super().__init__(Swelling, **kwargs, generator=generator)


class FracturedMorphoMNIST(PerturbedMorphoMNIST):
    def __init__(
        self,
        *,
        generator=PerturbedMorphoMNISTGenerator(
            Fracture, FractureArgsSampler()
        ),
        **kwargs
    ):
        super().__init__(Fracture, **kwargs, generator=generator)


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
