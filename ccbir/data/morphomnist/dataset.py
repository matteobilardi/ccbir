
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch import Tensor
from torch.utils.data import default_collate
import torch
from ccbir.tranforms import from_numpy_image
from ccbir.configuration import config
from ccbir.data.morphomnist.perturbsampler import FractureSampler, PerturbationSampler, SwellingSampler
from deepscm.datasets.morphomnist import MorphoMNISTLike, load_morphomnist_like, save_morphomnist_like
from deepscm.morphomnist.morpho import ImageMorphology
from deepscm.morphomnist.perturb import Fracture, Perturbation, Swelling
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import PIL
import multiprocessing
import numpy as np
import pandas as pd
import torchvision
import tqdm
import pickle


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
            self.images = torch.bernoulli(self.images)

        # copy labels so that torch doesn't complain about non-writable
        # underlying numpy array
        self.labels = torch.from_numpy(labels.copy())

        # enforce float32 metrics (instead of float64 of loaded numpy array) in
        # line with default pytorch tensors
        self.metrics = {
            metric_name: torch.tensor(
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
        image = self.images[index]

        if self.transform is not None:
            if isinstance(index, slice):
                # some torchvision transforms don't support batching so handle
                # slices explictly (and slowly)
                image = default_collate(list(map(self.transform, image)))
            else:
                image = self.transform(image)

        metrics = {k: v[index] for k, v in self.metrics.items()}

        return dict(image=image, label=self.labels[index], metrics=metrics)

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

    _THRESHOLD = .5
    _UP_FACTOR = 4

    def __init__(
        self,
        perturbation_type: Type[Perturbation],
        perturbation_sampler: PerturbationSampler,
        original_data_dir: str = str(config.original_mnist_data_path),
    ):

        if not issubclass(
            perturbation_sampler.perturbation_type,
            perturbation_type
        ):
            raise TypeError(
                f"{perturbation_sampler.perturbation_type=} is not a subclass "
                f"of {perturbation_type=}"
            )

        self.perturbation_type = perturbation_type
        self.perturbation_sampler = perturbation_sampler
        self.original_data_dir = original_data_dir

    def _perturb_image(
        self,
        perturbation_and_image: Tuple[Perturbation, np.ndarray]
    ) -> np.ndarray:
        perturbation, image = perturbation_and_image
        morph = ImageMorphology(image, self._THRESHOLD, self._UP_FACTOR)
        perturbed_image = morph.downscale(perturbation(morph))

        return perturbed_image

    def generate_dataset(
        self,
        out_dir: str,
        *,
        train: bool,
    ):
        original_dataset = MorphoMNISTLike(self.original_data_dir, train=train)
        original_images = original_dataset.images
        perturbations, perturbations_args = (
            self.perturbation_sampler.sample(len(original_images))
        )

        with multiprocessing.Pool() as pool:
            perturbed_images_gen = pool.imap(
                self._perturb_image,
                zip(perturbations, original_images),
                chunksize=128
            )

            # dispaly progress bar (technically not up to date because generator
            # blocks for correct result order)
            perturbed_images_gen = tqdm.tqdm(
                perturbed_images_gen,
                total=len(original_images),
                unit='img',
                ascii=True
            )

            perturbed_images = np.stack(list(perturbed_images_gen))

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        perturbations_args_path = (
            PerturbedMorphoMNIST.get_perturbations_args_path(out_path, train)
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

        perturbations_args.to_csv(perturbations_args_path, index_label='index')

        return perturbed_images, perturbations_args


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
        data_dir: Optional[str] = None,
        generator: Optional[PerturbedMorphoMNISTGenerator] = None,
        train: bool,
        overwrite: bool = False,
        **kwargs,
    ):
        self.perturbation_type = perturbation_type
        self.train = train

        if data_dir is None:
            self.data_path = config.project_data_path / self.perturbation_name
        else:
            self.data_path = Path(data_dir).resolve(strict=True)

        images_path = self.get_images_path(self.data_path, train)
        perturbations_args_path = (
            self.get_perturbations_args_path(self.data_path, train)
        )

        # Load dataset or generate it if it does not exist
        exists = images_path.exists() and perturbations_args_path.exists()
        if exists and not overwrite:
            self.perturbations_args = (
                pd.read_csv(perturbations_args_path, index_col='index')
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

            _images, self.perturbations_args = generator.generate_dataset(
                out_dir=str(self.data_path),
                train=train,
            )

        super().__init__(data_dir=str(self.data_path), train=train, **kwargs)

    @staticmethod
    def get_perturbations_args_path(data_path: Path, train: bool) -> Path:
        prefix = "train" if train else "t10k"
        return data_path / f"{prefix}-perturbations-args.csv"

    @property
    def perturbation_name(self) -> str:
        return str(self.perturbation_type.__name__).lower()

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['perturbations_args'] = (
            self.perturbations_args.iloc[index].to_dict()
        )

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
        generator=PerturbedMorphoMNISTGenerator(Swelling, SwellingSampler()),
        **kwargs,
    ):
        super().__init__(Swelling, **kwargs, generator=generator)


class FracturedMorphoMNIST(PerturbedMorphoMNIST):
    def __init__(
        self,
        *,
        generator=PerturbedMorphoMNISTGenerator(Fracture, FractureSampler()),
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
