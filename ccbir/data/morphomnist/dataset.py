
from torch import Tensor
from ccbir.data.dataset import CombinedDataset
from ccbir.configuration import config
from ccbir.data.morphomnist.perturbsampler import FractureSampler, PerturbationSampler, SwellingSampler
from deepscm.datasets.morphomnist import MorphoMNISTLike, save_morphomnist_like
from deepscm.morphomnist.morpho import ImageMorphology
from deepscm.morphomnist.perturb import Fracture, Perturbation, Swelling
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Type, Union
import PIL
import multiprocessing
import numpy as np
import pandas as pd
import torchvision
import tqdm


class MorphoMNIST(MorphoMNISTLike):

    def __init__(
        self,
        data_dir: str,
        train: bool,
        *,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        MorphoMNISTLike.__init__(
            self,
            root_dir=data_dir,
            train=train,
            columns=None  # include all metrics available
        )

        if transform is not None:
            self.images = transform(self.images)

    def __getitem__(self, index):
        item_old = MorphoMNISTLike.__getitem__(self, index)
        item = dict(image=item_old['image'], label=item_old['label'])

        metrics = {
            k: v for k, v in item_old.items()
            if k not in ('image', 'label')
        }
        if len(metrics) > 0:
            item['metrics'] = metrics

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
    ):
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
        perturbations, perturbations_args =  \
            self.perturbation_sampler.sample(len(original_images))

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

        perturbations_args_path = \
            PerturbedMorphoMNIST.get_perturbations_args_path(out_path, train)

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
        data_dir: Optional[str] = None,
        generator: Optional[PerturbedMorphoMNISTGenerator] = None,
        *,
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

        images_path = self.__class__.get_images_path(self.data_path, train)
        perturbations_args_path = \
            self.__class__.get_perturbations_args_path(self.data_path, train)

        # Load dataset or generate it if it does not exits
        exists = images_path.exists() and perturbations_args_path.exists()
        if exists and not overwrite:
            self.perturbations_args = \
                pd.read_csv(perturbations_args_path, index_col='index')
        elif generator is None:
            raise ValueError(
                f"generator cannot be None when the dataset does not exist"
                f"already in {data_dir=} or when overwrite=True"
            )
        else:
            dataset_type = 'train' if train else 'test'
            print(
                f"Generating {dataset_type} dataset for perturbation"
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
        item['perturbations_args'] = \
            self.perturbations_args.iloc[index].to_dict()

        return item


class PlainMorphoMNIST(MorphoMNIST):
    def __init__(
        self,
        data_dir: str = str(config.plain_morphomnist_data_path),
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, **kwargs)


class SwollenMorphoMNIST(PerturbedMorphoMNIST):
    def __init__(self, **kwargs):
        generator = PerturbedMorphoMNISTGenerator(Swelling, SwellingSampler())
        super().__init__(Swelling, **kwargs, generator=generator)


class FracturedMorphoMNIST(PerturbedMorphoMNIST):
    def __init__(self, **kwargs):
        generator = PerturbedMorphoMNISTGenerator(Fracture, FractureSampler())
        super().__init__(Fracture, **kwargs, generator=generator)


class PlainSwollenFracturedMorphoMNIST(CombinedDataset):
    def __init__(
        self,
        plain: Optional[PlainMorphoMNIST] = None,
        swollen: Optional[SwollenMorphoMNIST] = None,
        fractured: Optional[SwollenMorphoMNIST] = None,
    ):
        if plain is None:
            plain = PlainMorphoMNIST()
        if swollen is None:
            swollen = SwollenMorphoMNIST()
        if fractured is None:
            fractured = FracturedMorphoMNIST()

        super().__init__(datasets={
            "plain": plain,
            "swollen": swollen,
            "fractured": fractured
        })
