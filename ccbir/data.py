from __future__ import annotations
from sys import prefix

from ccbir.configuration import config
from deepscm.datasets import morphomnist
import deepscm
from deepscm.datasets.morphomnist import MorphoMNISTLike, _get_paths, save_morphomnist_like
from pathlib import Path
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from typing import Callable, List, Optional, Tuple
import multiprocessing
import pytorch_lightning as pl
import torchvision
import invertransforms
from torch.utils.data import Dataset
import PIL

from typing import Type, Union
from deepscm.morphomnist.morpho import ImageMorphology
from deepscm.morphomnist.perturb import Fracture, Perturbation, Swelling
import numpy as np
from functools import partial
import pandas as pd
import tqdm
import torchvision

from configuration import config

# As per generate_all script in morphomnist
THRESHOLD = .5
UP_FACTOR = 4


# TODO: not sure if the whole concept of a perturbation sampler and of saving to
# file the perturbation args makes sense. Might wanna remove if I conclusively
# establish that it is not needed
class PerturbationSampler:
    def __init__(self, perturbation_type: Type[Perturbation]):
        self.perturbation_type = perturbation_type

    def sample_args(self, num_samples: int) -> pd.DataFrame:
        raise NotImplementedError()

    def sample(
        self,
        num_samples: int
    ) -> Tuple[List[Perturbation], pd.DataFrame]:
        perturbations_args = self.sample_args(num_samples)
        perturbations = [
            # note that in generaral perturbation type constructors are
            # stochastic so that perturbations initialised with the same
            # arguments need not produce identical outputs
            self.perturbation_type(**kwargs)
            for kwargs in perturbations_args.to_dict(orient='records')
        ]

        return perturbations, perturbations_args


class SwellingSampler(PerturbationSampler):
    def __init__(self):
        super().__init__(Swelling)

    def sample_args(self, num_samples: int) -> pd.DataFrame:
        # TODO: for now we don't actually sample the arguments and instead rely
        # on the swelling location sampling internal to the Swelling class and
        # apply a Swelling initialised with the same arguments to all images.
        # Also, for now we don't pass such arguments to the twin network as they
        # are constant. And the treatment is instead a one-hot encoding of an
        # enum: either swelling fracture or no-treatment (i.e. identity
        # function)
        strength = np.full((num_samples,), 3.0)
        radius = np.full((num_samples,), 7.0)

        return pd.DataFrame.from_dict(dict(
            strength=strength,
            radius=radius,
        ))


class FractureSampler(PerturbationSampler):
    def __init__(self):
        super().__init__(Fracture)

    def sample_args(self, num_samples) -> pd.DataFrame:
        # TODO: see TODO in same location for SwellingSampler
        thickness = np.full((num_samples,), 1.5, dtype=float)
        prune = np.full((num_samples,), 2.0, dtype=float)
        num_frac = np.full((num_samples,), 3, dtype=int)

        return pd.DataFrame.from_dict(dict(
            thickness=thickness,
            prune=prune,
            num_frac=num_frac,
        ))


class PerturbedMorphoMNISTGenerator:
    """ Generate MorphoMNIST dataset after perturbations of a (single) given
    perturbation_type initilised with arguments obtained by sampling from the
    given perturbation_sampler and applied to the original MNIST dataset. """

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
            raise TypeError(f"""
                {perturbation_sampler.perturbation_type=} is not a subclass
                of {perturbation_type=}
            """)

        self.perturbation_type = perturbation_type
        self.perturbation_sampler = perturbation_sampler
        self.original_data_dir = original_data_dir

    def _perturb_image(
        self,
        perturbation_and_image: Tuple[Perturbation, np.ndarray]
    ):
        perturbation, image = perturbation_and_image
        morph = ImageMorphology(image, THRESHOLD, UP_FACTOR)
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

        perturbations_args.to_csv(perturbations_args_path)

        return perturbed_images, perturbations_args


# TODO: for now just subclass and hotfix deepscm MorphoMNISTLike for consistency
# with torchvision, ideally might want to write own logic avoid hacky use of
# multiple inheritance
class MorphoMNIST(MorphoMNISTLike, torchvision.datasets.VisionDataset):
    def __init__(
        self,
        data_dir: str,
        train: bool,
        *,
        transform: Optional[Callable] = None,
        metrics: Optional[List[str]] = None,  # None fetches all metrics
    ):
        torchvision.datasets.VisionDataset.__init__(
            self,
            root=data_dir,
            transform=transform,
        )

        MorphoMNISTLike.__init__(
            self,
            root_dir=data_dir,
            train=train,
            columns=metrics
        )

    def __getitem__(self, index):
        item = MorphoMNISTLike.__getitem__(self, index)
        image_tensor = item['image']
        # enforce consistent format with all other tochvision datasets
        image = PIL.Image.fromarray(image_tensor.numpy(), mode='L')

        if self.transform is not None:
            image = self.transform(image)

        item['image'] = image

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


class PlainMorphoMNIST(MorphoMNIST):
    def __init__(
        self,
        data_dir: str = str(config.plain_morphomnist_data_path),
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, **kwargs)


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
            self.perturbations_args = pd.read_csv(perturbations_args_path)
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


class SwollenMorphoMNIST(PerturbedMorphoMNIST):
    def __init__(self, **kwargs):
        generator = PerturbedMorphoMNISTGenerator(Swelling, SwellingSampler())
        super().__init__(Swelling, **kwargs, generator=generator)


class FracturedMorphoMNIST(PerturbedMorphoMNIST):
    def __init__(self, **kwargs):
        generator = PerturbedMorphoMNISTGenerator(Fracture, FractureSampler())
        super().__init__(Fracture, **kwargs, generator=generator)


"""
class TONAME_PROBABLY_IN_EXPRIMENT_SECTION(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.transform = self.transform

        # TODO: load from disk and apply transform
        ...

    @classmethod
    def generate(
        cls,
        data_dir: str,
        perturbations: List[SerializablePerturbation],
    ):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        perturbations = {
            'swelling': {
                'image': ...,
                # TODO: Probably data about the kind of swelling applied
            },
            'fracture': {
                'image': ...,
                # TODO: probably data about the kind of fracture applied

            }
        }

        original = {
            'original': {
                'image': ...,
                'label': ...,
                'metrics': ...,
            },
        }

        return {**original, **perturbations}
"""


# TODO: rename to be used with VAE and probably rerun training
# with dataset shwoing local perturbations (swelling + deformations)
class MorphoMNISTLikeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = str(config.local_morphomnist_data_path),
        num_workers: int = multiprocessing.cpu_count(),
        train_batch_size: int = 64,
        test_batch_size: int = 64,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        # check that data directory lives on disk, raise exception otherwise
        Path(self.data_dir).resolve(strict=True)

    def normalize(self, image: Tensor) -> Tensor:
        assert hasattr(self, '_normalize_transform')
        return self._normalize_transform(image)

    # TODO: if we never need to unnormalize, the dependency on invertransform
    # could be avoided
    def unnormalize(self, image: Tensor) -> Tensor:
        assert hasattr(self, '_normalize_transform')
        return self._normalize_transform.invert(image)

    def setup(self, stage: Optional[str] = None):

        # ISSUES TO ADDRESS:
        # TODO: the other morphometrics are not already part of the given
        # dataset. I should try to understand why that is? Also, I suspect
        # that we want to show to the twin network as much information about
        # the image that we have (intensity, slant, widht, thickness etc)
        # other than just the variables that we want to intervene upon. Notably
        # deep scm doesn't seem to include the other variables other than those
        # being m

        # TODO: with regards to interventions, can we know straight from the
        # deepscm dataset what covariates have been intervened upon and what the
        # original image was so that we can use the infromation to later train
        # the twin network?
        #
        # Well, the data generation process in gen_dataset seems to discard that
        # information and only generate images whose thickness and intensity has
        # been randomly changed. But then how do they test that a predicted
        # counterfactual is close to the ground truth one?

        # --------------------------------

        columns = [
            # 'area',
            # 'height',
            # 'length',
            # 'slant',
            # 'width',
            'intensity',
            'thickness',
        ]

        mnist_train = MorphoMNISTLike(
            self.data_dir, train=True, columns=columns
        )
        self.mnist_test = MorphoMNISTLike(
            self.data_dir, train=False, columns=columns
        )

        self._normalize_transform = invertransforms.Compose([
            # insert explicit gray channel and convert to float in range [0,1]
            # as typical for pytorch images
            invertransforms.Lambda(
                lambd=lambda image: image.unsqueeze(1).float() / 255.0,
                tf_inv=lambda image: (image * 255.0).byte().squeeze(1)
            ),
            # enforce range [-1, 1] in line with tanh NN output
            # see https://discuss.pytorch.org/t/understanding-transform-normalize/21730/2
            invertransforms.Normalize(mean=0.5, std=0.5)
        ])

        mnist_train.images = self.normalize(mnist_train.images)
        self.mnist_test.images = self.normalize(self.mnist_test.images)

        # Reserve split for validation
        num_val = len(mnist_train) // 10
        num_train = len(mnist_train) - num_val
        self.mnist_train, self.mnist_val = random_split(
            mnist_train, [num_train, num_val]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.test_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.test_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
