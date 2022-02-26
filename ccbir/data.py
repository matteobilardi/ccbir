from __future__ import annotations

from ccbir.configuration import config
from deepscm.datasets import morphomnist
from deepscm.datasets.morphomnist import MorphoMNISTLike
from pathlib import Path
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from typing import Callable, List, Optional, Tuple
import multiprocessing
import pytorch_lightning as pl
import torchvision
import invertransforms
from torch.utils.data import Dataset

from typing import Type, Union
from deepscm.morphomnist.morpho import ImageMorphology
from deepscm.morphomnist.perturb import Fracture, Perturbation, Swelling
import numpy as np
from functools import partial
import pandas as pd
import tqdm

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

    def sample_args(self, num_samples) -> pd.DataFrame:
        raise NotImplementedError()

    def sample(self, num_samples):
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

    def sample_args(self, num_samples) -> pd.DataFrame:
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


class PerturbedMorphoMNIST(Dataset):
    """ MorphoMNIST dataset after perturbations of a (single) given
    perturbation_type initilised with arguments obtained by sampling from the
    given stochastic model. """

    def __init__(
        self,
        perturbation_type: Type[Perturbation],
        data_dir: str = str(config.project_data_path),
        original_data_dir: str = str(config.original_mnist_data_path),
        *,
        train: bool,
        perturbation_sampler: Optional[PerturbationSampler] = None,
        transform: Optional[Callable] = None,
        overwrite: bool = False,
    ):
        super().__init__()
        self.perturbation_type = perturbation_type
        self.data_path = Path(data_dir).resolve(strict=True)
        self.original_data_dir = original_data_dir
        self.train = train
        self.transform = transform

        # Create folder for this dataset (if it does not exist already)
        self.dataset_path = self.data_path / self.perturbation_name
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # Set dataset paths
        prefix = "train" if self.train else "t10k"
        self.images_path = self.dataset_path / f"{prefix}-images-idx3-ubyte.gz"
        self.perturbations_args_path = \
            self.dataset_path / f"{prefix}-perturbations-args.csv"

        # Load dataset or generate it if it does not exits
        exists = self.images_path.exists() and self.dataset_path.exists()
        if exists and not overwrite:
            self._load_dataset()
        elif perturbation_sampler is None:
            raise ValueError(f"""
                perturbation_sampler cannot be None when the dataset does not
                exist already in {data_dir=} or when overwrite=True
            """)
        elif not issubclass(
            perturbation_sampler.perturbation_type,
            perturbation_type
        ):
            raise TypeError(f"""
                {perturbation_sampler.perturbation_type=} is not a subclass
                of {perturbation_type=}
            """)
        else:
            print(f"Generating dataset for perturbation {perturbation_type}")
            self._generate_dataset(perturbation_sampler)

    @property
    def perturbation_name(self) -> str:
        return str(self.perturbation_type.__name__).lower()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)

        perturbation_kwargs = self.perturbations_args.iloc[index].to_dict()

        return dict(image=image, perturbation_args=perturbation_kwargs)

    def _perturb_image(
        self,
        perturbation_and_image: Tuple[Perturbation, np.ndarray]
    ):
        perturbation, image = perturbation_and_image
        morph = ImageMorphology(image, THRESHOLD, UP_FACTOR)
        perturbed_image = morph.downscale(perturbation(morph))

        return perturbed_image

    def _load_dataset(self):
        self.images = morphomnist.io.load_idx(str(self.images_path))
        self.perturbations_args = pd.read_csv(self.perturbations_args_path)
        assert len(self.images) == len(self.perturbations_args)

    def _generate_dataset(
        self,
        perturbation_sampler: PerturbationSampler,
        persist: bool = True
    ):
        original_dataset = \
            MorphoMNISTLike(self.original_data_dir, train=self.train)
        images = original_dataset.images
        perturbations, perturbations_args =  \
            perturbation_sampler.sample(len(images))

        with multiprocessing.Pool() as pool:
            perturbed_images_gen = pool.imap(
                self._perturb_image,
                zip(perturbations, images),
                chunksize=128
            )

            # dispaly progress bar (technically not up to date because generator
            # blocks for correct result order)
            perturbed_images_gen = tqdm.tqdm(
                perturbed_images_gen,
                total=len(images),
                unit='img',
                ascii=True
            )

            perturbed_images = np.stack(list(perturbed_images_gen))

        if persist:
            morphomnist.io.save_idx(perturbed_images, str(self.images_path))
            perturbations_args.to_csv(self.perturbations_args_path)

        self.images = perturbed_images
        self.perturbations_args = perturbations_args


class SwelledMorphoMNIST(PerturbedMorphoMNIST):
    def __init__(
        self,
        *args,
        swelling_sampler: PerturbationSampler = SwellingSampler(),
        **kwargs,

    ):
        super().__init__(
            *args,
            **kwargs,
            perturbation_type=Swelling,
            perturbation_sampler=swelling_sampler,
        )


class FracturedMorphoMNIST(PerturbedMorphoMNIST):
    def __init__(
        self,
        *args,
        fracture_sampler: PerturbationSampler = FractureSampler(),
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
            perturbation_type=Fracture,
            perturbation_sampler=fracture_sampler,
        )


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


class MorphoMNISTLikeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = str(config.synth_mnist_data_path),
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
