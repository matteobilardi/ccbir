
from ccbir.configuration import config
from deepscm.datasets.morphomnist import MorphoMNISTLike
from pathlib import Path
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from typing import Optional
import multiprocessing
import pytorch_lightning as pl
import torchvision


class MorphoMNISTLikeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = str(config.synth_mnist_data_path),
        num_workers: int = multiprocessing.cpu_count(),
        train_batch_size: int = 64,
        test_batch_size: int = 64,
        pin_memory: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize

    def prepare_data(self):
        # check that data lives on disk, raise exception otherwise
        Path(self.data_dir).resolve(strict=True)

    # TODO: check if normalisation to gaussian improves performance.
    # https://github.com/gregunz/invertransforms/ looks useful
    def _normalize(self, image: Tensor) -> Tensor:
        # Normalize images to range [0, 1] and convert to float
        return image / 255.0

    def _preprocess(self, image: Tensor) -> Tensor:
        # insert explicit gray channel and convert to float
        image_ = image.unsqueeze(1).float()
        if self.normalize:
            image_ = self._normalize(image_)

        return image_

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

        mnist_train.images = self._preprocess(mnist_train.images)
        self.mnist_test.images = self._preprocess(self.mnist_test.images)

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

    # TODO: remove
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers
        )
