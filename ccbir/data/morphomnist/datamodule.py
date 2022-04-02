from __future__ import annotations
from ccbir.data.morphomnist.dataset import MorphoMNIST

from torch.utils.data import DataLoader, random_split
from typing import Callable, Optional
import multiprocessing
import pytorch_lightning as pl

from typing import Type, Union


class MorphoMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        dataset_ctor: Optional[Union[
            Type[MorphoMNIST],
            Callable[..., MorphoMNIST]
        ]],
        num_workers: int = multiprocessing.cpu_count(),
        batch_size: int,
        pin_memory: bool,
        transform,
    ):
        super().__init__()
        self.dataset_ctor = dataset_ctor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform

    def prepare_data(self):
        # check that the data lives on disk (else generate it)
        self.dataset_ctor(train=True)
        self.dataset_ctor(train=False)

    def setup(self, stage: Optional[str] = None):
        mnist_train = self.dataset_ctor(train=True, transform=self.transform)
        self.mnist_test = self.dataset_ctor(
            train=False,
            transform=self.transform,
        )

        # Reserve split for validation
        num_val = len(mnist_train) // 10
        num_train = len(mnist_train) - num_val
        self.mnist_train, self.mnist_val = random_split(
            mnist_train, [num_train, num_val]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
