from __future__ import annotations
from ccbir.data.util import BatchDict, random_split_repeated
from ccbir.data.morphomnist.dataset import MorphoMNIST

from torch.utils.data import DataLoader, random_split
from typing import Callable, Optional
import multiprocessing
import pytorch_lightning as pl
import torch

from typing import Type, Union


class MorphoMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        dataset_ctor: Optional[Union[
            Type[MorphoMNIST],
            Callable[..., MorphoMNIST]
        ]],
        batch_size: int,
        pin_memory: bool,
        transform: Callable[[BatchDict], BatchDict],
        num_workers: int = multiprocessing.cpu_count(),
    ):
        super().__init__()
        self.dataset_ctor = dataset_ctor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform

    def prepare_data(self):
        # check that the data lives on disk (else generate it)
        self.dataset_ctor(train=True, transform=self.transform)
        self.dataset_ctor(train=False, transform=self.transform)

    def setup(self, stage: Optional[str] = None):
        mnist_train = self.dataset_ctor(train=True, transform=self.transform)
        self.mnist_test = self.dataset_ctor(
            train=False,
            transform=self.transform,
        )

        self.mnist_train, self.mnist_val = mnist_train.train_val_random_split(
            train_ratio=0.9,
            # fix generator for reproducible split
            generator=torch.Generator().manual_seed(42),
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
