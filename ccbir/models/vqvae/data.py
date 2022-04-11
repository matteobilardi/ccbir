from functools import partial

from torch import default_generator
from ccbir.data.morphomnist.datamodule import MorphoMNISTDataModule
from ccbir.data.util import BatchDict, InterleaveDataset, random_split_repeated
from ccbir.data.morphomnist.dataset import (
    FracturedMorphoMNIST,
    MorphoMNIST,
    SwollenMorphoMNIST,
)
from torchvision import transforms
from typing import Callable, Generator, Tuple, Type
import toolz.curried as C
from torch.utils.data import Subset


class VQVAEDataset(InterleaveDataset):

    def __init__(
        self,
        train: bool,
        repeats: int = 4,
        transform: Callable[[BatchDict], BatchDict] = None,
    ) -> None:
        kwargs = dict(
            train=train,
            transform=transform,
            repeats=repeats,
        )

        super().__init__(datasets=[
            SwollenMorphoMNIST(**kwargs),
            FracturedMorphoMNIST(**kwargs),
        ])

        self.train = train
        self.repeats = repeats

    def __getitem__(self, index):
        return super().__getitem__(index)['image']

    def train_val_random_split(
        self,
        train_ratio: float,
        generator: Generator = default_generator,
    ) -> Tuple[Subset, Subset]:
        # NOTE: hacky but should work without data leaks: the interleave of
        # repeated datasets of equal length is the same as the repeated
        # interleave of the original datasets

        assert self.train
        return random_split_repeated(
            dataset=self,
            train_ratio=train_ratio,
            repeats=self.repeats,
            generator=generator
        )


class VQVAEMorphoMNISTDataModule(MorphoMNISTDataModule):
    def __init__(
        self,
        *,
        dataset_type: Type[MorphoMNIST] = VQVAEDataset,
        batch_size: int = 64,
        pin_memory: bool = True,
        **kwargs,
    ):
        normalize = transforms.Normalize(mean=0.5, std=0.5)
        super().__init__(
            dataset_ctor=dataset_type,
            batch_size=batch_size,
            pin_memory=pin_memory,
            transform=partial(
                BatchDict.map,
                func=lambda item: dict(image=normalize(item['image'])),
            ),
            **kwargs,
        )
