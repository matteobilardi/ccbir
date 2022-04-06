from functools import partial
from ccbir.data.morphomnist.datamodule import MorphoMNISTDataModule
from ccbir.data.dataset import BatchDict, InterleaveDataset
from ccbir.data.morphomnist.dataset import (
    FracturedMorphoMNIST,
    MorphoMNIST,
    SwollenMorphoMNIST,
)
from torchvision import transforms
from typing import Type
import toolz.curried as C


class VQVAEDataset(InterleaveDataset):

    def __init__(
        self,
        train: bool,
        transform=None,
    ) -> None:
        kwargs = dict(
            train=train,
            transform=transform,
        )

        super().__init__(datasets=[
            SwollenMorphoMNIST(**kwargs),
            FracturedMorphoMNIST(**kwargs),
        ])

    def __getitem__(self, index):
        return super().__getitem__(index)['image']


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
