from ccbir.data.morphomnist.datamodule import MorphoMNISTDataModule
from ccbir.data.dataset import InterleaveDataset
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


class VQVAEMorphoMNISTDataModule(MorphoMNISTDataModule):
    def __init__(
        self,
        *,
        dataset_type: Type[MorphoMNIST] = VQVAEDataset,
        batch_size: int = 64,
        pin_memory: bool = True,
    ):
        super().__init__(
            dataset_ctor=dataset_type,
            batch_size=batch_size,
            pin_memory=pin_memory,
            transform=transforms.Compose([
                C.get('image'),
                # enforce range [-1, 1] in line with tanh NN output
                # see https://discuss.pytorch.org/t/understanding-transform-normalize/21730/2
                transforms.Normalize(mean=0.5, std=0.5)
            ]),
        )
