from functools import cached_property, partial
from typing import Callable, Dict, Optional, Type, Union

from torch import Tensor
from ccbir.models.twinnet.data import PSFTwinNetDataModule
from ccbir.models.vqvae.data import OriginalMNIST_VQVAE_Dataset, SwellFractureVQVAE_Dataset, VQVAE_MorphoMNIST_DataModule
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class ExperimentData:
    """Provides convenient and fast access to the exact data used by a
    datamodule"""

    def __init__(
        self,
        datamodule_ctor: Union[
            Type[pl.LightningDataModule],
            Callable[..., pl.LightningDataModule],
        ],
    ):
        self.datamodule_ctor = datamodule_ctor

    @cached_property
    def datamodule(self) -> pl.LightningDataModule:
        print('Loading datamodule...')
        dm = self.datamodule_ctor()
        dm.prepare_data()
        dm.setup()

        return dm

    @cached_property
    def train_dataset(self):
        dm = self.datamodule
        dataset = dm.train_dataloader().dataset
        return dataset

    @cached_property
    def _val_dataset(self):
        dm = self.datamodule
        dataset = dm.val_dataloader().dataset
        return dataset

    @cached_property
    def test_dataset(self):
        dm = self.datamodule
        dataset = dm.test_dataloader().dataset
        return dataset

    def dataset(self, train: bool) -> Dataset:
        return self.train_dataset if train else self.test_dataset

    def dataloader(self, train: bool):
        if train:
            return self.datamodule.train_dataloader()
        else:
            return self.datamodule.test_dataloader()


class SwellFractureVQVAE_ExperimentData(ExperimentData):
    def __init__(self):
        super().__init__(
            datamodule_ctor=partial(
                VQVAE_MorphoMNIST_DataModule,
                dataset_type=SwellFractureVQVAE_Dataset,
            )
        )


class OriginalMNIST_VQVAE_ExperimentData(ExperimentData):
    def __init__(self):
        super().__init__(
            datamodule_ctor=partial(
                VQVAE_MorphoMNIST_DataModule,
                dataset_type=OriginalMNIST_VQVAE_Dataset,
            )
        )


class TwinNetExperimentData(ExperimentData):
    def __init__(self, embed_image: Callable[..., Tensor], repeats: int = 1):
        super().__init__(
            datamodule_ctor=partial(
                PSFTwinNetDataModule,
                embed_image=embed_image,
                repeats=repeats,
            )
        )

    def psf_items(self, train: bool, index: Optional[int] = None) -> Dict:
        # ugly accesses to train subset dataset
        dataset = self.dataset(train)
        if index is None:
            return (
                dataset.dataset.psf_items[dataset.indices] if train else
                dataset.psf_items.dict()
            )
        else:
            return (
                dataset.dataset.psf_items[dataset.indices[index]] if train else
                dataset.psf_items[index]
            )
