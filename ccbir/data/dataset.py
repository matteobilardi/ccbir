import operator
from itertools import starmap, repeat
from typing import List, Mapping, Tuple
from more_itertools import interleave_evenly, repeat_each, zip_equal
from torch.utils.data import Dataset, default_collate
import torch


class ZipDataset(Dataset):
    def __init__(self, datasets: Mapping[str, Dataset]):
        super().__init__()

        assert len(datasets) > 0

        # check that all datasets have equal length
        datasets_iter = iter(datasets.values())
        len_first = len(next(datasets_iter))
        assert all(len(dataset) == len_first for dataset in datasets_iter)

        self._len = len_first
        self.datasets = datasets

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index):
        return {
            dataset_name: dataset[index]
            for dataset_name, dataset in self.datasets.items()
        }


class InterleaveDataset(Dataset):
    """Inteleaves all items in all the given datasets. Assumes that
    items have the same type across datasets."""

    def __init__(self, datasets: List[Dataset]):
        super().__init__()
        self.datasets = datasets

        self.dataset_with_item_idx_for: Mapping[int, Tuple[Dataset, int]] = (
            list(interleave_evenly([
                list(zip(repeat(dataset), range(len(dataset))))
                for dataset in datasets
            ]))
        )

    def __len__(self) -> int:
        return len(self.dataset_with_item_idx_for)

    def __getitem__(self, index):
        dataset_with_item_idx = self.dataset_with_item_idx_for[index]
        if isinstance(index, slice):
            return default_collate([
                dataset[idx] for dataset, idx in dataset_with_item_idx
            ])
        else:
            dataset, item_idx = dataset_with_item_idx
            return dataset[item_idx]
