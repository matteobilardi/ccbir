import operator
from itertools import starmap, repeat
from typing import List, Mapping, Tuple
from more_itertools import all_equal, first, interleave_evenly, repeat_each, zip_equal
from torch.utils.data import Dataset, default_collate
import torch
from toolz import valmap
import toolz.curried as C


class ZipDataset(Dataset):
    def __init__(self, datasets: Mapping[str, Dataset]):
        assert len(datasets) > 0
        assert all_equal(map(len, datasets.values()))
        super().__init__()
        self.name_to_dataset = datasets

    def __len__(self) -> int:
        return len(first(self.name_to_dataset.values()))

    def __getitem__(self, index):
        return valmap(C.get(index), self.name_to_dataset)


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
