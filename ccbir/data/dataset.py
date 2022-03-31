from itertools import starmap
from typing import List, Mapping
from more_itertools import zip_equal
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
    """Randomly inteleaves all items in all the given datasets. Assumes that
    items have the same type across datasets. Equivalent to concatenating
    the dataset items into a list and shuffling"""

    def __init__(self, datasets: List[Dataset]):
        super().__init__()
        self.datasets = datasets
        self._len = sum(map(len, datasets))

        dataset_idx_for_idx = torch.cat([
            torch.tensor(idx).expand(len(dataset))
            for idx, dataset in enumerate(datasets)
        ])

        idx_in_dataset_for_idx = torch.cat([
            torch.arange(len(dataset)) for dataset in datasets
        ])

        shuffle_idxs = torch.randperm(self._len)
        self.dataset_idx_for_idx = dataset_idx_for_idx[shuffle_idxs]
        self.idx_in_dataset_for_idx = idx_in_dataset_for_idx[shuffle_idxs]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index):
        # we convert to numpy because an iterator over numpy arrays yields ints
        # rather than tensors of single elements, which avoids issues when
        # indexing dataframes (this is a concern when index is a slice)
        dataset_idx = self.dataset_idx_for_idx[index].numpy()
        idx_in_dataset = self.idx_in_dataset_for_idx[index].numpy()

        if isinstance(index, slice):
            return default_collate([
                self.datasets[d_idx][idx_in_d]
                for d_idx, idx_in_d in zip_equal(dataset_idx, idx_in_dataset)
            ])
        else:
            return self.datasets[dataset_idx][idx_in_dataset]
