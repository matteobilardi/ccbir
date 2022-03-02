from typing import Mapping
from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    def __init__(self, datasets: Mapping[str, Dataset]):
        super().__init__()

        assert len(datasets) > 0

        # check that all datasets have equal length
        datasets_iter = iter(datasets.values())
        len_first = len(next(datasets_iter))
        assert all(len(dataset) == len_first for dataset in datasets_iter)

        self.length = len_first
        self.datasets = datasets

    def __getitem__(self, index):
        return {
            dataset_name: dataset[index]
            for dataset_name, dataset in self.datasets.items()
        }

    def __len__(self) -> int:
        return self.length
