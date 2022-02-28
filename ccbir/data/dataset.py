from typing import Mapping
from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    def __init__(self, datasets: Mapping[str, Dataset]):
        super().__init__()

        assert len(datasets) > 0
        length = len(datasets.values()[0])
        assert all(len(dataset) == length for dataset in datasets.values())

        self.length = length
        self.datasets = datasets

    def __getitem__(self, index):
        return {
            dataset_name: dataset[index]
            for dataset_name, dataset in self.datasets.items()
        }
