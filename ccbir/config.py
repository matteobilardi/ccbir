from pathlib import Path


class Config:
    _MNIST_DATA_PATH = "/vol/bitbucket/mb8318/morphomnist_data/original"

    @property
    def mnist_data_path(self):
        return Path(self.__class__._MNIST_DATA_PATH)


config = Config()
