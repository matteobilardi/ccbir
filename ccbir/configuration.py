from pathlib import Path


# TODO: find a better, sensible way to keep track of all of the data and
# project directories, e.g. read object starting from yaml config
class Config:
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def original_mnist_data_path(self) -> Path:
        return Path('/vol/bitbucket/mb8318/morphomnist_data/original')

    @property
    def synth_mnist_data_path(self) -> Path:
        return self.project_root / 'submodules/deepscm/assets/data/morphomnist'


config = Config()
