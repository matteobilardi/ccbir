from email.policy import strict
from pathlib import Path
import sys


class Config:
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.resolve(strict=True)

    @property
    def project_data_path(self) -> Path:
        return self.project_root / 'assets/data'

    @property
    def morphomnist_data_path(self) -> Path:
        return Path(
            '/vol/bitbucket/mb8318/morphomnist_data/'
        ).resolve(strict=True)

    @property
    def original_mnist_data_path(self) -> Path:
        # TODO: find a better, sensible way to keep track of all of the data and
        # project directories, e.g. read object starting from yaml config
        return self.morphomnist_data_path / 'original'

    @property
    def global_morphomnist_data_path(self) -> Path:
        return self.morphomnist_data_path / 'global'

    @property
    def local_morphomnist_data_path(self) -> Path:
        return self.morphomnist_data_path / 'local'

    @property
    def plain_morphomnist_data_path(self) -> Path:
        return self.morphomnist_data_path / 'plain'

    @property
    def frac_morphomnist_data_path(self) -> Path:
        return self.morphomnist_data_path / 'frac'

    @property
    def swel_morphomnist_data_path(self) -> Path:
        return self.morphomnist_data_path / 'swel'

    """
    @property
    def synth_mnist_data_path(self) -> Path:
        return self.project_root / 'submodules/deepscm/assets/data/morphomnist'
    """

    @property
    def _submodules_path(self) -> Path:
        return self.project_root / 'submodules'

    def pythonpath_fix(self):
        """Extends current $PYTHONPATH environement variable to include modules
        that are part of the project but which haven't have been packaged and
        installed yet."""
        paths = [
            self.project_root,
            self.project_root / 'ccbir',
            self.project_root / 'ccbir/pytorch_vqvae',  # TODO: avoid copy-paste
            self._submodules_path / 'deepscm',
        ]
        sys.path.extend(str(p) for p in paths)


config = Config()

__all__ = ['config']
