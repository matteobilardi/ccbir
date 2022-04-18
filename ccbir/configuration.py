from pathlib import Path
import shutil
import sys
import os
import pytorch_lightning as pl
from typing import Type


def _mkdir_if_missing(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class Config:

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.resolve(strict=True)

    @property
    def project_data_path(self) -> Path:
        return self.project_root / 'assets/data'

    @property
    def checkpoints_path(self) -> Path:
        return self.project_root / 'assets/checkpoints'

    def checkpoints_path_for_model(
        self,
        model_type: Type[pl.LightningModule]
    ) -> Path:
        return self.checkpoints_path / model_type.__name__.lower()

    @property
    def morphomnist_data_path(self) -> Path:
        return Path(
            '/vol/bitbucket/mb8318/morphomnist_data/'
        ).resolve(strict=True)

    @property
    def original_mnist_data_path(self) -> Path:
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

    @property
    def temporary_data_path(self) -> Path:
        return _mkdir_if_missing(Path('/vol/bitbucket/mb8318/ccbir/tmp'))

    @property
    def logs_path(self) -> Path:
        return _mkdir_if_missing(Path('/vol/bitbucket/mb8318/ccbir/logs'))
    
    @property
    def tensorboard_logs_path(self) -> Path:
        return _mkdir_if_missing(self.logs_path / 'tb_logs')

    def clear_temporary_data(self):
        shutil.rmtree(str(self.temporary_data_path))

    def pythonpath_fix(self):
        """Extends current $PYTHONPATH environement variable to include modules
        that are part of the project but which haven't have been packaged and
        installed yet."""
        paths = [
            self.project_root,
            self.project_root / 'ccbir',
            self.project_root / 'ccbir/pytorch_vqvae',  # TODO: avoid copy-paste
            self._submodules_path,
            self._submodules_path / 'deepscm',
            self._submodules_path / 'shake_drop_pytorch',
        ]
        sys.path.extend(map(str, paths))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['LD_LIBRARY_PATH'] = '/vol/cuda/11.4.120-cudnn8.2.4/lib64'


config = Config()

__all__ = ['config']
