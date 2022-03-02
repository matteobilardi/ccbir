from turtle import mode
from typing import Type, TypeVar
from ccbir.configuration import config
import pytorch_lightning as pl

T = TypeVar("T", bound=pl.LightningModule)


def load_best_model(model_type: Type[T], eval: bool = True) -> T:
    path = config.checkpoints_path_for_model(model_type) / 'best.ckpt'
    path = path.resolve(strict=True)  # throws if does not exist
    model = model_type.load_from_checkpoint(str(path))
    if eval:
        model = model.eval()
    return model
