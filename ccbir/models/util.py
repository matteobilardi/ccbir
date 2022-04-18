from pathlib import Path
from turtle import mode
from typing import Type, TypeVar
import pyro
import pyro.infer

from torch import Tensor
from ccbir.configuration import config
import pytorch_lightning as pl

T = TypeVar("T", bound=pl.LightningModule)


def best_model_checkpoint_path(model_type: Type[T]) -> Path:
    path = config.checkpoints_path_for_model(model_type) / 'best.ckpt'
    path = path.resolve(strict=True)  # throws if does not exist
    return path


def load_best_model(model_type: Type[T], eval_only: bool = True) -> T:
    path = best_model_checkpoint_path(model_type)
    model = model_type.load_from_checkpoint(str(path))
    print(f"Loaded {model_type} from {path=}")
    if eval_only:
        model = model.eval()
    return model


def optimal_sigma(
    estimate_batch: Tensor,
    target_batch: Tensor,
) -> Tensor:
    """Sigma vae optimal sigma based on https://orybkin.github.io/sigma-vae/.
    Note that variance is calculated across all datapoints in the batch as is
    done in the sigma-vae paper."""
    assert estimate_batch.shape == target_batch.shape
    return (
        ((estimate_batch - target_batch) ** 2)
        .mean(dim=[0, 1, 2, 3], keepdim=True)
        .sqrt()
        .expand_as(estimate_batch)
    )
