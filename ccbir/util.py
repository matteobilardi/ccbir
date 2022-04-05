from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, Literal, Optional, TypeVar
from torch import Tensor
from toolz import curry, valmap
import toolz.curried as C
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from ccbir.configuration import config
from torch import nn


@curry
def maybe_unbatched_apply(
    func: Callable[[Tensor], Tensor],
    x: Tensor,
    single_item_dims: int = 3,
    **kwargs,
):
    """Extends the given func that only works on batched input of shape
    B X C X H X W to also work on single tensors of shape C x H x W
    """
    batched_item_dim = 1 + single_item_dims
    if x.dim() == single_item_dims:
        return func(x.unsqueeze(0), **kwargs).squeeze(0)
    elif x.dim() == batched_item_dim:
        return func(x, **kwargs)
    else:
        raise ValueError(
            f"{x.dim()=} but only {single_item_dims} and {batched_item_dim} "
            "are supported"
        )


T = TypeVar('T')


@curry
def star_apply(func: Callable[..., T], arg: Iterable) -> T:
    return func(*arg)


@curry
def _leaves_map(func, obj):
    if isinstance(obj, dict):
        return valmap(_leaves_map(func), obj)
    else:
        return func(obj)


def leaves_map(func: Callable, d: Dict):
    """Given a possbly-nested dictionary d, returns a new dictionary obtained by
    applying func to all the non-dictionary objects that appear in the values of
    any dictionary reachable form d, including d.

    d must not contain cycles.
    """
    if not isinstance(d, dict):
        raise TypeError(f"d must be a dictionary but was {type(d)}")
    return _leaves_map(func, d)


def tune_lr(
    lr_finder,
    model: pl.LightningModule,
    save_plot: bool = True,
    plot_dir: Optional[str] = None,
):
    print(lr_finder.results)

    if save_plot:
        if plot_dir is None:
            out_dir = config.checkpoints_path_for_model(type(model))
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            out_path = Path(plot_dir).resolve(strict=True)

        out_path = out_path / 'lr_finder_plot.png'
        fig = lr_finder.plot(suggest=True)
        fig.savefig(str(out_path))

    new_lr = lr_finder.suggestion()

    print(f"Updating lr to suggested {new_lr}")
    # TODO: it may not be necessary to set both of the following
    # attributes
    model.lr = new_lr
    model.hparams.lr = new_lr


ActivationFunc = Literal[
    'relu',
    'leakyrelu',
    'swish',
    'mish',
    'tanh',
    'sigmoid',
]

_activation_layer_ctor = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
    'swish': nn.SiLU,
    'silu': nn.SiLU,
    'mish': nn.Mish,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
}


def activation_layer_ctor(
    func_name: ActivationFunc,
    inplace: bool = True,
) -> nn.Module:
    try:
        layer_ctor = _activation_layer_ctor[func_name]
    except KeyError:
        raise TypeError(
            f"{func_name} is not a supported activation layer. Must be one of"
            f"{', '.join(_activation_layer_ctor.keys())}"
        )

    layer_ctor = partial(layer_ctor, inplace=inplace)

    return layer_ctor
