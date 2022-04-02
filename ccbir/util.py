from functools import partial
from itertools import starmap
from operator import getitem
from typing import Callable, Dict, Hashable
from torch import Tensor
from toolz import curry, valmap, get


class support_unbatched:
    """Extends the given func that only works on batched input of shape
    B X C X W X H to also work on single tensors of shape C x W x H
    """

    def __init__(self, func: Callable[[Tensor], Tensor]):
        self.func = func

    def __call__(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            return self.func(x.unsqueeze(0)).squeeze(0)
        elif x.dim() == 4:
            return self.func(x)
        else:
            raise ValueError(f"{x.dim()=} but only dim 3 and 4 supported")


class tupled_args:
    """Function that returns func(*x) when called on x"""

    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, arg):
        return self.func(*arg)


@curry
def _leaves_map(func, obj):
    if isinstance(obj, dict):
        return valmap(_leaves_map(func), obj)
    else:
        return func(obj)


def leaves_map(func: Callable, d):
    if not isinstance(d, dict):
        raise TypeError(f"d must be a dictionary but was {type(d)}")
    return _leaves_map(func, d)


def leaves_getitem(d: Dict, index) -> Dict:
    return leaves_map(partial(get, index), d)
