from typing import Callable
from torch import Tensor


def support_unbatched(
    func: Callable[[Tensor], Tensor]
) -> Callable[[Tensor], Tensor]:
    """Extends the given func that only works on batched input of shape
    B X C X W X H to also work on single tensors of shape C x W x H
    """
    def extended_func(x: Tensor):
        if x.dim() == 3:
            return func(x.unsqueeze(0)).squeeze(0)
        elif x.dim() == 4:
            return func(x)
        else:
            raise ValueError(f"{x.dim()=} but only dim 3 and 4 supported")

    return extended_func