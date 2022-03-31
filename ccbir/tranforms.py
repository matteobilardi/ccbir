from typing import Callable, Hashable, TypeVar
from torch import Tensor
import numpy as np
import torch


def from_numpy_image(
    image: np.ndarray,
    has_channel_dim: bool = True,
) -> Tensor:
    """Converts possibly batched, and possibly in range [0, 255] numpy images of
    shape HxWxC into a tensor image (also possibly batched) of shape CxHxW and
    range [0.0, 1.0]"""

    # enforce presence of channel dimension
    if not has_channel_dim:
        image = image[..., np.newaxis]

    assert image.ndim == 4 or image.ndim == 3

    # Following based on code from torchivion/transforms/functional.py
    default_float_dtype = torch.get_default_dtype()
    image = np.moveaxis(image, [-3, -2, -1], [-2, -1, -3])
    img = torch.from_numpy(image).contiguous()
    # backward compatibility
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=default_float_dtype).div(255)
    else:
        return img


class DictTransform:
    """Defines a transformation on the value under key for an item that
    is of type dictionary"""

    def __init__(self, key: Hashable, transform_value: Callable):
        self.key = key
        self.transform_value = transform_value

    def __call__(self, item):
        if not isinstance(item, dict):
            raise TypeError(f"expected dict, got f{type(item)=}")

        try:
            value = item[self.key]
        except KeyError as e:
            raise KeyError(f'Could not find key={self.key} in {item=}') from e

        # create new dictionary with previous key value overwritten by
        # transformed value
        return {**item, self.key: self.transform_value(value)}
