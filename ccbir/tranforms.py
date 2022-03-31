import numpy as np
from torch import Tensor, nn
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


"""
def binarize(tensor: Tensor, low, high, threshold, dtype):
    return torch.where(tensor < threshold, low, high).to(dtype)


class Binarize(nn.Module):
    def __init__(self, low, high, thresh, dtype=None):
        self.low = low
        self.high = high
        self.thresh = thresh
        self.dtype = dtype

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype if self.dtype is None else self.dtype
        return binarize(x, self.low, self.high, self.thresh, dtype)
"""
