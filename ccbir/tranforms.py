
from torch import Tensor, nn
import torch


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
