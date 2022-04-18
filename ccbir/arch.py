from typing_extensions import Self
from more_itertools import pairwise
from typing import Callable, Dict, Optional
from torch import nn
from torchvision.ops import SqueezeExcitation


class PreActResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: Callable[..., nn.Module] = nn.ReLU,
        use_se: bool = False,
    ):
        super().__init__()
        I = in_channels
        O = out_channels
        S = stride

        left_layers = [
            nn.BatchNorm2d(I),
            activation(),
            nn.Conv2d(I, O, 3, S, padding=1, bias=False),
            nn.BatchNorm2d(O),
            activation(),
            nn.Conv2d(O, O, 3, stride=1, padding=1, bias=False),
        ]

        if use_se:
            left_layers.append(SqueezeExcitation(O, O, activation))

        self.left = nn.Sequential(*left_layers)

        if S != 1 or I != O:
            self.shortcut = nn.Sequential(
                nn.Conv2d(I, O, 1, stride=S, padding=0, bias=False),
                nn.BatchNorm2d(O)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.left(x) + self.shortcut(x)
        return out

    @classmethod
    def multi_block(
        cls,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: Callable[..., nn.Module] = nn.ReLU,
        use_se: bool = False,
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        channels = (
            [in_channels] + [out_channels] * num_blocks
        )
        channels = pairwise(channels)
        return nn.Sequential(*(
            cls(in_c, out_c, stride, activation, use_se)
            for (in_c, out_c), stride in zip(channels, strides)
        ))
