from __future__ import annotations
from itertools import starmap, repeat
from more_itertools import all_equal, first, interleave_evenly
from toolz import valmap, curry, compose, do
from torch.utils.data import Dataset, default_collate, default_convert
from typing import Any, Callable, Dict, Generic, Hashable, List, Mapping, Sequence, Tuple, TypeVar, Union
from typing_extensions import Self
import numpy as np
import pandas as pd
import toolz.curried as C
import torch
from torch import Tensor

from ccbir.util import leaves_map, strict_update_in

BatchDictLike = Dict[Any, Union[Sequence, 'BatchDictLike']]


class BatchDict:
    """Dataframe like data structure that holds """

    def __init__(self, features: BatchDictLike):
        assert isinstance(features, dict)
        lengths = self._get_features_lengths(features)
        assert len(lengths) > 0
        assert all_equal(lengths)

        self._len = lengths[0]
        self._dict = features

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index) -> Dict:
        return leaves_map(C.get(index), self.dict())

    def dict(self) -> Dict:
        return self._dict

    @curry
    def map(self, func: Callable[[BatchDictLike], BatchDictLike]) -> Self:
        return self.__class__(func(self.dict()))

    @classmethod
    def zip(cls, batch_dicts: Dict[Any, BatchDict]) -> Self:
        return cls(valmap(cls.dict, batch_dicts))

    @classmethod
    def _get_features_lengths(cls, features: Any) -> List[int]:
        lengths = []
        _ = leaves_map(compose(lengths.append, len), features, strict=False)
        return lengths

    @curry
    def map_feature(
        self,
        keys: Sequence,
        func: Callable[[Sequence], Sequence],
        strict: bool = True,
    ) -> Self:
        update_in = strict_update_in if strict else C.update_in
        return self.map(update_in(keys=keys, func=func))

    @curry
    def set_feature(
        self,
        keys: Sequence,
        value: Sequence,
    ) -> Self:
        return self.map(C.assoc_in(keys=keys, value=value))


class InterleaveDataset(Dataset):
    """Inteleaves all items in all the given datasets. Assumes that
    items have the same type across datasets."""

    def __init__(self, datasets: List[Dataset]):
        super().__init__()
        self.datasets = datasets

        self.dataset_with_item_idx_for: Mapping[int, Tuple[Dataset, int]] = (
            list(interleave_evenly([
                list(zip(repeat(dataset), range(len(dataset))))
                for dataset in datasets
            ]))
        )

    def __len__(self) -> int:
        return len(self.dataset_with_item_idx_for)

    def __getitem__(self, index):
        dataset_with_item_idx = self.dataset_with_item_idx_for[index]
        if isinstance(index, slice):
            return default_collate([
                dataset[idx] for dataset, idx in dataset_with_item_idx
            ])
        else:
            dataset, item_idx = dataset_with_item_idx
            return dataset[item_idx]


def default_convert_series(s: pd.Series) -> Union[np.ndarray, Tensor]:
    """Attempts conversion to torch tensor type, casting floats to default
    torch datatype (float32)"""

    values = s.to_numpy()
    if np.issubdtype(values.dtype, np.floating):
        default_float_dtype = torch.get_default_dtype()
        return torch.as_tensor(values, dtype=default_float_dtype)
    else:
        return default_convert(values)


def tensor_from_numpy_image(
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