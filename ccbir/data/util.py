from __future__ import annotations
from itertools import starmap, repeat
import math
from more_itertools import all_equal, first, interleave_evenly
from sklearn.preprocessing import StandardScaler
from toolz import valmap, curry, compose, do, identity
from torch.utils.data import Dataset, default_collate, default_convert, Subset
from typing import Any, Callable, Dict, Generator, Generic, Hashable, List, Mapping, Sequence, Tuple, TypeVar, Union
from typing_extensions import Self
import numpy as np
import pandas as pd
import toolz.curried as C
import torch
from torch import Tensor, default_generator

from ccbir.util import NestedDict, array_like_ncycles, numpy_ncycles, leaves_map, strict_update_in, tensor_ncycles

BatchDictLike = NestedDict[Hashable, Sequence]


class BatchDict:
    """Convenince dataframe-like data structure that contains possibly nested
    dictionary of equally sized sequence objects (ideally tensors). This is
    similar to the batch that would yielded by a dataloader during training for
    a dataset whose __getitem__ method returns a dictionary object. """

    def __init__(self, features: BatchDictLike):
        assert isinstance(features, dict)
        lengths = self._get_features_lengths(features)
        assert len(lengths) > 0
        assert all_equal(lengths)
        super().__init__()

        self._len = lengths[0]
        self._dict = features

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index) -> Dict:
        return leaves_map(C.get(index), self.dict())

    def dict(self) -> Dict:
        return self._dict

    def map(self, func: Callable[[BatchDictLike], BatchDictLike]) -> Self:
        return self.__class__(func(self.dict()))

    def map_features(self, func: Callable[[Sequence], Sequence]) -> Self:
        return self.map(leaves_map(func))

    def map_feature(
        self,
        keys: Sequence,
        func: Callable[[Sequence], Sequence],
        strict: bool = True,
    ) -> Self:
        update_in = strict_update_in if strict else C.update_in
        return self.map(update_in(keys=keys, func=func))

    def set_feature(
        self,
        keys: Sequence,
        value: Sequence,
    ) -> Self:
        return self.map(C.assoc_in(keys=keys, value=value))

    def ncycles(self, n) -> Self:
        return self.map_features(array_like_ncycles(n=n))

    @classmethod
    def zip(cls, batch_dicts: Dict[Any, BatchDict]) -> Self:
        return cls(valmap(cls.dict, batch_dicts))

    @classmethod
    def _get_features_lengths(cls, features: BatchDictLike) -> List[int]:
        lengths = []
        _ = leaves_map(compose(lengths.append, len), features, strict=False)
        return lengths


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


def random_split_repeated(
    dataset: Dataset,
    train_ratio: float,
    repeats: int,
    generator: Generator = default_generator,
) -> Tuple[Subset, Subset]:
    """Randomly split (in a why that avoids data leakage) an augmented :dataset
    that was obtained by cyclying :repeats times an original dataset. If
    :repeats is 1, this function is equivalent to random_split as it's assumed
    that the dataset hasn't been augmented.

    If we split naively via torch random_split, two or more different items
    obtained from augmenting the same original item could occur in both the
    validation set and in the training set, we should be avoided for fair
    evaluation."""

    assert repeats >= 1

    original_len_f = len(dataset) / repeats
    assert original_len_f.is_integer()
    original_len = int(original_len_f)

    num_train = math.ceil(original_len * train_ratio)
    num_val = original_len - num_train
    assert num_train > 0 and num_val > 0

    shuffled_orig_idxs = torch.randperm(original_len, generator=generator)
    orig_train_idxs, orig_val_idxs = (
        torch.split(shuffled_orig_idxs, [num_train, num_val])
    )

    train_idxs = []
    val_idxs = []
    for cycle_idx in range(repeats):
        shift = cycle_idx * original_len
        train_idxs.append(orig_train_idxs + shift)
        val_idxs.append(orig_val_idxs + shift)

    train_idxs = torch.cat(train_idxs)
    val_idxs = torch.cat(val_idxs)

    return Subset(dataset, train_idxs), Subset(dataset, val_idxs)


def default_convert_series(
    series: pd.Series,
    all_numeric_to_float: bool,
) -> Union[np.ndarray, Tensor]:
    """Attempts conversion to torch tensor type, casting floats to default
    torch datatype (float32)"""

    values = series.to_numpy()
    if np.issubdtype(values.dtype, np.number) and (
        np.issubdtype(values.dtype, np.floating) or all_numeric_to_float
    ):
        default_float_dtype = torch.get_default_dtype()
        return torch.as_tensor(values, dtype=default_float_dtype)
    else:
        return default_convert(values)


def standard_scaler_for(seq: Sequence) -> Callable[[Sequence], Sequence]:
    """If :seq is a tensor, returns a scaling function that applies the standard
    scaler transformation fitted on the tensor data, otherwise returns the
    identity function"""

    if isinstance(seq, Tensor):
        assert seq.dim() == 1, seq.dim()
        scaler = StandardScaler()
        scaler = scaler.fit(seq.view(-1, 1).numpy())

        def scale(t: Tensor):
            assert t.dim() == 1, t.dim()
            array = t.view(-1, 1).numpy()
            normalized_array = scaler.transform(array).flatten()

            return torch.as_tensor(
                data=normalized_array,
                dtype=torch.float32,
            )

        return scale
    else:
        return identity


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
