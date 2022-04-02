from __future__ import annotations
from ctypes import Union

from attr import has
from deepscm.morphomnist.morpho import ImageMorphology
from deepscm.morphomnist.perturb import Fracture, Perturbation, Swelling
from typing import Any, Dict, List, Mapping, Tuple, Type
import numpy as np
import pandas as pd

import deepscm.morphomnist.skeleton as skeleton


# TODO: not sure if the whole concept of a perturbation sampler and of saving to
# file the perturbation args makes sense. Might wanna remove if I conclusively
# establish that it is not needed
class PerturbationArgsSampler:
    def __init__(self, perturbation_type: Type[Perturbation]):
        self.perturbation_type = perturbation_type

    def sample_args(self, num_samples: int) -> pd.DataFrame:
        raise NotImplementedError


class SwellingArgsSampler(PerturbationArgsSampler):
    def __init__(self):
        super().__init__(Swelling)

    def sample_args(self, num_samples: int) -> pd.DataFrame:
        # TODO: for now we don't actually sample the arguments and instead rely
        # on the swelling location sampling internal to the Swelling class and
        # apply a Swelling initialised with the same arguments to all images.
        # Also, for now we don't pass such arguments to the twin network as they
        # are constant. And the treatment is instead a one-hot encoding of an
        # enum: either swelling fracture or no-treatment (i.e. identity
        # function)
        strength = np.full((num_samples,), 3.0)
        radius = np.full((num_samples,), 7.0)

        return pd.DataFrame.from_dict(dict(
            strength=strength,
            radius=radius,
        ))


class FractureArgsSampler(PerturbationArgsSampler):
    def __init__(self):
        super().__init__(Fracture)

    def sample_args(self, num_samples) -> pd.DataFrame:
        # TODO: see TODO in same location for SwellingSampler
        thickness = np.full((num_samples,), 1.5, dtype=float)
        prune = np.full((num_samples,), 2.0, dtype=float)
        num_frac = np.full((num_samples,), 3, dtype=int)

        return pd.DataFrame.from_dict(dict(
            thickness=thickness,
            prune=prune,
            num_frac=num_frac,
        ))


# HACK START -----------------------------------------------------------------
# monkey patch LocationSampler to save sampled locations
OriginalLocationSampler = skeleton.LocationSampler


class SavingLocationSampler:
    """Location sampler that stores the sampled locations for later retrieval.
    Can only be used once."""

    def __init__(self, single_use: bool = True):
        self._single_use = single_use
        self._saved_locations = []

    def set_params(self, prune_tips: float = None, prune_forks: float = None):
        self.prune_tips = prune_tips
        self.prune_forks = prune_forks

    def sample(self, morph: ImageMorphology, num: int = None) -> np.ndarray:
        assert hasattr(self, 'prune_tips') and hasattr(self, 'prune_forks')
        if self._single_use and len(self._saved_locations) > 0:
            raise RuntimeError(
                'SavingLocationSampler can only be used to sample once.'
            )

        loc_sampler = OriginalLocationSampler(
            prune_tips=self.prune_tips,
            prune_forks=self.prune_forks,
        )

        locations = loc_sampler.sample(morph, num)

        # enforce np.ndarray in saved_location even when tuple is returned
        saved_location = np.array(locations).reshape(-1, 2)
        self._saved_locations.append(saved_location)

        return locations

    def saved_locations(self) -> np.ndarray:
        return np.concatenate(self._saved_locations)


# HACK END --------------------------------------------------------------------


def perturb_image(
    image: np.ndarray,
    perturbation_type: Type[Perturbation],
    perturbation_args: Mapping[str, Any],
    threshold: float = 0.5,
    up_factor: int = 4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    morph = ImageMorphology(image, threshold, up_factor)

    # save sampled locations
    loc_sampler = SavingLocationSampler()

    def loc_sampler_ctor(prune_tips=None, prune_forks=None):
        loc_sampler.set_params(prune_tips, prune_forks)
        return loc_sampler

    # apply monkey patch
    skeleton.LocationSampler = loc_sampler_ctor

    perturbation = perturbation_type(**perturbation_args)

    perturbed_image = morph.downscale(perturbation(morph))
    perturbation_data = dict(
        type=perturbation_type.__name__.lower(),
        args=perturbation_args,
    )

    perturbation_locations = loc_sampler.saved_locations()
    if len(perturbation_locations) > 0:
        perturbation_data['locations'] = {
            # converting location_idx to string for later json serialisation
            str(location_idx): dict(x=x, y=y)
            for location_idx, [x, y] in enumerate(perturbation_locations)
        }

    return perturbed_image, perturbation_data
