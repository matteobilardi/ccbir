from __future__ import annotations

from deepscm.morphomnist.morpho import ImageMorphology
from deepscm.morphomnist.perturb import Perturbation
from typing import Any, Dict, List, Mapping, Tuple, Type
import numpy as np
import pandas as pd

import deepscm.morphomnist.skeleton as skeleton


# HACK START -----------------------------------------------------------------
# monkey patch LocationSampler to save sampled locations
OriginalLocationSampler = skeleton.LocationSampler


class SavingLocationSampler:
    """Location sampler that stores the sampled locations for later retrieval.
    Note that this class shouldn't subclass LocationSampler as it's meant to
    monkey patch it.
    """

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
