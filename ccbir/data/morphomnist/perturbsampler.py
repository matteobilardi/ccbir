
from deepscm.morphomnist.perturb import Fracture, Perturbation, Swelling
from typing import List, Tuple, Type
import numpy as np
import pandas as pd


# TODO: not sure if the whole concept of a perturbation sampler and of saving to
# file the perturbation args makes sense. Might wanna remove if I conclusively
# establish that it is not needed
class PerturbationSampler:
    def __init__(self, perturbation_type: Type[Perturbation]):
        self.perturbation_type = perturbation_type

    def sample_args(self, num_samples: int) -> pd.DataFrame:
        raise NotImplementedError()

    def sample(
        self,
        num_samples: int
    ) -> Tuple[List[Perturbation], pd.DataFrame]:
        perturbations_args = self.sample_args(num_samples)
        perturbations = [
            # note that in generaral perturbation type constructors are
            # stochastic so that perturbations initialised with the same
            # arguments need not produce identical outputs
            self.perturbation_type(**kwargs)
            for kwargs in perturbations_args.to_dict(orient='records')
        ]

        return perturbations, perturbations_args


class SwellingSampler(PerturbationSampler):
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


class FractureSampler(PerturbationSampler):
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
