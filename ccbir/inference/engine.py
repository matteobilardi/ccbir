from abc import ABC, abstractmethod
from ccbir.models.util import load_best_model
from ccbir.models.vqvae.model import VQVAE
from ccbir.data.util import BatchDict
from ccbir.models.twinnet.model import PSFTwinNet, TwinNet
from ccbir.util import star_apply
from einops import rearrange, repeat, reduce
from functools import partial
from toolz import keyfilter, merge_with, valmap
from torch import Tensor
from typing import Dict, Optional
import torch
import torch.nn.functional as F


class InferenceEngine(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def outcomes_distance(
        self,
        outcome_samples: Tensor,
        target_outcome: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_outcomes(
        self,
        treatments: Dict[str, Tensor],
        confounders: Tensor,
        num_samples: int,
    ) -> BatchDict:
        """Generate num_samples joint outcomes for the given treatment and
        confounders"""
        raise NotImplementedError

    def _top_k_outcomes(
        self,
        sampled_outcomes: BatchDict,
        observed_factual_outcomes: Dict[str, Tensor],
        top_k: int,
    ) -> BatchDict:
        assert len(sampled_outcomes) >= top_k
        sampled_factual_outcomes = keyfilter(
            observed_factual_outcomes.__contains__,
            sampled_outcomes.dict,
        )
        distance_from_factual_outcome: Dict[str, Tensor] = merge_with(
            star_apply(self.outcomes_distance),
            sampled_factual_outcomes,
            observed_factual_outcomes,
        )
        distance = (
            torch.stack(list(distance_from_factual_outcome.values()))
            .mean(dim=0)
        )
        assert distance.shape[0] == len(sampled_outcomes)

        _, top_k_idxs = torch.topk(distance, k=top_k, largest=False)
        top_k_sampled_outcomes = BatchDict(sampled_outcomes[top_k_idxs])

        return top_k_sampled_outcomes

    def infer_outcomes(
        self,
        treatments: Dict[str, Tensor],
        confounders: Tensor,
        num_samples: int,
        observed_factual_outcomes: Dict[str, Tensor],
        top_k: int = 1,
        noise_scale: float = 1.0,
    ) -> BatchDict:
        """Generate num_samples joint outcomes for the given treatment and 
        confounders and return only the top_k outcomes whose factual outcomes
        are closest to the given factual_outcomes are returned."""
        assert (
            set(observed_factual_outcomes.keys())
            .issubset(treatments.keys())
        )

        sampled_outcomes: BatchDict = self.sample_outcomes(
            treatments,
            confounders,
            num_samples,
            noise_scale,
        )

        return self._top_k_outcomes(
            sampled_outcomes,
            observed_factual_outcomes,
            top_k,
        )


class VQVAE_Engine(InferenceEngine):
    def __init__(
        self,
        vqvae: VQVAE,
    ):
        super().__init__()
        self.vqvae = vqvae

    def outcomes_distance(
        self,
        outcome_samples: Tensor,
        outcome_target: Tensor,
    ) -> Tensor:
        if outcome_samples.dim() == 2:
            outcome_samples = outcome_samples.unsqueeze(0)

        assert outcome_samples.dim() == outcome_target.dim() + 1
        num_samples = outcome_samples.shape[0]
        outcome_targets = repeat(
            tensor=outcome_target,
            pattern='h w -> s h w',
            s=num_samples,
        )

        return self.vqvae.vq.latents_distance(outcome_samples, outcome_targets)


class SwollenFracturedMorphoMNIST_Engine(VQVAE_Engine):
    def __init__(
        self,
        vqvae: Optional[VQVAE] = None,
        twinnet: Optional[PSFTwinNet] = None,
    ):
        if twinnet is None:
            twinnet = load_best_model(PSFTwinNet)
        if vqvae is None:
            vqvae = twinnet.vqvae
        
        super().__init__(vqvae)
        self.twinnet = twinnet

    def sample_outcomes(
        self,
        treatments: Dict[str, Tensor],
        confounders: Tensor,
        num_samples: int,
        noise_scale: float = 1.0,
    ) -> BatchDict:
        worlds = {'fracture', 'swell'}
        assert worlds == set(treatments.keys())

        repeat_num_samples = partial(repeat, pattern='d -> s d', s=num_samples)

        x = repeat_num_samples(treatments['swell'])
        x_star = repeat_num_samples(treatments['fracture'])
        z = repeat_num_samples(confounders)
        u_y = noise_scale * torch.randn(
            size=(num_samples, self.twinnet.outcome_noise_dim),
            device=x.device,
        )
        input = dict(
            factual_treatment=x,
            counterfactual_treatment=x_star,
            confounders=z,
            outcome_noise=u_y,
        )

        with torch.no_grad():
            # FIXME: avoid accessing twinnet object interanl to twinnet
            # and decide what the output of the twinnet should be
            # consistently
            y, y_star = self.twinnet.twin_net.forward_discrete(**input)

        return BatchDict(dict(
            swell=y,
            fracture=y_star,
        ))
