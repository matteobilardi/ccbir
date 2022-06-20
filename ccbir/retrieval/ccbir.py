from abc import ABC
from functools import partial
from pprint import pprint
from typing import Dict, Optional, Tuple
from einops import repeat
from more_itertools import all_equal, first
from torch import LongTensor, Tensor
import torch
from ccbir.util import A, leaves_map, star_apply
from ccbir.data.util import BatchDict, BatchDictLike
from ccbir.inference.engine import InferenceEngine
from ccbir.retrieval.cbir import CBIR
from ccbir.models.twinnet.model_vq_elbo import PSFTwinNet, V
from toolz import valmap, merge_with


class CCBIR_Base(ABC):
    def __init__(
        self,
        cbir_for_treatment_type: Dict[str, CBIR],
    ):
        cbirs = cbir_for_treatment_type

        # check that the pipelines share the same methods for embedding
        # extraction and distance calculation i.e. they should only differ in
        # the images they contain
        assert all_equal(
            cbir.__class__.extract_embedding for cbir in cbirs.values())
        assert all_equal(
            cbir.__class__.embeddings_distance for cbir in cbirs.values()
        )
        self.cbir_for_treatment_type = cbir_for_treatment_type
        self.extract_embedding = first(cbirs.values()).extract_embedding

    def find_closest_counterfactuals(
        self,
        treatments: Dict[str, Tensor],
        confounders: Tensor,
        observed_factual_images: Dict[str, Tensor],
        top_k: Optional[int] = 1,
        num_samples=1024,
    ) -> BatchDict:
        raise NotImplementedError


class ELBO_CCBIR(CCBIR_Base):
    def __init__(
        self,
        twinnet: PSFTwinNet,
        cbir_for_treatment_type: Dict[str, CBIR],
    ):
        super().__init__(cbir_for_treatment_type)
        self.twinnet = twinnet

    def find_closest_counterfactuals(
        self,
        treatments: Dict[str, Tensor],
        confounders: Tensor,
        observed_factual_images: Dict[str, Tensor],
        factual_treatment_type: str,
        counterfactual_treatment_type: str,
        top_k: Optional[int],
        num_samples: int,
    ) -> Tuple[Tensor, LongTensor]:
        assert factual_treatment_type != counterfactual_treatment_type
        assert counterfactual_treatment_type not in observed_factual_images

        observed_factual_embedding = valmap(
            func=self.extract_embedding,
            d=observed_factual_images,
        )[factual_treatment_type]

        cbir = self.cbir_for_treatment_type[counterfactual_treatment_type]
        candidates_counterfactual_embeddings = cbir.embeddings

        # TODO: CHECK IF THIS IS JUST UNSQUEEZED DIMENSIONS THAT NEEDS TO BE INCREASED
        repeat_num_embeddings = partial(
            repeat,
            pattern='... -> n ...',
            n=len(candidates_counterfactual_embeddings),
        )
        if top_k is None:
            top_k = len(candidates_counterfactual_embeddings)

        x = {
            V.factual_treatment:
                treatments['swell'],
            V.counterfactual_treatment:
                treatments['fracture'],
            V.confounders:
                confounders,
        }

        x = valmap(repeat_num_embeddings, x)

        if factual_treatment_type == 'swell':
            y = {
                V.factual_outcome: repeat_num_embeddings(observed_factual_embedding),
                V.counterfactual_outcome: candidates_counterfactual_embeddings,
            }
        elif factual_treatment_type == 'fracture':
            y = {
                V.factual_outcome: candidates_counterfactual_embeddings,
                V.counterfactual_outcome: repeat_num_embeddings(observed_factual_embedding),
            }
        else:
            raise RuntimeError()

        y = valmap(self.twinnet.vqvae.vq.quantize_encoding, y)

        def eval_elbo(d: BatchDictLike) -> BatchDictLike:
            x = d['x']
            y = d['y']
            return dict(
                elbo=self.twinnet.eval_elbo(
                    batch=(x, y),
                    num_samples=num_samples,
                )
            )
        elbo = BatchDict(dict(x=x, y=y)).split_map_cat(
            func=eval_elbo,
            split_size=4096,
        ).dict['elbo']

        top_elbo, top_idxs = torch.topk(elbo, k=top_k, largest=True)
        #torch.set_printoptions(profile='full')
        print(top_elbo)
        #torch.set_printoptions(profile='default')

        return cbir.images[top_idxs], top_idxs


class CCBIR(CCBIR_Base):
    def __init__(
        self,
        inference_engine: InferenceEngine,
        cbir_for_treatment_type: Dict[str, CBIR],
    ):
        super().__init__(cbir_for_treatment_type)
        self.inference_engine = inference_engine

    def find_closest_counterfactuals(
        self,
        treatments: Dict[str, Tensor],
        confounders: Tensor,
        observed_factual_images: Dict[str, Tensor],
        factual_treatment_type: str,
        counterfactual_treatment_type: str,
        top_k: int,
        num_samples: int,
    ) -> Tuple[Tensor, LongTensor]:
        assert factual_treatment_type != counterfactual_treatment_type
        assert counterfactual_treatment_type not in observed_factual_images

        observed_factual_embeddings = valmap(
            func=self.extract_embedding,
            d=observed_factual_images,
        )
        best_sampled_embeddings = self.inference_engine.infer_outcomes(
            treatments=treatments,
            confounders=confounders,
            num_samples=num_samples,
            observed_factual_outcomes=observed_factual_embeddings,
            top_k=1,
        ).dict
        best_sampled_embeddings = valmap(
            lambda t: t.squeeze(0),
            best_sampled_embeddings,
        )

        cbir = self.cbir_for_treatment_type[counterfactual_treatment_type]
        embedding = best_sampled_embeddings[counterfactual_treatment_type]
        return cbir.find_closest_by_embedding(embedding, top_k=top_k)

    """
    def find_closest_counterfactuals(
        self,
        treatments: Dict[str, Tensor],
        confounders: Tensor,
        observed_factual_images: Dict[str, Tensor],
        top_k: Optional[int] = 1,
        num_samples=1024,
    ) -> BatchDict:
        observed_factual_embeddings = valmap(
            func=self.extract_embedding,
            d=observed_factual_images,
        )
        best_sampled_embeddings = self.inference_engine.infer_outcomes(
            treatments=treatments,
            confounders=confounders,
            num_samples=num_samples,
            observed_factual_outcomes=observed_factual_embeddings,
            top_k=1,
        ).dict
        best_sampled_embeddings = valmap(
            lambda t: t.squeeze(0),
            best_sampled_embeddings,
        )

        def _find_closest_by_embedding(
            cbir: CBIR,
            embedding: Tensor,
        ) -> Tuple[Tensor, LongTensor]:
            return cbir.find_closest_by_embedding(
                embedding=embedding,
                top_k=top_k,
            )

        closest_images_with_idx: Dict[str, Tuple[Tensor, LongTensor]] = (
            merge_with(
                star_apply(_find_closest_by_embedding),
                self.cbir_for_treatment_type,
                best_sampled_embeddings,
            )
        )

        return BatchDict(closest_images_with_idx)
    """
