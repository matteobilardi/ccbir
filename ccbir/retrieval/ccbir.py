from typing import Dict, Optional, Tuple
from more_itertools import all_equal, first
from torch import LongTensor, Tensor
from ccbir.util import leaves_map, star_apply
from ccbir.data.util import BatchDict
from ccbir.inference.engine import InferenceEngine
from ccbir.retrieval.cbir import CBIR
from toolz import valmap, merge_with


class CCBIR:
    def __init__(
        self,
        inference_engine: InferenceEngine,
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

        self.inference_engine = inference_engine
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
        observed_factual_embeddings = valmap(
            func=self.extract_embedding,
            d=observed_factual_images,
        )
        best_sampled_embeddings = self.inference_engine.infer_outcomes(
            treatments=treatments,
            confounders=confounders,
            num_samples=num_samples,
            observed_factual_outcomes=observed_factual_embeddings,
            top_k=1,  # this is meant to be 1
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
