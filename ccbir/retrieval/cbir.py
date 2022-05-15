from abc import ABC, abstractmethod
from functools import partial
from ccbir.util import maybe_unbatched_apply
from ccbir.data.morphomnist.dataset import SwollenMorphoMNIST
from ccbir.data.util import BatchDict
from ccbir.models.util import load_best_model
from ccbir.models.vqvae.model import VQVAE
from einops import reduce, repeat
from torch import LongTensor, Tensor
from torchmetrics.functional import structural_similarity_index_measure as ssim
from typing import Callable, Dict, Optional, Tuple
import torch


class CBIR(ABC):
    def __init__(
        self,
        images: Tensor,
        embeddings: Optional[Tensor] = None
    ) -> None:
        super().__init__()

        if embeddings is None:
            print('Extracting embeddings...')
            embeddings = self.extract_embedding(images)
            print('Extracted embedding')

        assert len(images) == len(embeddings)
        self.images = images
        self.embeddings = embeddings

    def find_closest(
        self,
        image: Tensor,
        top_k: Optional[int] = 1,
    ) -> Tuple[Tensor, LongTensor]:
        """Returns top-k images in the dataset that are most similar to the
        given image in descending order of similarity, along with their indices

        If top_k is None, all images are returned in ranked order
        """
        embedding = self.extract_embedding(image)
        return self.find_closest_by_embedding(embedding, top_k)

    def find_closest_by_embedding(
        self,
        embedding: Tensor,
        top_k: Optional[int] = 1,
    ) -> Tuple[Tensor, LongTensor]:
        if top_k is None:
            top_k = len(self.embeddings)

        embedding = repeat(embedding, '... -> n ...', n=len(self.embeddings))
        distances = self.embeddings_distance(embedding, self.embeddings)
        _, closest_idxs = torch.topk(distances, k=top_k, largest=False)
        return self.images[closest_idxs], closest_idxs

    @abstractmethod
    def embeddings_distance(
        self,
        embedding1: Tensor,
        embedding2: Tensor
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def extract_embedding(self, image: Tensor) -> Tensor:
        raise NotImplementedError


class SSIM_CBIR(CBIR):
    def __init__(self, images: Tensor, embeddings: Optional[Tensor] = None):
        super().__init__(images, embeddings)

    def embeddings_distance(
        self,
        embedding1: Tensor,
        embedding2: Tensor,
    ) -> Tensor:
        assert embedding1.shape == embedding2.shape
        neg_ssim = -ssim(embedding1, embedding2, reduction=None)

        # need to reduce manually since ssim would otherwise produce
        # a mean over the whole batch
        neg_ssim = reduce(neg_ssim, 'b ... -> b', 'mean')

        return neg_ssim

    def extract_embedding(self, image: Tensor) -> Tensor:
        return image


class VQVAE_CBIR(CBIR):
    def __init__(
        self,
        images: Tensor,
        vqvae: VQVAE,
    ):
        self.vqvae = vqvae
        super().__init__(images)

    def embeddings_distance(
        self,
        embedding1: Tensor,
        embedding2: Tensor,
    ) -> Tensor:
        return self.vqvae.vq.latents_distance(embedding1, embedding2)

    def extract_embedding(self, image: Tensor) -> Tensor:
        with torch.no_grad():
            return maybe_unbatched_apply(self.vqvae.encode, image)
