from xml.etree.ElementInclude import include
from ccbir.util import leaves_map, maybe_unbatched_apply
from ccbir.experiment.util import plot_tsne, show_tensor
from ccbir.inference.engine import SwollenFracturedMorphoMNISTEngine
from ccbir.experiment.data import TwinNetExperimentData, VQVAEExperimentData
from ccbir.models.twinnet.data import PSFTwinNetDataModule, PSFTwinNetDataset
from ccbir.models.twinnet.model import PSFTwinNet
from ccbir.models.twinnet.train import vqvae_embed_image
from ccbir.models.vqvae.data import VQVAEMorphoMNISTDataModule
from ccbir.models.vqvae.model import VQVAE
from functools import cached_property, partial
from more_itertools import interleave
from torch import Tensor
from torchvision import transforms
from toolz import first
from torchvision.utils import make_grid
from typing import Callable, Dict, List, Literal, Optional, Type, Union
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch

# TODO: rename file


class VQVAEExperiment:
    def __init__(self, vqvae: VQVAE):
        self.vqvae = vqvae
        self.device = vqvae.device
        self.data = VQVAEExperimentData()

        # TODO: hacky - this should not be needed in VQVAE experiments
        self.twinnet_data = TwinNetExperimentData(
            embed_image=partial(vqvae_embed_image, vqvae),
        )

    def show_vqvae_recons(
        self,
        num_images: int = 32,
        train: bool = False,
        dpi: int = 200,
    ):
        images = first(self.data.dataloader(train))[:num_images]
        images = images.to(self.device)

        with torch.no_grad():
            recons, _z_e, _z_q = self.vqvae(images)

        # TODO remove
        print(f"{_z_q.shape=}")

        show_tensor(make_grid(
            tensor=torch.cat((images, recons)),
            normalize=True,
            value_range=(-1, 1),
        ), dpi=dpi)

    def _vqvae_z_q(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            _x_hat, _z_e_x, z_q_x = self.vqvae(x.to(self.device))
            return z_q_x.to(x.device)

    # TODO: use perturbation types instead of literals

    def plot_vqvae_tsne(
        self,
        include_perturbations: List[Literal[
            'swollen', 'fractured'
        ]] = ['plain', 'swollen', 'fractured'],
        perplexity: int = 30,
        n_iter: int = 1000,
        num_points: int = 500,
        train: bool = False,
    ):
        psf_items = self.twinnet_data.psf_items(train, slice(num_points))

        latents_for_perturbation = {
            perturbation: self._vqvae_z_q(psf_items[perturbation]['image'])
            for perturbation in include_perturbations
        }

        labels = psf_items['plain']['label']

        plot_tsne(latents_for_perturbation, labels, perplexity, n_iter)


class PSFTwinNetExperiment:
    def __init__(
        self,
        vqvae: VQVAE,
        twinnet: PSFTwinNet,
    ):
        assert vqvae.device == twinnet.device
        self.device = vqvae.device
        self.vqvae = vqvae
        self.twinnet = twinnet
        self.data = TwinNetExperimentData(
            embed_image=partial(vqvae_embed_image, vqvae),
        )
        self.infer_engine = SwollenFracturedMorphoMNISTEngine(
            vqvae=vqvae,
            twinnet=twinnet,
        )

    def vqvae_recon(self, z_e_x: Tensor) -> Tensor:
        """z_e_x is the encoding of the input produced by the encoder network of
        the vqvae before it's mapped to the indices of specific codewords via
        the learned codebook"""

        with torch.no_grad():
            _, e_x, _ = self.vqvae.model.vq(z_e_x.to(self.device))
            x_recon = self.vqvae.decode(e_x)

        return x_recon.to(z_e_x.device)

    def encode(self, image: Tensor) -> Tensor:
        with torch.no_grad():
            return maybe_unbatched_apply(self.vqvae.encode, image)

    def decode(self, latent: Tensor) -> Tensor:
        with torch.no_grad():
            return maybe_unbatched_apply(
                self.vqvae.decode,
                latent,
                single_item_dims=2,
            )

    def prepare_query(
        self,
        item_idx: int,
        train: bool,
        condition_on_factual_outcome: bool,
        include_metadata: bool,
        include_ground_truth_strip: bool,
    ) -> Dict:
        dataset = self.data.dataset(train)
        X, _y = dataset[item_idx]
        psf_item = self.data.psf_items(train, item_idx)
        swollen_image = psf_item['swollen']['image']
        if include_ground_truth_strip:
            assert include_metadata

        swollen_embedding = self.encode(swollen_image)

        if include_metadata:
            original_image = psf_item['plain']['image']
            fractured_image = psf_item['fractured']['image']
            fractured_embedding = self.encode(fractured_image)
            metadata = dict(
                original_image=original_image,
                swollen_image=swollen_image,
                swollen_embedding=swollen_embedding,
                fractured_image=fractured_image,
                fractured_embedding=fractured_embedding,
            )

            if include_ground_truth_strip:
                ground_truths = torch.stack((
                    original_image,
                    swollen_image,
                    self.decode(swollen_embedding),
                    fractured_image,
                    self.decode(fractured_embedding),
                ))

                ground_truth_strip = make_grid(
                    tensor=ground_truths,
                    normalize=True,
                    value_range=(-1, 1)
                )
                metadata['ground_truth_strip'] = ground_truth_strip

        X = leaves_map(lambda t: t.to(self.device), X)
        query = dict(
            treatments=dict(
                swell=X['factual_treatment'],
                fracture=X['counterfactual_treatment'],
            ),
            confounders=X['confounders'],
        )
        if condition_on_factual_outcome:
            query['observed_factual_outcomes'] = dict(
                swell=swollen_embedding
            )

        return query, metadata

    def benchmark_generative_model(self):
        pass

    def show_twinnet_samples(
        self,
        item_idx: int,
        num_samples=32,
        noise_scale: float = 1,
        condition_on_factual_outcome: bool = False,
        top_k: Optional[int] = None,
        train=False,
        dpi=300,
    ):
        assert (
            (not condition_on_factual_outcome and top_k is None) or
            (condition_on_factual_outcome and top_k is not None)
        )
        query, metadata = self.prepare_query(
            item_idx=item_idx,
            train=train,
            condition_on_factual_outcome=condition_on_factual_outcome,
            include_metadata=True,
            include_ground_truth_strip=True,
        )

        if condition_on_factual_outcome:
            outcomes = self.infer_engine.infer_outcomes(
                **query,
                num_samples=num_samples,
                noise_scale=noise_scale,
                top_k=top_k,
            )
        else:
            outcomes = self.infer_engine.sample_outcomes(
                **query,
                num_samples=num_samples,
                noise_scale=noise_scale,
            )

        swollen_embedding_hat = outcomes.dict['swell']
        fractured_embedding_hat = outcomes.dict['fracture']

        # make the swollen and fractured embedding sampled from the same input
        # and noise show up one after the other to ease visual inspection
        paired_up_embeddings = torch.stack(list(
            interleave(swollen_embedding_hat, fractured_embedding_hat)
        ))

        # paired up images sampled via the twinnet
        images_hat = self.decode(paired_up_embeddings)

        show_tensor(metadata['ground_truth_strip'])
        show_tensor(
            make_grid(
                tensor=images_hat,
                normalize=True,
                value_range=(-1, 1),
            ),
            dpi=dpi,
        )

    def _show_twinnet_samples(
        self,
        item_idx: int,
        num_samples=32,
        noise_scale: float = 1.0,
        train=False,
        dpi=300,
    ):
        dataset = self.data.dataset(train)
        X, y = dataset[item_idx]
        psf_item = self.data.psf_items(train, item_idx)
        original_image = psf_item['plain']['image'].unsqueeze(0)
        swollen_image = psf_item['swollen']['image'].unsqueeze(0)
        fractured_image = psf_item['fractured']['image'].unsqueeze(0)
        swollen_embedding = y['factual_outcome'].unsqueeze(0)
        fractured_embedding = (
            y['counterfactual_outcome'].unsqueeze(0)
        )

        print(f"{swollen_embedding.shape=}")
        print(f"{fractured_embedding.shape=}")

        ground_truths = torch.cat((
            original_image,
            swollen_image,
            self.vqvae_recon(swollen_embedding),
            fractured_image,
            self.vqvae_recon(fractured_embedding),
        ))

        # repeat input num_samples time
        X = {
            k: (
                v.unsqueeze(0)
                .clone()
                .repeat_interleave(num_samples, dim=0)
            )
            for k, v in X.items()
        }

        X['outcome_noise'] = noise_scale * torch.randn(
            (num_samples, self.twinnet.outcome_noise_dim),
        )

        swollen_embedding_hat, fractured_embedding_hat = self.twinnet(X)

        # make the swollen and fractured embedding sampled from the same input
        # and noise show up one after the other to ease visual inspection
        paired_up_embeddings = torch.stack(list(
            interleave(swollen_embedding_hat, fractured_embedding_hat)
        ))

        # paired up images sampled via the twinnet
        images_hat = self.vqvae_recon(paired_up_embeddings)

        print('original, swollen, swollen_vqvae, fractured, fractured_vqvae')
        show_tensor(make_grid(
            tensor=ground_truths,
            normalize=True,
            value_range=(-1, 1)
        ))
        print('Twin network outputted swollen-fractured pairs')
        show_tensor(
            make_grid(
                tensor=images_hat,
                normalize=True,
                value_range=(-1, 1),
            ),
            dpi=dpi,
        )

    def plot_twinnet_tsne(
        self,
        perplexity: int = 30,
        n_iter: int = 1000,
        num_points: int = 500,
        train: bool = False,
    ):
        # for each original image, generate factual and counterfactual
        # embedding, flatten, run tsne split and plot
        x, _y = self.data.dataset(train)[:num_points]
        batch_size = x['factual_treatment'].shape[0]
        x['outcome_noise'] = torch.randn(
            (batch_size, self.twinnet.outcome_noise_dim)
        )
        swollen, fractured = self.twinnet.forward(x)
        latents_for_perturbations = dict(
            swollen=swollen,
            fractured=fractured,
        )
        psf_items = self.data.psf_items(train, slice(num_points))
        labels = psf_items['plain']['label']

        plot_tsne(latents_for_perturbations, labels, perplexity, n_iter)
