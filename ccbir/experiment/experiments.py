from matplotlib import image
from ccbir.data.morphomnist.dataset import (
    FracturedMorphoMNIST,
    SwollenMorphoMNIST
)
from ccbir.data.util import BatchDict
from ccbir.experiment.data import TwinNetExperimentData, VQVAEExperimentData
from ccbir.experiment.util import hit_rate, ndcg, plot_tsne, reciprocal_rank, show_tensor
from ccbir.inference.engine import SwollenFracturedMorphoMNIST_Engine
from ccbir.models.twinnet.model import PSFTwinNet
from ccbir.models.twinnet.train import vqvae_embed_image
from ccbir.models.vqvae.model import VQVAE
from ccbir.retrieval.cbir import (
    CBIR_Pipeline,
    SSIM_CBIR_Pipeline,
    VQVAE_CBIR_Pipeline,
)
from ccbir.retrieval.ccbir import CCBIR_Pipeline
from ccbir.util import leaves_map, maybe_unbatched_apply, star_apply
from functools import partial
from more_itertools import interleave
from toolz import first, merge_with, valmap, compose
from torch import LongTensor, Tensor
from torchmetrics.functional import (
    structural_similarity_index_measure as ssim,
)
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from typing import Callable, Dict, Iterable, List, Literal, Optional, Type, Union
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import toolz.curried as C
import torch


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
        self.infer_engine = SwollenFracturedMorphoMNIST_Engine(
            vqvae=vqvae,
            twinnet=twinnet,
        )

    def vqvae_recon(self, image: Tensor) -> Tensor:
        def recon(x: Tensor):
            with torch.no_grad():
                x_recon, _, _ = self.vqvae(x.to(self.device))
                return x_recon.to(x.device)

        return maybe_unbatched_apply(recon, image)

    def encode(self, image: Tensor) -> Tensor:
        with torch.no_grad():
            return maybe_unbatched_apply(
                self.vqvae.encode,
                image.to(self.device),
            ).to(image.device)

    def decode(self, latent: Tensor) -> Tensor:
        with torch.no_grad():
            return maybe_unbatched_apply(
                self.vqvae.decode,
                latent.to(self.device),
                single_item_dims=2,
            ).to(latent.device)

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
        swollen_image = psf_item['swollen']['image'].to(self.device)
        if include_ground_truth_strip:
            assert include_metadata

        swollen_embedding = self.encode(swollen_image)

        if include_metadata:
            original_image = psf_item['plain']['image'].to(self.device)
            fractured_image = psf_item['fractured']['image'].to(self.device)
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

        X = leaves_map(partial(Tensor.to, device=self.device), X)
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

    @torch.no_grad()
    def benchmark_generative_model(
        self,
        num_samples: int = 1024,
        noise_scale: float = 1.0,
        max_items: int = 20000,
    ) -> Dict:
        prepare_query = partial(
            self.prepare_query,
            train=False,
            condition_on_factual_outcome=True,
            include_metadata=True,
            include_ground_truth_strip=False,
        )
        num_items = min(max_items, len(self.data.dataset(train=False)))
        item_idxs = range(num_items)
        queries = map(prepare_query, item_idxs)
        cumulative_metrics = dict()

        for query, metadata in tqdm(queries, total=num_items):
            outcomes_ = self.infer_engine.infer_outcomes(
                **query,
                num_samples=num_samples,
                noise_scale=noise_scale,
                top_k=1,
            ).dict

            original_img = dict(
                swell=metadata['swollen_image'],
                fracture=metadata['fractured_image'],
            )
            outcome = dict(
                swell=metadata['swollen_embedding'],
                fracture=metadata['fractured_embedding'],
            )
            original_vqvae_img = valmap(self.decode, outcome)
            outcome_ = valmap(partial(Tensor.squeeze, dim=0), outcomes_)
            fake_vqvae_img = valmap(self.decode, outcome_)

            _outcomes_distance = star_apply(compose(
                Tensor.item,
                self.infer_engine.outcomes_distance,
            ))

            # MSE between observed discrete latents and reconstructed latents
            embedding_mse = merge_with(_outcomes_distance, outcome_, outcome)
            # cosine similarity between observed discrete latents and
            # reconstructed latents
            embedding_cosine_similarity = valmap(
                lambda mse: 1 - mse / 2,
                embedding_mse,
            )

            def _ssim(args) -> float:
                [preds, target] = args
                return ssim(
                    preds=preds.unsqueeze(0),
                    target=target.unsqueeze(0),
                    data_range=2.0,  # 1.0 - (-1.0)
                ).item()

            ssim_dicts = C.merge_with(_ssim)

            metrics = dict(
                embedding_mse=embedding_mse,
                embedding_cosine_similarity=embedding_cosine_similarity,
                # ssim between original and vqvae reconsturction
                ssim_orig_vqvae=ssim_dicts(original_vqvae_img, original_img),
                # ssim between vqvae reconstruction and best sampled outcomes
                ssim_vqvae_fake=ssim_dicts(fake_vqvae_img, original_vqvae_img),
                # ssim between original and best sampled outcomes
                ssim_orig_fake=ssim_dicts(fake_vqvae_img, original_img),
            )

            cumulative_metrics = merge_with(
                C.merge_with(sum),
                cumulative_metrics,
                metrics,
            )

        avg_metrics = leaves_map(
            lambda x: x / len(item_idxs),
            cumulative_metrics,
        )

        return avg_metrics

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


class RetrievalExperiment:
    def __init__(
        self,
        vqvae: VQVAE,
        twinnet: PSFTwinNet,
        train_datasets: bool = False,
        num_images: int = -1,  # include all images by default
    ):
        assert vqvae.device == twinnet.device
        self.device = vqvae.device

        normalize = transforms.Normalize(mean=0.5, std=0.5)
        kwargs = dict(
            train=train_datasets,
            transform=partial(
                BatchDict.map,
                func=lambda item: dict(image=normalize(item['image'])),
            ),
            repeats=1,
        )

        print('Loading datasets')
        datasets = dict(
            swell=SwollenMorphoMNIST(**kwargs),
            fracture=FracturedMorphoMNIST(**kwargs),
        )

        self.twinnet_data = TwinNetExperimentData(
            embed_image=partial(vqvae_embed_image, vqvae),
        )

        print('Loaded datasets')

        self.images = valmap(
            lambda d: d[:num_images]['image'].to(self.device),
            datasets,
        )
        self.ground_truth_cbir_pipelines: Dict[str, CBIR_Pipeline] = (
            valmap(SSIM_CBIR_Pipeline, self.images)
        )
        self.cbir_pipelines: Dict[str, CBIR_Pipeline] = (
            valmap(partial(VQVAE_CBIR_Pipeline, vqvae=vqvae), self.images)
        )
        self.ccbir_pipeline = CCBIR_Pipeline(
            inference_engine=SwollenFracturedMorphoMNIST_Engine(
                vqvae=vqvae,
                twinnet=twinnet,
            ),
            cbir_for_treatment_type=self.cbir_pipelines,
        )

    def _benchmark_cbir_pipeline(
        self,
        query_images: Tensor,
        cbir: CBIR_Pipeline,
        ground_truth_cbir: CBIR_Pipeline,
    ) -> Dict:
        cumulative_metrics = dict()
        for image in tqdm(query_images):
            _, pred_idxs = cbir.find_closest(image, top_k=None)
            target_images, target_idxs = (
                ground_truth_cbir.find_closest(image, top_k=None)
            )
            assert torch.allclose(image, target_images[0])

            relevant_result = target_idxs[0].item()
            metrics = dict(
                ndcg=ndcg(pred_idxs, target_idxs),
                hit_rate_at_1=hit_rate(pred_idxs, relevant_result, k=1),
                hit_rate_at_5=hit_rate(pred_idxs, relevant_result, k=5),
                reciprocal_rank=reciprocal_rank(pred_idxs, relevant_result),
            )

            cumulative_metrics = merge_with(sum, cumulative_metrics, metrics)

        avg_metrics = valmap(
            lambda x: x / len(query_images),
            cumulative_metrics,
        )

        return avg_metrics

    def benchmark_cbir_vqvae(
        self,
        num_queries: int = 1000,
    ) -> Dict:
        query_images = valmap(lambda imgs: imgs[:num_queries], self.images)
        results = merge_with(
            star_apply(self._benchmark_cbir_pipeline),
            query_images,
            self.cbir_pipelines,
            self.ground_truth_cbir_pipelines,
        )

        return results

    def _benchmark_ccbir_pipeline(
        self,
        factual_treatment_type: str,
        counterfactual_treatment_type: str,
        queries: BatchDict,
        num_samples: int,
    ) -> Dict:
        assert factual_treatment_type != counterfactual_treatment_type

        gt_cbir_star = self.cbir_pipelines[counterfactual_treatment_type]
        cumulative_metrics = dict()

        for query in tqdm(queries.iter_rows(), total=len(queries)):
            gt_images = query['ground_truth_image']
            gt_image_star: Tensor = gt_images[counterfactual_treatment_type]
            target_images, target_idxs = gt_cbir_star.find_closest(
                image=gt_image_star,
                top_k=None,
            )
            assert torch.allclose(gt_image_star, target_images[0])

            preds = self.ccbir_pipeline.find_closest_counterfactuals(
                treatments=query['treatments'],
                confounders=query['confounders'],
                observed_factual_images={
                    factual_treatment_type: gt_images[factual_treatment_type],
                },
                num_samples=num_samples,
                top_k=None,
            ).dict
            _pred_images, pred_idxs = preds[counterfactual_treatment_type]

            relevant_result = target_idxs[0].item()
            metrics = dict(
                ndcg=ndcg(pred_idxs, target_idxs),
                hit_rate_at_1=hit_rate(pred_idxs, relevant_result, k=1),
                hit_rate_at_5=hit_rate(pred_idxs, relevant_result, k=5),
                hit_rate_at_10=hit_rate(pred_idxs, relevant_result, k=10),
                hit_rate_at_15=hit_rate(pred_idxs, relevant_result, k=15),
                reciprocal_rank=reciprocal_rank(pred_idxs, relevant_result),
            )
            cumulative_metrics = merge_with(sum, cumulative_metrics, metrics)

        avg_metrics = valmap(lambda x: x / len(queries), cumulative_metrics)

        return avg_metrics

    def benchmark_ccbir(
        self,
        num_queries: int = 1000,
        num_samples: int = 1024,
    ) -> Dict:
        psf_items = (
            self.twinnet_data.psf_items(train=False, index=slice(num_queries))
        )
        x, _ = self.twinnet_data.dataset(train=False)[:num_queries]

        queries = BatchDict(dict(
            treatments=dict(
                swell=x['factual_treatment'],
                fracture=x['counterfactual_treatment'],
            ),
            ground_truth_image=dict(
                swell=psf_items['swollen']['image'],
                fracture=psf_items['fractured']['image'],
            ),
            confounders=x['confounders'],
        ))

        queries = queries.map_features(lambda tensor: tensor.to(self.device))

        results = dict(
            swell=self._benchmark_ccbir_pipeline(
                factual_treatment_type='swell',
                counterfactual_treatment_type='fracture',
                queries=queries,
                num_samples=num_samples,
            ),
            fracture=self._benchmark_ccbir_pipeline(
                factual_treatment_type='fracture',
                counterfactual_treatment_type='swell',
                queries=queries,
                num_samples=num_samples,
            ),
        )

        return results
