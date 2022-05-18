import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
from sympy import sequence
from ccbir.data.morphomnist.dataset import (
    FracturedMorphoMNIST,
    SwollenMorphoMNIST
)
from ccbir.data.util import BatchDict
from ccbir.experiment.data import OriginalMNIST_VQVAE_ExperimentData, TwinNetExperimentData, SwellFractureVQVAE_ExperimentData
from ccbir.experiment.util import avg_precision, hit_rate, ndcg, plot_tsne, reciprocal_rank, show_tensor
from ccbir.inference.engine import SwollenFracturedMorphoMNIST_Engine
from ccbir.models.twinnet.model import PSFTwinNet
from ccbir.models.twinnet.train import vqvae_embed_image
from ccbir.models.vqvae.model import VQVAE
from ccbir.retrieval.cbir import (
    CBIR,
    SSIM_CBIR,
    VQVAE_CBIR,
)
from ccbir.retrieval.ccbir import CCBIR
from ccbir.util import leaves_map, maybe_unbatched_apply, star_apply
from functools import partial
from more_itertools import all_equal, interleave, unzip
from toolz import first, merge_with, valmap, compose, concat, memoize
from torch import BoolTensor, LongTensor, Tensor, randperm
from torch.utils.data import default_collate
from torchmetrics.functional import (
    structural_similarity_index_measure as ssim,
)
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from typing import Callable, Dict, Iterable, List, Literal, Optional, Set, Tuple, Type, Union
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import toolz.curried as C
import torch


class VQVAEExperiment:
    def __init__(self, vqvae: VQVAE):
        self.vqvae = vqvae
        self.device = vqvae.device
        self.data = SwellFractureVQVAE_ExperimentData()

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
                    value_range=(-1, 1),
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
        train: bool = False,
    ) -> Dict:
        prepare_query = partial(
            self.prepare_query,
            train=train,
            condition_on_factual_outcome=True,
            include_metadata=True,
            include_ground_truth_strip=False,
        )
        num_items = min(max_items, len(self.data.dataset(train=train)))
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
        vqvae_original_mnist: VQVAE,
        twinnet: PSFTwinNet,
        #     num_images: int = -1,  # include all images by default
    ):
        assert all_equal((
            vqvae.device,
            twinnet.device,
            vqvae_original_mnist.device,
        ))

        self.device = vqvae.device
        self.vqvae = vqvae
        self.twinnet = twinnet
        self.twinnet_data = TwinNetExperimentData(
            embed_image=partial(vqvae_embed_image, vqvae),
        )

        queries_data, partial_gallery_data, full_gallery_data = (
            self._load_queries_and_gallery_data()
        )

        self.queries_data = queries_data
        self.partial_gallery_data = partial_gallery_data
        self.full_gallery_data = full_gallery_data

        full_gallery = full_gallery_data.dict['ground_truth']['image']
        partial_gallery = partial_gallery_data.dict['ground_truth']['image']

        self.ground_truth_cbirs: Dict[str, CBIR] = dict(
            swell=SSIM_CBIR(full_gallery['swell']),
            fracture=SSIM_CBIR(full_gallery['fracture']),
            original=SSIM_CBIR(partial_gallery['original']),
        )
        self.cbirs: Dict[str, CBIR] = dict(
            swell=VQVAE_CBIR(full_gallery['swell'], vqvae),
            fracture=VQVAE_CBIR(full_gallery['fracture'], vqvae),
            original=VQVAE_CBIR(
                images=partial_gallery['original'],
                vqvae=vqvae_original_mnist,
            ),
        )
        self.ccbir = CCBIR(
            inference_engine=SwollenFracturedMorphoMNIST_Engine(
                vqvae=vqvae,
                twinnet=twinnet,
            ),
            cbir_for_treatment_type=self.cbirs,
        )

    def _load_queries_and_gallery_data(
        self,
        num_queries: int = 1000,
    ) -> Tuple[BatchDict, BatchDict, BatchDict]:
        twinnet_data = TwinNetExperimentData(
            embed_image=partial(vqvae_embed_image, self.vqvae),
        )
        assert num_queries % 10 == 0

        original_mnist_data = OriginalMNIST_VQVAE_ExperimentData()
        original_mnist_train_val = original_mnist_data.train_dataset.dataset
        original_mnist_test = original_mnist_data.test_dataset

        def cat(dicts_or_sequences):
            dicts = all(map(lambda d: isinstance(d, dict), dicts_or_sequences))
            if dicts:
                return merge_with(cat, *dicts_or_sequences)
            else:
                sequences = dicts_or_sequences
                assert all_equal(map(type, sequences))
                seq1 = sequences[0]
                if isinstance(seq1, Tensor):
                    return torch.cat(sequences)
                elif isinstance(seq1, np.ndarray):
                    return np.concatenate(sequences)
                elif isinstance(seq1, list):
                    return list(concat(sequences))
                else:
                    raise TypeError(f"Unsupported type {type(seq1)}")

        original_mnist_images = cat((
            original_mnist_test.get_items().dict,
            original_mnist_train_val.get_items().dict,
        ))['image']

        test_dataset = twinnet_data.test_dataset
        train_val_dataset = twinnet_data.train_dataset.dataset
        psf_items_test = test_dataset.psf_items.dict
        psf_items_train = train_val_dataset.psf_items.dict
        psf_items = cat((psf_items_test, psf_items_train))

        items = cat((
            test_dataset.get_items().dict,
            train_val_dataset.get_items().dict,
        ))
        x = items['x']

        labels = psf_items['plain']['label']

        data = BatchDict(dict(
            treatments=dict(
                swell=x['factual_treatment'],
                fracture=x['counterfactual_treatment'],
            ),
            ground_truth=dict(
                image=dict(
                    swell=psf_items['swollen']['image'],
                    fracture=psf_items['fractured']['image'],
                    original=original_mnist_images,
                ),
                label=labels,
            ),
            confounders=x['confounders'],
        )).map_features(lambda t: t.to(self.device))

        generator = torch.Generator(self.device).manual_seed(42)

        test_labels = psf_items_test['plain']['label'].to(self.device)
        num_queries_per_class = num_queries // 10

        def random_subset_query_idxs(class_queries_idxs):
            rand_idxs = torch.randperm(
                n=len(class_queries_idxs),
                generator=generator,
                device=class_queries_idxs.device,
            )
            _queries_idxs = rand_idxs[:num_queries_per_class]
            _test_gallery_idxs = rand_idxs[num_queries_per_class:]
            return (
                class_queries_idxs[_queries_idxs],
                class_queries_idxs[_test_gallery_idxs]
            )

        query_idxs_for_class, test_gallery_idxs_for_class = unzip(
            random_subset_query_idxs(
                (digit == test_labels).nonzero().flatten()
            ) for digit in range(10)
        )
        query_idxs = torch.cat(list(query_idxs_for_class))
        test_gallery_idxs = torch.cat(list(test_gallery_idxs_for_class))

        assert len(query_idxs) + len(test_gallery_idxs) == len(test_labels)
        assert torch.all(torch.eq(
            torch.unique(torch.cat([query_idxs, test_gallery_idxs])),
            torch.arange(len(test_dataset), device=self.device)
        ))

        queries_data = BatchDict(data[query_idxs])
        assert len(queries_data) == num_queries

        train_gallery_idxs = len(test_labels) + torch.randperm(
            n=len(train_val_dataset),
            generator=generator,
            device=self.device,
        )
        partial_gallery_idxs = torch.cat([
            test_gallery_idxs,
            train_gallery_idxs,
        ])

        partial_gallery_data = BatchDict(data[partial_gallery_idxs])
        full_gallery_idxs = torch.cat([
            query_idxs,
            partial_gallery_idxs,
        ])
        assert len(full_gallery_idxs) == len(data)
        full_gallery_data = BatchDict(data[full_gallery_idxs])
        assert (
            len(queries_data) + len(partial_gallery_data)
            == len(full_gallery_data)
        )
        assert len(full_gallery_data) == 70_000

        return queries_data, partial_gallery_data, full_gallery_data

    def cbir_show_query_result(
        self,
        treatment_type: Literal['swell', 'fracture'],
        query_idx: int,
        top_k: int = 64,
    ):
        cbir = self.cbirs[treatment_type]
        image = (
            self.queries_data
            [query_idx]['ground_truth']['image'][treatment_type]
        )
        show_tensor(image, dpi=50)
        images, _idxs = cbir.find_closest(image, top_k=top_k)
        show_tensor(
            make_grid(tensor=images, normalize=True, value_range=(-1, 1))
        )

    def ccbir_show_query_result(
        self,
        query_idx: int,
        factual_treatment_type: Literal['swell', 'fracture'],
        counterfactual_treatment_type: Literal['swell', 'fracture'],
        top_k: int = 64,
    ):
        factual_treatment_type != counterfactual_treatment_type

        query = self.data[query_idx]
        images = query['ground_truth']['image']
        for img in images.values():
            show_tensor(img, dpi=50)

        observed_factual_images = {
            factual_treatment_type: images[factual_treatment_type],
        }

        results = self.ccbir.find_closest_counterfactuals(
            treatments=query['treatments'],
            confounders=query['confounders'],
            observed_factual_images=observed_factual_images,
            top_k=top_k,
        ).dict

        images, _idxs = results[counterfactual_treatment_type]

        show_tensor(
            make_grid(tensor=images, normalize=True, value_range=(-1, 1))
        )

    @memoize
    def _is_relevant_idx(
        self,
        relevant_label: int,
        full_gallery: bool,
    ) -> Tensor:
        gallery_data = (
            self.full_gallery_data if full_gallery else
            self.partial_gallery_data
        )
        label = gallery_data.dict['ground_truth']['label']
        is_relvant_idx = label == relevant_label

        return is_relvant_idx

    def _query_metrics(
        self,
        pred_idxs: LongTensor,
        target_idxs: LongTensor,
        ground_truth_idx: int,
        ground_truth_label: int,
        full_gallery: bool,
    ) -> Dict:
        is_relvant_idx = self._is_relevant_idx(
            ground_truth_label,
            full_gallery,
        )

        return dict(
            hit_rate_at_1=hit_rate(pred_idxs, ground_truth_idx, k=1),
            hit_rate_at_5=hit_rate(pred_idxs, ground_truth_idx, k=5),
            hit_rate_at_10=hit_rate(pred_idxs, ground_truth_idx, k=10),
            hit_rate_at_15=hit_rate(pred_idxs, ground_truth_idx, k=15),
            reciprocal_rank=reciprocal_rank(pred_idxs, ground_truth_idx),
            ssim_ndcg=ndcg(pred_idxs, target_idxs),
            label_avg_precision=avg_precision(pred_idxs, is_relvant_idx),
            label_avg_precision_at_1000=(
                avg_precision(pred_idxs, is_relvant_idx, k=1000)
            ),
        )

    def show_precision_recall_curve(
        self,
        benchmark_results: Dict,
    ):
        raise NotImplementedError
        PrecisionRecallDisplay(
            precision=benchmark_results['precision'],
            recall=benchmark_results['recall']
        ).plot()
        plt.show()

    def _benchmark_cbir(
        self,
        queries: BatchDict,
        cbir: CBIR,
        ground_truth_cbir: CBIR,
        full_gallery: Optional[bool] = True
    ) -> Dict:
        cumulative_metrics = dict()
        for query in tqdm(queries.iter_rows(), total=len(queries)):
            image = query['image']
            _, pred_idxs = cbir.find_closest(image, top_k=None)
            target_images, target_idxs = (
                ground_truth_cbir.find_closest(image, top_k=None)
            )

            if full_gallery:
                assert torch.allclose(image, target_images[0])

            metrics = self._query_metrics(
                pred_idxs=pred_idxs,
                target_idxs=target_idxs,
                ground_truth_idx=target_idxs[0].item(),
                ground_truth_label=query['label'].item(),
                full_gallery=full_gallery,
            )

            cumulative_metrics = merge_with(sum, cumulative_metrics, metrics)

        avg_metrics = valmap(
            lambda x: x / len(queries),
            cumulative_metrics,
        )

        return avg_metrics

    @torch.no_grad()
    def benchmark_cbir(
        self,
        num_queries: int = 1000,
    ) -> Dict:
        queries = BatchDict(self.queries_data[:num_queries]['ground_truth'])

        def queries_for(treatment_type) -> BatchDict:
            return queries.map(
                C.update_in(keys=['image'], func=C.get(treatment_type))
            )

        results = dict()
        for dataset in ('original', ):  # , 'swell', 'fracture'):
            results[dataset] = self._benchmark_cbir(
                queries=queries_for(dataset),
                cbir=self.cbirs[dataset],
                ground_truth_cbir=self.ground_truth_cbirs[dataset],
                full_gallery=(dataset != 'original'),
            )

        return results

    def _benchmark_ccbir(
        self,
        factual_treatment_type: str,
        counterfactual_treatment_type: str,
        queries: BatchDict,
        num_samples: int,
    ) -> Dict:
        assert factual_treatment_type != counterfactual_treatment_type

        gt_cbir_star = self.cbirs[counterfactual_treatment_type]
        cumulative_metrics = dict()

        for query in tqdm(queries.iter_rows(), total=len(queries)):
            gt_images = query['ground_truth']['image']
            gt_image_star: Tensor = gt_images[counterfactual_treatment_type]
            target_images, target_idxs = gt_cbir_star.find_closest(
                image=gt_image_star,
                top_k=None,
            )
            assert torch.allclose(gt_image_star, target_images[0])

            preds = self.ccbir.find_closest_counterfactuals(
                treatments=query['treatments'],
                confounders=query['confounders'],
                observed_factual_images={
                    factual_treatment_type: gt_images[factual_treatment_type],
                },
                num_samples=num_samples,
                top_k=None,
            ).dict
            _pred_images, pred_idxs = preds[counterfactual_treatment_type]

            metrics = self._query_metrics(
                pred_idxs=pred_idxs,
                target_idxs=target_idxs,
                ground_truth_idx=target_idxs[0].item(),
                ground_truth_label=query['ground_truth']['label'].item(),
                full_gallery=True,
            )
            cumulative_metrics = merge_with(sum, cumulative_metrics, metrics)

        avg_metrics = valmap(lambda x: x / len(queries), cumulative_metrics)

        return avg_metrics

    @torch.no_grad()
    def benchmark_ccbir(
        self,
        num_queries: int = 1000,
        num_samples: int = 1024,
    ) -> Dict:
        queries = BatchDict(self.data[:num_queries])
        results = dict(
            factual_swell_counterfactual_fracture=self._benchmark_ccbir(
                factual_treatment_type='swell',
                counterfactual_treatment_type='fracture',
                queries=queries,
                num_samples=num_samples,
            ),
            factual_fracture_counterfactual_swell=self._benchmark_ccbir(
                factual_treatment_type='fracture',
                counterfactual_treatment_type='swell',
                queries=queries,
                num_samples=num_samples,
            ),
        )

        return results
