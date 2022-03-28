from functools import cached_property, partial
from typing import Any, Callable, List, Literal, Optional, Type, Union
from sklearn.manifold import TSNE
from torch import Tensor
from ccbir.models.vqvae import VQVAE, VQVAEMorphoMNISTDataModule
from ccbir.models.twin_network import (
    PSFTwinNet,
    PSFTwinNetDataModule,
    PSFTwinNetDataset,
    vqvae_embed_image,
)
import torch
from torch.utils.data import default_collate
from torchvision.utils import make_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from deepscm.submodules.morphomnist.morphomnist.perturb import Perturbation
from more_itertools import interleave
import pandas as pd


def pil_from_tensor(tensor):
    return transforms.ToPILImage()(tensor)


def tensor_from_pil(img):
    return transforms.ToTensor()(img)


def show_tensor(tensor_img, dpi=150):
    plt.figure(dpi=dpi)
    plt.imshow(pil_from_tensor(tensor_img), cmap='gray', vmin=0, vmax=255)


class ExperimentData:
    def __init__(
        self,
        datamodule_ctor: Union[
            Type[pl.LightningDataModule],
            Callable[..., pl.LightningDataModule],
        ],
    ):
        self.datamodule_ctor = datamodule_ctor

    @cached_property
    def datamodule(self) -> pl.LightningDataModule:
        print('Loading datamodule...')
        dm = self.datamodule_ctor()
        dm.prepare_data()
        dm.setup()

        return dm

    @cached_property
    def _train_data(self):
        dm = self.datamodule
        print('Loading training data...')
        dataset = dm.train_dataloader().dataset
        return default_collate(list(dataset))

    @cached_property
    def _test_data(self):
        dm = self.datamodule
        print('Loading test data...')
        dataset = dm.test_dataloader().dataset
        return default_collate(list(dataset))

    def batched(self, train: bool) -> Any:
        return self._train_data if train else self._test_data


class VQVAEExperiment:
    def __init__(self, vqvae: VQVAE):
        self.vqvae = vqvae
        self.data = ExperimentData(VQVAEMorphoMNISTDataModule)

        # TODO: hacky - this should not be needed in VQVAE experiments
        self.psf_data = ExperimentData(
            datamodule_ctor=partial(
                PSFTwinNetDataModule,
                embed_image=partial(vqvae_embed_image, vqvae),
            )
        )

    def show_vqvae_recons(self, num_images: int = 32, train: bool = False):
        data = self.data.batched(train)
        images = data['image'][:num_images]
        
        with torch.no_grad():
            recons, _z_e, _z_q = self.vqvae(images)

        show_tensor(make_grid(
            tensor=torch.cat((images, recons)),
            normalize=True,
            value_range=(-1, 1),
        ))

    def _vqvae_z_q(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            _x_hat, _z_e_x, z_q_x = self.vqvae(x)
            return z_q_x

    def plot_vqvae_tsne(
        self,
        include_perturbations: List[Literal[
            'plain', 'swollen', 'fractured'
        ]] = ['plain', 'swollen', 'fractured'],
        perplexity: int = 30,
        num_points: int = 500,
        train: bool = False,
    ):
        assert len(include_perturbations) > 0
        perturbations = sorted(include_perturbations)

        _x, _y, psf_items = self.psf_data.batched(train)

        # concat z_q of all perturbations to compute TSNE for all embeddings
        z_q_all = torch.cat([
            self._vqvae_z_q(
                psf_items[perturbation]['image'][:num_points]
            )
            for perturbation in perturbations
        ])

        print('Computing TSNE...')
        tsne = TSNE(perplexity=perplexity, n_jobs=-1)
        z_q_embedded_all = torch.as_tensor(tsne.fit_transform(
            # latents need to be flattened to vectors to be processed by TSNE
            z_q_all.flatten(start_dim=1).detach().numpy()
        ))
        z_q_embedded_for_perturbation = dict(zip(
            perturbations,
            torch.chunk(z_q_embedded_all, len(perturbations))
        ))

        labels = psf_items['plain']['label'][:num_points]


        data = pd.DataFrame({
            'digit': labels,
            **z_q_embedded_for_perturbation
        })


        """
        plt.gca().set_aspect('auto', 'box')

        for perturbation in perturbations:
            z_q_embedded = z_q_embedded_for_perturbation[perturbation]
            for digit in range(10):
                embedding_for_digit = z_q_embedded[labels == digit]
                x = embedding_for_digit[:, 0]
                y = embedding_for_digit[:, 1]
                plt.scatter(x, y, label=f"{digit}-{perturbation}")

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.figure()
        plt.show()
        """


class PSFTwinNetExperiment:
    def __init__(
        self,
        vqvae: VQVAE,
        twinnet: PSFTwinNet,
    ):
        self.vqvae = vqvae
        self.twinnet = twinnet
        self.data = ExperimentData(
            datamodule_ctor=partial(
                PSFTwinNetDataModule,
                embed_image=partial(vqvae_embed_image, vqvae),
            )
        )

    def vqvae_recon(self, z_e_x: Tensor) -> Tensor:
        """z_e_x is the encoding of the input produced by the encoder network of
        the vqvae before it's mapped to the indices of specific codewords via
        the learned codebook"""

        with torch.no_grad():
            e_x = self.vqvae.model.codebook(z_e_x)
            x_recon = self.vqvae.decode(e_x)

        return x_recon

    def show_twinnet_samples(
        self,
        item_idx: int,
        num_samples=32,
        noise_rescale: float = 1.0,
        train=False,
    ):
        batch = self.data.batched(train)
        X, y, psf_item = batch

        original_image = psf_item['plain']['image'][item_idx].unsqueeze(0)
        swollen_image = psf_item['swollen']['image'][item_idx].unsqueeze(0)
        fractured_image = psf_item['fractured']['image'][item_idx].unsqueeze(0)
        swollen_embedding = y['factual_outcome'][item_idx].unsqueeze(0)
        fractured_embedding = (
            y['counterfactual_outcome'][item_idx].unsqueeze(0)
        )

        ground_truths = torch.cat((
            original_image,
            swollen_image,
            self.vqvae_recon(swollen_embedding),
            fractured_image,
            self.vqvae_recon(fractured_embedding),
        ))

        print('original, swollen, swollen_vqvae, fractured, fractured_vqvae')
        show_tensor(make_grid(
            tensor=ground_truths,
            normalize=True,
            value_range=(-1, 1)
        ))

        # repeat input num_samples time
        X = {
            k: (
                v[item_idx]
                .unsqueeze(0)
                .clone()
                .repeat_interleave(num_samples, dim=0)
            )
            for k, v in X.items()
        }

        # resample noise, possibly rescaling it for higher/lower variance
        X['outcome_noise'] = (
            torch.randn_like(X['outcome_noise']) * noise_rescale
        )

        swollen_embedding_hat, fractured_embedding_hat = self.twinnet(X)

        # make the swollen and fractured embedding sampled from the same input
        # and noise show up one after the other to ease visual inspection
        paired_up_embeddings = torch.stack(list(
            interleave(swollen_embedding_hat, fractured_embedding_hat)
        ))

        # paired up images sampled via the twinnet
        images_hat = self.vqvae_recon(paired_up_embeddings)
        print('Twin network outputted swollen-fractured pairs')
        show_tensor(
            make_grid(
                tensor=images_hat,
                normalize=True,
                value_range=(-1, 1),
            ),
            dpi=250,
        )

    def plot_twinnet_tsne(self):
        # TODO
        raise NotImplementedError
