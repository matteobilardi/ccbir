from ccbir.models.twinnet.data import PSFTwinNetDataModule, PSFTwinNetDataset
from ccbir.models.twinnet.model import PSFTwinNet
from ccbir.models.twinnet.train import vqvae_embed_image
from ccbir.models.vqvae.data import VQVAEMorphoMNISTDataModule
from ccbir.models.vqvae.model import VQVAE
from functools import cached_property, partial
from more_itertools import interleave
from sklearn.manifold import TSNE
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from toolz import first
from torchvision.utils import make_grid
from typing import Callable, Dict, List, Literal, Optional, Type, Union
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch

# TODO: rename file


def pil_from_tensor(tensor):
    return transforms.ToPILImage()(tensor)


def tensor_from_pil(img):
    return transforms.ToTensor()(img)


def show_tensor(tensor_img, dpi=150):
    plt.figure(dpi=dpi)
    plt.imshow(pil_from_tensor(tensor_img), cmap='gray', vmin=0, vmax=255)


def _plot_tsne(
    latents_for_perturbation: Dict[
        Literal['plain', 'swollen', 'fractured'],
        Tensor,
    ],
    labels: Tensor,
    perplexity: int = 30,
    n_iter: int = 1000,
):
    perturbations = sorted(latents_for_perturbation.keys())
    assert len(perturbations) > 0
    latents_all = torch.cat([
        latents_for_perturbation[p] for p in perturbations
    ])
    print('Computing TSNE...')
    tsne = TSNE(perplexity=perplexity, n_jobs=-1)
    latents_embedded_all = torch.as_tensor(tsne.fit_transform(
        # latents need to be flattened to vectors to be processed by TSNE
        # in case they are not already flat
        latents_all.flatten(start_dim=1).detach().numpy()
    ))

    latents_embedded_for_perturbation = dict(zip(
        perturbations,
        torch.chunk(latents_embedded_all, len(perturbations)),
    ))

    df = pd.concat((
        pd.DataFrame(dict(
            perturbation=[perturbation] * len(latents_embedded),
            digit=labels,
            x=latents_embedded[:, 0],
            y=latents_embedded[:, 1],
        ))
        for perturbation, latents_embedded in
        latents_embedded_for_perturbation.items()
    ))

    df = df.sort_values(by=['digit', 'perturbation'])

    plt.figure(dpi=200)
    # sns.set_style('white')
    g = sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='digit',
        style='perturbation',
        legend='full',
        palette='bright',
    )
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    g.set(title=f"TSNE: {perplexity=}, {n_iter=}")


class ExperimentData:
    """Provides convenient and fast access to the exact data used by a
    datamodule"""

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
    def _train_dataset(self):
        dm = self.datamodule
        dataset = dm.train_dataloader().dataset
        return dataset

    @cached_property
    def _test_dataset(self):
        dm = self.datamodule
        dataset = dm.test_dataloader().dataset
        return dataset

    def dataset(self, train: bool) -> Dataset:
        return self._train_dataset if train else self._test_dataset

    def dataloader(self, train: bool):
        if train:
            return self.datamodule.train_dataloader()
        else:
            return self.datamodule.test_dataloader()


class VQVAEExperimentData(ExperimentData):
    def __init__(self):
        super().__init__(VQVAEMorphoMNISTDataModule)


class TwinNetExperimentData(ExperimentData):
    def __init__(self, embed_image):
        super().__init__(
            datamodule_ctor=partial(
                PSFTwinNetDataModule,
                embed_image=embed_image,
                num_workers=1,
            )
        )

    def psf_items(self, train: bool, index: Optional[int] = None) -> Dict:
        # ugly accesses to train subset dataset
        dataset = self.dataset(train)
        if index is None:
            return (
                dataset.dataset.psf_items[dataset.indices] if train else
                dataset.psf_items.dict()
            )
        else:
            return (
                dataset.dataset.psf_items[dataset.indices[index]] if train else
                dataset.psf_items[index]
            )


class VQVAEExperiment:
    def __init__(self, vqvae: VQVAE):
        self.vqvae = vqvae
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
            _x_hat, _z_e_x, z_q_x = self.vqvae(x)
            return z_q_x

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

        _plot_tsne(latents_for_perturbation, labels, perplexity, n_iter)


class PSFTwinNetExperiment:
    def __init__(
        self,
        vqvae: VQVAE,
        twinnet: PSFTwinNet,
    ):
        self.vqvae = vqvae
        self.twinnet = twinnet
        self.data = TwinNetExperimentData(
            embed_image=partial(vqvae_embed_image, vqvae),
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
        noise_scale: float = 0.25,
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

        # resample noise, possibly with differen scale for higher/lower
        # variance
        X['outcome_noise'] = PSFTwinNetDataset.sample_outcome_noise(
            sample_shape=X['outcome_noise'].shape,
            scale=noise_scale,
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
        swollen, fractured = self.twinnet.forward(x)
        latents_for_perturbations = dict(
            swollen=swollen,
            fractured=fractured,
        )
        psf_items = self.data.psf_items(train, slice(num_points))
        labels = psf_items['plain']['label']

        _plot_tsne(latents_for_perturbations, labels, perplexity, n_iter)
