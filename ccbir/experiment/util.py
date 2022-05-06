from sklearn.manifold import TSNE
from torch import Tensor
from torchvision import transforms
from typing import Dict, Literal
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def pil_from_tensor(tensor):
    return transforms.ToPILImage()(tensor)


def tensor_from_pil(img):
    return transforms.ToTensor()(img)


def show_tensor(tensor_img, dpi=150):
    plt.figure(dpi=dpi)
    plt.imshow(pil_from_tensor(tensor_img), cmap='gray', vmin=0, vmax=255)


def plot_tsne(
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
