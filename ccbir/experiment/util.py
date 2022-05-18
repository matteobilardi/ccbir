from sklearn.manifold import TSNE
from torch import BoolTensor, LongTensor, Tensor
from torchmetrics.functional import (
    retrieval_normalized_dcg,
    retrieval_average_precision,
    precision_recall_curve
)
from torchvision import transforms
from typing import Dict, Iterable, Literal, Optional, Set
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


def ndcg(pred_idxs: LongTensor, target_idxs: LongTensor) -> float:
    assert len(pred_idxs) == len(target_idxs)

    num_results = len(pred_idxs)
    relevance_scores = torch.arange(
        start=num_results - 1,
        end=-1,
        step=-1,
        dtype=torch.float32,
    )

    # TODO: there might be a cleaner way than explicit allocation
    pred_relevance = torch.full((num_results,), -1.0)
    pred_relevance[pred_idxs] = relevance_scores
    target_relevance = torch.full((num_results,), -1.0)
    target_relevance[target_idxs] = relevance_scores
    assert torch.all(pred_relevance >= 0)
    assert torch.all(target_relevance >= 0)

    ndgc_ = retrieval_normalized_dcg(pred_relevance, target_relevance)

    return ndgc_.item()


def hit_rate(
    ranked_result_idxs: LongTensor,
    relevant_result_idx: int,
    k: Optional[int] = None,
) -> float:
    if k is None:
        k = -1
    else:
        assert 0 <= k <= len(ranked_result_idxs)

    return float(relevant_result_idx in ranked_result_idxs[:k])


def reciprocal_rank(
    ranked_result_idxs: Tensor,
    relevant_result_idx: int,
) -> float:
    rank = (ranked_result_idxs == relevant_result_idx).nonzero().item()
    reciprocal_rank_ = 1 / (1 + rank)
    return reciprocal_rank_


def avg_precision(
    ranked_result_idxs: Tensor,
    is_relevant_idx: BoolTensor,
    k: Optional[int] = None,
) -> float:
    num_results = len(ranked_result_idxs)
    num_relevant = is_relevant_idx.count_nonzero()
    if k is None:
        k = num_results
    else:
        assert 0 <= k <= num_results

    relevant_preds = torch.cat((
        torch.ones(num_relevant),
        torch.zeros(num_results - num_relevant),
    ))
    relevant_target = is_relevant_idx[ranked_result_idxs]

    return retrieval_average_precision(
        preds=relevant_preds[:k],
        target=relevant_target[:k],
    ).item()
