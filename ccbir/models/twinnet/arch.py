from functools import partial
from typing import Callable, Literal, Tuple
from torch import Tensor
import torch
from torch import nn
from ccbir.arch import PreActResBlock
from ccbir.models.vqvae.model import VQVAE
import torch.nn.functional as F

# TODO: ideally break direct dependence on VQVAE and reduce
# hardcoding of architecture input/output sizes


class DeepTwinNet(nn.Module):
    """Twin network for DAG: X -> Y <- Z that allows sampling from
    P(Y, Y* | X, X*, Z)"""

    def __init__(
        self,
        treatment_dim: int,
        confounders_dim: int,
        outcome_noise_dim: int,
        outcome_shape: torch.Size,
        vqvae: VQVAE,
        weight_sharing: bool,
        activation: Callable[..., nn.Module],
    ):
        super().__init__()
        self.data_dim = treatment_dim + confounders_dim
        self.weight_sharing = weight_sharing
        self.vqvae = vqvae
        self.input_dim = self.data_dim + outcome_noise_dim

        resblocks = partial(
            PreActResBlock.multi_block,
            activation=activation,
            use_se=True,
        )

        def make_branch():
            return nn.Sequential(
                nn.Unflatten(1, (-1, 1, 1)),
                nn.Upsample(scale_factor=32, mode='nearest'),
                nn.Conv2d(self.input_dim, 128, 3, 1, 1, bias=False),
                # num_blocks, in_channels, out_channels, stride
                resblocks(2, 128, 64),
                resblocks(2, 64, 64, 2),
                resblocks(2, 64, 64, 2),
                nn.Conv2d(64, 1024, 1),
                nn.Softmax2d(),
            )

        self.predict_y = make_branch()
        self.predict_y_star = (
            self.predict_y if weight_sharing else make_branch()
        )

        # init lazy modules
        dummy_batch_size = 64
        x = torch.randn(dummy_batch_size, treatment_dim)
        x_star = x.clone()
        z = torch.randn(dummy_batch_size, confounders_dim)
        u_y = torch.randn(dummy_batch_size, outcome_noise_dim)

        dummy_y, dummy_y_star = self.forward(
            factual_treatment=x,
            counterfactual_treatment=x_star,
            confounders=z,
            outcome_noise=u_y,
        )

        assert dummy_y.shape[1:] == outcome_shape, dummy_y.shape
        assert dummy_y_star.shape[1:] == outcome_shape, dummy_y_star.shape

    def forward(
        self,
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        outcome_noise: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        y_discrete, y_star_discrete = self.forward_discrete(
            factual_treatment=factual_treatment,
            counterfactual_treatment=counterfactual_treatment,
            confounders=confounders,
            outcome_noise=outcome_noise,
        )
        # TODO: refactor trainwrecks/stop dependening on vqvae
        y = self.vqvae.model.codebook.embedding(y_discrete).permute(0, 3, 1, 2)
        y_star = self.vqvae.model.codebook.embedding(
            y_star_discrete).permute(0, 3, 1, 2)

        return y, y_star

    def forward_discrete(
        self,
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        outcome_noise: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        y_probs, y_star_probs = self.forward_probs(
            factual_treatment=factual_treatment,
            counterfactual_treatment=counterfactual_treatment,
            confounders=confounders,
            outcome_noise=outcome_noise,
        )

        y_discrete = y_probs.argmax(dim=-1)
        y_star_discrete = y_star_probs.argmax(dim=-1)

        return y_discrete, y_star_discrete

    def forward_probs(
        self,
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        outcome_noise: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        factual_input = torch.cat(
            tensors=(
                factual_treatment,
                confounders,
                outcome_noise,
            ),
            dim=-1,
        )
        counterfactual_input = torch.cat(
            tensors=(
                counterfactual_treatment,
                confounders,
                outcome_noise,
            ),
            dim=-1,
        )

        # pyro/pytorch distribution like the index in the probability vector
        # to be on the last dimension but softmax2d was applied to the
        # channel dimentsion so make channel dimension the last one
        factual_outcome = self.predict_y(factual_input).permute(0, 2, 3, 1)
        counterfactual_outcome = self.predict_y_star(
            counterfactual_input).permute(0, 2, 3, 1)

        return factual_outcome, counterfactual_outcome


class DeepTwinNetNoiseEncoder(nn.Module):
    def __init__(
        self,
        treatment_dim: int,
        confounders_dim: int,
        outcome_noise_dim: int,
        outcome_shape: torch.Size,
        activation: Callable[..., nn.Module],
        include_non_descendants: bool = True,
    ):
        super().__init__()
        self.treatment_dim = treatment_dim
        self.confounders_dim = confounders_dim
        self.outcome_noise_dim = outcome_noise_dim
        self.outcome_shape = outcome_shape
        self.activation = activation
        self.include_non_descendants = include_non_descendants

        resblocks = partial(
            PreActResBlock.multi_block,
            activation=activation,
            use_se=True,
        )

        YD = outcome_shape.numel()
        YC = outcome_shape[0]  # channels outcome
        input_channels = 2 * YC  # factual + counterfactual feature map
        if include_non_descendants:
            input_channels += 2 * treatment_dim + confounders_dim

        self.predict_u_y = nn.Sequential(
            # num_blocks, in_channels, out_channels, stride
            nn.Conv2d(input_channels, 256, 3, stride=1, padding=1, bias=False),
            resblocks(1, 256, 256, 1),
            resblocks(2, 256, 256, 2),
            resblocks(3, 256, 128, 2),
            resblocks(2, 128, 64, 2),
            nn.Conv2d(64, outcome_noise_dim * 2, 1),
            nn.Flatten(),
        )
        # init lazy modules
        dummy_batch_size = 64
        x = torch.randn(dummy_batch_size, treatment_dim)
        x_star = x.clone()
        z = torch.randn(dummy_batch_size, confounders_dim)
        y = torch.randn(dummy_batch_size, *outcome_shape)
        y_star = y.clone()

        dummy_u_y_loc, dummy_u_y_scale = self.forward(
            factual_treatment=x,
            counterfactual_treatment=x_star,
            confounders=z,
            factual_outcome=y,
            counterfactual_outcome=y_star,
        )

        assert dummy_u_y_loc.shape == (dummy_batch_size, outcome_noise_dim)
        assert dummy_u_y_scale.shape == (dummy_batch_size, outcome_noise_dim)

    def forward(
        self,
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        factual_outcome: Tensor,
        counterfactual_outcome: Tensor,
    ) -> Tensor:
        condition_vars = [factual_outcome, counterfactual_outcome]
        if self.include_non_descendants:
            height = self.outcome_shape[1]
            width = self.outcome_shape[2]
            non_descendants = torch.cat(
                tensors=(
                    factual_treatment,
                    counterfactual_treatment,
                    confounders,
                ),
                dim=-1,
            )
            batch_size = non_descendants.shape[0]
            non_descendants = non_descendants.view(batch_size, -1, 1, 1)
            non_descendants = F.upsample(
                input=non_descendants,
                size=(height, width),
                mode='nearest',
            )
            condition_vars.append(non_descendants)

        # stack feature maps from all conditioning vars
        condition = torch.cat(condition_vars, dim=1)
        u_y_loc, log_u_y_scale = (
            torch.chunk(self.predict_u_y(condition), 2, dim=-1)
        )
        u_y_scale = torch.exp(log_u_y_scale)

        return u_y_loc, u_y_scale
