from functools import partial
from typing import Callable, Literal, Tuple
from einops import rearrange
from torch import Tensor
import torch
from torch import nn
from ccbir.arch import PreActResBlock
from ccbir.models.vqvae.model import VQVAE
import torch.nn.functional as F
from einops.layers.torch import Rearrange

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
        output_type: Literal['categorical', 'continuous'] = 'categorical',
    ):
        super().__init__()
        self.weight_sharing = weight_sharing
        self.vqvae = vqvae
        self.output_type = output_type

        resblocks = partial(
            PreActResBlock.multi_block,
            activation=activation,
            use_se=True,
        )

        trunk_in_channels = confounders_dim + outcome_noise_dim
        trunk_out_channels = 16
        branch_in_channels = trunk_out_channels + treatment_dim

        def make_trunk():
            return nn.Sequential(
                nn.Unflatten(1, (-1, 1, 1)),
                nn.Upsample(scale_factor=8, mode='nearest'),
                nn.Conv2d(trunk_in_channels, 32, 3, 1, 1, bias=False),
                # num_blocks, in_channels, out_channels, stride
                resblocks(3, 32, 16),
                resblocks(1, 16, trunk_out_channels),
            )

        def make_branch():
            branch = [
                resblocks(3, branch_in_channels, 32),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64),
                activation(),
            ]
            if output_type == 'continuous':
                output = [
                    nn.Conv2d(64, vqvae.latent_dim, 1),
                    Rearrange('b d h w -> b (h w) d'),
                ]
            elif output_type == 'categorical':
                output = [
                    nn.Conv2d(64, vqvae.codebook_size, 1),
                    nn.Softmax2d(),
                    Rearrange('b k h w -> b (h w) k'),
                ]
            else:
                raise NotImplementedError

            return nn.Sequential(*branch, *output)

        self.shared_trunk = make_trunk()
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
        assert dummy_y_star.shape[1:] == outcome_shape, dummy_y_star.shapez

    def forward(
        self,
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        outcome_noise: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        if self.output_type == 'categorical':
            y_discrete, y_star_discrete = self.forward_discrete(
                factual_treatment=factual_treatment,
                counterfactual_treatment=counterfactual_treatment,
                confounders=confounders,
                outcome_noise=outcome_noise,
            )
            # TODO: refactor trainwrecks/stop dependening on vqvae
            y = self.vqvae.model.vq.quantize_encoding(y_discrete)
            y_star = self.vqvae.model.vq.quantize_encoding(y_star_discrete)

        elif self.output_type == 'continuous':
            y, y_star = self.forward_probs(
                factual_treatment=factual_treatment,
                counterfactual_treatment=counterfactual_treatment,
                confounders=confounders,
                outcome_noise=outcome_noise,
            )
            y, _, _, _ = self.vqvae.vq(y)
            y_star, _, _, _ = self.vqvae.vq(y_star)
        else:
            raise NotImplementedError

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

        if self.output_type == 'categorical':
            y_discrete = y_probs.argmax(dim=-1)
            y_star_discrete = y_star_probs.argmax(dim=-1)
        elif self.output_type == 'continuous':
            _z_q, y_discrete, _loss, _metrics = self.vqvae.vq(y_probs)
            _z_q, y_star_discrete, _loss, _metrics = (
                self.vqvae.vq(y_star_probs)
            )
        else:
            raise NotImplementedError

        return y_discrete, y_star_discrete

    def forward_probs(
        self,
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        outcome_noise: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        shared_input = torch.cat((confounders, outcome_noise), dim=-1)
        shared_output = self.shared_trunk(shared_input)
        b, _c, h, w = shared_output.shape
        factual_treatment_fmap = F.upsample(
            input=factual_treatment.view(b, -1, 1, 1),
            size=(h, w),
            mode='nearest',
        )
        factual_input = torch.cat(
            tensors=(factual_treatment_fmap, shared_output),
            dim=1,
        )
        counterfactual_treatment_fmap = F.upsample(
            input=counterfactual_treatment.view(b, -1, 1, 1),
            size=(h, w),
            mode='nearest',
        )
        counterfactual_input = torch.cat(
            tensors=(counterfactual_treatment_fmap, shared_output),
            dim=1,
        )

        factual_outcome = self.predict_y(factual_input)
        counterfactual_outcome = self.predict_y_star(counterfactual_input)

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
        print(f'{outcome_shape=}')
        YC = outcome_shape[1]  # channels outcome
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
        h, w = (8, 8)
        quantized_latent_to_fmap = partial(
            rearrange,
            pattern='b (h w) d -> b d h w',
            h=8,
            w=8,
        )
        condition_vars = [
            quantized_latent_to_fmap(factual_outcome),
            quantized_latent_to_fmap(counterfactual_outcome),
        ]
        if self.include_non_descendants:
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
                size=(h, w),
                mode='nearest',
            )
            condition_vars.append(non_descendants)

        # stack feature maps from all conditioning vars
        condition = torch.cat(condition_vars, dim=1)
        u_y_loc, log_u_y_scale = (
            torch.chunk(self.predict_u_y(condition), chunks=2, dim=-1)
        )
        u_y_scale = torch.exp(log_u_y_scale)

        return u_y_loc, u_y_scale
