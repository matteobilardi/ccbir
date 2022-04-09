
from ccbir.pytorch_vqvae.modules import ResBlock
from ccbir.models.twinnet.data import PSFTwinNetDataset
from ccbir.util import ActivationFunc, activation_layer_ctor
import pytorch_lightning as pl
from torch import Tensor, nn
from typing import Callable, Literal, Tuple
import torch
import torch.nn.functional as F


class DeepTwinNetComponent(nn.Module):
    """Twin network for DAG: X -> Y <- Z that allows sampling from
    P(Y, Y* | X, X*, Z)"""

    def __init__(
        self,
        treatment_dim: int,
        confounders_dim: int,
        outcome_noise_dim: int,
        outcome_shape: torch.Size,
        noise_inject_mode: Literal['concat', 'add', 'multiply'],
        use_combine_net: bool,
        weight_sharing: bool,
        activation: Callable[..., nn.Module],
        batch_norm: bool,
    ):
        super().__init__()
        self.use_combine_net = use_combine_net
        self.data_dim = treatment_dim + confounders_dim
        self.weight_sharing = weight_sharing

        if noise_inject_mode == 'concat':
            self.input_dim = self.data_dim * 2
            self.inject_noise = (
                lambda data, noise: torch.cat((data, noise), dim=-1)
            )
        elif noise_inject_mode == 'add':
            self.input_dim = self.data_dim
            self.inject_noise = torch.add
        elif noise_inject_mode == 'multiply':
            self.input_dim = self.data_dim
            self.inject_noise = torch.mul
        else:
            raise ValueError(f"Invalid {noise_inject_mode=}")

        # TODO: For clarity consider not using lazy layers once the architecture
        # is stabilised
        def linear_layer(out_features: int, batch_norm: bool = batch_norm):
            layers = []
            layers.append(nn.LazyLinear(out_features=out_features))
            if batch_norm:
                layers.append(nn.LazyBatchNorm1d())

            return nn.Sequential(*layers)

        self.combine_treatment_confounders_net = nn.Sequential(
            linear_layer(self.data_dim),
            activation(),
            linear_layer(self.data_dim),
            # activation(),
            # linear_layer(),
        ) if use_combine_net else None

        # to inject noise via multiplication/addition, we currently force
        # the reparametrised noise to have the same number of dimensions as the
        # data (i.e. [treatement, confounders]) regardless of the value of
        # outcome_noise_dim, which is just the size of the input to the
        # reparametrize_noise_net. For pure convenience, currently this is the
        # case even when we inject noise by concatenation
        self.reparametrize_noise_net = nn.Sequential(
            linear_layer(self.data_dim),
            activation(),
            linear_layer(self.data_dim),
            # activation(),
            # linear_layer(self.data_dim),
        )

        # NOTE: so far best architecture: obtained  MSE ~0.075, without
        # weight sharing, beating previous best (version 75)
        def make_branch_():
            return nn.Sequential(
                linear_layer(1024),
                activation(),
                linear_layer(1024),
                activation(),
                linear_layer(1024),
                activation(),
                linear_layer(256),
                activation(),
                linear_layer(128),
                nn.Unflatten(1, outcome_shape)
            )

        def make_branch_():
            # around 0.25 loss
            return nn.Sequential(
                nn.Unflatten(1, (-1, 1, 1)),
                nn.LazyConvTranspose2d(128, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConvTranspose2d(128, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConvTranspose2d(128, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConvTranspose2d(128, 3),
                ResBlock(128, activation),
                ResBlock(128, activation),
                ResBlock(128, activation),
                nn.LazyConv2d(2, 2),
            )

        # best so far ~ 0.035 loss on training set
        def make_branch():
            return nn.Sequential(
                nn.Unflatten(1, (-1, 1, 1)),
                nn.LazyConvTranspose2d(1024, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConvTranspose2d(512, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConvTranspose2d(256, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConvTranspose2d(128, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                ResBlock(128, activation),
                ResBlock(128, activation),
                nn.LazyConv2d(64, 1),
                ResBlock(64, activation),
                nn.LazyConv2d(2, 2),
            )

        self.factual_branch = make_branch()
        self.counterfactual_branch = (
            self.factual_branch if weight_sharing else make_branch()
        )

        # force lazy init
        dummy_X_Z_output = self.combine_treatment_confounders_net(
            torch.rand(64, self.data_dim)
        )
        dummy_noise_output = self.reparametrize_noise_net(
            torch.rand(64, outcome_noise_dim)
        )
        dummy_output_fact = self.factual_branch(torch.rand(64, self.input_dim))
        dummy_output_counterfact = (
            self.counterfactual_branch(torch.rand(64, self.input_dim))
        )

        assert dummy_X_Z_output.shape[1] == self.data_dim
        assert dummy_noise_output.shape[1] == self.data_dim
        assert dummy_output_fact.shape[1:] == outcome_shape
        assert dummy_output_counterfact.shape[1:] == outcome_shape

    def combine_treatment_confounders(
        self,
        treatment: Tensor,
        confounders: Tensor,
    ) -> Tensor:
        combined = torch.cat((treatment, confounders), dim=-1)
        if self.use_combine_net:
            combined = self.combine_treatment_confounders_net(combined)

        return combined

    def forward(
        self,
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        outcome_noise: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        factual_data = self.combine_treatment_confounders(
            factual_treatment,
            confounders,
        )

        counterfactual_data = self.combine_treatment_confounders(
            counterfactual_treatment,
            confounders
        )

        noise = self.reparametrize_noise_net(outcome_noise)

        factual_input = self.inject_noise(factual_data, noise)
        counterfactual_input = self.inject_noise(counterfactual_data, noise)

        # factual_branch and counterfactual_branch are the same network
        # when weight sharing is on
        factual_outcome = self.factual_branch(factual_input)
        counterfactual_outcome = (
            self.counterfactual_branch(counterfactual_input)
        )

        return factual_outcome, counterfactual_outcome


class TwinNet(pl.LightningModule):
    def __init__(
        self,
        treatment_dim: int,
        confounders_dim: int,
        outcome_noise_dim: int,
        outcome_size: torch.Size,
        lr: float,
        noise_inject_mode: Literal['concat', 'add', 'multiply'] = 'concat',
        use_combine_net: bool = True,
        weight_sharing: bool = True,
        activation: ActivationFunc = 'mish',
        batch_norm: bool = False,
    ):
        super().__init__()
        self.twin_net = DeepTwinNetComponent(
            treatment_dim=treatment_dim,
            confounders_dim=confounders_dim,
            outcome_noise_dim=outcome_noise_dim,
            outcome_shape=outcome_size,
            noise_inject_mode=noise_inject_mode,
            use_combine_net=use_combine_net,
            weight_sharing=weight_sharing,
            activation=activation_layer_ctor(activation),
            batch_norm=batch_norm,
        )
        self.outcome_noise_dim = outcome_noise_dim
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        return self.twin_net(**x)

    def _step(self, batch):
        X, y = batch

        factual_outcome_hat, counterfactual_outcome_hat = self(X)

        factual_outcome = y['factual_outcome']
        counterfactual_outcome = y['counterfactual_outcome']

        factual_loss = F.mse_loss(factual_outcome_hat, factual_outcome)
        counterfactual_loss = F.mse_loss(
            counterfactual_outcome_hat,
            counterfactual_outcome,
        )

        loss = factual_loss + counterfactual_loss

        # FIXME: this helps pytorch-lighnting with ambiguos batch sizes
        # but is otherwise unnecessary
        batch_size = y['factual_outcome'].shape[0]
        self.log('batch_size_fix', -1.0, batch_size=batch_size)

        return loss, dict(
            loss=loss,
            factual_loss=factual_loss.item(),
            counterfactual_loss=counterfactual_loss.item(),
        )

    def training_step(self, batch, _batch_idx):
        loss, _metrics = self._step(batch)
        return loss

    def validation_step(self, batch, _batch_idx):
        _loss, metrics = self._step(batch)
        self.log_dict({f"val_{k}": v for k, v in metrics.items()})

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.twin_net.parameters(), lr=self.lr)


class PSFTwinNet(TwinNet):
    def __init__(
        self,
        outcome_size: torch.Size,
        # 4.952220800885215e-08
        # 0.0005,  # 1.7013748158991985e-06 # 4.4157044735331275e-05
        # 0.001  # 4.952220800885215e-08  # 0.0005  # 0.0005  # 1.7013748158991985e-06
        lr: float = 0.001
    ):
        super().__init__(
            outcome_size=outcome_size,
            treatment_dim=PSFTwinNetDataset.treatment_dim(),
            confounders_dim=PSFTwinNetDataset.confounders_dim(),
            outcome_noise_dim=PSFTwinNetDataset.outcome_noise_dim,
            lr=lr,
        )
        self.save_hyperparameters()
