from enum import Enum
from functools import partial
import pyro
import pyro.optim
import pyro.infer
from pyro.distributions import Normal
from ccbir.arch import PreActResBlock
from ccbir.data.util import BatchDictLike
from ccbir.pytorch_vqvae.modules import ResBlock
from ccbir.models.twinnet.data import PSFTwinNetDataset
from ccbir.util import ActivationFunc, activation_layer_ctor
import pytorch_lightning as pl
from torch import Tensor, nn
from typing import Callable, Dict, Literal, Tuple
import torch
import torch.nn.functional as F
from toolz import dissoc, keymap


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
            self.input_dim = self.data_dim + outcome_noise_dim
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

        resblocks = partial(
            PreActResBlock.multi_block,
            stride=1,
            activation=activation,
            use_se=True,
        )

        def make_branch():
            return nn.Sequential(
                nn.Unflatten(1, (-1, 1, 1)),
                nn.Upsample(scale_factor=9, mode='nearest'),
                nn.Conv2d(self.input_dim, 128, 3, 1, 1, bias=False),
                # num_blocks, in_channels, out_channels, stride
                #resblocks(1, 256, 256),
                resblocks(2, 128, 64),
                resblocks(4, 64, 32),
                resblocks(4, 32, 16),
                resblocks(2, 16, 8),
                nn.Conv2d(8, 2, 2),
            )

        self.predict_y = make_branch()
        self.predict_y_star = (
            self.predict_y if weight_sharing else make_branch()
        )

        # force lazy init
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

        assert dummy_y.shape[1:] == outcome_shape
        assert dummy_y_star.shape[1:] == outcome_shape

    def forward(
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
        include_non_descendants: bool,
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


class V:
    # TODO: there must be a better way to refer to random variables in pyro
    # but alas it's strings for now...
    factual_treatment: str = 'factual_treatment'
    counterfactual_treatment: str = 'counterfactual_treatment'
    confounders: str = 'confounders'
    factual_outcome: str = 'factual_outcome'
    counterfactual_outcome: str = 'counterfactual_outcome'
    outcome_noise: str = 'outcome_noise'


class CustomELBO(pyro.infer.TraceMeanField_ELBO):
    # inspired by deepscm codebase

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trace_storage = {'model': None, 'guide': None}

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = (
            super()._get_trace(model, guide, args, kwargs)
        )

        self.trace_storage['model'] = model_trace
        self.trace_storage['guide'] = guide_trace

        return model_trace, guide_trace

    def get_metrics(self) -> Dict:
        model = self.trace_storage['model']
        guide = self.trace_storage['guide']

        def log_prob(model_or_guide, variable) -> float:
            log_prob = model_or_guide.nodes[variable]['log_prob'].mean()
            return log_prob.item()

        model_random_vars = [
            V.factual_outcome,
            V.counterfactual_outcome,
            V.outcome_noise,
        ]
        guide_random_vars = [V.outcome_noise]

        model_probs = {v: log_prob(model, v) for v in model_random_vars}
        guide_probs = {v: log_prob(guide, v) for v in guide_random_vars}
        kl = guide_probs[V.outcome_noise] - model_probs[V.outcome_noise]

        model_metrics = keymap(lambda v: f"log p({v})", model_probs)
        guide_metrics = keymap(lambda v: f"log q({v})", guide_probs)

        return {
            **model_metrics,
            **guide_metrics,
            # TODO: not completely sure this is what I'm extracting from pyro
            f"KL(q({V.outcome_noise}|everything)||p({V.outcome_noise}))": kl,
        }


def compute_optimal_sigma(
    estimate_batch: Tensor,
    target_batch: Tensor,
) -> Tensor:
    assert estimate_batch.shape == target_batch.shape
    # based on https://orybkin.github.io/sigma-vae/
    # note that variance is calculated across all datapoints in the batch
    # as is done in the sigma-vae paper
    return (
        ((estimate_batch - target_batch) ** 2)
        .mean(dim=[0, 1, 2, 3], keepdim=True)
        .sqrt()
        .expand_as(estimate_batch)
    )


class Stage(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    PREDICT = 'predict'


class ValidationOptimalSigma:
    def __init__(
        self,
        stage: Stage,
    ) -> None:
        self._val_variances = []
        self._stage = stage
        self._val_optimal_sigma = None

    def set_stage(self, stage: Stage):
        if self._stage == stage:
            return

        if stage == Stage.TRAIN and len(self._val_variances) > 0:
            # element-wise mean sigma of the validation sigmas
            val_optimal_variance = torch.mean(
                input=torch.stack(self._val_variances),
                dim=0,
            )
            self._val_optimal_sigma = val_optimal_variance.sqrt()
        elif stage == Stage.VAL:
            self._val_optimal_sigma = None
        else:
            # predicting or testing
            pass

        self._val_variances.clear()
        self._stage = stage

    def _compute_optimal_variance(
        self,
        estimate_batch: Tensor,
        target_batch: Tensor,
    ) -> Tensor:
        assert estimate_batch.shape == target_batch.shape
        n = estimate_batch.numel()
        unbiased_variance = (
            ((estimate_batch - target_batch) ** 2).sum() / (n - 1)
        )

        # NOTE: this is a scalar
        return unbiased_variance

    def __call__(
        self,
        estimate_batch: Tensor,
        target_batch: Tensor,
    ) -> Tensor:
        if self._val_optimal_sigma is None:
            if self._stage == Stage.TRAIN:
                # validation hasn't run yet, so just compute the optimal sigma
                # on the training set
                return compute_optimal_sigma(estimate_batch, target_batch)
            elif self._stage == Stage.VAL:
                # accumulate optimal variances during validation
                variance = self._compute_optimal_variance(
                    estimate_batch,
                    target_batch,
                )

                self._val_variances.append(variance)
                sigma = variance.sqrt().expand_as(estimate_batch)
                return sigma
            else:
                # predicting or testing, don't use optimal sigma
                return torch.ones_like(estimate_batch)
        else:
            if self._stage == Stage.TRAIN:
                return self._val_optimal_sigma.expand_as(estimate_batch)
            elif self._stage == Stage.VAL:
                raise RuntimeError(
                    "Unreachable code: we don't compute the optimal sigma to"
                    "be used during training while still in validation stage"
                )
            else:
                # predicting or testing, don't use optimal sigma
                return torch.ones_like(estimate_batch)


class TwinNet(pl.LightningModule):
    def __init__(
        self,
        treatment_dim: int,
        confounders_dim: int,
        outcome_noise_dim: int,
        outcome_size: torch.Size,
        lr: float,
        encoder_lr: float,
        noise_inject_mode: Literal['concat', 'add', 'multiply'] = 'concat',
        use_combine_net: bool = True,
        weight_sharing: bool = False,
        activation: ActivationFunc = 'mish',
        batch_norm: bool = False,
    ):
        super().__init__()
        # TODO: may want to refactor verbose argument passing
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
        self.infer_noise_net = DeepTwinNetNoiseEncoder(
            treatment_dim=treatment_dim,
            confounders_dim=confounders_dim,
            outcome_noise_dim=outcome_noise_dim,
            outcome_shape=outcome_size,
            activation=activation_layer_ctor(activation),
            include_non_descendants=False,
        )
        self.outcome_noise_dim = outcome_noise_dim
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.save_hyperparameters()

        self.stage = Stage.TRAIN
        self.optimal_sigma = compute_optimal_sigma

        def optimizer_kwargs_for_model_param(module_name, param_name):
            if module_name == 'twin_net':
                return dict(lr=lr)
            elif module_name == 'infer_noise_net':
                return dict(lr=encoder_lr)
            else:
                raise RuntimeError(
                    f'Unexpected parameter {module_name=}, {param_name=}'
                )

        # analytical KL ELBO
        self.svi_loss = CustomELBO(
            num_particles=1,
            # NOTE: I'm not sure it's possible/easy to vectorize particles
            # when convolutions are being used as a dimension
            # gets added to the sampled variables
            vectorize_particles=False,
        )
        self.svi = pyro.infer.SVI(
            model=self.model,
            guide=self.guide,
            optim=pyro.optim.Adam(optimizer_kwargs_for_model_param),
            loss=self.svi_loss
        )

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        return self.twin_net(**x)

    def model(self, batch):
        pyro.module('twin_net', self.twin_net)
        input, output = batch
        batch_size = input[V.factual_treatment].shape[0]

        y: Tensor = output[V.factual_outcome]
        y_star: Tensor = output[V.counterfactual_outcome]
        u_y_loc = torch.zeros(self.outcome_noise_dim, device=self.device)
        u_y_scale = torch.ones(self.outcome_noise_dim, device=self.device)

        with pyro.plate('data', size=batch_size):
            u_y_dist = Normal(u_y_loc, u_y_scale).to_event(1)
            u_y = pyro.sample(V.outcome_noise, u_y_dist)

            input = {**input, V.outcome_noise: u_y}
            y_, y_star_ = self.forward(input)

            y_event_dim = y_.dim() - 1
            assert y_event_dim == y_star_.dim() - 1
            y_loc = y_
            y_scale = self.optimal_sigma(y_, y)
            y_dist = Normal(y_loc, y_scale).to_event(y_event_dim)

            y_star_loc = y_star_
            y_star_scale = self.optimal_sigma(y_star_, y_star)
            y_star_dist = (
                Normal(y_star_loc, y_star_scale).to_event(y_event_dim)
            )

            self._log_optimal_sigmas(
                factual_sigma=y_scale,
                counterfactual_sigma=y_star_scale,
            )

            pyro.sample(V.factual_outcome, y_dist, obs=y)
            pyro.sample(V.counterfactual_outcome, y_star_dist, obs=y_star)

    def guide(self, batch):
        pyro.module('infer_noise_net', self.infer_noise_net)
        input, output = batch
        batch_size = input[V.factual_treatment].shape[0]

        # make sure that the inference network doesn't see the outcome noise
        assert V.outcome_noise not in input

        u_y_loc_, u_y_scale_ = self.infer_noise_net.forward(
            factual_treatment=input[V.factual_treatment],
            counterfactual_treatment=input[V.counterfactual_treatment],
            confounders=input[V.confounders],
            factual_outcome=output[V.factual_outcome],
            counterfactual_outcome=output[V.counterfactual_outcome],
        )

        with pyro.plate('data', size=batch_size):
            u_y_dist = Normal(u_y_loc_, u_y_scale_).to_event(1)
            pyro.sample(V.outcome_noise, u_y_dist)

    def backward(self, *args, **kwargs):
        # no loss to backpropagate as pyro handles optimisation
        pass

    def configure_optimizers(self):
        # optimisation handled in pyro
        pass

    def set_stage(self, stage: Stage):
        self.stage = stage
        if isinstance(self.optimal_sigma, ValidationOptimalSigma):
            self.optimal_sigma.set_stage(stage)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.set_stage(Stage.TRAIN)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.set_stage(Stage.VAL)

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.set_stage(Stage.TEST)

    def on_predict_epoch_start(self) -> None:
        super().on_predict_epoch_start()
        self.set_stage(Stage.PREDICT)

    def training_step(self, batch, _batch_idx):
        loss, metrics = self._step(batch, train=True)
        train_metrics = keymap('train/'.__add__, metrics)
        # needed without / to include in the checkpoint filename
        train_metrics['train_loss'] = metrics['loss']
        self.log_dict(train_metrics)
        return loss

    def validation_step(self, batch, _batch_idx):
        _loss, metrics = self._step(batch, train=False)
        val_metrics = keymap('val/'.__add__, metrics)
        # needed without / to include in the checkpoint filename
        val_metrics['val_loss'] = metrics['loss']
        self.log_dict(val_metrics, on_epoch=True)

    def test_step(self, batch, batch_idx):
        _loss, metrics = self._step(batch, train=False)
        test_metrics = keymap('test/'.__add__, metrics)
        self.log_dict(test_metrics, on_epoch=True)

    def _step(
        self,
        batch: Tuple[BatchDictLike, BatchDictLike],
        *,
        train: bool,
    ):
        x, y = batch

        # FIXME: for now just ignoring dataset injected noise this should
        # probably not appear in the dataset from the start
        x = dissoc(x, V.outcome_noise)
        batch = (x, y)

        # NOTE: this helps pytorch-lighnting with ambiguos batch sizes
        # but is otherwise unnecessary
        batch_size = y[V.factual_outcome].shape[0]
        self.log('batch_size_fix', float(batch_size), batch_size=batch_size)

        if train:
            loss = self.svi.step(batch)
        else:
            loss = self.svi.evaluate_loss(batch)

        loss = torch.as_tensor(loss / batch_size)
        svi_metrics = self.svi_loss.get_metrics()

        return loss, dict(
            loss=loss,
            **svi_metrics,
        )

    def _log_optimal_sigmas(
        self,
        factual_sigma: Tensor,
        counterfactual_sigma: Tensor
    ):
        factual_sigma_: Tensor = factual_sigma.unique()
        counterfactual_sigma_: Tensor = counterfactual_sigma.unique()
        assert factual_sigma_.numel() == 1
        assert counterfactual_sigma_.numel() == 1
        factual_sigma_scalar = factual_sigma_[0].item()
        counterfactual_sigma_scalar = counterfactual_sigma_[0].item()
        stage = self.stage.value
        self.log_dict({
            f"{stage}/factual_sigma": factual_sigma_scalar,
            f"{stage}/counterfactual_sigma": counterfactual_sigma_scalar,
        })


class PSFTwinNet(TwinNet):
    def __init__(
        self,
        outcome_size: torch.Size,
        # 4.952220800885215e-08
        # 0.0005,  # 1.7013748158991985e-06 # 4.4157044735331275e-05
        # 0.001  # 4.952220800885215e-08  # 0.0005  # 0.0005  # 1.7013748158991985e-06
        lr: float = 0.0002,  # 0.0005, # try with 0.0001 next
        encoder_lr: float = 0.0002,
    ):
        super().__init__(
            outcome_size=outcome_size,
            treatment_dim=PSFTwinNetDataset.treatment_dim(),
            confounders_dim=PSFTwinNetDataset.confounders_dim(),
            # FIXME: change dataset
            outcome_noise_dim=12,  # 16,  # 32,  # PSFTwinNetDataset.outcome_noise_dim,
            lr=lr,
            encoder_lr=encoder_lr,
        )
        self.save_hyperparameters()
