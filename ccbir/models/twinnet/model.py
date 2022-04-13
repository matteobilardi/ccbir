
from base64 import encode
from numpy import var
import pyro
import pyro.optim
import pyro.infer
from pyro.distributions import Normal
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
            linear_layer(512),
            activation(),
            linear_layer(outcome_noise_dim),
        )

        D = 32  # 64  # base dim
        # best so far ~ 0.035 loss on training set

        def make_branch():
            return nn.Sequential(
                nn.Unflatten(1, (-1, 1, 1)),
                nn.LazyConvTranspose2d(16 * D, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConvTranspose2d(8 * D, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConvTranspose2d(4 * D, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConvTranspose2d(2 * D, 3),
                nn.LazyBatchNorm2d(),
                activation(),
                ResBlock(2 * D, activation),
                ResBlock(2 * D, activation),
                nn.LazyConv2d(D, 1),
                ResBlock(D, activation),
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
        assert dummy_noise_output.shape[1] == outcome_noise_dim
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


class DeepTwinNetNoiseEncoder(nn.Module):
    def __init__(
        self,
        treatment_dim: int,
        confounders_dim: int,
        outcome_noise_dim: int,
        outcome_shape: torch.Size,
        activation: Callable[..., nn.Module],
        weight_sharing: bool,
    ):
        super().__init__()
        self.treatment_dim = treatment_dim
        self.confounders_dim = confounders_dim
        self.outcome_noise_dim = outcome_noise_dim
        self.outcome_shape = outcome_shape
        self.activation = activation
        self.weight_sharing = weight_sharing

        YD = outcome_shape.numel()
        YC = outcome_shape[0]  # channels outcome

        # NOTE: for now this doesn't reduce the overall dimensions,
        # but just flattens the outcome space to a vector
        def outcome_encoder():
            return nn.Sequential(
                nn.LazyConv2d(YD, 3, 1),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConv2d(4 * YD, 3, 1),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConv2d(4 * YD, 2, 1),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConv2d(2 * YD, 2, 1),
                nn.LazyBatchNorm2d(),
                activation(),
                nn.LazyConv2d(YD, 2, 1),
                nn.Flatten(),
            )

        ND = outcome_noise_dim
        NW = 16  # (stands for noise width) scales architecture easily

        def noise_encoder():
            return nn.Sequential(
                nn.LazyLinear(512),
                nn.LazyBatchNorm1d(),
                activation(),
                nn.LazyLinear(2 * NW * ND),
                nn.LazyBatchNorm1d(),
                activation(),
                nn.LazyLinear(2 * ND),
            )

        self.encode_y = outcome_encoder()
        self.encode_y_star = (
            self.encode_y if weight_sharing else outcome_encoder()
        )
        self.predict_u_y = noise_encoder()

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
        y_enc = self.encode_y(factual_outcome)
        y_star_enc = self.encode_y_star(counterfactual_outcome)

        inputs = torch.cat((
            factual_treatment,
            counterfactual_treatment,
            confounders,
            y_enc,
            y_star_enc,
        ), dim=-1)

        u_y_loc, log_u_y_scale = (
            torch.chunk(self.predict_u_y(inputs), 2, dim=-1)
        )
        u_y_scale = torch.exp(log_u_y_scale)

        return u_y_loc, u_y_scale

    def forward(
        self,
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        factual_outcome: Tensor,
        counterfactual_outcome: Tensor,
    ) -> Tensor:
        y_enc = self.encode_y(factual_outcome)
        y_star_enc = self.encode_y_star(counterfactual_outcome)

        inputs = torch.cat((
            factual_treatment,
            counterfactual_treatment,
            confounders,
            y_enc,
            y_star_enc,
        ), dim=-1)

        u_y_loc, log_u_y_scale = (
            torch.chunk(self.predict_u_y(inputs), 2, dim=-1)
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
        weight_sharing: bool = True,
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
            weight_sharing=True,
        )
        self.outcome_noise_dim = outcome_noise_dim
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.save_hyperparameters()

        def optimizer_kwargs_for_model_param(module_name, param_name):
            if module_name == 'twin_net':
                return dict(lr=lr)
            elif module_name == 'infer_noise_net':
                return dict(lr=encoder_lr)
            else:
                raise RuntimeError(
                    f'Unexpected parameter {module_name=}, {param_name=}')

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

    def _decoder_sigma(
        self,
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

            y_dist = Normal(
                loc=y_,
                scale=self._decoder_sigma(y_, y),
            ).to_event(y_event_dim)

            y_star_dist = Normal(
                loc=y_star_,
                scale=self._decoder_sigma(y_star_, y_star),
            ).to_event(y_event_dim)

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
        self.log('batch_size_fix', -1.0, batch_size=batch_size)

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

    def training_step(self, batch, _batch_idx):
        loss, metrics = self._step(batch, train=True)
        train_metrics = keymap('train/'.__add__, metrics)
        self.log_dict(train_metrics)
        return loss

    def validation_step(self, batch, _batch_idx):
        _loss, metrics = self._step(batch, train=False)
        val_metrics = keymap('val/'.__add__, metrics)
        self.log_dict(val_metrics, on_epoch=True)

    def test_step(self, batch, batch_idx):
        _loss, metrics = self._step(batch, train=False)
        test_metrics = keymap('test/'.__add__, metrics)
        self.log_dict(test_metrics, on_epoch=True)

    def configure_optimizers(self):
        # optimisation handled in pyro
        pass


class PSFTwinNet(TwinNet):
    def __init__(
        self,
        outcome_size: torch.Size,
        # 4.952220800885215e-08
        # 0.0005,  # 1.7013748158991985e-06 # 4.4157044735331275e-05
        # 0.001  # 4.952220800885215e-08  # 0.0005  # 0.0005  # 1.7013748158991985e-06
        lr: float = 0.0005,  # 0.0005, # try with 0.0001 next
        encoder_lr: float = 0.0001,
    ):
        super().__init__(
            outcome_size=outcome_size,
            treatment_dim=PSFTwinNetDataset.treatment_dim(),
            confounders_dim=PSFTwinNetDataset.confounders_dim(),
            # FIXME: change dataset
            outcome_noise_dim=64,  # PSFTwinNetDataset.outcome_noise_dim,
            lr=lr,
            encoder_lr=encoder_lr,
        )
        self.save_hyperparameters()
