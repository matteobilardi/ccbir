from enum import Enum
from functools import partial
import pyro
import pyro.optim
import pyro.infer
from pyro.distributions import Normal, Categorical
from ccbir.models.twinnet.arch import (
    DeepTwinNet,
    DeepTwinNetNoiseEncoder,
)
from ccbir.models.util import load_best_model
from ccbir.models.vqvae.model_vq import VQVAE
from ccbir.data.util import BatchDictLike
from ccbir.models.twinnet.data import PSFTwinNetDataset
from ccbir.util import ActivationFunc, activation_layer_ctor
import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Literal, Optional, Tuple
import torch
from toolz import dissoc, keymap


class V:
    """Model variables"""
    # TODO: there must be a better way to refer to random variables in pyro
    # but alas it's strings for now...
    factual_treatment: str = 'factual_treatment'
    counterfactual_treatment: str = 'counterfactual_treatment'
    confounders: str = 'confounders'
    factual_outcome: str = 'factual_outcome'
    counterfactual_outcome: str = 'counterfactual_outcome'
    outcome_noise: str = 'outcome_noise'


class CustomELBO(pyro.infer.TraceMeanField_ELBO):
    """Analytical KL ELBO that tracks metrics about the model variables"""

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

    def get_batched_elbo(
        self,
    ) -> Tensor:
        model = self.trace_storage['model']
        guide = self.trace_storage['guide']
        model_log_probs = []
        for var, site in model.nodes.items():
            if site['type'] == 'sample':
                model_log_probs.append(site['fn'].log_prob(site['value']))

        guide_log_probs = []
        for site in guide.nodes.values():
            if site['type'] == 'sample':
                guide_log_probs.append(site['fn'].log_prob(site['value']))

        elbo = sum(model_log_probs) - sum(guide_log_probs)

        return elbo

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
            f"KL(q({V.outcome_noise})||p({V.outcome_noise}))": kl,
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
        vqvae: VQVAE,
        weight_sharing: bool = False,
        activation: ActivationFunc = 'mish',
        batch_norm: bool = False,  # TODO remove
    ):
        super().__init__()
        kwargs = dict(
            treatment_dim=treatment_dim,
            confounders_dim=confounders_dim,
            outcome_noise_dim=outcome_noise_dim,
            outcome_shape=outcome_size,
            activation=activation_layer_ctor(activation),
        )
        self.twin_net = DeepTwinNet(
            **kwargs,
            vqvae=vqvae,
            weight_sharing=weight_sharing,
        )
        self.infer_noise_net = DeepTwinNetNoiseEncoder(
            **kwargs,
            include_non_descendants=True,
        )
        self.outcome_noise_dim = outcome_noise_dim
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.save_hyperparameters(ignore=['vqvae'])
        self.vqvae = vqvae

        def optimizer_kwargs_for_model_param(module_name, param_name):
            if module_name == 'twin_net':
                return dict(lr=lr)
            elif module_name == 'infer_noise_net':
                return dict(lr=encoder_lr)
            else:
                raise RuntimeError(
                    f'Unexpected parameter {module_name=}, {param_name=}'
                )

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

    def _discrete_latent(self, z_q: Tensor) -> Tensor:
        with torch.no_grad():
            z_q_, e, _loss, _metrics = self.vqvae.model.vq(z_q)
            return e

    def model(self, batch):
        pyro.module('twin_net', self.twin_net)
        input, output = batch
        batch_size = input[V.factual_treatment].shape[0]

        y = self._discrete_latent(output[V.factual_outcome])
        y_star = self._discrete_latent(output[V.counterfactual_outcome])
        u_y_loc = torch.zeros(self.outcome_noise_dim, device=self.device)
        u_y_scale = torch.ones(self.outcome_noise_dim, device=self.device)

        with pyro.plate('data', size=batch_size):
            u_y = pyro.sample(
                name=V.outcome_noise,
                fn=Normal(u_y_loc, u_y_scale).to_event(1),
            )

            input = {**input, V.outcome_noise: u_y}
            y_, y_star_ = self.twin_net.forward_probs(**input)

            pyro.sample(
                name=V.factual_outcome,
                fn=Categorical(probs=y_).to_event(1),
                obs=y,
            )
            pyro.sample(
                name=V.counterfactual_outcome,
                fn=Categorical(probs=y_star_).to_event(1),
                obs=y_star,
            )

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

    def eval_elbo(
        self,
        batch: Tuple[BatchDictLike, BatchDictLike],
        num_samples: int,
    ) -> Tensor:
        svi_loss = CustomELBO(num_particles=num_samples)
        svi = pyro.infer.SVI(
            model=self.model,
            guide=self.guide,
            optim=pyro.optim.Adam({}),
            loss=svi_loss,
        )
        _ = svi.evaluate_loss(batch)
        return svi_loss.get_batched_elbo()

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
        metrics = dict(loss=loss, **svi_metrics)
        return loss, metrics


class PSFTwinNet(TwinNet):
    def __init__(
        self,
        vqvae: VQVAE,
        outcome_size: torch.Size,
        lr: float = 0.0002,
        encoder_lr: float = 0.0002,
    ):
        super().__init__(
            outcome_size=outcome_size,
            treatment_dim=PSFTwinNetDataset.treatment_dim(),
            confounders_dim=PSFTwinNetDataset.confounders_dim(),
            # FIXME: change dataset
            # 16, #16,  # 12,  # 8,  # 16,  # 32,  # PSFTwinNetDataset.outcome_noise_dim,
            outcome_noise_dim=16,
            lr=lr,
            encoder_lr=encoder_lr,
            vqvae=vqvae,
        )
        self.save_hyperparameters(ignore=['vqvae'])
