import enum
from turtle import forward
from typing import Callable, Dict, Mapping, Tuple
import pytorch_lightning as pl
from torch import Tensor, nn
from torch.distributions import Distribution
import torch.nn.functional as F
import torch


class BaseSimpleTwinNet(nn.Module):
    """Twin network for DAG: X -> Y <- Z that allows sampling from
    P(Y, Y* | X, X*, Z)"""

    def __init__(self):
        super().__init__()

    def forward(
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        outcome_noise: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()
        return factual_outcome, counterfactual_outcome


class DeepSimpleTwinNet(BaseSimpleTwinNet):
    def __init__(
        self,
        treatment_dim: int,
        confounders_dim: int,
        outcome_dim: int,
        outcome_noise_dim: int,
    ):
        super().__init__()

    def forward(
        self,
        factual_treatment: Tensor,
        counterfactual_treatment: Tensor,
        confounders: Tensor,
        outcome_noise: Tensor
    ) -> Tuple[Tensor, Tensor]:

        # TODO
        ...


class CBIRDatabase:
    def __init__(
        self,
        extract_feature_vector: Callable[[Tensor], Tensor],
    ):
        pass

    def insert(image):
        pass

    def find_closest_images(img_feature_vector, top_k):
        pass


class PlainSwollenFracturedTwinNet(pl.LightningModule):
    # NOTE: For now relying on binary treatments: either swelling or
    # fracture
    treatments = ["swell", "fracture"]
    _treatement_to_index = {t: idx for idx, t in enumerate(treatments)}

    # TODO: probaby better move metrics and labels somewhere more appropriate
    metrics = ['area', 'length', 'thickness', 'slant', 'width', 'height']
    labels = list(range(10))

    def __init__(
        self,
        feature_vect_dim: int,
        outcome_noise_dim: int,
    ):
        super().__init__()
        self.twin_net = DeepSimpleTwinNet(
            treatment_dim=len(self.treatments),
            confounders_dim=len(self.metrics) + len(self.labels),
            outcome_dim=feature_vect_dim,
            outcome_noise_dim=outcome_noise_dim,
        )
        self.outcome_noise_dim = outcome_noise_dim

    def treatment_to_index(self, treatment: str) -> int:
        try:
            return self._treatement_to_index[treatment]
        except KeyError:
            raise ValueError(
                f"{treatment=} is not a valid treatment: must be one of "
                f"{', '.join(self.treatments)}"
            )

    def one_hot_treatment(self, treatment: str) -> Tensor:
        return F.one_hot(
            torch.as_tensor(self.treatement_to_index(treatment)),
            num_classes=len(self.treatments),
        )

    def forward(
        self,
        plain_image_label: Tensor,
        plain_image_metrics: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        batch_size = plain_image_label.shape[0]

        # Produce one-hot encodings of binary treatement (either swell or
        # fracture)
        swell = self.one_hot_treatment('swell').float().repeat(batch_size)
        fracture = \
            self.one_hot_treatment('fracture').float().repeat(batch_size)

        label = F.one_hot(
            plain_image_label,
            num_classes=len(self.labels)
        ).float()

        # NOTE: for now, no metric normalisation is occurring
        # Make a tensor merging the metrics sorted by metric name
        # to ensure consitent order
        metrics = torch.hstack([
            plain_image_metrics[metric]
            for metric in sorted(plain_image_metrics.keys())
        ])

        swollen_fv, fractured_fv = self.twin_net(
            factual_treatment=swell,
            counterfactual_treatment=fracture,
            confounders=torch.hstack((label, metrics)),
            outcome_noise=torch.randn((batch_size, self.outcome_noise_dim)),
        )

        return swollen_fv, fractured_fv

    def training_step(self, batch, _batch_idx):
        swollen_fv = batch['swollen']['feature_vector']
        fractured_fv = batch['fractured']['feature_vector']
        label = batch['plain']['label']
        metrics = batch['plain']['metrics']

        swollen_fv_hat, fractured_fv_hat = self(label, metrics)

        swollen_fv_loss = F.mse_loss(swollen_fv_hat, swollen_fv)
        fractured_fv_loss = F.mse_loss(fractured_fv_hat, fractured_fv)
        loss = swollen_fv_loss + fractured_fv_loss

        return dict(
            loss=loss,
            swollen_fv_loss=swollen_fv_loss,
            fractured_fv_loss=fractured_fv_loss,
        )


class PlainSwollenFracturedCCBIR:
    def __init__(
        self,
        encode: Callable[[Tensor], Tensor],
        decode: Callable[[Tensor], Tensor],
        database: CBIRDatabase,
    ) -> None:
        pass

    def counterfactual_fractured_img(
        original_img_info: dict,
        swollen_img: Tensor,
    ):
        swollen_fv = self.encode(swollen_img)

        pass

    def retrieve_counterfactual_fractured(
        original_img_info,
        swollen_img,
    ):
        pass


class PlainSwellFracTwinNet(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()


"""
class DeepTwinNet(BaseTwinNet):
    def __init__(self):
        super().__init__()

    def forward(self, observed_factual_intensity, intervention_counterfactual_intensity):
        noise_thickness = sample_gaussian()
        sampled_factual_thickness = f_thickness(
            observed_factual_intensity, noise_thickness)
        sampled_counterfactual_thickness = f_thickness(
            do_counterfactual_intensity, noise_thickness)

        noise_image = sample_gaussian())
        sampled_factual_image = f_image(
            observed_factual_intensity, sampled_factual_thickness, noise_image)
        sampled_counterfactual_image = f_image(
            intervention_counterfactual_intensity, sampled_counterfactual_thickness, noise_image)


        output = {
            'factual': [observed_factual_intensity, sampled_factual_thickness, sampled_factual_image]
            'counterfactual': [intervention_counterfactual_intensity, sampled_counterfactual_thickness, sampled_counterfactual_image]
        }

        return output

    def sample(self, observed_intensity, observed_thickness, observed_image, intervention_intensity):
        while True:
            output = self.forward(observed_intensity, intervention_intensity)
            _, sampled_thickness, sampled_image =  output['factual']
            if sampled_thickness.isclose(observed_thickness) and sampled_image.isclose(observed_image):
                _, _, counterfactual_image = output['counterfactual']
                return counterfactual_image"
"""
