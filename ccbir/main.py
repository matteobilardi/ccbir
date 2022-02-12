from torch import unsqueeze
from configuration import config
from deepscm.datasets.morphomnist import MorphoMNISTLike
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from typing import Optional
import pytorch_lightning as pl
import pl_bolts
from torch.nn import Conv2d


class MorphoMNISTLikeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = str(config.synth_mnist_data_path),
        train_batch_size: int = 64,
        test_batch_size: int = 64,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def prepare_data(self):
        # check that data lives on disk, raise exception otherwise
        Path(self.data_dir).resolve(strict=True)

    def setup(self, stage: Optional[str] = None):

        # TODO: the other morphometrics are not already part of the given
        # dataset. I should try to understand why that is? Also, I suspect
        # that we want to show to the twin network as much information about
        # the image that we have (intensity, slant, widht, thickness etc)
        # other than just the variables that we want to intervene upon. Notably
        # deep scm doesn't seem to include the other variables other than those
        # being m
        columns = [
            # 'area',
            # 'height',
            # 'length',
            # 'slant',
            # 'width',
            'intensity',
            'thickness',
        ]

        if stage == 'fit' or stage is None:
            mnist_train = MorphoMNISTLike(
                self.data_dir, train=True, columns=columns
            )

            num_val = len(mnist_train) // 10
            num_train = len(mnist_train) - num_val
            self.mnist_train, self.mnist_val = random_split(
                mnist_train, [num_train, num_val]
            )

        if stage == 'test' or stage is None:
            self.mnist_test = MorphoMNISTLike(
                self.data_dir, train=False, columns=columns
            )

        # TODO: Does any normalisation need to occur?

        # TODO: with regards to interventions, can we know straight from the
        # deepscm dataset what covariates have been intervened upon and what the
        # original image was so that we can use the infromation to later train
        # the twin network?
        #
        # Well, the data generation process in gen_dataset seems to discard that
        # information and only generate images whose thickness and intensity has
        # been randomly changed. But then how do they test that a predicted
        # counterfactual is close to the ground truth one?

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train, batch_size=self.train_batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.test_batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.test_batch_size)

    # TODO: remove
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class MyDummyVAE(pl_bolts.models.VAE):
    def __init__(self):
        super().__init__(input_height=28, first_conv=False, maxpool1=False)

        # based on https://discuss.pytorch.org/t/no-sample-variety-on-mnist-for-pl-bolts-models-autoencoders-vae/111298
        # changing first and last convolution to deal with 28x28 grayscale image
        self.encoder.conv1 = Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.decoder.conv1 = Conv2d(
            64 * self.decoder.expansion,
            1,
            kernel_size=3,
            stride=1,
            padding=3,
            bias=False
        )

    # TODO: there is probably a better way to apply a function to the the
    # dataset before passing it to the VAE but this should do for now.
    def _prep_batch(self, batch):
        ## Remove all morphometrics, keeping only the image, and convert to RGB
        #X = batch['image'].unsqueeze(1).repeat(1, 3, 1, 1).float()
        X = batch['image'].unsqueeze(1).float()
        y = None  # self-supervised setting so label
        return X, y

    def training_step(self, batch, batch_idx):
        batch = self._prep_batch(batch)
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        batch = self._prep_batch(batch)
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        batch = self._prep_batch(batch)
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        batch = self._prep_batch(batch)
        return super().predict_step(batch, batch_idx, dataloader_idx)


def main():
    from pytorch_lightning.utilities.cli import LightningCLI
    from pl_bolts.callbacks import PrintTableMetricsCallback

    cli = LightningCLI(
        MyDummyVAE,
        MorphoMNISTLikeDataModule,
        save_config_overwrite=True,
        run=False,  # used to de-activate automatic fitting.
        trainer_defaults=dict(
            callbacks=PrintTableMetricsCallback(),
            max_epochs=10,
            gpus=1,
        ),
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    predictions = cli.trainer.predict(
        ckpt_path="best",
        datamodule=cli.datamodule
    )
    print(predictions[0])


if __name__ == '__main__':
    main()
