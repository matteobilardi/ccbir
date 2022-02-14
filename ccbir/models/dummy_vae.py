import pytorch_lightning as pl
import pl_bolts
from torch.nn import Conv2d
import pl_bolts

class MNISTDummyVAE(pl_bolts.models.VAE):
    def __init__(self):
        super().__init__(input_height=28)

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
        # Remove all morphometrics, keeping only the image, and convert to RGB
        #X = batch['image'].unsqueeze(1).repeat(1, 3, 1, 1).float()
        X = batch['image']
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
        # TODO: idiosyncrasies of pl_bolt's VAE: do better in own implementation
        X, _y = self._prep_batch(batch)
        batch = X
        return super().predict_step(batch, batch_idx, dataloader_idx)
