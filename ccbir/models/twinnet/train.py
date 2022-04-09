from ccbir.configuration import config
from ccbir.models.twinnet.data import PSFTwinNetDataModule
from ccbir.models.twinnet.model import PSFTwinNet
from ccbir.models.util import load_best_model
from ccbir.models.vqvae.model import VQVAE
from ccbir.util import maybe_unbatched_apply, tune_lr
from functools import partial
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI
from torch import Tensor
import torch
import torch.multiprocessing
import shutil


def vqvae_embed_image(vqvae: VQVAE, image: Tensor):
    embed = partial(vqvae.embed, latent_type='decoder_input')

    with torch.no_grad():
        return maybe_unbatched_apply(embed, image)


def main():
    vqvae = load_best_model(VQVAE)
    embedding_size = vqvae.encoder_net(torch.rand((64, 1, 28, 28))).shape[1:]
    print(f"{embedding_size}")

    # clean up possible database caches from interrupted previous run
    # config.clear_temporary_data()

    # FIXME: initialsiation, traning, saving and storing a bit hacky currently
    # Functions signatures needed by LightningCLI so can't use functools.partial
    def Model() -> PSFTwinNet:
        return PSFTwinNet(outcome_size=embedding_size)

    def Datamodule() -> PSFTwinNetDataModule:
        return PSFTwinNetDataModule(
            embed_image=partial(vqvae_embed_image, vqvae)
        )

    cli = LightningCLI(
        model_class=Model,
        datamodule_class=Datamodule,
        save_config_overwrite=True,
        run=False,  # deactivate automatic fitting
        trainer_defaults=dict(
            callbacks=[
                ModelCheckpoint(
                    monitor='val_loss',
                    filename='twinnet-{epoch:03d}-{val_loss:.7f}',
                    save_top_k=3,
                    save_last=True,
                ),
            ],
            max_epochs=2000,
            gpus=1,
            # profiler='simple',
        ),
    )

    """

    lr_finder = cli.trainer.tuner.lr_find(
        model=cli.model,
        datamodule=cli.datamodule,
        max_lr=0.01,
        num_training=10000,
        early_stop_threshold=6.0,
    )

    tune_lr(lr_finder, cli.model)

    """

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
