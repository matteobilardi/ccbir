from configuration import config
config.pythonpath_fix()
from ccbir.models.vqvae import VQVAE
from ccbir.data import MorphoMNISTLikeDataModule
from ccbir.models.dummy_vae import MNISTDummyVAE


def main():
    from pytorch_lightning.utilities.cli import LightningCLI
    from pytorch_lightning.callbacks import ModelCheckpoint

    cli = LightningCLI(
        VQVAE,
        MorphoMNISTLikeDataModule,
        save_config_overwrite=True,
        run=False,  # deactivate automatic fitting
        trainer_defaults=dict(
            callbacks=[
                ModelCheckpoint(
                    monitor='val_loss',
                    filename='synth-mnist-{epoch:03d}-{val_loss:.7f}',
                    save_top_k=3,
                )
            ],
            max_epochs=50,
            gpus=1,
        ),
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, ckpt_path="best", datamodule=cli.datamodule)


if __name__ == '__main__':
    main()
