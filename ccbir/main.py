from configuration import config
config.pythonpath_fix()
from ccbir.models.dummy_vae import MNISTDummyVAE
from ccbir.data import MorphoMNISTLikeDataModule
#from ccbir.models.vqvae import VQVAE


def main():
    from pytorch_lightning.utilities.cli import LightningCLI
    from pytorch_lightning.callbacks import ModelCheckpoint

    cli = LightningCLI(
        MNISTDummyVAE,
        MorphoMNISTLikeDataModule,
        save_config_overwrite=True,
        run=False,  # deactivate automatic fitting
        trainer_defaults=dict(
            callbacks=[
                ModelCheckpoint(monitor='val_loss', save_top_k=1)
            ],
            max_epochs=1,
            gpus=1,
        ),
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, ckpt_path="best", datamodule=cli.datamodule)
    predictions = cli.trainer.predict(
        ckpt_path="best",
        datamodule=cli.datamodule
    )


if __name__ == '__main__':
    main()
