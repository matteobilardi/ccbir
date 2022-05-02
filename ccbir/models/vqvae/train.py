from ccbir.configuration import config
config.pythonpath_fix()
from ccbir.models.vqvae.data import VQVAEMorphoMNISTDataModule
from ccbir.models.vqvae.model import VQVAE
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    from pytorch_lightning.utilities.cli import LightningCLI
    from pytorch_lightning.callbacks import ModelCheckpoint

    cli = LightningCLI(
        VQVAE,
        VQVAEMorphoMNISTDataModule,
        save_config_overwrite=True,
        run=False,  # deactivate automatic fitting
        trainer_defaults=dict(
            logger=TensorBoardLogger(
                save_dir=str(config.tensorboard_logs_path),
                name='vqvae',
            ),
            callbacks=[
                ModelCheckpoint(
                    monitor='val_loss',
                    filename='vqvae-morphomnist-{epoch:03d}-{val_loss:.7f}',
                    save_top_k=3,
                    save_last=True,
                )
            ],
            max_epochs=3000,
            gpus=1,
        ),
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, ckpt_path="best", datamodule=cli.datamodule)


if __name__ == '__main__':
    main()
