from ccbir.models.vqvae.data import VQVAEMorphoMNISTDataModule
from ccbir.models.vqvae.model import VQVAE


def main():
    from pytorch_lightning.utilities.cli import LightningCLI
    from pytorch_lightning.callbacks import ModelCheckpoint

    cli = LightningCLI(
        VQVAE,
        VQVAEMorphoMNISTDataModule,
        save_config_overwrite=True,
        run=False,  # deactivate automatic fitting
        trainer_defaults=dict(
            callbacks=[
                ModelCheckpoint(
                    monitor='val_loss',
                    # dirpath=str(config.checkpoints_path_for_model(
                    #    model_type=VQVAE
                    # )),
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
