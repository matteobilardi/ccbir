from configuration import config
config.pythonpath_fix()
from ccbir.models.vqvae import VQVAE

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
