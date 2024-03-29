import torch.multiprocessing
import torch
from torch import Tensor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
from functools import partial
from ccbir.util import maybe_unbatched_apply, split_apply_cat, tune_lr
from ccbir.models.vqvae.model import VQVAE
from ccbir.models.util import load_best_model
from ccbir.models.twinnet.model import PSFTwinNet
from ccbir.models.twinnet.data import PSFTwinNetDataModule
from ccbir.configuration import config
config.pythonpath_fix()


def vqvae_embed_image(vqvae: VQVAE, image: Tensor) -> Tensor:
    embed = partial(vqvae.embed, latent_type='decoder_input')
    embed = split_apply_cat(embed)
    embed = maybe_unbatched_apply(embed)

    with torch.no_grad():
        return embed(image.to(vqvae.device)).to(image.device)


def main():

    vqvae = VQVAE.load_from_checkpoint(
        '/vol/bitbucket/mb8318/ccbir/logs/tb_logs/vqvae/version_327/checkpoints/vqvae-morphomnist-epoch=322-val_loss=0.0011884.ckpt'
        # '/vol/bitbucket/mb8318/ccbir/logs/tb_logs/vqvae/version_320/checkpoints/vqvae-morphomnist-epoch=155-val_loss=0.0015304.ckpt'
        # '/vol/bitbucket/mb8318/ccbir/logs/tb_logs/vqvae/version_324/checkpoints/vqvae-morphomnist-epoch=245-val_loss=0.0012822.ckpt'
        # '/vol/bitbucket/mb8318/ccbir/logs/tb_logs/vqvae/version_69/checkpoints/last.ckpt'
        # '/vol/bitbucket/mb8318/ccbir/logs/tb_logs/vqvae/version_61/checkpoints/last.ckpt'
        # '/vol/bitbucket/mb8318/ccbir/logs/tb_logs/vqvae/version_67/checkpoints/last.ckpt'
    )
    vqvae.eval()
    #vqvae = load_best_model(VQVAE)
    embedding_size = vqvae.encoder_net(torch.rand((64, 1, 28, 28))).shape[1:]
    print(f"{embedding_size}")

    # clean up possible database caches from interrupted previous run
    # config.clear_temporary_data()

    def Model() -> PSFTwinNet:
        return PSFTwinNet(outcome_size=embedding_size, vqvae=vqvae)

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
            logger=TensorBoardLogger(
                save_dir=str(config.tensorboard_logs_path),
                name='twinnet',
            ),
            callbacks=[
                ModelCheckpoint(
                    monitor='val/loss',
                    filename='twinnet-{epoch:03d}-{val_loss:.7f}',
                    save_top_k=3,
                    save_last=True,
                ),
                ModelCheckpoint(
                    monitor='train/loss',
                    filename='twinnet-{epoch:03d}-{train_loss:.7f}',
                    save_top_k=3,
                    save_last=False,
                    save_on_train_epoch_end=True,
                ),
            ],
            max_epochs=5000,
            gpus=1,
            # profiler='advanced',
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
