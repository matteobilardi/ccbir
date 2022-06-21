from ccbir.models.vqvae import model
from ccbir.util import reset_random_seed
from ccbir.models.twinnet.model import PSFTwinNet
from ccbir.experiment.experiments import PSFTwinNetExperiment, RetrievalExperiment
from ccbir.models.util import load_best_model
from ccbir.experiment.experiments import VQVAEExperiment
import torch
from pprint import pprint
from configuration import config
config.pythonpath_fix()
device = torch.device('cuda:0')

reset_random_seed(42)
vqvae = model.VQVAE.load_from_checkpoint(
    "CHECKPOINT_PATH_HERE"
).to(device)
vqvae.eval()


twinnet = PSFTwinNet.load_from_checkpoint(
    'CHECKPOINT_PATH_HERE',
    vqvae=vqvae.to(torch.device('cpu')),
    map_location=device,
).to(device)
twinnet.eval()


exp = PSFTwinNetExperiment(
    vqvae=vqvae,
    twinnet=twinnet,
)

args = dict(
    num_samples=2**14,
    train=False,
)
pprint(args)
metrics = exp.benchmark_generative_model(**args)


pprint(metrics)
