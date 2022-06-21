from ccbir.ccbir.models.vqvae import model_pq_rt_1
from ccbir.models.vqvae import model_vq
from ccbir.util import reset_random_seed
from ccbir.models.util import load_best_model, best_model_checkpoint_path
from ccbir.models.twinnet.model_vq import PSFTwinNet
from ccbir.experiment.experiments import  RetrievalExperiment
from ccbir.models.util import load_best_model
import torch
from pprint import pprint
from configuration import config
config.pythonpath_fix()
device = torch.device('cuda:1')


reset_random_seed(42)
vqvae = model_vq.VQVAE.load_from_checkpoint(
    'CHECKPOINT_PATH_HERE'
).to(device)
vqvae.eval()

twinnet = PSFTwinNet.load_from_checkpoint(
    'CHECKPOINT_PATH_HERE',
    vqvae=vqvae.to(torch.device('cpu')),
    map_location=device,
).to(device)
twinnet.eval()

vqvae_original_mnsit = model_pq_rt_1.VQVAE.load_from_checkpoint(
    'CHECKPOINT_PATH_HERE',
).to(device)
vqvae_original_mnsit.eval()

exp = RetrievalExperiment(
    vqvae=vqvae,
    twinnet=twinnet,
    vqvae_original_mnist=vqvae_original_mnsit,
)

metrics = exp.benchmark_cbir()

pprint(metrics)
