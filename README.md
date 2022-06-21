# Counterfactual content-based image retrieval
This repo contains the code in support of the final year project report for my Master of Engineering in Computing (Artificial Intelligence and Machinge Learning) on _Counterfactual Content-Based Image Retrieval via Variational Twin Networks and VQ-VAEs_


## What is counterfactual image retrieval?

---


Counterfactual reasoning about an image allows one to hypothesise how that image would have looked like, had something about the world in which it was produced been different: 'How would have my passport photo looked like, had I had blue eyes?' or 'How would this MRI scan have looked like, had the patient in it been 10 years older?'. Methods for counterfactual image generation enable sampling from the distribution over such counterfactual images. This work proposes, in addition to a generative method, a framework to retrieve the images which have the highest probability of being valid counterfactuals from a pre-existing dataset. Then, the two previous questions become 'Which photo, within a dataset, would I be most likely to have on my passport, had I had blue eyes?' and 'Which MRI scan, within a medical database, would be the most likely to be produced for the patient in this MRI scan, had they been 10 years older?'. 

## Codebase structure
---
An outline of the more important parts of the codebase is provided below.

```
├── ccbir
│   ├── data                # generation and loading of the project's dataset        
│   ├── models
│   │   ├── twinnet
│   │   │   ├── arch.py     # VTN's neural architecture components
│   │   │   ├── data.py     # data processing for VTN model
│   │   │   ├── model.py    # definition of VTN's probablistic model
│   │   │   ├── train.py    # training entry point
│   │   │   └── ... 
│   │   └── vqvae
│   │       ├── data.py     # data processing for VQ-VAE
│   │       ├── model.py    # PQ-VAE model
│   │       ├── train.py    # training entry point
│   │       ├── vq.py       # vector and product quantisers with discrete noise injection 
│   │       └── ...
│   ├── retrieval 
│   │   ├── cbir.py         # standard image retrieval logic
│   │   ├── ccbir.py        # counterfactual image retrieval logic
│   │   └── ...
│   ├── experiment
│   │   ├── benchmarks      # benchmarks' entry points
│   │   ├── experiments.py  # experiments' implementation
│   │   └── ...  
├── submodules              # third-party repos used
│   ├── deepscm
│   └── Morpho-MNIST
└── ...
```

## How to run?
--- 

### Datasets
---
Make sure you have all parts of the original MNIST database in a folder called `original` and a copy of the plain Morpho-MNIST dataset in a folder called `plain`. Links for these can be found at https://github.com/dccastro/Morpho-MNIST. Specify the location where you have placed these folders under the `morphomnist_data_path` in `ccbir/configuration.py`.


### Training
---
First, specify the location where you wish your checkpoints to be placed (which will be the ones under  `tensorboard_logs_path` in `ccbir/configuration.py`)

A VQ-VAE needs to be trained first, to do so run
```bash
python ccbir/models/vqvae/train.py
```
If it's the first time you train a VQ-VAE model, the dataset will be constructed automatically starting from MNIST; this process has been parallelised, but may still take a while depending on your processor.

Then, specify the model and checkpoint of the trained VQ-VAE at the entry point for the training of the VTN, which is found in `ccbir/models/twinnet/train.py`, and train the VTN with
```bash
python ccbir/models/twinnet/train.py
``` 

If it's the first time you train a VTN model, a dataset preprocessing step will run for a while, and its result  will then be cached under the `temporary_data_path` found in `ccbir/configuration.py` for future training runs. If you train a new VQ-VAE and then wish to train a VTN with it, you will need to delete or move the `temporary_data_path`, otherwise the cache for the preprocessed dataset for the VTN will have become stale.

### Benchmarking
---
Entrypoints for each benchmark type can be found in
`ccbir/experiment/benchmarks/*`. Make sure sure you specify the checkpoints of your trained models in the file for the benchmark you have selcted before running it, for example as
```bash
python ccbir/experiment/benchmarks/benchmark_ccbir_sample_similarity_rank.py 2>&1 | tee results.txt

```

Note that this codebase is still in experimental state, so feel free to create a new issue if something isn't working as you would expect it to.
