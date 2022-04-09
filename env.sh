export CCBIR_VENV="/vol/bitbucket/mb8318/ccbir_venv"
export CCBIR_ROOT=$PWD
export PYTHONPATH=$CCBIR_ROOT:"$CCBIR_ROOT/ccbir":"$CCBIR_ROOT/ccbir/pytorch_vqvae":"$CCBIR_ROOT/submodules/deepscm"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export LD_LIBRARY_PATH="/vol/cuda/11.4.120-cudnn8.2.4/lib64"

source "$CCBIR_VENV/bin/activate"