#!/bin/bash
#SBATCH --job-name=install_torch_pyg_clean
#SBATCH --partition=log_gpu_24gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/venvs/gnn-thesis-gpu/bin/activate

export TMPDIR=/var/tmp/$USER/tmp
export PIP_CACHE_DIR=/var/tmp/$USER/pip-cache
export TORCH_HOME=/var/tmp/$USER/torch-cache
export MPLCONFIGDIR=/var/tmp/$USER/matplotlib
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$MPLCONFIGDIR" /var/tmp/$USER/pyg-data

python --version
which python
pip --version

pip install --upgrade pip setuptools wheel

# Clean GPU install of PyTorch for Linux + CUDA 12.8
pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Minimal PyG install first
pip install --no-cache-dir --force-reinstall torch_geometric

# Matching PyG extension wheels for torch 2.7.x + cu128
pip install --no-cache-dir --force-reinstall pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

python - <<'PY'
import torch
import torch_geometric
print("torch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("torch_geometric:", torch_geometric.__version__)
PY
