#!/bin/bash
#SBATCH --job-name=fix_verify_torch_pyg
#SBATCH --partition=log_gpu_24gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
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
nvidia-smi || true

# Remove mismatched PyG extension wheels
pip uninstall -y pyg_lib torch_scatter torch_sparse

# Reinstall matching wheels for torch 2.11.0 + cu128
pip install --no-cache-dir pyg_lib torch_scatter torch_sparse \
  -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

# Verify torch + cuda + pyg
python - <<'PY'
import torch
import torch_geometric

print("torch:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("torch_geometric:", torch_geometric.__version__)

x = torch.randn(4, 4, device="cuda")
print("cuda tensor ok:", x.device, x.shape)

from torch_geometric.datasets import Planetoid
dataset = Planetoid(root="/tmp/Planetoid", name="Cora")
data = dataset[0]
print("Cora loaded:", data)
print("num_features:", dataset.num_features)
print("num_classes:", dataset.num_classes)
PY
