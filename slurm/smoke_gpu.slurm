#!/bin/bash
#SBATCH --job-name=gnn_smoke
#SBATCH --partition=log_gpu_24gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/venvs/gnn-thesis/bin/activate

export TMPDIR=/var/tmp/$USER/tmp
export PIP_CACHE_DIR=/var/tmp/$USER/pip-cache
export TORCH_HOME=/var/tmp/$USER/torch-cache
export MPLCONFIGDIR=/var/tmp/$USER/matplotlib
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$MPLCONFIGDIR" /var/tmp/$USER/pyg-data

echo "HOSTNAME=$(hostname)"
echo "PYTHON=$(which python)"
python --version
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

nvidia-smi || true

python - <<'PY'
import os, sys, platform
print("Python:", sys.version)
print("Platform:", platform.platform())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
