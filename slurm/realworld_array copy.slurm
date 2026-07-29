#!/bin/bash
#SBATCH --job-name=realworld_grid
#SBATCH --partition=log_gpu_24gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-119
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source ~/venvs/gnn-thesis-gpu/bin/activate
export TMPDIR=/var/tmp/$USER/tmp
export PIP_CACHE_DIR=/var/tmp/$USER/pip-cache
export TORCH_HOME=/var/tmp/$USER/torch-cache
export MPLCONFIGDIR=/var/tmp/$USER/matplotlib
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$MPLCONFIGDIR" /var/tmp/$USER/pyg-data

cd ~/gnn_thesis_cluster

python scripts/run_realworld.py \
  --grid configs/realworld_grid.csv \
  --task-id ${SLURM_ARRAY_TASK_ID} \
  --epochs 200 \
  --data-root data \
  --out-dir runs/realworld
