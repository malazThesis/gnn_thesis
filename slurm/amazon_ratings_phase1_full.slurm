#!/bin/bash
#SBATCH --job-name=amrat_p1_full
#SBATCH --partition=log_gpu_24gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-239%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

source ~/venvs/gnn-thesis-gpu/bin/activate

export TMPDIR=/work/log1/$USER/tmp
export PIP_CACHE_DIR=/work/log1/$USER/cache/pip
export TORCH_HOME=/work/log1/$USER/cache/torch
export MPLCONFIGDIR=/work/log1/$USER/cache/matplotlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p \
  "$TMPDIR" \
  "$PIP_CACHE_DIR" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  /work/log1/$USER/pyg-data \
  logs \
  runs/amazon_ratings_phase1_full

cd /work/log1/$USER/gnn_thesis_cluster

python scripts/run_realworld.py \
  --grid configs/amazon_ratings_phase1_full.csv \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --epochs 200 \
  --data-root /work/log1/$USER/pyg-data \
  --out-dir runs/amazon_ratings_phase1_full
