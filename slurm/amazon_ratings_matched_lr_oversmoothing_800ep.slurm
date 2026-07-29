#!/bin/bash
#SBATCH --job-name=ar8_os_lr3
#SBATCH --partition=log_gpu_24gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-39%2
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
  runs/amazon_ratings_fair_lr_800ep

cd /work/log1/$USER/gnn_thesis_cluster

python scripts/run_realworld.py \
  --grid "${GRID:?GRID not set}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --epochs 800 \
  --data-root /work/log1/$USER/pyg-data \
  --track-oversmoothing \
  --oversmoothing-epochs "0,1,10,25,50,100,200,400,600,800" \
  --oversmoothing-max-nodes 2048 \
  --oversmoothing-max-edges 200000 \
  --out-dir "${OUT_DIR:?OUT_DIR not set}"
