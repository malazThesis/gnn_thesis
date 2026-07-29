#!/bin/bash
#SBATCH --job-name=rw_os
#SBATCH --partition=log_gpu_24gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

source ~/venvs/gnn-thesis-gpu/bin/activate

export TMPDIR=/work/log1/$USER/tmp
export PIP_CACHE_DIR=/work/log1/$USER/cache/pip
export TORCH_HOME=/work/log1/$USER/cache/torch
export MPLCONFIGDIR=/work/log1/$USER/cache/matplotlib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/log1/$USER/gnn_thesis_cluster

export PYTHONPATH="/work/log1/$USER/gnn_thesis_cluster:${PYTHONPATH:-}"

: "${GRID:?GRID environment variable is required}"
: "${OUT_DIR:?OUT_DIR environment variable is required}"

METRIC_EPOCHS="${METRIC_EPOCHS:-1,2,4,8,16,32,50,100,150,200}"
MAX_NODES="${MAX_NODES:-1024}"
MAX_EDGES="${MAX_EDGES:-100000}"

mkdir -p \
  "$TMPDIR" \
  "$PIP_CACHE_DIR" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  /work/log1/$USER/pyg-data \
  logs \
  "$OUT_DIR"

python scripts/run_realworld.py \
  --grid "$GRID" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --epochs 200 \
  --data-root /work/log1/$USER/pyg-data \
  --out-dir "$OUT_DIR" \
  --track-oversmoothing \
  --oversmoothing-epochs "$METRIC_EPOCHS" \
  --oversmoothing-max-nodes "$MAX_NODES" \
  --oversmoothing-max-edges "$MAX_EDGES"
