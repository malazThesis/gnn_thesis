#!/bin/bash
#SBATCH --job-name=synth_ring_p1
#SBATCH --partition=log_gpu_24gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-107
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source ~/venvs/gnn-thesis-gpu/bin/activate

export TMPDIR=/var/tmp/$USER/tmp
export PIP_CACHE_DIR=/var/tmp/$USER/pip-cache
export TORCH_HOME=/var/tmp/$USER/torch-cache
export MPLCONFIGDIR=/var/tmp/$USER/matplotlib
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$MPLCONFIGDIR"

cd ~/gnn_thesis_cluster

python scripts/run_synthetic_ring.py \
  --grid configs/synthetic_ring_grid_phase1.csv \
  --task-id ${SLURM_ARRAY_TASK_ID} \
  --epochs 150 \
  --out-dir runs/synthetic_ring_phase1
