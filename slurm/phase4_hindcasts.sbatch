#!/bin/bash
# Phase 4: the training hindcasts (41_run_hindcasts.py) on 4 H100s at once.
# Each of the 4 jobs takes a fixed quarter of the start dates; finished files are
# skipped, so after a time-out or failure simply submit again.
#
#   (once, on the login node)  python scripts/40_hindcast_plan.py
#   sbatch slurm/phase4_hindcasts.sbatch
#   squeue --me
#   tail -n 3 slurm/logs/hind_*_*.out        (progress of each share)
#
# Fewer GPUs free? Change --array=0-3 to 0-1 (2 jobs) - the split adapts.
#SBATCH --job-name=hindcasts
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/hind_%A_%a.out
#SBATCH --error=slurm/logs/hind_%A_%a.out
set -uo pipefail
source slurm/env.sh

if [ ! -f results/phase4/hindcast_plan.csv ]; then
  echo "results/phase4/hindcast_plan.csv missing - run on the login node first:"
  echo "    python scripts/40_hindcast_plan.py"
  exit 1
fi
nvidia-smi --query-gpu=name --format=csv,noheader
python scripts/41_run_hindcasts.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$SLURM_ARRAY_TASK_COUNT"
