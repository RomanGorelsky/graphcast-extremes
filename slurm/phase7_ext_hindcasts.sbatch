#!/bin/bash
# Phase 7, prospects 1+2: training hindcasts for the bias corrections (PBC, QM, TailQM)
# on the EXTENDED set. One GPU per task, each a fixed share of the start dates (2 or 3
# tasks: --array=0-1 or 0-2; the split follows the number of tasks); resumable.
# First, on the login node (seconds):
#     EXTREMES_SET=ext python scripts/40_hindcast_plan.py --years 2019 2018
#
#   sbatch slurm/phase7_ext_hindcasts.sbatch                 (2 GPUs)
#   sbatch --array=0-2 slurm/phase7_ext_hindcasts.sbatch     (3 GPUs)
#   (always resubmit with the same number of tasks, so the split stays the same)
#   tail -n 2 slurm/logs/p7hind_*_*.out
#SBATCH --job-name=p7hind
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/p7hind_%A_%a.out
#SBATCH --error=slurm/logs/p7hind_%A_%a.out
set -uo pipefail
source slurm/env.sh
export EXTREMES_SET=ext
[ -f results/sets/ext/phase4/hindcast_plan.csv ] || {
  echo "plan missing - run: EXTREMES_SET=ext python scripts/40_hindcast_plan.py --years 2019 2018"; exit 1; }
python scripts/41_run_hindcasts.py --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$SLURM_ARRAY_TASK_COUNT"
echo "== finished $(date '+%F %T')"
