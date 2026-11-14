#!/bin/bash
# Phase 5, step 3: R0's scores and the four fine-tuning runs, one GPU each, at most 2 at
# a time, all with identical data, steps and learning rate:
#   task 0 = R0 (original, scored only) | 1 = R1 control | 2 = R2 + dry-air mass |
#   3 = R3 + moisture | 4 = R4 + energy
# Every run is scored on 6-h, 1-day and 5-day forecasts of 2015 and keeps its weights
# at the end of stage 1. Checkpoint every 250 steps; cluster limit 12 h per job - a run
# that needs longer stops at 12 h; submit again and it continues.
#
#   sbatch slurm/phase5_train.sbatch
#   tail -n 2 slurm/logs/train_*_*.out
#SBATCH --job-name=p5train
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/train_%A_%a.out
#SBATCH --error=slurm/logs/train_%A_%a.out
set -uo pipefail
source slurm/env.sh

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  python scripts/51_train.py --baseline
else
  python scripts/51_train.py --run "R${SLURM_ARRAY_TASK_ID}"
fi
