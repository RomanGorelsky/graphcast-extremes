#!/bin/bash
# Phase 5, step 3: the four fine-tuning runs, one GPU each (at most 2 at a time), all with identical data,
# steps and learning rate:  R1 control | R2 + dry-air mass | R3 + moisture | R4 + energy
# Each run saves a checkpoint every 250 steps; after a time-out simply submit again.
#
#   sbatch slurm/phase5_train.sbatch
#   tail -n 2 slurm/logs/train_*_*.out
#
# Run lengths: set from the benchmark, e.g. add  --stage1-steps 4000  to the python line.
#SBATCH --job-name=p5train
#SBATCH --array=1-4%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/train_%A_%a.out
#SBATCH --error=slurm/logs/train_%A_%a.out
set -uo pipefail
source slurm/env.sh

python scripts/51_train.py --run "R${SLURM_ARRAY_TASK_ID}"
