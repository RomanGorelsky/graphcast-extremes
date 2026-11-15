#!/bin/bash
# Phase 5, step 3: R0's scores and the four fine-tuning runs, one GPU each, at most 2 at
# a time, all with identical data, steps and learning rate:
#   task 0 = R0 (original, scored only) | 1 = R1 control | 2 = R2 + dry-air mass |
#   3 = R3 + moisture | 4 = R4 + energy
# Every stage runs in a fresh process (GPU memory left over from one stage made the next
# fail) and resumes from the run's checkpoint; JAX may use 95 % of the GPU (default 75 %).
# Cluster limit 12 h per job: a run that is stopped continues when submitted again.
#
#   sbatch slurm/phase5_train.sbatch              (all)
#   sbatch --array=1-4%2 slurm/phase5_train.sbatch  (runs only, R0 already scored)
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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

ID=$SLURM_ARRAY_TASK_ID
if [ "$ID" -eq 0 ]; then
  python scripts/51_train.py --baseline
  exit $?
fi
for attempt in 1 2 3 4 5 6; do                      # one stage per process
  [ -f "results/phase5/runs/R${ID}/params_R${ID}.ckpt" ] && break   # finished
  python scripts/51_train.py --run "R${ID}" --one-stage || exit 1
done
