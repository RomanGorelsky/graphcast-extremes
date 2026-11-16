#!/bin/bash
# Phase 6 extension: 52 weekly 10-day forecasts of 2020 with every model, scored on
# ordinary weather (RMSE, ACC, sharpness, global budgets). One GPU per model, at most 2
# at a time. Task 0 = R0, then ERA5's own budgets on the same days. Resumable.
#   task 0 = R0 (+ ERA5 budgets) | 1 = R1 | 2 = R2 | 3 = R3 | 4 = R4
#
#   sbatch slurm/phase6_ordinary.sbatch
#   tail -n 2 slurm/logs/p6ord_*_*.out
#SBATCH --job-name=p6ord
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/p6ord_%A_%a.out
#SBATCH --error=slurm/logs/p6ord_%A_%a.out
set -uo pipefail
source slurm/env.sh

RUN="R${SLURM_ARRAY_TASK_ID}"
python scripts/62_ordinary_weather.py --model "$RUN" || exit 1
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  python scripts/62_ordinary_weather.py --truth-budgets
fi
echo "== finished $(date '+%F %T')"
