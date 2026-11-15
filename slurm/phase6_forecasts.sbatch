#!/bin/bash
# Phase 6: forecasts of the 14 case studies (starts -1/-3/-5/-7 d, 10 days each) and of
# the 12 ordinary dates, with each fine-tuned run - exactly as Phase 3 did for R0.
# One GPU per run, at most 2 at a time. Resumable: finished forecasts are skipped.
#   task 1 = R1 | 2 = R2 | 3 = R3 | 4 = R4
#
#   sbatch slurm/phase6_forecasts.sbatch
#   tail -n 2 slurm/logs/p6fc_*_*.out
#SBATCH --job-name=p6fc
#SBATCH --array=1-4%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/p6fc_%A_%a.out
#SBATCH --error=slurm/logs/p6fc_%A_%a.out
set -uo pipefail
source slurm/env.sh

RUN="R${SLURM_ARRAY_TASK_ID}"
echo "== case-study forecasts, $RUN"; python scripts/31_run_case_forecasts.py --model "$RUN" || exit 1
echo "== ordinary dates, $RUN";      python scripts/32_run_noharm.py --model "$RUN" --label "$RUN"
echo "== finished $(date '+%F %T')"
