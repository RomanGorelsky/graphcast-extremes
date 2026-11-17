#!/bin/bash
# Phase 7, prospect 3: R5 on the 14 case studies (as Phase 6 did for R1-R4) and on the
# 52 ordinary dates of 2020. One GPU, ~8-9 h; resumable (submit again if stopped).
#
#   sbatch slurm/phase7_r5_eval.sbatch
#   tail -n 2 slurm/logs/p7r5eval_*.out
#SBATCH --job-name=p7r5eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/p7r5eval_%j.out
#SBATCH --error=slurm/logs/p7r5eval_%j.out
set -uo pipefail
source slurm/env.sh
[ -f results/phase5/runs/R5/params_R5.ckpt ] || { echo "R5 not trained yet (slurm/phase7_train_r5.sbatch)"; exit 1; }
echo "== case studies";  python scripts/31_run_case_forecasts.py --model R5 || exit 1
echo "== ordinary dates"; python scripts/62_ordinary_weather.py --model R5
echo "== finished $(date '+%F %T')"
