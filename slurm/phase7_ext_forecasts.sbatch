#!/bin/bash
# Phase 7, prospect 2: forecasts of the EXTENDED case-study set (case_studies_ext.csv),
# one model per task, one GPU each, at most 2 at a time:
#   task 0 = R0 | 1 = R1 (fine-tuned control) | 2 = R4 (+ energy) | 3 = R5 (sharpness loss)
# (R2 and R3 matched R1 on the 14 events; add them with MODELS if wanted.)
# ~360 forecasts per model at ~1.2 min -> ~7 h per task; resumable (submit again).
# R5 needs slurm/phase7_train_r5.sbatch first: run the R5 task later with --array=3.
#
#   sbatch --array=0-2%2 slurm/phase7_ext_forecasts.sbatch
#   sbatch --array=3 slurm/phase7_ext_forecasts.sbatch          (after R5 is trained)
#   tail -n 2 slurm/logs/p7ext_*_*.out
#SBATCH --job-name=p7ext
#SBATCH --array=0-2%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/p7ext_%A_%a.out
#SBATCH --error=slurm/logs/p7ext_%A_%a.out
set -uo pipefail
source slurm/env.sh
export EXTREMES_SET=ext
[ -f results/phase2/case_studies_ext.csv ] || { echo "missing results/phase2/case_studies_ext.csv (phase7_events)"; exit 1; }
read -r -a MODELS <<< "${MODELS:-R0 R1 R4 R5}"
RUN="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "== extended set, $RUN"
python scripts/31_run_case_forecasts.py --model "$RUN"
echo "== finished $(date '+%F %T')"
