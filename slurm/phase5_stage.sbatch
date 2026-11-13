#!/bin/bash
# Phase 5, step 1: stage the training data (2010-2014) and the validation year (2015),
# one job per year, CPU only. Resumable per month: if a job stops, submit again.
#
#   sbatch slurm/phase5_stage.sbatch
#   squeue --me
#   tail -n 3 slurm/logs/stage_*.out
#
# If the cluster refuses a job without a GPU, add the line  #SBATCH --gres=gpu:1
#SBATCH --job-name=stage
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/stage_%A_%a.out
#SBATCH --error=slurm/logs/stage_%A_%a.out
set -uo pipefail
source slurm/env.sh
export JAX_PLATFORMS=cpu

YEAR=$((2010 + SLURM_ARRAY_TASK_ID))   # 0-4 -> 2010-2014 (training), 5 -> 2015 (validation)
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && [ ! -d data/phase5/era5_1deg/statics.zarr ]; then
  python scripts/50_stage_data.py --statics
fi
python scripts/50_stage_data.py --year "$YEAR"
