#!/bin/bash
# Phase 5, step 1: stage the training data (2010-2014) and the validation year (2015).
# 3 CPU jobs (cluster limit: 3 running, 5 submitted, 12 h each), two years each, one
# after the other (~2.5 h per year). Resumable per month: if a job stops, submit again.
#
#   sbatch slurm/phase5_stage.sbatch
#   squeue --me
#   grep -h "times in" slurm/logs/stage_*.out | sort | tail -n 8
#
# If the cluster refuses a job without a GPU, add the line  #SBATCH --gres=gpu:1
#SBATCH --job-name=stage
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/stage_%A_%a.out
#SBATCH --error=slurm/logs/stage_%A_%a.out
set -uo pipefail
source slurm/env.sh
export JAX_PLATFORMS=cpu

# task 0 -> 2010, 2011 | task 1 -> 2012, 2013 | task 2 -> 2014, 2015 (validation)
FIRST=$((2010 + 2 * SLURM_ARRAY_TASK_ID))
if [ ! -d data/phase5/era5_1deg/statics.zarr ]; then
  python scripts/50_stage_data.py --statics
fi
for YEAR in "$FIRST" "$((FIRST + 1))"; do
  python scripts/50_stage_data.py --year "$YEAR"
done
