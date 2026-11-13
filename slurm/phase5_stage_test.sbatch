#!/bin/bash
# Phase 5, step 0: stage ONE month (January 2015) and check it, before the full download.
# Shows how long a month takes (-> how long staging will take) and whether the fluxes
# have the right units and signs. The month is kept: the full staging job skips it.
#
#   sbatch slurm/phase5_stage_test.sbatch
#   cat slurm/logs/stagetest_<JOBID>.out
#
# If the cluster refuses a job without a GPU, add the line  #SBATCH --gres=gpu:1
#SBATCH --job-name=stagetest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/stagetest_%j.out
#SBATCH --error=slurm/logs/stagetest_%j.out
set -uo pipefail
source slurm/env.sh
export JAX_PLATFORMS=cpu

echo "== 1. terrain height and land-sea mask"; python scripts/50_stage_data.py --statics
echo "== 2. January 2015";                      python scripts/50_stage_data.py --year 2015 --months 1
echo "== 3. check";                             python scripts/52_check_staged.py --month 2015-01
du -sh data/phase5/era5_1deg/2015-01.zarr
echo "== finished $(date '+%F %T')"
