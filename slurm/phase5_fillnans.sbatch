#!/bin/bash
# Phase 5 repair: fill the isolated blank cells, check that none are left, and recompute
# the flux statistics. Run AFTER all staging jobs have finished.
#   sbatch slurm/phase5_fillnans.sbatch
#   cat $(ls -t slurm/logs/fillnans_*.out | head -1) | grep -v "checked\|warn"
#SBATCH --job-name=fillnans
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/fillnans_%j.out
#SBATCH --error=slurm/logs/fillnans_%j.out
set -uo pipefail
source slurm/env.sh
export JAX_PLATFORMS=cpu

echo "== 1. fill";            python scripts/54_fill_nans.py
echo "== 2. scan again";      python scripts/53_find_nans.py
echo "== 3. flux statistics"; rm -f results/phase5/flux_stats.nc
python scripts/50_stage_data.py --stats --years 2010 2011 2012 2013 2014
echo "== finished $(date '+%F %T')"
