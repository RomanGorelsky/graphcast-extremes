#!/bin/bash
# Phase 5 diagnosis: find missing values (NaN) in the staged data.
#   sbatch slurm/phase5_nanscan.sbatch
#   cat $(ls -t slurm/logs/nanscan_*.out | head -1) | grep -v "checked\|warn"
#SBATCH --job-name=nanscan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/nanscan_%j.out
#SBATCH --error=slurm/logs/nanscan_%j.out
set -uo pipefail
source slurm/env.sh
export JAX_PLATFORMS=cpu
python scripts/53_find_nans.py
