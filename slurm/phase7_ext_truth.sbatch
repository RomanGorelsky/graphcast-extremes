#!/bin/bash
# Phase 7: ERA5 truth for the extended set's hindcasts (42_truth_cache.py). CPU only,
# lots of memory; resumable per event. Needs the hindcast plan (see phase7_ext_hindcasts).
#
#   sbatch slurm/phase7_ext_truth.sbatch
#
# If the cluster refuses a job without a GPU, add the line  #SBATCH --gres=gpu:1
#SBATCH --job-name=p7truth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/p7truth_%j.out
#SBATCH --error=slurm/logs/p7truth_%j.out
set -uo pipefail
source slurm/env.sh
export JAX_PLATFORMS=cpu
export EXTREMES_SET=ext
python scripts/42_truth_cache.py
echo "== finished $(date '+%F %T')"
