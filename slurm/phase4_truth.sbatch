#!/bin/bash
# Phase 4: ERA5 truth for the hindcasts (42_truth_cache.py). Needs no GPU, but a lot
# of memory (it crashed Colab's 12.7 GB), so it runs as a job, not on the login node.
# Resumable per event: if it stops, submit again.
#
#   sbatch slurm/phase4_truth.sbatch
#   tail -f slurm/logs/truth_<JOBID>.out     (Ctrl+C stops watching, not the job)
#
# If the cluster refuses a job without a GPU, add the line  #SBATCH --gres=gpu:1
#SBATCH --job-name=truth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/truth_%j.out
#SBATCH --error=slurm/logs/truth_%j.out
set -uo pipefail
source slurm/env.sh
export JAX_PLATFORMS=cpu

if [ ! -f results/phase4/hindcast_plan.csv ]; then
  echo "results/phase4/hindcast_plan.csv missing - run: python scripts/40_hindcast_plan.py"
  exit 1
fi
python scripts/42_truth_cache.py
