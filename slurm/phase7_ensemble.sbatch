#!/bin/bash
# Phase 7, prospect 4: perturbed-start ensembles of GraphCast_small (R0) for the 14 case
# studies, 20 members + the R0 control (scripts/72_ensemble_forecasts.py). One GPU,
# ~3-4 h; resumable. ALPHA = size of the perturbation (default 0.1); for a second size:
#   sbatch --export=ALL,ALPHA=0.2 slurm/phase7_ensemble.sbatch
#
#   sbatch slurm/phase7_ensemble.sbatch
#   tail -n 2 slurm/logs/p7ens_*.out
#SBATCH --job-name=p7ens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/p7ens_%j.out
#SBATCH --error=slurm/logs/p7ens_%j.out
set -uo pipefail
source slurm/env.sh
python scripts/72_ensemble_forecasts.py --members 20 --alpha "${ALPHA:-0.1}"
echo "== finished $(date '+%F %T')"
