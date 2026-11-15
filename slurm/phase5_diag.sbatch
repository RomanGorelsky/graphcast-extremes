#!/bin/bash
# Phase 5 diagnosis: why does fine-tuning degrade the 5-day forecasts?
# Continue R1 from its end-of-stage-1 weights through stage 2 (1,000 steps, 2-step
# forecasts, learning rate 3e-6) exactly as round 3 did - except that the flux errors do
# NOT count (--flux-weight 0). Round 3's R1 reached 5-day core 4.42 (R0: 3.70) at this point.
#   ~3.7 -> the flux outputs pulled the shared layers away (fix: fluxes only in stage 1)
#   ~4.4 -> the cause is elsewhere
#
#   sbatch slurm/phase5_diag.sbatch
#   grep -h "forecasts\|validation loss" $(ls -t slurm/logs/p5diag_*.out | head -1)
#SBATCH --job-name=p5diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/p5diag_%j.out
#SBATCH --error=slurm/logs/p5diag_%j.out
set -uo pipefail
source slurm/env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

python scripts/51_train.py --run R1 --init-params results/phase5/runs/R1/params_stage1.pkl \
  --skip-stages 1 --only-stages 1 --flux-weight 0 --out results/phase5/diag/R1_noflux
