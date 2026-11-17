#!/bin/bash
# Phase 7, prospect 3: R5 = the R1 control trained with the sharpness-preserving loss
# (finetune.amse_per_level) instead of plain squared error. Same data, steps and
# learning rate as R1; it starts from R1's weights after the flux warm-up (stage 1,
# which trains only the flux outputs), so R5 and R1 differ ONLY in the loss of the
# 2-, 4- and 8-step stages. One GPU; every stage in a fresh process (as Phase 5).
# 12 h limit: if stopped, submit again (it resumes).
#
#   sbatch slurm/phase7_train_r5.sbatch
#   tail -n 2 slurm/logs/p7r5_*.out
#SBATCH --job-name=p7r5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/p7r5_%j.out
#SBATCH --error=slurm/logs/p7r5_%j.out
set -uo pipefail
source slurm/env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
INIT=results/phase5/runs/R1/params_stage1.pkl
[ -f "$INIT" ] || { echo "missing $INIT (R1's weights after the flux warm-up)"; exit 1; }
for attempt in 1 2 3 4 5; do                      # one stage per process
  [ -f results/phase5/runs/R5/params_R5.ckpt ] && break
  python scripts/51_train.py --run R5 --init-params "$INIT" --skip-stages 1 --one-stage || exit 1
done
echo "== finished $(date '+%F %T')"
