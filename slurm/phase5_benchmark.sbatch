#!/bin/bash
# Phase 5, step 2: after staging - flux statistics, R0's validation loss, and a short
# R4 training run to measure speed and memory (sets the length of the real runs).
#
#   sbatch slurm/phase5_benchmark.sbatch
#   cat slurm/logs/p5bench_<JOBID>.out
#SBATCH --job-name=p5bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/p5bench_%j.out
#SBATCH --error=slurm/logs/p5bench_%j.out
set -uo pipefail
source slurm/env.sh

echo "== 1. flux statistics (training years)"
[ -f results/phase5/flux_stats.nc ] || python scripts/50_stage_data.py --stats --years 2010 2011 2012 2013 2014
echo "== 2. R0 on the validation year";  python scripts/51_train.py --baseline
echo "== 3. R4, 100 training steps (benchmark only, separate folder)"
rm -rf results/phase5/benchmark
python scripts/51_train.py --run R4 --max-steps 100 --eval-every 50 --out results/phase5/benchmark
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "== finished $(date '+%F %T')"
