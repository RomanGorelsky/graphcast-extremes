#!/bin/bash
# Phase 7, prospect 2, step 1 (CPU, ~4-6 h): records and events of 2021 and 2022 with
# exactly the Phase 1-2 methods, then the extended case-study list.
#   z500 1979-2022 (regimes) -> records 2021, 2022 -> DBSCAN -> Isolation Forest ->
#   K-means regimes -> results/phase2/case_studies_ext.csv
# Needs results/phase1/thresholds_1979-2015_1deg.nc and results/phase2/records_2020_labeled.csv.gz
# on the cluster (git add -f them on the Mac, see README). Resumable: submit again if stopped.
#
#   sbatch slurm/phase7_events.sbatch
#   tail -f slurm/logs/p7events_<JOBID>.out
#
# If the cluster refuses a job without a GPU, add the line  #SBATCH --gres=gpu:1
#SBATCH --job-name=p7events
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/p7events_%j.out
#SBATCH --error=slurm/logs/p7events_%j.out
set -uo pipefail
source slurm/env.sh
export JAX_PLATFORMS=cpu
for f in results/phase1/thresholds_1979-2015_1deg.nc results/phase2/records_2020_labeled.csv.gz; do
  [ -f "$f" ] || { echo "missing $f - add it to git on the Mac (README, Phase 7) and git pull"; exit 1; }
done
echo "== z500 1979-2022";  python scripts/20_fetch_z500.py --years 1979 2022 || exit 1
for Y in 2021 2022; do
  echo "== records $Y";    python scripts/12_find_records.py --year "$Y" || exit 1
  echo "== DBSCAN $Y";     python scripts/21_detect_events.py --year "$Y" || exit 1
  echo "== IForest $Y";    python scripts/22_outliers.py --year "$Y" || exit 1
  echo "== regimes $Y";    python scripts/23_regimes.py --year "$Y" --z500 z500_daily_1979-2022_2p8deg.nc || exit 1
done
echo "== extended set";    python scripts/70_select_extended.py
echo "== finished $(date '+%F %T')"
