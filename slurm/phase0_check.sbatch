#!/bin/bash
# Phase 0 on the cluster: does everything work on one H100?
#   sbatch slurm/phase0_check.sbatch          (from ~/graphcast-extremes)
#   squeue --me                               (wait until it disappears)
#   cat slurm/logs/phase0_<JOBID>.out
#SBATCH --job-name=phase0-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/phase0_%j.out
#SBATCH --error=slurm/logs/phase0_%j.out
set -uo pipefail
source slurm/env.sh

echo "== 1. GPU";            nvidia-smi --query-gpu=name,memory.total --format=csv
echo "== 2. JAX sees it";    python -c "import jax; print(jax.devices())"
echo "== 3. offline tests";  python -m pytest -q tests
echo "== 4. ERA5 reachable from this machine"
python - <<'EOF'
from extremes import era5
ds = era5.open_store(era5.WB2_ERA5)
v = float(ds["2m_temperature"].sel(time="2020-07-01T00").isel(lat=0, lon=0).values)
print("WeatherBench2 ERA5 OK, one value:", round(v, 2), "K")
EOF
echo "== 5. smoke test (must end with PASSED)"; python scripts/01_smoke_test.py
echo "== 6. speed";          python scripts/02_benchmark.py --label cluster-h100
echo "== finished $(date '+%F %T')"
