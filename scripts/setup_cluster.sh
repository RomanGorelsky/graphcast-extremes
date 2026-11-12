#!/usr/bin/env bash
# Cluster setup (MBZUAI Shared-Cluster-v3, H100). Run ONCE on the login node,
# from the repository folder:
#
#     cd ~/graphcast-extremes
#     bash scripts/setup_cluster.sh
#
# What it does (all light work, allowed on the login node):
#   1. finds Python 3.12 - the system one if present, otherwise a private copy
#      installed by 'uv' (a small Python installer; no admin rights needed)
#   2. creates the environment ~/venvs/extremes
#   3. installs the pinned packages + the GPU (CUDA 12) build of JAX 0.7.2
#   4. downloads the model, statistics and example data (~0.4 GB) to ./data
# It does NOT use the GPU - the GPU check runs as a job: sbatch slurm/phase0_check.sbatch
# Takes ~10 minutes and ~5 GB in your home folder. Safe to run again.
set -euo pipefail

VENV="$HOME/venvs/extremes"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
# Your home folder is copied to S3 automatically; keep download caches out of it.
export UV_NO_CACHE=1 PIP_NO_CACHE_DIR=1

echo "== 1. Python 3.12"
PY=""
for c in python3.12 python3.13; do
  if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
done
if [ -z "$PY" ]; then
  if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "   no Python 3.12 on the system - installing uv into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh \
      || python3 -m pip install --user --quiet uv
  fi
  export PATH="$HOME/.local/bin:$PATH"
  uv python install 3.12
  PY="$(uv python find 3.12)"
fi
echo "   using $PY ($("$PY" --version))"

echo "== 2. environment $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  mkdir -p "$(dirname "$VENV")"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip

echo "== 3. packages (the GPU build of JAX is the large part)"
python -m pip install --quiet -r requirements.txt
python -m pip install --quiet "jax[cuda12]==0.7.2"
python -m pip install --quiet -e .

echo "== 4. model + statistics + example data"
python scripts/00_download_assets.py

echo
python - <<'EOF'
import sys, jax, xarray, weathernext, extremes
print("python  ", sys.version.split()[0])
print("jax     ", jax.__version__, "(the login node has no GPU; the job checks the GPU)")
print("xarray  ", xarray.__version__)
print("extremes", extremes.__version__)
EOF
mkdir -p slurm/logs
echo
echo "Setup done. Next:  sbatch slurm/phase0_check.sbatch   then   squeue --me"
