#!/usr/bin/env bash
# Phase 0, step 2: create the Mac environment.
#
# Run from the repository folder in Terminal:
#     cd ~/Documents/GitHub/graphcast-extremes
#     bash scripts/setup_mac.sh
#
# Creates a conda environment "extremes" with Python 3.12 (the GraphCast
# library needs >= 3.12), installs the pinned dependencies and this project
# in editable mode. Takes ~5-10 minutes and ~1.5 GB of disk.
set -euo pipefail

ENV_NAME=extremes

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Open a new Terminal after installing Anaconda/Miniconda, or run:"
  echo "  source ~/anaconda3/bin/activate   (or ~/miniconda3/bin/activate)"
  exit 1
fi

# Make 'conda activate' work inside this script.
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' already exists - reusing it."
else
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

# Register the environment as a Jupyter kernel (for notebooks in VS Code / Jupyter).
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo
python - <<'EOF'
import sys, jax, xarray, weathernext, extremes
print("python     ", sys.version.split()[0])
print("jax        ", jax.__version__, jax.devices())
print("xarray     ", xarray.__version__)
print("extremes   ", extremes.__version__)
EOF
echo
echo "Done. In every new Terminal:  conda activate $ENV_NAME"
df -h . | tail -1 | awk '{print "Free disk space here: " $4}'
