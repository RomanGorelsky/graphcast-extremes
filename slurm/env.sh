# Sourced by every job in this folder: activates the environment made by
# scripts/setup_cluster.sh and sets where data and results go.
# Results stay inside the repository folder in your home, which the cluster copies
# to S3 automatically - from the Mac they can be fetched with 'aws s3 sync' (README).
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
# shellcheck disable=SC1091
source "$HOME/venvs/extremes/bin/activate"
export EXTREMES_DATA="$REPO/data"
export EXTREMES_RESULTS="$REPO/results"
export PYTHONUNBUFFERED=1
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%F %T'), repo $REPO"
