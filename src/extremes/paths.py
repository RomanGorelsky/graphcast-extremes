"""Project-wide locations and constants.

Everything that differs between the Mac, Colab and the cluster is read from
environment variables, so the same code runs in all three places.

  EXTREMES_DATA   where large-ish local files live (default: <repo>/data)
  EXTREMES_RESULTS where small result files live (default: <repo>/results)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("EXTREMES_DATA", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("EXTREMES_RESULTS", REPO_ROOT / "results"))
ASSETS_DIR = DATA_DIR / "assets"          # model checkpoint, normalisation stats, example data

# Public bucket published by the GraphCast authors (read anonymously).
GCS_BUCKET = "dm_graphcast"
GCS_PREFIX = "graphcast/"

# The model used in this study (matched against the bucket listing, see model.py).
MODEL_KEY = "GraphCast_small"
MODEL_RESOLUTION = 1.0
MODEL_LEVELS = 13
STATS_FILES = ("diffs_stddev_by_level.nc", "mean_by_level.nc", "stddev_by_level.nc")
