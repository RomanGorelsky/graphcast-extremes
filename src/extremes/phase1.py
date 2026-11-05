"""Shared settings of Phase 1 (record thresholds and 2020 records)."""
import json

import xarray

from . import model, paths, records

OUT = paths.RESULTS_DIR / "phase1"
YEARLY = OUT / "yearly"                   # monthly extremes of each year (resumable)
REF_START, REF_END = 1979, 2015           # GraphCast_small's training years


def thresholds_path(start: int = REF_START, end: int = REF_END):
    return OUT / f"thresholds_{start}-{end}_1deg.nc"


def chosen_method() -> str:
    f = OUT / "regrid_check.json"
    if not f.exists():
        raise FileNotFoundError("run scripts/10_check_regridding.py first")
    return json.loads(f.read_text())["chosen_method"]


def land_mask() -> tuple[xarray.DataArray, xarray.DataArray]:
    """Land-sea mask of the model's own 1-deg data, and the analysis mask (land, > 60 S)."""
    ex, _ = model.load_example(min_steps=1)
    lsm = ex["land_sea_mask"]
    lsm = lsm.isel({d: 0 for d in lsm.dims if d not in ("lat", "lon")}, drop=True).sortby("lat").load()
    return lsm, records.analysis_mask(lsm).rename("analysis_mask")
