"""Shared settings and helpers of Phase 3 (baseline forecasts of the case studies)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray

from . import paths

OUT = paths.RESULTS_DIR / "phase3"
FORECASTS = OUT / "forecasts"
N_STEPS = 40                                    # 10 days of 6-h steps
LEADS_DAYS = (1, 3, 5, 7)
VERIFY_VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
               "mean_sea_level_pressure"]


def case_studies() -> pd.DataFrame:
    """results/phase2/case_studies.csv - from the results folder, else from the repo copy."""
    for base in (paths.RESULTS_DIR, paths.REPO_ROOT / "results"):
        f = base / "phase2" / "case_studies.csv"
        if f.exists():
            return pd.read_csv(f, parse_dates=["peak_time", "start", "end"] +
                               [f"init_minus_{d}d" for d in LEADS_DAYS])
    raise FileNotFoundError("results/phase2/case_studies.csv not found (Phase 2, script 24)")


def forecast_path(event_id: str, lead_days: int):
    return FORECASTS / f"{event_id}_minus{lead_days}d.nc"


def crop(da: xarray.DataArray, lat_min, lat_max, lon_min, lon_max) -> xarray.DataArray:
    """Cut a (…, lat, lon) field to a box; the longitude range may cross 0 deg."""
    lat_ok = (da.lat >= lat_min) & (da.lat <= lat_max)
    lon = da.lon.values
    lon_ok = (lon >= lon_min) & (lon <= lon_max) if lon_min <= lon_max else (lon >= lon_min) | (lon <= lon_max)
    return da.isel(lat=np.nonzero(lat_ok.values)[0], lon=np.nonzero(lon_ok)[0])


def box_of(case) -> tuple[float, float, float, float]:
    return case.box_lat_min, case.box_lat_max, case.box_lon_min, case.box_lon_max
