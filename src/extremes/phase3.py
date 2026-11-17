"""Shared settings and helpers of Phase 3 (baseline forecasts of the case studies)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray

from . import paths

OUT = paths.set_dir("phase3")                   # results/phase3, or results/sets/<SET>/phase3
FORECASTS = OUT / "forecasts"
N_STEPS = 40                                    # 10 days of 6-h steps
LEADS_DAYS = (1, 3, 5, 7)
VERIFY_VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
               "mean_sea_level_pressure"]


def case_studies_file() -> str:
    return "case_studies.csv" if not paths.SET else f"case_studies_{paths.SET}.csv"


def case_studies() -> pd.DataFrame:
    """results/phase2/case_studies.csv (or case_studies_<SET>.csv when EXTREMES_SET is set)
    - from the results folder, else from the repo copy.

    Every table gets the columns `year` and `source_event_id` (the event's id in
    events_<year>.csv); for the original 14 case studies these are 2020 and event_id."""
    name = case_studies_file()
    for base in (paths.RESULTS_DIR, paths.REPO_ROOT / "results"):
        f = base / "phase2" / name
        if f.exists():
            cs = pd.read_csv(f, parse_dates=["peak_time", "start", "end"] +
                             [f"init_minus_{d}d" for d in LEADS_DAYS])
            if "year" not in cs:
                cs["year"] = 2020
            if "source_event_id" not in cs:
                cs["source_event_id"] = cs["event_id"]
            return cs
    raise FileNotFoundError(f"results/phase2/{name} not found (Phase 2 script 24, or Phase 7 script 70)")


def event_tables(years) -> dict:
    """{year: (events_<year>.csv, records_<year>_labeled.csv.gz)} from results/phase2."""
    p2 = paths.RESULTS_DIR / "phase2"
    return {int(y): (pd.read_csv(p2 / f"events_{int(y)}.csv"),
                     pd.read_csv(p2 / f"records_{int(y)}_labeled.csv.gz", parse_dates=["time"]))
            for y in sorted(set(int(y) for y in years))}


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
