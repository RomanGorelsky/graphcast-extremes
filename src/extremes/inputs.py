"""Building GraphCast_small inputs from ERA5 (Phase 3).

`build_batch` reproduces the layout of the authors' example files: dims
(batch=1, time, level, lat, lon), `time` = timedelta from the first input time,
`datetime` = (batch, time) coordinate. It holds

  * the two input times (init - 6 h, init) for all surface and pressure-level
    variables (incl. 6-h precipitation, which GraphCast_small uses as input),
  * placeholders (NaN) for those variables at the forecast times - the model only
    needs their shape,
  * top-of-atmosphere solar radiation at ALL times (a known-in-advance forcing),
  * the static fields (terrain height, land-sea mask).

ERA5 comes from the 0.25-deg WeatherBench2 store, put on the 1-deg grid by
subsampling - the processing that matches the model's training data exactly
(Phase 1). `scripts/30_check_inputs.py` verifies the whole batch against the
authors' example before any forecast is run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray

from weathernext.weathernext1_graph import graphcast

from . import era5

SURFACE = list(graphcast.TARGET_SURFACE_VARS)          # incl. total_precipitation_6hr
ATMOS = list(graphcast.TARGET_ATMOSPHERIC_VARS)
LEVELS = list(graphcast.TASK_13.pressure_levels)
TISR = "toa_incident_solar_radiation"
STATIC = ["geopotential_at_surface", "land_sea_mask"]
STEP = pd.Timedelta(hours=6)


def times_for(init, n_steps: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    init = pd.Timestamp(init)
    inputs = pd.DatetimeIndex([init - STEP, init])
    targets = pd.DatetimeIndex([init + STEP * k for k in range(1, n_steps + 1)])
    return inputs, targets


def _static(ds: xarray.Dataset, var: str, method: str) -> xarray.DataArray:
    da = ds[var]
    da = da.isel({d: 0 for d in da.dims if d not in ("lat", "lon")}, drop=True)
    da = da.astype(np.float32).transpose("lat", "lon").load()
    return da if da.sizes["lat"] == 181 else era5.regrid(da, method)


def build_batch(ds: xarray.Dataset, init, n_steps: int, method: str = "subsample",
                statics: dict | None = None, io_threads: int = 32) -> xarray.Dataset:
    """Example-format dataset for a forecast starting at `init` with `n_steps` 6-h steps.

    `statics` can pass pre-loaded static fields ({name: DataArray(lat, lon)}) to avoid
    re-reading them for every forecast.
    """
    t_in, t_out = times_for(init, n_steps)
    t_all = t_in.append(t_out)

    data = {}
    for var in SURFACE:
        x = era5.load_1deg(ds, var, t_in.values, method, io_threads)            # (time, lat, lon)
        pad = xarray.full_like(x.isel(time=[0] * len(t_out)), np.nan)
        data[var] = xarray.concat([x, pad.assign_coords(time=t_out.values)], "time")
    for var in ATMOS:
        x = era5.load_1deg(ds.sel(level=LEVELS), var, t_in.values, method, io_threads)  # (time, level, lat, lon)
        pad = xarray.full_like(x.isel(time=[0] * len(t_out)), np.nan)
        data[var] = xarray.concat([x, pad.assign_coords(time=t_out.values)], "time")
    if TISR in ds:
        data[TISR] = era5.load_1deg(ds, TISR, t_all.values, method, io_threads)
    statics = statics or {v: _static(ds, v, method) for v in STATIC}

    out = xarray.Dataset()
    for var, da in data.items():
        da = da.transpose("time", ...).expand_dims(batch=1)
        out[var] = da.drop_vars([c for c in da.coords if c not in ("time", "level", "lat", "lon")])
    for var, da in statics.items():
        out[var] = da.drop_vars([c for c in da.coords if c not in ("lat", "lon")])
    out = out.assign_coords(time=(t_all - t_all[0]).values,
                            datetime=(("batch", "time"), t_all.values[None]))
    out = out.assign_coords(level=np.asarray(LEVELS))
    return out.transpose("batch", "time", "level", "lat", "lon")


def load_statics(ds: xarray.Dataset, method: str = "subsample") -> dict:
    return {v: _static(ds, v, method) for v in STATIC}
