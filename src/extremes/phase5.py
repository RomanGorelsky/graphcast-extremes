"""Phase 5 locations and data sources.

Training data are staged once, month by month, to EXTREMES_DATA/phase5/era5_1deg/
(<YYYY-MM>.zarr, plus statics.zarr): every variable GraphCast_small reads or predicts,
plus the 7 flux variables, on the 1-degree grid (subsampled, as the model's own data).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray

from . import era5, paths, physics

OUT = paths.RESULTS_DIR / "phase5"
STAGED = paths.DATA_DIR / "phase5" / "era5_1deg"
FLUX_STATS = OUT / "flux_stats.nc"

# Google's analysis-ready ERA5 (hourly, 0.25 deg, every ERA5 variable), read anonymously.
ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# our name -> name in the ARCO store (hourly accumulations ending at the time stamp;
# summed over the 6 hours ending at each 6-hourly time, as total_precipitation_6hr is)
FLUX_SOURCE = {
    physics.EVAP: "evaporation",
    physics.TOA_SOLAR: "top_net_solar_radiation",
    physics.TOA_THERMAL: "top_net_thermal_radiation",
    physics.SFC_SOLAR: "surface_net_solar_radiation",
    physics.SFC_THERMAL: "surface_net_thermal_radiation",
    physics.SFC_SENSIBLE: "surface_sensible_heat_flux",
    physics.SFC_LATENT: "surface_latent_heat_flux",
}


def open_arco(concurrency: int = 32) -> xarray.Dataset:
    """The ARCO store, only the 7 flux variables, WITHOUT dask.

    The store holds 273 variables hourly since 1940; opened with dask (as the much
    smaller WeatherBench2 store is) the read plan alone needs > 64 GB. Without dask,
    xarray reads exactly the chunks a selection touches, zarr fetching them in parallel."""
    import zarr
    zarr.config.set({"async.concurrency": concurrency})
    ds = xarray.open_zarr(ARCO_ERA5, storage_options={"token": "anon"}, chunks=None)
    return era5.standardise(ds[list(FLUX_SOURCE.values())])


def month_path(year: int, month: int) -> Path:
    return STAGED / f"{year}-{month:02d}.zarr"


def run_dir(run: str) -> Path:
    return OUT / "runs" / run


def open_staged(years=None) -> xarray.Dataset:
    """All staged months (optionally only some years) as one lazy dataset + statics."""
    files = sorted(p for p in STAGED.glob("????-??.zarr")
                   if years is None or int(p.name[:4]) in set(years))
    if not files:
        raise FileNotFoundError(f"no staged months in {STAGED}; run scripts/50_stage_data.py")
    ds = xarray.concat([xarray.open_zarr(f, chunks={}) for f in files], "time")
    st = xarray.open_zarr(STAGED / "statics.zarr", chunks={})
    return ds.merge(st)


def flux_stats(ds: xarray.Dataset, every: int = 7) -> tuple[xarray.Dataset, xarray.Dataset]:
    """Mean and standard deviation of each flux variable (one number each), from every
    `every`-th time step - the normalisation GraphCast uses for target-only variables."""
    mean, std = {}, {}
    for v in physics.FLUX_VARS:                      # one variable at a time: little memory
        x = ds[v].isel(time=slice(None, None, every)).load().values.astype(np.float64)
        mean[v], std[v] = x.mean(), x.std()
    return (xarray.Dataset({v: ((), np.float32(m)) for v, m in mean.items()}),
            xarray.Dataset({v: ((), np.float32(s)) for v, s in std.items()}))


def fill_isolated_nans(a: np.ndarray) -> np.ndarray:
    """Replace NaN cells of a (lat, lon) field by the mean of their valid neighbours
    (3 x 3 box, longitude wraps around). The ARCO hourly fluxes have a few isolated
    blank cells (Phase 5 scan: 33 single cells in 6 years)."""
    out = np.array(a, dtype=np.float64, copy=True)
    for i, j in zip(*np.where(np.isnan(out))):
        rows = slice(max(i - 1, 0), min(i + 2, out.shape[0]))
        cols = [(j + d) % out.shape[1] for d in (-1, 0, 1)]
        box = np.asarray(a, np.float64)[rows][:, cols]
        out[i, j] = np.nanmean(box) if np.isfinite(box).any() else 0.0
    return out.astype(np.asarray(a).dtype)


def staged_years() -> list[int]:
    return sorted({int(p.name[:4]) for p in STAGED.glob("????-??.zarr")})


def hours_before(t: pd.DatetimeIndex, n: int = 6) -> np.ndarray:
    """For each 6-hourly time: the n hourly times ending at it (t-5h ... t)."""
    offs = pd.to_timedelta(np.arange(n - 1, -1, -1), unit="h")
    return (t.values[:, None] - offs.values[None, :]).ravel()
