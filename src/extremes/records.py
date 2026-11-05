"""Record thresholds and record detection (following Zhang et al., Sci. Adv. 2026).

A record is defined per variable, per grid cell and per CALENDAR MONTH against a
reference period (here 1979-2015, GraphCast_small's training years), using the
6-hourly values at 00/06/12/18 UTC:

  heat  2 m temperature  above the highest value of that month in 1979-2015
  cold  2 m temperature  below the lowest  value of that month in 1979-2015
  wind  10 m wind speed  above the highest value of that month in 1979-2015

Exceedance = how far beyond the threshold (K or m/s, always positive).
Only land cells (land-sea mask > 0.5) north of 60 S are kept (no Antarctica).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray

KINDS = ("heat", "cold", "wind")
UNITS = {"heat": "K", "cold": "K", "wind": "m/s"}


def wind_speed(u: xarray.DataArray, v: xarray.DataArray) -> xarray.DataArray:
    return np.hypot(u, v).rename("10m_wind_speed")


def month_extremes(t2m: xarray.DataArray, wsp: xarray.DataArray) -> xarray.Dataset:
    """Extremes over the time axis of ONE month of data -> (lat, lon) fields."""
    return xarray.Dataset({
        "t2m_max": t2m.max("time"), "t2m_min": t2m.min("time"), "wind_max": wsp.max("time"),
        "n_times": xarray.DataArray(t2m.sizes["time"]),
    })


def stack_months(per_month: dict[int, xarray.Dataset]) -> xarray.Dataset:
    """{month: Dataset(lat, lon)} -> Dataset(month, lat, lon)."""
    months = sorted(per_month)
    return xarray.concat([per_month[m] for m in months], dim=pd.Index(months, name="month"))


def combine_years(yearly: list[xarray.Dataset]) -> xarray.Dataset:
    """Monthly extremes of several years -> thresholds (extreme over all years)."""
    all_years = xarray.concat(yearly, dim="year")
    out = xarray.Dataset({
        "t2m_max": all_years.t2m_max.max("year"),
        "t2m_min": all_years.t2m_min.min("year"),
        "wind_max": all_years.wind_max.max("year"),
        "n_times": all_years.n_times.sum("year"),
    })
    out["n_years"] = xarray.DataArray(len(yearly))
    return out


def analysis_mask(land_sea_mask: xarray.DataArray) -> xarray.DataArray:
    """True for land cells (> 50 % land) north of 60 S."""
    return (land_sea_mask > 0.5) & (land_sea_mask.lat > -60)


def find_records(t2m: xarray.DataArray, wsp: xarray.DataArray, thresholds: xarray.Dataset,
                 mask: xarray.DataArray) -> pd.DataFrame:
    """All record-breaking (time, cell) pairs in the given data (any months).

    Returns one row per record: kind, time, lat, lon, value, threshold, exceedance.
    """
    month = t2m["time"].dt.month
    th = thresholds.sel(month=month)                     # threshold of each time's month
    checks = {
        "heat": (t2m, th.t2m_max, t2m - th.t2m_max),
        "cold": (t2m, th.t2m_min, th.t2m_min - t2m),
        "wind": (wsp, th.wind_max, wsp - th.wind_max),
    }
    rows = []
    # mask on exactly the data's grid (same order of lat/lon)
    m = mask.reindex(lat=t2m["lat"], lon=t2m["lon"], method="nearest").values.astype(bool)
    for kind, (val, thr, exc) in checks.items():
        e = exc.transpose("time", "lat", "lon").values
        hit = (e > 0) & m[None]
        it, ilat, ilon = np.nonzero(hit)
        if it.size == 0:
            continue
        rows.append(pd.DataFrame({
            "kind": kind,
            "time": t2m["time"].values[it],
            "lat": t2m["lat"].values[ilat],
            "lon": t2m["lon"].values[ilon],
            "value": val.transpose("time", "lat", "lon").values[it, ilat, ilon],
            "threshold": thr.transpose("time", "lat", "lon").values[it, ilat, ilon],
            "exceedance": e[it, ilat, ilon],
        }))
    cols = ["kind", "time", "lat", "lon", "value", "threshold", "exceedance"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
