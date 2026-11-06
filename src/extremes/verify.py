"""Verification of forecasts on record-breaking cells (Phase 3 onwards).

Sign convention ("towards the extreme" = s * value, s = +1 for heat and wind, -1 for cold):
  intensity_error   = s * (forecast - observed)       negative = forecast too mild (underestimate)
  obs_exceedance    = s * (observed - threshold)      > 0 for every record
  fc_exceedance     = s * (forecast - threshold)      > 0 if the forecast also breaks the record
  hit               = fc_exceedance > 0
Zhang et al. report the raw bias (forecast - observed); for cold records that is the
negative of intensity_error. Both are kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray

SIGN = {"heat": 1.0, "cold": -1.0, "wind": 1.0}
EXCEEDANCE_BINS = [0, 0.5, 1, 2, 4, np.inf]


def field(ds: xarray.Dataset, which: str, kind: str) -> xarray.DataArray:
    """'forecast' or 'truth' field relevant for a record kind (t2m, or 10 m wind speed)."""
    if kind in ("heat", "cold"):
        return ds[f"{which}_2m_temperature"]
    return np.hypot(ds[f"{which}_10m_u_component_of_wind"], ds[f"{which}_10m_v_component_of_wind"])


def record_errors(fc: xarray.Dataset, recs: pd.DataFrame, kind: str, init) -> pd.DataFrame:
    """One row per record cell-time of the event that lies inside the forecast (after init)."""
    s = SIGN[kind]
    f, o = field(fc, "forecast", kind), field(fc, "truth", kind)
    times = pd.DatetimeIndex(fc.time.values)
    r = recs[(recs.time > pd.Timestamp(init)) & recs.time.isin(times)].copy()
    r = r[r.lat.isin(fc.lat.values) & r.lon.isin(fc.lon.values)]
    if r.empty:
        return r.assign(forecast=[], observed=[], intensity_error=[], raw_bias=[], obs_exceedance=[],
                        fc_exceedance=[], hit=[], lead_h=[])
    sel = dict(time=xarray.DataArray(r.time.values, dims="r"), lat=xarray.DataArray(r.lat.values, dims="r"),
               lon=xarray.DataArray(r.lon.values, dims="r"))
    r["forecast"] = f.sel(**sel).values
    r["observed"] = o.sel(**sel).values
    r["raw_bias"] = r.forecast - r.observed
    r["intensity_error"] = s * r.raw_bias
    r["obs_exceedance"] = s * (r.observed - r.threshold)
    r["fc_exceedance"] = s * (r.forecast - r.threshold)
    r["hit"] = r.fc_exceedance > 0
    r["lead_h"] = (r.time - pd.Timestamp(init)).dt.total_seconds() / 3600
    r["truth_check"] = np.abs(r.observed - r.value)          # ~0: our truth = Phase 1 record value
    return r


def box_exceedances(fc: xarray.Dataset, thresholds: xarray.Dataset, mask: xarray.DataArray,
                    kind: str, t0, t1) -> dict:
    """Over the whole box and the event period [t0, t1] (forecast times only):
    counts of forecast / observed record exceedances, for precision (false alarms)."""
    s = SIGN[kind]
    win = fc.sel(time=slice(pd.Timestamp(t0), pd.Timestamp(t1)))
    if win.sizes["time"] == 0:
        return dict(n_fc_records=0, n_obs_records=0, n_both=0, precision=np.nan)
    th_var = {"heat": "t2m_max", "cold": "t2m_min", "wind": "wind_max"}[kind]
    th = thresholds[th_var].sel(month=win.time.dt.month).sel(lat=win.lat, lon=win.lon)
    m = mask.sel(lat=win.lat, lon=win.lon).astype(bool)
    fe = (s * (field(win, "forecast", kind) - th) > 0) & m
    oe = (s * (field(win, "truth", kind) - th) > 0) & m
    n_fc, n_obs, n_both = int(fe.sum()), int(oe.sum()), int((fe & oe).sum())
    return dict(n_fc_records=n_fc, n_obs_records=n_obs, n_both=n_both,
                precision=n_both / n_fc if n_fc else np.nan)


def case_summary(err: pd.DataFrame) -> dict:
    if err.empty:
        return dict(n_verified=0)
    return dict(
        n_verified=len(err),
        mean_intensity_error=float(err.intensity_error.mean()),
        rmse=float(np.sqrt((err.raw_bias ** 2).mean())),
        mean_obs_exceedance=float(err.obs_exceedance.mean()),
        mean_fc_exceedance=float(err.fc_exceedance.mean()),
        capture_ratio=float(err.fc_exceedance.mean() / err.obs_exceedance.mean()),
        hit_rate=float(err.hit.mean()),
    )


def by_exceedance(err: pd.DataFrame, by: list) -> pd.DataFrame:
    """Mean intensity error per observed-exceedance bin (the Zhang et al. Fig. 2 curve)."""
    e = err.copy()
    e["exc_bin"] = pd.cut(e.obs_exceedance, EXCEEDANCE_BINS, right=False)
    g = e.groupby(by + ["exc_bin"], observed=True).intensity_error
    return g.agg(["mean", "std", "count"]).reset_index()
