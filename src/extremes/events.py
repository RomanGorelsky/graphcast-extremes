"""From record cells to events (Phase 2).

1. DBSCAN groups record cells that are close in space AND time into events.
   Each record (lat, lon, time) is embedded in a Euclidean space where one unit
   is `space_km` in distance or `time_h` in time:
       (x, y, z) = Earth-surface chord coordinates / space_km      (no longitude seam)
       t         = hours since start / time_h
   so DBSCAN's eps = 1 means "about space_km apart, or time_h apart".
2. `event_features` summarises each event (size, duration, intensity, centre, ...).
3. `isolation_flags` marks unusual events with an Isolation Forest, repeated over
   many seeds for stability; 'record-shattering' = consistently unusual AND
   above-median peak exceedance (IF also flags unusually *small* events).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

EARTH_R_KM = 6371.0
CELL_KM2_AT_EQUATOR = (2 * np.pi * EARTH_R_KM / 360) ** 2      # one 1x1 deg cell, ~12,364 km2

# Regions for the weather-regime analysis (lon in 0..360). An event belongs to the
# first region containing its centre; events elsewhere get sector "other".
SECTORS = {
    "north_america": dict(lat=(20, 80), lon=(190, 310)),
    "euro_atlantic": dict(lat=(30, 80), lon=(300, 60)),       # wraps through 0 deg
    "asia_siberia": dict(lat=(30, 85), lon=(60, 180)),
    "south_america": dict(lat=(-55, 10), lon=(275, 330)),
}


def embed(df: pd.DataFrame, space_km: float, time_h: float) -> np.ndarray:
    lat, lon = np.deg2rad(df.lat.values), np.deg2rad(df.lon.values)
    xyz = EARTH_R_KM * np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)
    hours = (pd.to_datetime(df.time) - pd.Timestamp("2000-01-01")).dt.total_seconds().values / 3600
    return np.column_stack([xyz / space_km, hours / time_h])


def k_distance(df: pd.DataFrame, k: int, space_km: float, time_h: float) -> np.ndarray:
    """Sorted distance to the k-th neighbour (for choosing eps: look for the 'knee')."""
    X = embed(df, space_km, time_h)
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return np.sort(d[:, -1])


def cluster_records(df: pd.DataFrame, space_km: float = 250, time_h: float = 12,
                    eps: float = 1.0, min_samples: int = 5) -> np.ndarray:
    """DBSCAN labels (-1 = noise) for the records of ONE kind."""
    if len(df) == 0:
        return np.array([], dtype=int)
    X = embed(df, space_km, time_h)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def sector_of(lat: float, lon: float) -> str:
    for name, box in SECTORS.items():
        (a, b), (c, d) = box["lat"], box["lon"]
        in_lon = (c <= lon <= d) if c <= d else (lon >= c or lon <= d)
        if a <= lat <= b and in_lon:
            return name
    return "other"


def _circular_mean_lon(lon: np.ndarray, w: np.ndarray) -> float:
    r = np.deg2rad(lon)
    return float(np.rad2deg(np.arctan2((w * np.sin(r)).sum(), (w * np.cos(r)).sum())) % 360)


def event_features(df: pd.DataFrame, land_sea_mask=None) -> pd.DataFrame:
    """One row per event from labelled records (columns kind, event, time, lat, lon, exceedance).

    `land_sea_mask` (optional xarray, 1-deg) adds `coastal_fraction`: the share of
    the event's cells that are only partly land (0.5 < mask < 0.9), a warning sign
    for coastline / sea-ice artefacts.
    """
    rows = []
    for (kind, ev), g in df[df.event >= 0].groupby(["kind", "event"]):
        cells = g[["lat", "lon"]].drop_duplicates()
        area = float((CELL_KM2_AT_EQUATOR * np.cos(np.deg2rad(cells.lat))).sum())
        t = pd.to_datetime(g.time)
        peak = g.loc[g.exceedance.idxmax()]
        w = g.exceedance.values
        lat_c = float(np.average(g.lat, weights=w))
        lon_c = _circular_mean_lon(g.lon.values, w)
        row = dict(kind=kind, event=int(ev), n_records=len(g), n_cells=len(cells), area_km2=area,
                   start=t.min(), end=t.max(), duration_h=(t.max() - t.min()).total_seconds() / 3600 + 6,
                   peak_time=pd.Timestamp(peak.time), peak_lat=float(peak.lat), peak_lon=float(peak.lon),
                   peak_exceedance=float(g.exceedance.max()), mean_exceedance=float(g.exceedance.mean()),
                   total_exceedance=float(g.exceedance.sum()), centre_lat=lat_c, centre_lon=lon_c,
                   month=int(pd.Timestamp(peak.time).month), sector=sector_of(lat_c, lon_c))
        if land_sea_mask is not None:
            m = land_sea_mask.sel(lat=xarray.DataArray(cells.lat.values, dims="cell"),
                                  lon=xarray.DataArray(cells.lon.values, dims="cell"),
                                  method="nearest").values
            row["coastal_fraction"] = float(((m > 0.5) & (m < 0.9)).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["kind", "total_exceedance"], ascending=[True, False]).reset_index(drop=True)
        out["event_id"] = [f"{k}-{i:03d}" for k, i in zip(out.kind, out.groupby("kind").cumcount() + 1)]
    return out


IF_FEATURES = ["log_area", "log_duration", "peak_exceedance", "mean_exceedance", "log_records"]


def _if_matrix(ev: pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame({
        "log_area": np.log10(ev.area_km2), "log_duration": np.log10(ev.duration_h),
        "peak_exceedance": ev.peak_exceedance, "mean_exceedance": ev.mean_exceedance,
        "log_records": np.log10(ev.n_records)})[IF_FEATURES]
    return StandardScaler().fit_transform(X.values)


def isolation_flags(ev: pd.DataFrame, contamination: float = 0.1, n_seeds: int = 20,
                    stable: float = 0.8) -> pd.DataFrame:
    """Adds if_score (mean anomaly score, higher = more unusual), if_flag_rate,
    unusual (flagged in >= `stable` of the seeds) and record_shattering, per kind."""
    out = []
    for kind, g in ev.groupby("kind"):
        g = g.copy()
        if len(g) < 5:                                   # too few events to judge
            g["if_score"], g["if_flag_rate"] = np.nan, 0.0
        else:
            X = _if_matrix(g)
            scores, flags = [], []
            for seed in range(n_seeds):
                f = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(X)
                scores.append(-f.score_samples(X))
                flags.append(f.predict(X) == -1)
            g["if_score"] = np.mean(scores, 0)
            g["if_flag_rate"] = np.mean(flags, 0)
        g["unusual"] = g.if_flag_rate >= stable
        g["record_shattering"] = g.unusual & (g.peak_exceedance > g.peak_exceedance.median())
        out.append(g)
    return pd.concat(out).sort_index()
