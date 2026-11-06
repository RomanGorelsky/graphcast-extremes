"""Weather regimes from daily 500 hPa geopotential anomalies (Phase 2).

Per region ('sector', see events.SECTORS), year-round, in the spirit of
Grams et al. (2017, Nature Climate Change) year-round regimes:
  1. anomalies: remove a smooth seasonal cycle (mean + 3 annual harmonics) and a
     linear trend, both fitted on the reference years 1979-2015 only (so 2020 is
     not used to define 'normal'; the trend keeps warming from looking like a regime);
  2. weight by sqrt(cos(latitude)) (equal-area), PCA on reference days;
  3. K-means on the leading PCs, k chosen by the silhouette score;
  4. every day (incl. 2020) gets the regime of its nearest centroid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .events import SECTORS

REF_START, REF_END = "1979-01-01", "2015-12-31"


def _design(time: pd.DatetimeIndex, n_harm: int = 3) -> np.ndarray:
    doy = time.day_of_year.values + time.hour.values / 24
    years = (time - pd.Timestamp("1997-01-01")).days.values / 365.25
    cols = [np.ones(len(time)), years]
    for k in range(1, n_harm + 1):
        cols += [np.cos(2 * np.pi * k * doy / 365.25), np.sin(2 * np.pi * k * doy / 365.25)]
    return np.column_stack(cols)


def anomalies(z: xarray.DataArray) -> xarray.DataArray:
    """z(time, lat, lon) minus seasonal cycle + linear trend fitted on the reference years."""
    t = pd.DatetimeIndex(z.time.values)
    X = _design(t)
    ref = (t >= REF_START) & (t <= REF_END)
    Y = z.values.reshape(len(t), -1).astype(np.float64)
    beta, *_ = np.linalg.lstsq(X[ref], Y[ref], rcond=None)
    anom = (Y - X @ beta).reshape(z.shape).astype(np.float32)
    return z.copy(data=anom).rename("z500_anomaly")


def sector_box(a: xarray.DataArray, sector: str) -> xarray.DataArray:
    box = SECTORS[sector]
    lo, hi = box["lon"]
    lon_ok = (a.lon >= lo) & (a.lon <= hi) if lo <= hi else (a.lon >= lo) | (a.lon <= hi)
    lat_ok = (a.lat >= box["lat"][0]) & (a.lat <= box["lat"][1])
    return a.isel(lon=np.nonzero(lon_ok.values)[0], lat=np.nonzero(lat_ok.values)[0])


def fit_sector(anom: xarray.DataArray, sector: str, n_pcs: int = 10, k_range=range(3, 9),
               seed: int = 0) -> dict:
    """PCA + K-means for one sector. Returns labels for all days and diagnostics."""
    a = sector_box(anom, sector)
    w = np.sqrt(np.clip(np.cos(np.deg2rad(a.lat.values)), 0, None))[:, None]
    X = (a.values * w[None]).reshape(a.sizes["time"], -1)
    t = pd.DatetimeIndex(a.time.values)
    ref = (t >= REF_START) & (t <= REF_END)
    pca = PCA(n_components=n_pcs, random_state=seed).fit(X[ref])
    pcs = pca.transform(X)
    rng = np.random.default_rng(seed)
    sample = rng.choice(np.nonzero(ref)[0], size=min(4000, ref.sum()), replace=False)
    sil = {}
    models = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(pcs[ref])
        models[k] = km
        sil[k] = float(silhouette_score(pcs[sample], km.predict(pcs[sample])))
    k_best = max(sil, key=sil.get)
    km = models[k_best]
    labels = km.predict(pcs)
    freq = np.bincount(labels[ref], minlength=k_best) / ref.sum()
    # composite anomaly map of each regime (reference days), for plotting
    comp = xarray.concat([a.isel(time=np.nonzero(ref & (labels == r))[0]).mean("time")
                          for r in range(k_best)], dim=pd.Index(range(k_best), name="regime"))
    return {"sector": sector, "k": k_best, "silhouette": sil,
            "explained_variance": pca.explained_variance_ratio_.round(4).tolist(),
            "frequency": freq.round(4).tolist(),
            "labels": pd.Series(labels, index=t, name=sector), "composites": comp}
