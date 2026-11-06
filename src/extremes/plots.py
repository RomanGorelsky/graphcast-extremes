"""Report figures that are drawn from saved results."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray

from .events import SECTORS

G = 9.80665


def _continuous_lon(lon: np.ndarray, lo: float) -> np.ndarray:
    """Longitudes of a box that may cross 0 deg, made increasing (e.g. 300..420)."""
    return np.where(lon >= lo, lon, lon + 360)


def _lon_label(x: float) -> str:
    x = ((x + 180) % 360) - 180
    return "0°" if abs(x) < 1e-6 else (f"{abs(x):.0f}°E" if x > 0 else f"{abs(x):.0f}°W")


def _lat_label(y: float) -> str:
    return "0°" if abs(y) < 1e-6 else (f"{abs(y):.0f}°N" if y > 0 else f"{abs(y):.0f}°S")


def _box(da: xarray.DataArray, sector: str) -> xarray.DataArray:
    """Cut a (lat, lon) field to a sector box, with increasing continuous longitudes."""
    box = SECTORS[sector]
    lo, hi = box["lon"]
    lon = da.lon.values
    keep = (lon >= lo) & (lon <= hi) if lo <= hi else (lon >= lo) | (lon <= hi)
    lat_ok = (da.lat >= box["lat"][0]) & (da.lat <= box["lat"][1])
    sub = da.isel(lon=np.nonzero(keep)[0], lat=np.nonzero(lat_ok.values)[0])
    sub = sub.assign_coords(lon=_continuous_lon(sub.lon.values, lo)).sortby("lon").sortby("lat")
    return sub


def regime_composites(composites: dict, frequencies: dict, land_sea_mask: xarray.DataArray, path) -> None:
    """One row per sector, one panel per regime: mean 500 hPa height anomaly (gpm)
    with coastlines (from the model's land-sea mask), lat/lon ticks, shared colour scale per row."""
    sectors = list(composites)
    kmax = max(c.sizes["regime"] for c in composites.values())
    fig = plt.figure(figsize=(4.3 * kmax + 1.2, 3.3 * len(sectors)))
    gs = fig.add_gridspec(len(sectors), kmax + 1, width_ratios=[1] * kmax + [0.05], wspace=0.25, hspace=0.45)
    for i, sector in enumerate(sectors):
        comp = composites[sector] / G
        comp = comp.assign_coords(lon=_continuous_lon(comp.lon.values, SECTORS[sector]["lon"][0])) \
            .sortby("lon").sortby("lat")
        lsm = _box(land_sea_mask, sector)
        lim = float(np.ceil(np.abs(comp).max() / 10) * 10)
        lat0, lat1 = SECTORS[sector]["lat"]
        lon0 = SECTORS[sector]["lon"][0]
        lon1 = _continuous_lon(np.array([SECTORS[sector]["lon"][1]]), lon0)[0]
        for j in range(kmax):
            ax = fig.add_subplot(gs[i, j])
            if j >= comp.sizes["regime"]:
                ax.axis("off"); continue
            c = comp.isel(regime=j)
            im = ax.pcolormesh(c.lon, c.lat, c.values, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
            ax.contour(c.lon, c.lat, c.values, levels=np.arange(-lim, lim + 1, 20), colors="k",
                       linewidths=0.4, alpha=0.5)
            ax.contour(lsm.lon, lsm.lat, lsm.values, levels=[0.5], colors="0.15", linewidths=0.8)
            ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)
            ax.set_aspect(1 / np.cos(np.deg2rad((lat0 + lat1) / 2)))
            xt = np.arange(np.ceil(lon0 / 30) * 30, lon1 + 0.1, 30)
            yt = np.arange(np.ceil(lat0 / 20) * 20, lat1 + 0.1, 20)
            ax.set_xticks(xt); ax.set_xticklabels([_lon_label(x) for x in xt], fontsize=7)
            ax.set_yticks(yt); ax.set_yticklabels([_lat_label(y) for y in yt], fontsize=7)
            ax.set_title(f"{sector.replace('_', ' ')}  R{j}  ({frequencies[sector][j]:.0%} of days)", fontsize=9)
        cax = fig.add_subplot(gs[i, kmax])
        fig.colorbar(im, cax=cax, label="gpm")
    fig.suptitle("Weather regimes: mean 500 hPa height anomaly per regime (1979–2015)\n"
                 "red = higher than normal (ridge / high), blue = lower (trough / low); "
                 "contours every 20 gpm; coastlines from the model's land-sea mask", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
