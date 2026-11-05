"""Phase 1, step 4: sanity check of the 2020 records against Zhang et al. (Fig. 1A-B).

Draws, for heat / cold / wind: a map of how many 6-hourly time steps broke the
record in each cell, and the number of records per latitude. Prints a check on
a well-known event: the Siberian heatwave of spring-summer 2020.
Writes results/phase1/records_<year>_maps.png.

    python scripts/13_sanity_maps.py
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

from extremes import phase1, records


def count_grid(df: pd.DataFrame, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    g = np.zeros((lat.size, lon.size))
    if len(df):
        il = np.searchsorted(lat, df.lat.values)
        jl = np.searchsorted(lon, df.lon.values)
        np.add.at(g, (il, jl), 1)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2020)
    a = ap.parse_args()
    df = pd.read_csv(phase1.OUT / f"records_{a.year}.csv.gz", parse_dates=["time"])
    th = xarray.load_dataset(phase1.thresholds_path())
    lat, lon = th.lat.values, th.lon.values
    land = th["analysis_mask"].values

    # shift to -180..180 for plotting
    shift = lon.size // 2
    lon_p = np.concatenate([lon[shift:] - 360, lon[:shift]])
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), gridspec_kw={"width_ratios": [5, 1]})
    for row, kind in enumerate(records.KINDS):
        sub = df[df.kind == kind]
        g = count_grid(sub, lat, lon)
        g = np.where(land, g, np.nan)
        g = np.concatenate([g[:, shift:], g[:, :shift]], axis=1)
        ax = axes[row, 0]
        ax.contour(lon_p, lat, np.concatenate([land[:, shift:], land[:, :shift]], 1).astype(float),
                   levels=[0.5], colors="0.6", linewidths=0.5)
        im = ax.pcolormesh(lon_p, lat, np.where(g > 0, g, np.nan), cmap="YlOrRd" if kind == "heat"
                           else "Blues" if kind == "cold" else "Purples", vmin=1,
                           vmax=max(np.nanpercentile(g[g > 0], 99), 2) if np.nansum(g) else 2, shading="auto")
        fig.colorbar(im, ax=ax, shrink=0.8, label="record time steps")
        ax.set_title(f"{kind} records {a.year}: {len(sub):,} cell-times in "
                     f"{sub[['lat', 'lon']].drop_duplicates().shape[0]:,} cells")
        ax.set_ylim(-60, 90); ax.set_xlim(-180, 180)
        per_lat = sub.groupby("lat").size().reindex(lat, fill_value=0)
        axes[row, 1].plot(per_lat.values, lat)
        axes[row, 1].set_ylim(-60, 90); axes[row, 1].set_title("count per latitude")
    fig.tight_layout()
    out = phase1.OUT / f"records_{a.year}_maps.png"
    fig.savefig(out, dpi=110)

    heat = df[df.kind == "heat"]
    sib = heat[(heat.lat >= 55) & (heat.lat <= 75) & (heat.lon >= 60) & (heat.lon <= 180)
               & heat.time.dt.month.isin([5, 6, 7])]
    print(f"Siberia (55-75N, 60-180E, May-Jul): {len(sib):,} heat-record cell-times in "
          f"{sib[['lat', 'lon']].drop_duplicates().shape[0]:,} cells; "
          f"max exceedance {sib.exceedance.max() if len(sib) else float('nan'):.1f} K")
    print("(expected: clearly non-zero - the 2020 Siberian heatwave is one of Zhang et al.'s named events)")
    for kind in records.KINDS:
        e = df[df.kind == kind].exceedance
        if len(e):
            print(f"{kind}: median exceedance {e.median():.2f}, 95th pct {e.quantile(0.95):.2f}, "
                  f"max {e.max():.2f} {records.UNITS[kind]}")
    print("figure:", out)


if __name__ == "__main__":
    main()
