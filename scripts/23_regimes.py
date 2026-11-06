"""Phase 2, step 3 (Mac): K-means weather regimes and the regime of each event.

Uses results/phase2/z500_daily_1979-2020_2p8deg.nc (from 20_fetch_z500.py, Colab).
Per sector: anomalies -> PCA (10 PCs) -> K-means with k = 3..8, best silhouette.
Fitted on 1979-2015; every day 1979-2020 gets a regime.

Writes to results/phase2/:
  regimes_daily.csv          date, regime per sector
  regimes_summary.json       k, silhouette scores, regime frequencies, explained variance
  regimes_composites.png     mean anomaly map of each regime (what the regimes look like)
and adds regime + regime_frequency to events_2020.csv (regime of the event's peak day
in its sector; events outside the four sectors get none).

    python scripts/23_regimes.py
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

from extremes import events, paths, regimes

OUT = paths.RESULTS_DIR / "phase2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--n-pcs", type=int, default=10)
    a = ap.parse_args()
    z = xarray.load_dataarray(OUT / "z500_daily_1979-2020_2p8deg.nc")
    print(f"z500: {z.sizes['time']} days, grid {z.sizes['lat']} x {z.sizes['lon']}", flush=True)
    anom = regimes.anomalies(z)

    fits, daily = {}, {}
    for sector in events.SECTORS:
        r = regimes.fit_sector(anom, sector, n_pcs=a.n_pcs)
        fits[sector] = r
        daily[sector] = r["labels"]
        print(f"{sector:14s} k={r['k']}  silhouette " +
              " ".join(f"{k}:{s:.3f}" for k, s in r["silhouette"].items()) +
              f"  | frequencies {r['frequency']}", flush=True)

    d = pd.DataFrame(daily)
    d.index = d.index.normalize()
    d.index.name = "date"
    d.to_csv(OUT / "regimes_daily.csv")
    (OUT / "regimes_summary.json").write_text(json.dumps(
        {s: {k: v for k, v in r.items() if k not in ("labels", "composites")} for s, r in fits.items()},
        indent=1))

    # composites figure: one row per sector
    kmax = max(r["k"] for r in fits.values())
    fig, axes = plt.subplots(len(fits), kmax, figsize=(3.2 * kmax, 2.6 * len(fits)), squeeze=False)
    for i, (sector, r) in enumerate(fits.items()):
        comp = r["composites"] / 9.80665                          # m2/s2 -> geopotential metres
        lim = float(np.abs(comp).max())
        for j in range(kmax):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= r["k"]:
                ax.axis("off"); continue
            c = comp.isel(regime=j)
            lon = c.lon.values
            order = np.argsort(np.where(lon >= events.SECTORS[sector]["lon"][0], lon, lon + 360))
            ax.pcolormesh(np.arange(lon.size), c.lat.values, c.values[:, order], cmap="RdBu_r",
                          vmin=-lim, vmax=lim, shading="auto")
            ax.set_title(f"{sector} R{j}  ({r['frequency'][j]:.0%})", fontsize=8)
    fig.suptitle("Regime composites: 500 hPa height anomaly (gpm), red = high, blue = low", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "regimes_composites.png", dpi=100)

    # regime of each event
    f = OUT / f"events_{a.year}.csv"
    ev = pd.read_csv(f, parse_dates=["start", "end", "peak_time"])
    reg, freq = [], []
    for r in ev.itertuples():
        if r.sector in fits:
            day = pd.Timestamp(r.peak_time).normalize()
            lab = int(d.loc[day, r.sector])
            reg.append(f"{r.sector}-R{lab}")
            freq.append(fits[r.sector]["frequency"][lab])
        else:
            reg.append("none"); freq.append(np.nan)
    ev["regime"], ev["regime_frequency"] = reg, freq
    ev.to_csv(f, index=False)
    print("\nevents per regime:\n" + ev[ev.regime != "none"].groupby(["kind", "regime"]).size()
          .unstack(0, fill_value=0).to_string())
    print("\nupdated:", f)


if __name__ == "__main__":
    main()
