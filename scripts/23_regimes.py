"""Phase 2, step 3 (Mac): K-means weather regimes and the regime of each event.

Uses results/phase2/z500_daily_1979-2020_2p8deg.nc (from 20_fetch_z500.py, Colab).
Per sector: anomalies -> PCA (10 PCs) -> K-means with k = 3..8, best silhouette.
Fitted on 1979-2015; every day 1979-2020 gets a regime.

Writes to results/phase2/:
  regimes_daily.csv          date, regime per sector
  regimes_summary.json       k, silhouette scores, regime frequencies, explained variance
  regimes_composites.png     mean anomaly map of each regime (what the regimes look like)
  regimes_composite_<sector>.nc  the maps' data (redraw with 23b_plot_regimes.py)
and adds regime + regime_frequency to events_2020.csv (regime of the event's peak day
in its sector; events outside the sectors get none).

    python scripts/23_regimes.py
"""
import argparse
import json

import numpy as np
import pandas as pd
import xarray

from extremes import events, paths, phase1, plots, regimes

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

    # composites: saved per sector (redraw any time with 23b_plot_regimes.py) and plotted
    for sector, r in fits.items():
        r["composites"].rename("z500_anomaly").to_netcdf(OUT / f"regimes_composite_{sector}.nc")
    lsm = xarray.load_dataset(phase1.thresholds_path())["land_sea_mask"]
    plots.regime_composites({s: r["composites"] for s, r in fits.items()},
                            {s: r["frequency"] for s, r in fits.items()}, lsm, OUT / "regimes_composites.png")

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
