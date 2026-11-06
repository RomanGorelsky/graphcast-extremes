"""Phase 2, step 1 (Mac): DBSCAN - group 2020 record cells into events.

For each kind (heat, cold, wind) separately. Two records are neighbours if they are
about --space-km apart at the same time, or in the same cell --time-h apart
(eps = 1 in scaled units). A cluster needs --min-samples records; the rest is noise.

Writes to results/phase2/:
  kdistance.png               k-distance curves (choose eps at the 'knee'; ~1 expected)
  records_2020_labeled.csv.gz records with their event number (-1 = noise)
  events_2020.csv             one row per event with its features
  events_2020_map.png         event centres (size = area, colour = kind)

    python scripts/21_detect_events.py
    python scripts/21_detect_events.py --space-km 300 --time-h 12 --min-samples 5
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

from extremes import events, paths, phase1, records

OUT = paths.RESULTS_DIR / "phase2"
COLOURS = {"heat": "tab:red", "cold": "tab:blue", "wind": "tab:purple"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--space-km", type=float, default=250)
    ap.add_argument("--time-h", type=float, default=12)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--min-samples", type=int, default=5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(phase1.OUT / f"records_{a.year}.csv.gz", parse_dates=["time"])
    th = xarray.load_dataset(phase1.thresholds_path())

    # k-distance diagnostic
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in records.KINDS:
        sub = df[df.kind == kind]
        if len(sub) > a.min_samples:
            d = events.k_distance(sub, a.min_samples, a.space_km, a.time_h)
            ax.plot(np.linspace(0, 100, d.size), d, label=kind, color=COLOURS[kind])
    ax.axhline(a.eps, ls="--", color="k", lw=0.8, label=f"eps = {a.eps}")
    ax.set_ylim(0, 4); ax.set_xlabel("records, sorted (%)")
    ax.set_ylabel(f"distance to {a.min_samples}-th neighbour (scaled)")
    ax.set_title(f"k-distance ({a.space_km:.0f} km / {a.time_h:.0f} h = 1)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "kdistance.png", dpi=110); plt.close(fig)

    # DBSCAN per kind
    df["event"] = -1
    for kind in records.KINDS:
        idx = df.index[df.kind == kind]
        df.loc[idx, "event"] = events.cluster_records(df.loc[idx], a.space_km, a.time_h, a.eps, a.min_samples)
    df.to_csv(OUT / f"records_{a.year}_labeled.csv.gz", index=False)

    ev = events.event_features(df, th["land_sea_mask"])
    ev.to_csv(OUT / f"events_{a.year}.csv", index=False)

    summary = {"params": vars(a), "per_kind": {}}
    for kind in records.KINDS:
        sub, e = df[df.kind == kind], ev[ev.kind == kind]
        summary["per_kind"][kind] = {
            "records": int(len(sub)), "events": int(len(e)),
            "share_in_events": round(float((sub.event >= 0).mean()), 3) if len(sub) else 0,
            "largest_event_cells": int(e.n_cells.max()) if len(e) else 0,
            "median_event_cells": float(e.n_cells.median()) if len(e) else 0}
        print(f"{kind}: {len(sub):,} records -> {len(e)} events "
              f"({summary['per_kind'][kind]['share_in_events']:.0%} of records in events); "
              f"median {summary['per_kind'][kind]['median_event_cells']:.0f} cells, "
              f"largest {summary['per_kind'][kind]['largest_event_cells']}")
    (OUT / f"events_{a.year}_summary.json").write_text(json.dumps(summary, indent=1, default=str))

    print("\nTop 5 events per kind (by area-weighted exceedance):")
    cols = ["event_id", "peak_time", "centre_lat", "centre_lon", "n_cells", "area_km2", "duration_h",
            "peak_exceedance", "weighted_exceedance", "sector"]
    for kind in records.KINDS:
        top = ev[ev.kind == kind].head(5)[cols].copy()
        num = top.select_dtypes("number").columns
        top[num] = top[num].round(1)
        print(top.to_string(index=False))

    # map of event centres
    fig, ax = plt.subplots(figsize=(13, 6))
    land = th["analysis_mask"].values.astype(float)
    lon = th.lon.values
    lon_p = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon_p)
    ax.contour(lon_p[order], th.lat.values, land[:, order], levels=[0.5], colors="0.6", linewidths=0.5)
    for kind in records.KINDS:
        e = ev[ev.kind == kind]
        x = np.where(e.centre_lon > 180, e.centre_lon - 360, e.centre_lon)
        ax.scatter(x, e.centre_lat, s=np.clip(e.n_cells, 5, 400), alpha=0.5, color=COLOURS[kind],
                   label=f"{kind} ({len(e)})", edgecolor="k", linewidth=0.3)
    for name, box in events.SECTORS.items():
        (y0, y1), (x0, x1) = box["lat"], box["lon"]
        x0, x1 = (x0 - 360 if x0 > 180 else x0), (x1 - 360 if x1 > 180 else x1)
        ax.add_patch(plt.Rectangle((x0, y0), (x1 - x0) % 360 or 360, y1 - y0, fill=False, ls=":", lw=1))
        ax.text(x0 + 1, y1 - 4, name, fontsize=8)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 90); ax.legend(loc="lower left")
    ax.set_title(f"{a.year} record events (DBSCAN): centre, size = number of cells")
    fig.tight_layout(); fig.savefig(OUT / f"events_{a.year}_map.png", dpi=110)
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
