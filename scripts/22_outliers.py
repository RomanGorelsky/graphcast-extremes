"""Phase 2, step 2 (Mac): Isolation Forest - which events are 'record-shattering'?

Per kind, on standardised event features (log area, log duration, peak and mean
exceedance, log number of records). Repeated over --seeds random seeds; an event is
'unusual' if flagged in >= 80 % of them, and 'record-shattering' if it is unusual AND
its peak exceedance is above the median of its kind.

Adds columns if_score, if_flag_rate, unusual, record_shattering to
results/phase2/events_2020.csv and draws events_2020_outliers.png.

    python scripts/22_outliers.py
    python scripts/22_outliers.py --contamination 0.05
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from extremes import events, paths, records

OUT = paths.RESULTS_DIR / "phase2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--contamination", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    f = OUT / f"events_{a.year}.csv"
    ev = pd.read_csv(f, parse_dates=["start", "end", "peak_time"])
    ev = ev.drop(columns=[c for c in ("if_score", "if_flag_rate", "unusual", "record_shattering") if c in ev])
    ev = events.isolation_flags(ev, contamination=a.contamination, n_seeds=a.seeds)
    ev.to_csv(f, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, kind in zip(axes, records.KINDS):
        e = ev[ev.kind == kind]
        for flag, style in ((False, dict(c="0.6", s=18, label="ordinary")),
                            (True, dict(c="tab:red", s=40, label="record-shattering"))):
            s = e[e.record_shattering == flag]
            ax.scatter(s.area_km2, s.peak_exceedance, edgecolor="k", linewidth=0.3, **style)
        u = e[e.unusual & ~e.record_shattering]
        ax.scatter(u.area_km2, u.peak_exceedance, facecolor="none", edgecolor="tab:orange", s=40,
                   label="unusual, not intense")
        for r in e[e.record_shattering].itertuples():
            ax.annotate(r.event_id, (r.area_km2, r.peak_exceedance), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("area (km²)")
        ax.set_ylabel(f"peak exceedance ({records.UNITS[kind]})")
        ax.set_title(f"{kind}: {int(e.record_shattering.sum())} of {len(e)} record-shattering")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"events_{a.year}_outliers.png", dpi=110)

    cols = ["event_id", "peak_time", "centre_lat", "centre_lon", "n_cells", "duration_h",
            "peak_exceedance", "if_flag_rate", "sector"]
    for kind in records.KINDS:
        e = ev[(ev.kind == kind) & ev.record_shattering].sort_values("if_score", ascending=False)
        print(f"\n{kind}: {len(e)} record-shattering events")
        if len(e):
            print(e[cols].round(2).to_string(index=False))
    print("\nupdated:", f)


if __name__ == "__main__":
    main()
