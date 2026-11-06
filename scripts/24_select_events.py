"""Phase 2, step 4 (Mac): propose the case-study events.

Rules (all printed, so the choice can be discussed and changed):
  * skip likely artefacts: coastal_fraction > --max-coastal (mostly partly-land cells)
  * skip tiny events: fewer than --min-cells cells
  * per kind, pick record-shattering and ordinary events (counts below), largest total
    exceedance first, preferring a regime/sector not yet used for that kind
  * forecasts start 1, 3, 5 and 7 days before the peak (00 UTC of those days)
  * evaluation box = the event's record cells +- --margin degrees

Writes results/phase2/case_studies.csv and candidates_<kind>.csv (top 15 per kind).

    python scripts/24_select_events.py
    python scripts/24_select_events.py --heat 2 2 --cold 1 2 --wind 1 2   # (shattering, ordinary)
    python scripts/24_select_events.py --force heat-003 --exclude wind-001
"""
import argparse

import numpy as np
import pandas as pd

from extremes import paths, records

OUT = paths.RESULTS_DIR / "phase2"
LEADS_DAYS = (1, 3, 5, 7)


def lon_range(lons: np.ndarray) -> tuple[float, float]:
    """Smallest longitude interval (0..360, may wrap) containing all lons."""
    u = np.sort(np.unique(lons % 360))
    gaps = np.diff(np.r_[u, u[0] + 360])
    i = int(np.argmax(gaps))                       # the largest empty gap is outside the event
    return float(u[(i + 1) % u.size]), float(u[i])


def pick(cands: pd.DataFrame, n: int, used: set) -> list:
    chosen = []
    pool = cands.sort_values("total_exceedance", ascending=False)
    while len(chosen) < n and len(pool):
        fresh = pool[~pool.regime.isin(used) | (pool.regime == "none")]
        row = (fresh if len(fresh) else pool).iloc[0]
        chosen.append(row.event_id)
        used.add(row.regime)
        pool = pool[pool.event_id != row.event_id]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--heat", nargs=2, type=int, default=[2, 2], metavar=("SHATTERING", "ORDINARY"))
    ap.add_argument("--cold", nargs=2, type=int, default=[1, 2], metavar=("SHATTERING", "ORDINARY"))
    ap.add_argument("--wind", nargs=2, type=int, default=[1, 2], metavar=("SHATTERING", "ORDINARY"))
    ap.add_argument("--min-cells", type=int, default=5)
    ap.add_argument("--max-coastal", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--force", nargs="*", default=[], help="event_ids to include anyway")
    ap.add_argument("--exclude", nargs="*", default=[], help="event_ids to leave out")
    a = ap.parse_args()

    ev = pd.read_csv(OUT / f"events_{a.year}.csv", parse_dates=["start", "end", "peak_time"])
    lab = pd.read_csv(OUT / f"records_{a.year}_labeled.csv.gz", parse_dates=["time"])
    for col in ("record_shattering", "regime"):
        if col not in ev:
            raise RuntimeError(f"column {col} missing: run 22_outliers.py and 23_regimes.py first")
    ok = (ev.n_cells >= a.min_cells) & (ev.coastal_fraction.fillna(0) <= a.max_coastal) \
        & ~ev.event_id.isin(a.exclude)
    print(f"{len(ev)} events; {int((ev.n_cells < a.min_cells).sum())} too small, "
          f"{int((ev.coastal_fraction > a.max_coastal).sum())} mostly coastal (possible artefacts), "
          f"{len(a.exclude)} excluded by hand -> {int(ok.sum())} eligible")

    chosen = []
    for kind in records.KINDS:
        n_shat, n_ord = getattr(a, kind)
        e = ev[ok & (ev.kind == kind)]
        used = set()
        chosen += pick(e[e.record_shattering], n_shat, used)
        chosen += pick(e[~e.record_shattering], n_ord, used)
        cols = ["event_id", "peak_time", "centre_lat", "centre_lon", "n_cells", "duration_h",
                "peak_exceedance", "total_exceedance", "record_shattering", "regime", "coastal_fraction"]
        ev[ev.kind == kind].sort_values("total_exceedance", ascending=False).head(15)[cols] \
            .to_csv(OUT / f"candidates_{kind}.csv", index=False)
    chosen += [f for f in a.force if f not in chosen]

    rows = []
    for eid in chosen:
        r = ev[ev.event_id == eid].iloc[0]
        cells = lab[(lab.kind == r.kind) & (lab.event == r.event)]
        lo0, lo1 = lon_range(cells.lon.values)
        peak_day = pd.Timestamp(r.peak_time).normalize()
        rows.append(dict(
            event_id=eid, kind=r.kind, peak_time=r.peak_time, start=r.start, end=r.end,
            centre_lat=round(r.centre_lat, 1), centre_lon=round(r.centre_lon, 1), n_cells=r.n_cells,
            peak_exceedance=round(r.peak_exceedance, 2), record_shattering=bool(r.record_shattering),
            regime=r.regime, sector=r.sector,
            box_lat_min=max(cells.lat.min() - a.margin, -90), box_lat_max=min(cells.lat.max() + a.margin, 90),
            box_lon_min=(lo0 - a.margin) % 360, box_lon_max=(lo1 + a.margin) % 360,
            **{f"init_minus_{d}d": peak_day - pd.Timedelta(days=d) for d in LEADS_DAYS},
            reason="forced" if eid in a.force else ("record-shattering" if r.record_shattering else "ordinary")))
    cs = pd.DataFrame(rows)
    cs.to_csv(OUT / "case_studies.csv", index=False)
    print(f"\nproposed case studies ({len(cs)}):")
    print(cs[["event_id", "kind", "peak_time", "centre_lat", "centre_lon", "n_cells", "peak_exceedance",
              "reason", "regime"]].to_string(index=False))
    print(f"\n{len(cs) * len(LEADS_DAYS)} forecasts to run in Phase 3 "
          f"(starts {', '.join(f'-{d} d' for d in LEADS_DAYS)} before each peak)")
    print("written:", OUT / "case_studies.csv")


if __name__ == "__main__":
    main()
