"""Phase 7, prospect 2 (Mac or a CPU job): the extended set of case studies.

All record-shattering events of 2020, 2021 and 2022 that pass the same rules as the 14
original case studies (script 24: at least 5 cells, at most half coastal cells, centre
south of 75 N), the largest first, at most --heat/--cold/--wind per year (default
15/8/10; --all = every eligible one). The 14 original case studies are added too
(--no-original leaves them out), so the old results can be compared one to one.

Needs, per year, results/phase2/events_<year>.csv (with record_shattering and regime:
scripts 21, 22, 23) and records_<year>_labeled.csv.gz (script 21).
Writes results/phase2/case_studies_ext.csv. Every Phase 3/4/6 script then works on
this set when started with EXTREMES_SET=ext (results go to results/sets/ext/).

    python scripts/70_select_extended.py
    python scripts/70_select_extended.py --years 2020 2021 --heat 10 --cold 5 --wind 5
    python scripts/70_select_extended.py --all
"""
import argparse

import pandas as pd

from extremes import extended, paths

P2 = paths.RESULTS_DIR / "phase2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022])
    ap.add_argument("--heat", type=int, default=extended.CAPS["heat"])
    ap.add_argument("--cold", type=int, default=extended.CAPS["cold"])
    ap.add_argument("--wind", type=int, default=extended.CAPS["wind"])
    ap.add_argument("--all", action="store_true", help="no caps: every eligible record-shattering event")
    ap.add_argument("--no-original", action="store_true", help="do not add the 14 original case studies")
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--name", default="ext", help="set name: writes case_studies_<name>.csv")
    ap.add_argument("--minutes-per-forecast", type=float, default=1.2)
    a = ap.parse_args()
    caps = None if a.all else {"heat": a.heat, "cold": a.cold, "wind": a.wind}

    rows = []
    for y in a.years:
        ev = pd.read_csv(P2 / f"events_{y}.csv", parse_dates=["start", "end", "peak_time"])
        lab = pd.read_csv(P2 / f"records_{y}_labeled.csv.gz", parse_dates=["time"])
        for col in ("record_shattering", "regime"):
            if col not in ev:
                raise RuntimeError(f"events_{y}.csv has no column {col}: run 22_outliers.py and 23_regimes.py "
                                   f"--year {y} first")
        n_shat = int(ev.record_shattering.sum())
        sel = extended.choose(ev, caps)
        dropped = 0
        for e in sel.itertuples():
            cells = lab[(lab.kind == e.kind) & (lab.event == e.event)]
            r = extended.case_row(e, cells, y, a.margin, "record-shattering")
            if extended.fits_in_data(r):
                rows.append(r)
            else:
                dropped += 1
        print(f"{y}: {len(ev)} events, {n_shat} record-shattering, {int(extended.eligible(ev).sum())} eligible "
              f"(any kind) -> chosen " + ", ".join(f"{k} {int((sel.kind == k).sum())}" for k in ("heat", "cold", "wind"))
              + (f"; {dropped} dropped (forecast would run past the end of the data)" if dropped else ""))

    if not a.no_original and 2020 in a.years:
        orig = pd.read_csv(P2 / "case_studies.csv")
        ev = pd.read_csv(P2 / "events_2020.csv", parse_dates=["start", "end", "peak_time"])
        lab = pd.read_csv(P2 / "records_2020_labeled.csv.gz", parse_dates=["time"])
        have = {r["event_id"] for r in rows}
        added = 0
        for eid in orig.event_id:
            if f"2020-{eid}" in have:
                continue
            e = ev[ev.event_id == eid].iloc[0]
            cells = lab[(lab.kind == e.kind) & (lab.event == e.event)]
            rows.append(extended.case_row(e, cells, 2020, a.margin, "original case study"))
            added += 1
        print(f"+ {added} original case studies not chosen above")

    cs = pd.DataFrame(rows).sort_values(["year", "kind", "event_id"]).reset_index(drop=True)
    out = P2 / f"case_studies_{a.name}.csv"
    cs.to_csv(out, index=False)
    n_fc = len(cs) * len(extended.LEADS_DAYS)
    print(f"\n{len(cs)} case studies ({', '.join(f'{k} {int((cs.kind == k).sum())}' for k in ('heat', 'cold', 'wind'))}; "
          f"{int(cs.record_shattering.sum())} record-shattering) -> {n_fc} forecasts per model")
    print(f"GPU time per model at ~{a.minutes_per_forecast} min per forecast: {n_fc * a.minutes_per_forecast / 60:.1f} h")
    print(cs.groupby(["year", "kind"]).size().unstack(fill_value=0).to_string())
    print("written:", out, f"\nnext: EXTREMES_SET={a.name} for the Phase 3/4/6 scripts (see README, Phase 7)")


if __name__ == "__main__":
    main()
