"""Phase 7, prospect 2 check (Mac, seconds): does it matter that the Isolation Forest was
fitted again for every year?

In the extended set, 22_outliers.py fitted its standardisation and its 20 Isolation
Forests on each year's own events, so "record-shattering" means "exceptional among that
year's record events". Here the alternative "fixed yardstick": fit the standardisation
and the 20 forests on the 2020 events only (per kind, same settings), score the 2021 and
2022 events with them, and use 2020's median peak exceedance for the second rule. An
event is then record-shattering if it is more unusual than 2020's typical record event.

Prints, per year and kind: record-shattering events with the per-year fit and with the
2020 yardstick, how many both agree on, and how many of the events CHOSEN for the
extended set (case_studies_ext.csv) are record-shattering under the 2020 yardstick too.
Writes results/phase2/fixed_yardstick.csv. Changes nothing else.

    python scripts/74_fixed_yardstick.py
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from extremes import events, paths

P2 = paths.RESULTS_DIR / "phase2"


def features(ev: pd.DataFrame) -> np.ndarray:
    return pd.DataFrame({
        "log_area": np.log10(ev.area_km2), "log_duration": np.log10(ev.duration_h),
        "peak_exceedance": ev.peak_exceedance, "mean_exceedance": ev.mean_exceedance,
        "log_records": np.log10(ev.n_records)})[events.IF_FEATURES].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-year", type=int, default=2020)
    ap.add_argument("--years", type=int, nargs="+", default=[2021, 2022])
    ap.add_argument("--contamination", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--stable", type=float, default=0.8)
    a = ap.parse_args()
    base = pd.read_csv(P2 / f"events_{a.base_year}.csv")
    cs = P2 / "case_studies_ext.csv"
    chosen = pd.read_csv(cs) if cs.exists() else None

    rows = []
    for kind, b in base.groupby("kind"):
        scaler = StandardScaler().fit(features(b))
        forests = [IsolationForest(n_estimators=300, contamination=a.contamination, random_state=s)
                   .fit(scaler.transform(features(b))) for s in range(a.seeds)]
        median = b.peak_exceedance.median()
        for y in a.years:
            ev = pd.read_csv(P2 / f"events_{y}.csv")
            e = ev[ev.kind == kind]
            if e.empty:
                continue
            X = scaler.transform(features(e))
            rate = np.mean([f.predict(X) == -1 for f in forests], 0)
            fixed = (rate >= a.stable) & (e.peak_exceedance.values > median)
            own = e.record_shattering.astype(bool).values
            r = dict(year=y, kind=kind, events=len(e), shattering_own_fit=int(own.sum()),
                     shattering_2020_yardstick=int(fixed.sum()), both=int((own & fixed).sum()),
                     median_peak_own=round(float(e.peak_exceedance.median()), 2),
                     median_peak_2020=round(float(median), 2))
            if chosen is not None:
                ids = set(chosen[(chosen.year == y) & (chosen.kind == kind)].source_event_id)
                pick = e.event_id.isin(ids).values
                r.update(chosen=int(pick.sum()), chosen_also_2020_yardstick=int((pick & fixed).sum()))
            rows.append(r)

    t = pd.DataFrame(rows).sort_values(["year", "kind"])
    t.to_csv(P2 / "fixed_yardstick.csv", index=False)
    print(f"record-shattering events: Isolation Forest fitted on each year's own events vs on {a.base_year}'s\n")
    print(t.to_string(index=False))
    tot = t[["shattering_own_fit", "shattering_2020_yardstick", "both"]].sum()
    print(f"\nall: {tot.shattering_own_fit} (own fit) vs {tot.shattering_2020_yardstick} ({a.base_year} yardstick), "
          f"{tot.both} on both lists")
    if "chosen" in t:
        c = t[["chosen", "chosen_also_2020_yardstick"]].sum()
        print(f"chosen for the extended set: {c.chosen}, of which {c.chosen_also_2020_yardstick} "
              f"({100 * c.chosen_also_2020_yardstick / max(c.chosen, 1):.0f}%) are record-shattering "
              f"under the {a.base_year} yardstick too")
    print("written:", P2 / "fixed_yardstick.csv")


if __name__ == "__main__":
    main()
