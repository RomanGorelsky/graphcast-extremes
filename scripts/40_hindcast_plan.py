"""Phase 4, step 1: which hindcasts are needed?

For every case study and forecast start (-1/-3/-5/-7 d), the training start dates
are 00 UTC on the same calendar date +-7 days in 2016-2019. Each date is run once
and serves all events whose windows contain it.

Writes results/phase4/hindcast_plan.csv (init, event_ids) and prints the number of
runs, the number of training samples per event start, and a time estimate.

    python scripts/40_hindcast_plan.py
"""
import argparse

import pandas as pd

from extremes import phase3, phase4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes-per-run", type=float, default=1.5)
    a = ap.parse_args()
    p = phase4.plan()
    runs = (p.groupby("init").event_id.apply(lambda s: " ".join(sorted(set(s)))).rename("event_ids")
            .reset_index())
    runs["year"] = runs.init.dt.year
    runs = runs.sort_values(["year", "init"], ascending=[False, True]).reset_index(drop=True)
    phase4.OUT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(phase4.OUT / "hindcast_plan.csv", index=False)

    per = p.groupby(["event_id", "start_days"]).size()
    print(f"{len(runs)} hindcast runs ({', '.join(f'{y}: {n}' for y, n in runs.year.value_counts().sort_index(ascending=False).items())})")
    print(f"training samples per event start: min {per.min()}, max {per.max()} "
          f"(= {phase4.HALF_WINDOW * 2 + 1} days x {len(phase4.YEARS)} years)")
    print(f"events served per run: mean {runs.event_ids.str.split().str.len().mean():.1f}")
    h = len(runs) * a.minutes_per_run / 60
    print(f"estimated GPU time at {a.minutes_per_run} min/run: {h:.0f} h "
          f"(~{h / 3:.0f} Colab sessions of 3 h)")
    print("written:", phase4.OUT / "hindcast_plan.csv")


if __name__ == "__main__":
    main()
