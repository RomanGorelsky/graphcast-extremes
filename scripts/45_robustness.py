"""Phase 4, check 1 (Mac, ~1 min): do the Phase 4 conclusions hold across events?

Uses the files written by 33_verify.py for R0, Debias, Persistence, PBC
(results/phase3/<label>_record_errors.csv.gz). Compares each correction with R0 on
exactly the same record cell-times and asks, per kind and start day:
  - delta: change of the mean intensity error (negative = correction made the
    underestimate worse) and of the hit rate,
  - 95% range from resampling whole events (bootstrap),
  - range when one event is left out (is one event driving it?),
  - how many events got worse / better.

Writes results/phase4/robustness_events.csv (per event), robustness.csv (summary)
and robustness_events.png.

    python scripts/45_robustness.py
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from extremes import phase3, phase4, records, robust


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["R0", "Debias", "Persistence", "PBC"])
    ap.add_argument("--bootstrap", type=int, default=5000)
    a = ap.parse_args()
    errs = {lab: pd.read_csv(phase3.OUT / f"{lab}_record_errors.csv.gz") for lab in a.labels}
    p = robust.paired(errs)
    n0 = len(errs["R0"])
    print(f"{len(p):,} record cell-times forecast by all of {', '.join(a.labels)} "
          f"(R0 alone: {n0:,}; fewer = forecasts skipped in 43 for lack of hindcasts)")

    ev = robust.event_table(p, a.labels)
    ev.to_csv(phase4.OUT / "robustness_events.csv", index=False)
    rt = robust.robustness_table(p, a.labels, B=a.bootstrap)
    rt.to_csv(phase4.OUT / "robustness.csv", index=False)

    methods = [lab for lab in a.labels if lab != "R0"]
    for metric, title, scale, unit in (("ie", "change of mean intensity error vs R0", 1, "K or m/s"),
                                       ("hit", "change of hit rate vs R0", 100, "percentage points")):
        print(f"\n{title} ({unit}; negative = worse than R0)")
        print("  delta [95% range from resampling events] {events worse/total}; "
              "range with one event left out")
        for (kind, d), g in rt[rt.metric == metric].groupby(["kind", "start_days"]):
            cells = []
            for m in methods:
                r = g[g.method == m].iloc[0]
                sure = "*" if (r.ci_high < 0 or r.ci_low > 0) else " "
                cells.append(f"{m} {r.delta * scale:+6.2f} [{r.ci_low * scale:+.2f},{r.ci_high * scale:+.2f}]{sure}"
                             f" {{{r.events_worse}/{r.n_events}}} loo {r.loo_min * scale:+.2f}..{r.loo_max * scale:+.2f}")
            print(f"  {kind:4s} -{d} d | " + "\n           | ".join(cells))
    print("\n* = the 95% range excludes zero (the change is consistent across events)")
    print("Caution: with 4-5 events per kind these ranges are rough; read them together with the "
          "per-event file robustness_events.csv")

    # figure: per-event change of intensity error, one panel per kind
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=False)
    marks = dict(zip(methods, ["s", "^", "D", "o", "v"]))
    for ax, kind in zip(axes, records.KINDS):
        e = ev[ev.kind == kind]
        for k, m in enumerate(methods):
            x = e.start_days + (k - (len(methods) - 1) / 2) * 0.35
            ax.scatter(x, e[f"delta_{m}"], marker=marks[m], label=m, alpha=0.75)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(list(phase3.LEADS_DAYS)); ax.set_xlabel("forecast start, days before the peak")
        ax.set_ylabel(f"change of intensity error vs R0 ({records.UNITS[kind]})")
        ax.set_title(f"{kind}: one dot per event ({e.event_id.nunique()} events); < 0 = worse")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(phase4.OUT / "robustness_events.png", dpi=120)
    print("\nwritten:", phase4.OUT / "robustness.csv", ", robustness_events.csv, robustness_events.png")


if __name__ == "__main__":
    main()
