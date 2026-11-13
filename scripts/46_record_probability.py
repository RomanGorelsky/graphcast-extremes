"""Phase 4, check 2 (Mac, a few minutes): does a probabilistic version of the corrections
give useful probabilities of breaking the record?

For each case study and start, as in 43_apply_pbc.py, the corrections are learned
from the training hindcasts - now together with how far off each one typically was.
Each forecast then gives, for every land cell and time of the event period in the
event box, the probability that the cell breaks its 1979-2015 record (see
extremes/probrec.py for the five versions). Scored against what ERA5 did:
  brier             mean squared difference between probability and outcome (0/1);
                    lower is better; R0's is the share of cells it gets wrong
  skill_vs_R0       1 - brier / brier(R0): > 0 = better than the deterministic R0
  mean_p_on_records average probability given to the cells that DID break a record
  pod_0.2/0.5       share of records "warned" at probability >= 20% / 50%
  precision_0.2/0.5 share of warnings that were right

Needs results/phase4/{hindcasts,truth}/, hindcast_plan.csv, the R0 forecasts
(results/phase3/forecasts) and the Phase 1 thresholds. Writes
results/phase4/record_probability.csv (pooled per kind and start),
record_probability_cases.csv (per event and start) and record_probability.png.

    python scripts/46_record_probability.py
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

from extremes import pbc, phase1, phase3, phase4, probrec, records, verify

T2M, U, V = "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"
METHODS = ("R0", "R0+spread", "Debias", "Persistence", "PBC")
TH_VAR = {"heat": "t2m_max", "cold": "t2m_min", "wind": "wind_max"}


def quantity(ds, which: str, kind: str) -> np.ndarray:
    if kind in ("heat", "cold"):
        return ds[f"{which}_{T2M}"].values
    return np.hypot(ds[f"{which}_{U}"].values, ds[f"{which}_{V}"].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hindcasts", type=int, default=20)
    a = ap.parse_args()
    cases = phase3.case_studies()
    th = xarray.load_dataset(phase1.thresholds_path())
    pooled = {}                                   # (kind, start, method) -> [p arrays], [y arrays]
    rows = []

    for c in cases.itertuples():
        tf = phase4.truth_path(c.event_id)
        if not tf.exists():
            print(f"{c.event_id}: no truth cache - skipped"); continue
        T = xarray.load_dataset(tf)
        s = verify.SIGN[c.kind]
        for d in phase3.LEADS_DAYS:
            fp = phase3.forecast_path(c.event_id, d)
            if not fp.exists():
                continue
            F = xarray.load_dataset(fp)
            t_e = pd.Timestamp(F.attrs["init"])
            inits = [t for t in phase4.training_inits(t_e) if phase4.hindcast_path(c.event_id, t).exists()]
            if len(inits) < a.min_hindcasts:
                print(f"{c.event_id} -{d} d: only {len(inits)} hindcasts - skipped"); continue
            H = [xarray.load_dataset(phase4.hindcast_path(c.event_id, t)) for t in inits]
            times = pd.DatetimeIndex(F.time.values)
            leads = times - t_e

            Fq = quantity(F, "forecast", c.kind)                                       # (K, Y, X)
            Hq = np.stack([quantity(h, "forecast", c.kind) for h in H])                # (J, K, Y, X)
            Oq = np.stack([quantity(T.sel(time=t + leads), "truth", c.kind) for t in inits])
            H0 = np.stack([quantity(T.sel(time=[t]), "truth", c.kind)[0] for t in inits])
            O0 = quantity(T.sel(time=[t_e]), "truth", c.kind)[0]
            truth = quantity(F, "truth", c.kind)

            D = pbc.debias(Fq, Hq, Oq)
            sd_D = pbc.debias_spread(Hq, Oq)
            P, sd_P = pbc.persistence(Fq, O0, Hq, Oq, H0, return_spread=True)
            with np.errstate(invalid="ignore"):
                rms = np.sqrt(np.nanmean((Hq - Oq) ** 2, 0))

            thr = th[TH_VAR[c.kind]].sel(month=xarray.DataArray(times.month, dims="time"))
            thr = thr.sel(lat=F.lat, lon=F.lon).transpose("time", "lat", "lon").values
            land = th["analysis_mask"].sel(lat=F.lat, lon=F.lon).values.astype(bool)
            in_period = (times > t_e) & (times >= max(t_e, pd.Timestamp(c.start))) & (times <= pd.Timestamp(c.end))
            valid = in_period[:, None, None] & land[None] & np.isfinite(truth) & np.isfinite(thr)
            y = (s * (truth - thr) > 0)[valid]

            pD = probrec.prob_exceed(D, sd_D, thr, s)
            pP = probrec.prob_exceed(P, sd_P, thr, s)
            probs = {"R0": probrec.prob_exceed(Fq, np.nan, thr, s),
                     "R0+spread": probrec.prob_exceed(Fq, rms, thr, s),
                     "Debias": pD, "Persistence": pP, "PBC": 0.5 * (pD + pP)}
            for m, pr in probs.items():
                pv = pr[valid]
                key = (c.kind, d, m)
                pooled.setdefault(key, ([], []))
                pooled[key][0].append(pv); pooled[key][1].append(y)
                rows.append(dict(event_id=c.event_id, kind=c.kind, start_days=d, method=m,
                                 n_hindcasts=len(inits), **probrec.scores(pv, y)))
            print(f"{c.event_id} -{d} d: {valid.sum():,} cell-times, {y.sum():,} records; Brier "
                  + ", ".join(f"{m} {rows[-len(METHODS) + i]['brier']:.3f}" for i, m in enumerate(METHODS)),
                  flush=True)

    cs = pd.DataFrame(rows)
    cs.to_csv(phase4.OUT / "record_probability_cases.csv", index=False)
    out = []
    for (kind, d, m), (ps, ys) in pooled.items():
        out.append(dict(kind=kind, start_days=d, method=m,
                        **probrec.scores(np.concatenate(ps), np.concatenate(ys))))
    t = pd.DataFrame(out)
    base = t[t.method == "R0"][["kind", "start_days", "brier"]].rename(columns={"brier": "brier_R0"})
    t = t.merge(base, on=["kind", "start_days"], how="left")
    t["skill_vs_R0"] = np.where(t.brier_R0 > 0, 1 - t.brier / t.brier_R0, np.nan)
    t.to_csv(phase4.OUT / "record_probability.csv", index=False)

    for col, title in (("skill_vs_R0", "Brier skill vs R0 (> 0 = better probabilities than R0)"),
                       ("mean_p_on_records", "average probability given to cells that broke a record"),
                       ("pod_0.2", "share of records warned at probability >= 20%"),
                       ("precision_0.2", "share of those warnings that were right")):
        print(f"\n{title}")
        print(t.pivot_table(index=["kind", "start_days"], columns="method", values=col)[list(METHODS)]
              .round(3).to_string())
    print("\nbase rate (share of cell-times in the boxes that broke a record):")
    print(t[t.method == "R0"].set_index(["kind", "start_days"]).base_rate.round(3).to_string())

    # figure: Brier skill by start day, and reliability (all starts pooled)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, kind in enumerate(records.KINDS):
        g = t[t.kind == kind]
        for m in METHODS[1:]:
            h = g[g.method == m].sort_values("start_days")
            axes[0, j].plot(h.start_days, h.skill_vs_R0, "-o", label=m)
        axes[0, j].axhline(0, color="k", lw=0.8); axes[0, j].set_title(f"{kind}: Brier skill vs R0 (> 0 = better)")
        axes[0, j].set_xlabel("forecast start, days before the peak"); axes[0, j].legend(fontsize=8)
        for m in METHODS[1:]:
            ps = np.concatenate([np.concatenate(pooled[(kind, d, m)][0]) for d in phase3.LEADS_DAYS
                                 if (kind, d, m) in pooled] or [np.array([])])
            ys = np.concatenate([np.concatenate(pooled[(kind, d, m)][1]) for d in phase3.LEADS_DAYS
                                 if (kind, d, m) in pooled] or [np.array([])])
            r = probrec.reliability(ps, ys)
            if r:
                axes[1, j].plot([x[0] for x in r], [x[1] for x in r], "-o", label=m)
        axes[1, j].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[1, j].set_xlabel("forecast probability of a record"); axes[1, j].set_ylabel("observed frequency")
        axes[1, j].set_title(f"{kind}: reliability (on the diagonal = honest probabilities)")
        axes[1, j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(phase4.OUT / "record_probability.png", dpi=120)
    print("\nwritten:", phase4.OUT / "record_probability.csv", ", record_probability_cases.csv, record_probability.png")


if __name__ == "__main__":
    main()
