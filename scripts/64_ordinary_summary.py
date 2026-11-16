"""Phase 6 extension (Mac): summary of the 52 ordinary-weather forecasts of R0-R4.

Reads results/phase6/ordinary/<model>.csv (62_ordinary_weather.py) and
truth_budgets.csv. Prints and writes (results/phase6/ordinary/):
  1. skill: RMSE and ACC by variable and lead day, and the change against R0 with a
     95 % range from resampling the 52 dates (paired: same dates for every model);
     share of dates on which the model beats R0
  2. sharpness: forecast / ERA5 anomaly activity (1 = as variable as reality)
  3. budget drift: how far the forecast's global dry-air mass, water vapour and total
     energy are from ERA5's on the same day (relative), by lead day
  figure ordinary_skill.png (RMSE change vs R0 and budget drift by lead day)

    python scripts/64_ordinary_summary.py
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extremes import paths

OUT = paths.RESULTS_DIR / "phase6" / "ordinary"
SHOW_LEADS = [1, 3, 5, 7, 10]
VARS = ["z500", "t850", "2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
        "10m_v_component_of_wind"]
SHORT = {"z500": "z500", "t850": "t850", "2m_temperature": "t2m", "mean_sea_level_pressure": "msl",
         "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10"}


def boot_ci(x: np.ndarray, B: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["R0", "R1", "R2", "R3", "R4"])
    a = ap.parse_args()
    parts = []
    for lab in a.labels:
        f = OUT / f"{lab}.csv"
        if f.exists():
            parts.append(pd.read_csv(f, parse_dates=["init"]).assign(model=lab))
        else:
            print(f"(missing {f})")
    d = pd.concat(parts)
    labels = [lab for lab in a.labels if lab in set(d.model)]
    sk = d[d["var"].isin(VARS)].copy()
    n_dates = sk.groupby("model").init.nunique()
    print("dates per model:", n_dates.to_dict())

    # 1. skill
    for metric, better in (("rmse", "lower"), ("acc", "higher")):
        if metric not in sk or sk[metric].isna().all():
            continue
        t = sk[sk.lead_days.isin(SHOW_LEADS)].pivot_table(index=["var", "lead_days"], columns="model",
                                                          values=metric)[labels]
        t.index = t.index.set_levels([SHORT.get(v, v) for v in t.index.levels[0]], level=0)
        print(f"\n1. {metric.upper()} ({better} = better), mean of {n_dates.min()} dates")
        print(t.round(3).to_string())
    rows = []
    base = sk[sk.model == "R0"].set_index(["init", "var", "lead_days"])
    for lab in labels:
        if lab == "R0":
            continue
        o = sk[sk.model == lab].set_index(["init", "var", "lead_days"])
        j = o.join(base, rsuffix="_R0", how="inner")
        for (var, lead), g in j.groupby(level=["var", "lead_days"]):
            rel = 100 * (g.rmse / g.rmse_R0 - 1)
            lo, hi = boot_ci(rel.values)
            row = dict(model=lab, var=SHORT.get(var, var), lead_days=lead, rmse_change_pct=rel.mean(),
                       ci_low=lo, ci_high=hi, share_dates_better=float((g.rmse < g.rmse_R0).mean()))
            if "acc" in g and g.acc.notna().any():
                row["acc_change"] = float((g.acc - g.acc_R0).mean())
            rows.append(row)
    ch = pd.DataFrame(rows)
    ch.to_csv(OUT / "ordinary_change_vs_R0.csv", index=False)
    print("\n   RMSE change vs R0 in % [95 % range over dates] (share of dates better than R0)")
    for (var, lead), g in ch[ch.lead_days.isin(SHOW_LEADS)].groupby(["var", "lead_days"]):
        cells = [f"{r.model} {r.rmse_change_pct:+5.1f} [{r.ci_low:+.1f},{r.ci_high:+.1f}]"
                 f"{'*' if r.ci_high < 0 or r.ci_low > 0 else ' '} ({100 * r.share_dates_better:.0f}%)"
                 for r in g.itertuples()]
        print(f"   {var:5s} d{lead:<2d} | " + " | ".join(cells))
    print("   * = 95 % range excludes zero")

    # 2. sharpness
    if "activity" in sk and sk.activity.notna().any():
        t = sk[sk.lead_days.isin(SHOW_LEADS)].pivot_table(index=["var", "lead_days"], columns="model",
                                                          values="activity")[labels]
        t.index = t.index.set_levels([SHORT.get(v, v) for v in t.index.levels[0]], level=0)
        print("\n2. activity: forecast / ERA5 anomaly variability (1 = as variable as reality, < 1 = smoother)")
        print(t.round(3).to_string())

    # 3. budget drift
    bt = OUT / "truth_budgets.csv"
    bud = d[d["var"].str.startswith("budget_")].copy()
    drift = None
    if bt.exists() and len(bud):
        tr = pd.read_csv(bt, parse_dates=["valid"]).drop_duplicates("valid").set_index("valid")
        bud["valid"] = bud.init + pd.to_timedelta(bud.lead_days, unit="D")
        bud["quantity"] = bud["var"].str.removeprefix("budget_")
        bud["truth"] = [tr.at[v, q] if v in tr.index else np.nan for v, q in zip(bud.valid, bud.quantity)]
        bud["rel_error"] = (bud.value - bud.truth) / bud.truth
        drift = bud.pivot_table(index=["quantity", "lead_days"], columns="model", values="rel_error")[labels]
        drift.to_csv(OUT / "budget_drift.csv")
        print("\n3. global budgets: (forecast - ERA5) / ERA5, mean over dates (0 = same as ERA5)")
        print(drift[drift.index.get_level_values(1).isin([0] + SHOW_LEADS)].map(lambda x: f"{x:+.2e}").to_string())
    else:
        print("\n3. (no truth_budgets.csv - run 62_ordinary_weather.py --truth-budgets)")

    # figure
    fig, axes = plt.subplots(1, 3 if drift is not None else 2, figsize=(16, 4.5))
    for ax, var in zip(axes[:2], ("z500", "t2m")):
        for lab in labels[1:]:
            g = ch[(ch.model == lab) & (ch["var"] == var)].sort_values("lead_days")
            ax.plot(g.lead_days, g.rmse_change_pct, "-o", ms=3, label=lab)
            ax.fill_between(g.lead_days, g.ci_low, g.ci_high, alpha=0.12)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{var}: RMSE change vs R0 (%), 52 dates of 2020")
        ax.set_xlabel("lead time (days)"); ax.legend(fontsize=8)
    if drift is not None:
        ax = axes[2]
        for lab in labels:
            s = drift.loc["vapour"][lab]
            ax.plot(s.index, 100 * s.values, "-o", ms=3, label=lab)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title("global water vapour: forecast vs ERA5 (%)"); ax.set_xlabel("lead time (days)")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "ordinary_skill.png", dpi=120)
    print("\nwritten:", OUT / "ordinary_change_vs_R0.csv", ", budget_drift.csv, ordinary_skill.png")


if __name__ == "__main__":
    main()
