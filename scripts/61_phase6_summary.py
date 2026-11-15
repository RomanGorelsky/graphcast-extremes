"""Phase 6 (Mac): the parts of the comparison that are not about record cells.

1. Ordinary weather ("do no harm"): global RMSE of R0-R4 on the 12 ordinary 2020 dates
   (results/phase3/noharm.csv for R0, results/phase6/noharm_<run>.csv for R1-R4),
   at lead days 1, 3, 5, 10, and the change against R0.
2. Sharpness: does a model smooth its forecasts? For every event box and forecast time,
   spatial standard deviation of the forecast / that of ERA5 (1 = as much detail as
   reality, < 1 = smoother). Averaged by lead day, for 2 m temperature and 10 m wind
   speed. Tests the "blurring" explanation of Phase 5 (short-rollout training made
   forecasts sharper and worse at 5 days in squared error) - and smoother forecasts
   cannot reach record values.

Writes results/phase6/noharm_comparison.csv, sharpness.csv, sharpness.png.

    python scripts/61_phase6_summary.py
    python scripts/61_phase6_summary.py --labels R0 R1 R4 PBC
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

from extremes import paths, phase3, phase4

OUT = paths.RESULTS_DIR / "phase6"
T2M, U, V = "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"


def folder(label: str):
    if label == "R0":
        return phase3.FORECASTS
    if label in ("Debias", "Persistence", "PBC"):
        return phase4.OUT / f"forecasts_{label}"
    return OUT / f"forecasts_{label}"


def field(ds, which, q):
    if q == "t2m":
        return ds[f"{which}_{T2M}"]
    return np.hypot(ds[f"{which}_{U}"], ds[f"{which}_{V}"])


def sharpness(label: str) -> pd.DataFrame:
    rows = []
    for f in sorted(folder(label).glob("*_minus*d.nc")):
        ds = xarray.load_dataset(f)
        init = pd.Timestamp(ds.attrs["init"])
        lead_day = np.ceil((pd.DatetimeIndex(ds.time.values) - init) / pd.Timedelta(days=1)).astype(int)
        for q in ("t2m", "wind"):
            sf = field(ds, "forecast", q).std(("lat", "lon")).values
            so = field(ds, "truth", q).std(("lat", "lon")).values
            for d, a, b in zip(lead_day, sf, so):
                rows.append(dict(label=label, event_id=ds.attrs["event_id"], quantity=q, lead_day=int(d),
                                 ratio=a / b if b > 0 else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["R0", "R1", "R2", "R3", "R4"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. ordinary weather
    parts = []
    for lab in a.labels:
        f = phase3.OUT / "noharm.csv" if lab == "R0" else OUT / f"noharm_{lab}.csv"
        if f.exists():
            d = pd.read_csv(f)
            parts.append(d[d.label == lab])
        else:
            print(f"(no ordinary-weather file for {lab}: {f})")
    if parts:
        nh = pd.concat(parts)
        t = nh.groupby(["var", "lead_days", "label"]).rmse.mean().unstack("label")
        t = t[[lab for lab in a.labels if lab in t.columns]]
        t.to_csv(OUT / "noharm_comparison.csv")
        print("1. ordinary weather: global RMSE, mean of 12 dates of 2020 (lower = better)")
        print(t.round(3).to_string())
        if "R0" in t:
            print("\n   change against R0 (%)")
            print((100 * (t.div(t["R0"], axis=0) - 1)).drop(columns="R0").round(1).to_string())

    # 2. sharpness
    sh = pd.concat([sharpness(lab) for lab in a.labels if folder(lab).exists()])
    sh.to_csv(OUT / "sharpness.csv", index=False)
    s = sh.groupby(["quantity", "lead_day", "label"]).ratio.mean().unstack("label")
    s = s[[lab for lab in a.labels if lab in s.columns]]
    print("\n2. sharpness: spatial variability of the forecast / of ERA5 in the event boxes "
          "(1 = as detailed as reality, < 1 = smoother)")
    print(s.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, q in zip(axes, ("t2m", "wind")):
        for lab in s.columns:
            ax.plot(s.loc[q].index, s.loc[q][lab], "-o", label=lab, ms=4)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xlabel("lead time (days)"); ax.set_ylabel("forecast / ERA5 spatial std")
        ax.set_title({"t2m": "2 m temperature", "wind": "10 m wind speed"}[q] + " - sharpness (1 = as detailed as ERA5)")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "sharpness.png", dpi=120)
    print("\nwritten:", OUT / "noharm_comparison.csv", ", sharpness.csv, sharpness.png")


if __name__ == "__main__":
    main()
