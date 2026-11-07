"""Phase 4, step 5 (Mac): R0 vs Debias vs Persistence vs PBC on the record cells.

Reads results/phase3/<label>_summary.json and <label>_record_errors.csv.gz written by
33_verify.py for each label, and results/phase4/harm_check.csv.
Writes results/phase4/comparison.csv and comparison_bias_vs_exceedance.png.

    python scripts/44_compare.py
    python scripts/44_compare.py --labels R0 PBC
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from extremes import phase3, phase4, records, verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["R0", "Debias", "Persistence", "PBC"])
    a = ap.parse_args()
    rows, errs = [], {}
    for lab in a.labels:
        s = json.loads((phase3.OUT / f"{lab}_summary.json").read_text())
        errs[lab] = pd.read_csv(phase3.OUT / f"{lab}_record_errors.csv.gz")
        for kind, per_start in s["by_kind_start"].items():
            for d, v in per_start.items():
                rows.append(dict(label=lab, kind=kind, start_days=int(d), **v))
            for grp, v in s["by_kind_shattering"][kind].items():
                rows.append(dict(label=lab, kind=kind, start_days="all", group=grp, **v))
    t = pd.DataFrame(rows)
    t.to_csv(phase4.OUT / "comparison.csv", index=False)

    for metric, unit in (("mean_intensity_error", "(negative = too mild)"), ("hit_rate", "")):
        print(f"\n{metric} {unit}")
        p = t[t.start_days != "all"].pivot_table(index=["kind", "start_days"], columns="label", values=metric)
        print(p[a.labels].round(2).to_string())
    g = t[t.start_days == "all"].pivot_table(index=["kind", "group"], columns="label", values="mean_intensity_error")
    print("\nmean intensity error, record-shattering vs ordinary\n" + g[a.labels].round(2).to_string())

    hf = phase4.OUT / "harm_check.csv"
    if hf.exists():
        h = pd.read_csv(hf)
        cols = [f"rmse_{lab}" for lab in a.labels if f"rmse_{lab}" in h]
        print("\nbox-wide RMSE (all cells and times; must not get worse)\n" +
              h.groupby(["quantity", "start_days"])[cols].mean().round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, kind in zip(axes, records.KINDS):
        for lab, style in zip(a.labels, ["-o", "--s", ":^", "-D"]):
            e = errs[lab]
            b = verify.by_exceedance(e[e.kind == kind], ["start_days"])
            b = b[b["count"] >= 10].groupby("exc_bin", observed=True)["mean"].mean()
            x = [iv.left + (min(iv.right, 8) - iv.left) / 2 for iv in b.index]
            ax.plot(x, b.values, style, label=lab)
        ax.axhline(0, color="k", lw=0.8); ax.set_title(f"{kind} (mean over starts)")
        ax.set_xlabel(f"observed exceedance ({records.UNITS[kind]})")
        ax.set_ylabel(f"intensity error ({records.UNITS[kind]})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(phase4.OUT / "comparison_bias_vs_exceedance.png", dpi=120)
    print("\nwritten:", phase4.OUT / "comparison.csv", "and comparison_bias_vs_exceedance.png")


if __name__ == "__main__":
    main()
