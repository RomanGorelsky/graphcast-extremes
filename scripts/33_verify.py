"""Phase 3, step 4 (Mac): how much does GraphCast_small (R0) underestimate the case studies?

Needs results/phase3/forecasts/*.nc (from Colab), results/phase2/{case_studies.csv,
events_2020.csv, records_2020_labeled.csv.gz} and the Phase 1 thresholds.

Writes to results/phase3/ (with --label, default R0, so later models get their own files):
  <label>_record_errors.csv.gz  every verified record cell-time
  <label>_case_metrics.csv      one row per event x start: intensity error, RMSE, capture
                                ratio, hit rate, precision
  <label>_summary.json          overall, by kind, by start day, record-shattering vs ordinary, regime
  <label>_bias_vs_exceedance.png   headline curve (cf. Zhang et al. Fig. 2E-G)
  <label>_by_start.png          intensity error and hit rate vs start day, shattering vs ordinary
  event_maps/<event>.png        truth vs forecasts at the peak, record cells outlined

    python scripts/33_verify.py
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

from extremes import paths, phase1, phase3, records, verify

P2 = paths.RESULTS_DIR / "phase2"
UNITS = records.UNITS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="R0")
    ap.add_argument("--forecasts", default=None, help="folder with forecast files (default results/phase3/forecasts)")
    ap.add_argument("--no-maps", action="store_true")
    a = ap.parse_args()
    fdir = phase3.FORECASTS if a.forecasts is None else type(phase3.FORECASTS)(a.forecasts)
    cases = phase3.case_studies()
    ev = pd.read_csv(P2 / "events_2020.csv")
    lab = pd.read_csv(P2 / "records_2020_labeled.csv.gz", parse_dates=["time"])
    th = xarray.load_dataset(phase1.thresholds_path())
    mask = th["analysis_mask"]

    all_err, rows = [], []
    for c in cases.itertuples():
        e = ev[ev.event_id == c.event_id].iloc[0]
        recs = lab[(lab.kind == c.kind) & (lab.event == e.event)]
        for d in phase3.LEADS_DAYS:
            p = fdir / phase3.forecast_path(c.event_id, d).name
            if not p.exists():
                print(f"missing forecast: {p.name}")
                continue
            fc = xarray.load_dataset(p)
            init = pd.Timestamp(fc.attrs["init"])
            err = verify.record_errors(fc, recs, c.kind, init)
            err = err.assign(event_id=c.event_id, start_days=d, record_shattering=bool(e.record_shattering),
                             regime=e.regime)
            all_err.append(err)
            row = dict(event_id=c.event_id, kind=c.kind, start_days=d, init=init,
                       record_shattering=bool(e.record_shattering), regime=e.regime,
                       sector=e.sector, **verify.case_summary(err),
                       **verify.box_exceedances(fc, th, mask, c.kind, max(init, c.start), c.end))
            if len(err):
                row["max_truth_check"] = float(err.truth_check.max())
            rows.append(row)
    err = pd.concat(all_err, ignore_index=True)
    cm = pd.DataFrame(rows)
    out = phase3.OUT
    err.to_csv(out / f"{a.label}_record_errors.csv.gz", index=False)
    cm.to_csv(out / f"{a.label}_case_metrics.csv", index=False)

    tc = float(err.truth_check.max()) if len(err) else float("nan")
    print(f"{len(cm)} forecasts, {len(err):,} verified record cell-times; "
          f"truth check (ERA5 here vs Phase 1 records): max diff {tc:.2e} "
          f"{'OK' if tc < 1e-3 else '- MISMATCH, tell Claude'}")

    # ---- summary
    def agg(g):
        return dict(n=int(len(g)), mean_intensity_error=round(float(g.intensity_error.mean()), 3),
                    hit_rate=round(float(g.hit.mean()), 3),
                    capture_ratio=round(float(g.fc_exceedance.mean() / g.obs_exceedance.mean()), 3))
    summary = {"label": a.label, "by_kind": {}, "by_kind_start": {}, "by_kind_shattering": {}, "by_regime": {}}
    for kind, g in err.groupby("kind"):
        summary["by_kind"][kind] = agg(g)
        summary["by_kind_start"][kind] = {int(d): agg(h) for d, h in g.groupby("start_days")}
        summary["by_kind_shattering"][kind] = {("record-shattering" if s else "ordinary"): agg(h)
                                               for s, h in g.groupby("record_shattering")}
    for reg, g in err.groupby("regime"):
        summary["by_regime"][reg] = agg(g)
    (out / f"{a.label}_summary.json").write_text(json.dumps(summary, indent=1))
    print("\nintensity error on record cells (negative = forecast too mild), hit rate, capture ratio:")
    for kind in records.KINDS:
        if kind not in summary["by_kind"]:
            continue
        s = summary["by_kind"][kind]
        print(f"  {kind}: {s['mean_intensity_error']:+.2f} {UNITS[kind]}, hit {s['hit_rate']:.0%}, "
              f"capture {s['capture_ratio']:.2f}  (n={s['n']:,})")
        for d, v in summary["by_kind_start"][kind].items():
            print(f"      start -{d} d: {v['mean_intensity_error']:+.2f}, hit {v['hit_rate']:.0%}, "
                  f"capture {v['capture_ratio']:.2f}")
        for k, v in summary["by_kind_shattering"][kind].items():
            print(f"      {k:17s}: {v['mean_intensity_error']:+.2f}, hit {v['hit_rate']:.0%}")

    # ---- figure 1: bias vs exceedance
    bx = verify.by_exceedance(err, ["kind", "start_days"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, kind in zip(axes, records.KINDS):
        for d in phase3.LEADS_DAYS:
            b = bx[(bx.kind == kind) & (bx.start_days == d) & (bx["count"] >= 10)]
            if len(b):
                x = [iv.left + (min(iv.right, 8) - iv.left) / 2 for iv in b.exc_bin]
                ax.plot(x, b["mean"], marker="o", label=f"start -{d} d")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(f"observed exceedance of the record ({UNITS[kind]})")
        ax.set_ylabel(f"intensity error ({UNITS[kind]}), < 0 = too mild")
        ax.set_title(kind); ax.legend(fontsize=8)
    fig.suptitle(f"{a.label}: error on record cells vs how far the record was broken (bins with >= 10 values)")
    fig.tight_layout(); fig.savefig(out / f"{a.label}_bias_vs_exceedance.png", dpi=120); plt.close(fig)

    # ---- figure 2: by start day, shattering vs ordinary
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for j, kind in enumerate(records.KINDS):
        g = err[err.kind == kind]
        for flag, style in ((True, "-o"), (False, "--s")):
            h = g[g.record_shattering == flag].groupby("start_days")
            lbl = "record-shattering" if flag else "ordinary"
            if len(g[g.record_shattering == flag]):
                axes[0, j].plot(h.intensity_error.mean(), style, label=lbl)
                axes[1, j].plot(h.hit.mean() * 100, style, label=lbl)
        axes[0, j].axhline(0, color="k", lw=0.8); axes[0, j].set_title(kind)
        axes[0, j].set_ylabel(f"intensity error ({UNITS[kind]})"); axes[1, j].set_ylabel("record hit rate (%)")
        axes[1, j].set_xlabel("forecast start, days before the peak"); axes[1, j].set_ylim(0, 100)
        axes[0, j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"{a.label}_by_start.png", dpi=120); plt.close(fig)

    # ---- figure 3: event maps
    if not a.no_maps:
        (out / "event_maps").mkdir(exist_ok=True)
        for c in cases.itertuples():
            files = [fdir / phase3.forecast_path(c.event_id, d).name for d in phase3.LEADS_DAYS]
            files = [f for f in files if f.exists()]
            if not files:
                continue
            e = ev[ev.event_id == c.event_id].iloc[0]
            recs = lab[(lab.kind == c.kind) & (lab.event == e.event) & (lab.time == pd.Timestamp(c.peak_time))]
            dsets = [xarray.load_dataset(f) for f in files]
            peak = pd.Timestamp(c.peak_time)
            truth = verify.field(dsets[0], "truth", c.kind).sel(time=peak)
            fcs = [verify.field(x, "forecast", c.kind).sel(time=peak) for x in dsets]
            vals = np.concatenate([truth.values.ravel()] + [f.values.ravel() for f in fcs])
            lo, hi = np.nanpercentile(vals, [2, 98])
            fig, axes = plt.subplots(1, 1 + len(fcs), figsize=(3.6 * (1 + len(fcs)), 3.4), squeeze=False)
            for ax, da, title in zip(axes[0], [truth] + fcs, ["ERA5 at peak"] +
                                     [f"start -{x.attrs['lead_days_before_peak']} d" for x in dsets]):
                lon = da.lon.values
                wraps = c.box_lon_min > c.box_lon_max            # box crosses 0 deg
                order = np.argsort(np.where(lon >= c.box_lon_min, lon, lon + 360) if wraps else lon)
                im = ax.pcolormesh(np.arange(lon.size), da.lat, da.values[:, order],
                                   cmap="RdYlBu_r" if c.kind != "wind" else "viridis", vmin=lo, vmax=hi, shading="auto")
                pos = {v: i for i, v in enumerate(lon[order])}
                ax.scatter([pos[x] for x in recs.lon], recs.lat, s=6, facecolor="none", edgecolor="k", lw=0.5)
                ax.set_title(title, fontsize=9); ax.set_xticks([])
            fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="K" if c.kind != "wind" else "m/s")
            fig.suptitle(f"{c.event_id} ({c.kind}), peak {peak:%Y-%m-%d %H}h; circles = record cells at the peak",
                         fontsize=10)
            fig.savefig(out / "event_maps" / f"{c.event_id}.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("\nwritten:", out)


if __name__ == "__main__":
    main()
