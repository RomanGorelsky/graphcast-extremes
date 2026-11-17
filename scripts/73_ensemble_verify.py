"""Phase 7, prospect 4 (Mac): how good are the perturbed-start ensembles?

Reads the R0 forecasts (Phase 3: control member + ERA5 truth) and the members from
72_ensemble_forecasts.py. For every case study and start:

  1. record probability = share of members (control included) that break the cell's
     1979-2015 record, for every land cell and time of the event period in the box -
     scored exactly like Phase 4 check 2 (46_record_probability.py): Brier score, skill
     vs R0's yes/no forecast, share of records warned at >= 20 % / 50 %, share of
     warnings that were right, reliability. Phase 4's "R0+spread" and "PBC" rows are
     printed next to it (from results/phase4/record_probability.csv).
  2. spread vs error: per lead day, the ensemble spread (standard deviation across
     members) and the RMSE of the ensemble mean, over the whole box (all cells and
     times). A well-dispersed ensemble has spread ~ RMSE (ratio ~ 1); GraphCast
     ensembles from perturbed starts are known to be under-dispersed (ratio < 1).
  3. the ensemble mean as a forecast file (results/phase7/forecasts_ENSmean_a<alpha>/),
     so it can be verified on the record cells like any model:
         python scripts/33_verify.py --label ENSmean --forecasts results/phase7/forecasts_ENSmean_a0.1 --no-maps
     (wind: mean u and v, rescaled to the members' mean wind SPEED).

Writes results/phase7/ensemble_a<alpha>_{scores,cases,spread}.csv and ensemble_a<alpha>.png.

    python scripts/73_ensemble_verify.py
    python scripts/73_ensemble_verify.py --alpha 0.2
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

from extremes import paths, phase1, phase3, phase4, probrec, records, verify

T2M, U, V, MSL = "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"
TH_VAR = {"heat": "t2m_max", "cold": "t2m_min", "wind": "wind_max"}


def qty(ds, which, kind_or_q) -> np.ndarray:
    if kind_or_q in ("heat", "cold", "t2m"):
        return ds[f"{which}_{T2M}"].values
    return np.hypot(ds[f"{which}_{U}"].values, ds[f"{which}_{V}"].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--members", type=int, default=20)
    a = ap.parse_args()
    root = paths.set_dir("phase7")
    edir = root / f"ensemble_a{a.alpha:g}"
    mean_dir = root / f"forecasts_ENSmean_a{a.alpha:g}"
    mean_dir.mkdir(parents=True, exist_ok=True)
    cases = phase3.case_studies()
    th = xarray.load_dataset(phase1.thresholds_path())
    pooled, rows, spread_rows = {}, [], []

    for c in cases.itertuples():
        s = verify.SIGN[c.kind]
        for d in phase3.LEADS_DAYS:
            fp = phase3.forecast_path(c.event_id, d)
            mfiles = [edir / f"m{j:02d}" / fp.name for j in range(1, a.members + 1)]
            if not fp.exists() or not all(f.exists() for f in mfiles):
                print(f"{c.event_id} -{d} d: R0 or members missing - skipped"); continue
            R0 = xarray.load_dataset(fp)
            mem = [R0] + [xarray.load_dataset(f) for f in mfiles]
            t_e = pd.Timestamp(R0.attrs["init"])
            times = pd.DatetimeIndex(R0.time.values)

            # 3. ensemble-mean file
            out = R0.copy(deep=True)
            for v in (T2M, MSL):
                out[f"forecast_{v}"].values[:] = np.mean([m[f"forecast_{v}"].values for m in mem], 0)
            mu, mv = (np.mean([m[f"forecast_{x}"].values for m in mem], 0) for x in (U, V))
            mean_speed = np.mean([qty(m, "forecast", "wind") for m in mem], 0)
            scale = np.where(np.hypot(mu, mv) > 0.1, mean_speed / np.maximum(np.hypot(mu, mv), 0.1), 1.0)
            out[f"forecast_{U}"].values[:], out[f"forecast_{V}"].values[:] = mu * scale, mv * scale
            out.attrs.update(model=f"GraphCast_small ensemble mean ({a.members}+1 members, alpha {a.alpha:g})")
            out.to_netcdf(mean_dir / fp.name)

            # 1. record probabilities
            X = np.stack([qty(m, "forecast", c.kind) for m in mem])                    # (M, K, Y, X)
            truth = qty(R0, "truth", c.kind)
            thr = th[TH_VAR[c.kind]].sel(month=xarray.DataArray(times.month, dims="time"))
            thr = thr.sel(lat=R0.lat, lon=R0.lon).transpose("time", "lat", "lon").values
            land = th["analysis_mask"].sel(lat=R0.lat, lon=R0.lon).values.astype(bool)
            in_period = (times > t_e) & (times >= max(t_e, pd.Timestamp(c.start))) & (times <= pd.Timestamp(c.end))
            valid = in_period[:, None, None] & land[None] & np.isfinite(truth) & np.isfinite(thr)
            y = (s * (truth - thr) > 0)[valid]
            p_ens = (s * (X - thr[None]) > 0).mean(0)[valid]
            p_r0 = (s * (X[0] - thr) > 0).astype(float)[valid]
            for name, pv in (("R0", p_r0), ("Ensemble", p_ens)):
                key = (c.kind, d, name)
                pooled.setdefault(key, ([], []))
                pooled[key][0].append(pv); pooled[key][1].append(y)
                rows.append(dict(event_id=c.event_id, kind=c.kind, start_days=d, method=name,
                                 **probrec.scores(pv, y)))

            # 2. spread vs error, whole box, per lead day
            lead_day = np.ceil((times - t_e) / pd.Timedelta(days=1)).astype(int)
            for q in ("t2m", "wind"):
                Z = np.stack([qty(m, "forecast", q) for m in mem])
                tq = qty(R0, "truth", q)
                for ld in np.unique(lead_day):
                    k = lead_day == ld
                    err = Z[:, k].mean(0) - tq[k]
                    spread_rows.append(dict(event_id=c.event_id, kind=c.kind, start_days=d, quantity=q,
                                            lead_day=int(ld), rmse_mean=float(np.sqrt(np.nanmean(err ** 2))),
                                            rmse_R0=float(np.sqrt(np.nanmean((Z[0, k] - tq[k]) ** 2))),
                                            spread=float(np.sqrt(np.nanmean(Z[:, k].var(0, ddof=1))))))
            print(f"{c.event_id} -{d} d: {valid.sum():,} cell-times, {int(y.sum()):,} records; Brier R0 "
                  f"{rows[-2]['brier']:.3f}, ensemble {rows[-1]['brier']:.3f}", flush=True)

    if not rows:
        print("nothing to verify yet"); return
    tag = f"ensemble_a{a.alpha:g}"
    pd.DataFrame(rows).to_csv(root / f"{tag}_cases.csv", index=False)
    t = pd.DataFrame([dict(kind=k, start_days=d, method=m, **probrec.scores(np.concatenate(ps), np.concatenate(ys)))
                      for (k, d, m), (ps, ys) in pooled.items()])
    base = t[t.method == "R0"][["kind", "start_days", "brier"]].rename(columns={"brier": "brier_R0"})
    t = t.merge(base, on=["kind", "start_days"])
    t["skill_vs_R0"] = np.where(t.brier_R0 > 0, 1 - t.brier / t.brier_R0, np.nan)
    p4 = phase4.OUT / "record_probability.csv"
    if p4.exists():                                          # Phase 4 check 2, same cells and scores
        q = pd.read_csv(p4)
        t = pd.concat([t, q[q.method.isin(["R0+spread", "PBC"])][t.columns.intersection(q.columns)]],
                      ignore_index=True)
    t.to_csv(root / f"{tag}_scores.csv", index=False)
    sp = pd.DataFrame(spread_rows)
    sp.to_csv(root / f"{tag}_spread.csv", index=False)

    order = [m for m in ("Ensemble", "R0+spread", "PBC") if m in set(t.method)]
    for col, title in (("skill_vs_R0", "Brier skill vs R0 (> 0 = better probabilities than R0's yes/no)"),
                       ("mean_p_on_records", "average probability given to cells that broke a record"),
                       ("pod_0.2", "share of records warned at probability >= 20%"),
                       ("precision_0.2", "share of those warnings that were right")):
        print(f"\n{title}")
        print(t.pivot_table(index=["kind", "start_days"], columns="method", values=col)[order].round(3).to_string())
    s2 = sp.groupby(["quantity", "lead_day"])[["spread", "rmse_mean", "rmse_R0"]].mean()
    s2["spread_to_error"] = s2.spread / s2.rmse_mean
    print("\nspread vs error over the event boxes (1 = well dispersed, < 1 = over-confident)")
    print(s2[s2.index.get_level_values(1).isin([1, 3, 5, 7, 10])].round(3).to_string())
    summary = {"alpha": a.alpha, "members": a.members + 1,
               "spread_to_error_day5": s2.xs(5, level=1).spread_to_error.round(3).to_dict()
               if 5 in s2.index.get_level_values(1) else None}
    (root / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, kind in enumerate(records.KINDS):
        g = t[t.kind == kind]
        for m in order:
            h = g[g.method == m].sort_values("start_days")
            axes[0, j].plot(h.start_days, h.skill_vs_R0, "-o", label=m)
        axes[0, j].axhline(0, color="k", lw=0.8)
        axes[0, j].set_title(f"{kind}: Brier skill vs R0 (> 0 = better)"); axes[0, j].legend(fontsize=8)
        axes[0, j].set_xlabel("forecast start, days before the peak")
        ps = [np.concatenate(pooled[(kind, d, "Ensemble")][0]) for d in phase3.LEADS_DAYS if (kind, d, "Ensemble") in pooled]
        ys = [np.concatenate(pooled[(kind, d, "Ensemble")][1]) for d in phase3.LEADS_DAYS if (kind, d, "Ensemble") in pooled]
        if ps:
            r = probrec.reliability(np.concatenate(ps), np.concatenate(ys))
            axes[1, j].plot([x[0] for x in r], [x[1] for x in r], "-o", label="Ensemble")
        axes[1, j].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[1, j].set_title(f"{kind}: reliability"); axes[1, j].set_xlabel("forecast probability of a record")
        axes[1, j].set_ylabel("observed frequency")
    fig.tight_layout(); fig.savefig(root / f"{tag}.png", dpi=120)
    print("\nwritten:", root / f"{tag}_scores.csv", f", {tag}_cases.csv, {tag}_spread.csv, {tag}.png;",
          "ensemble-mean files in", mean_dir)


if __name__ == "__main__":
    main()
