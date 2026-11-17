"""Phase 7, prospect 1 (Mac, no GPU): tail-targeted correction by quantile mapping.

Same inputs as 43_apply_pbc.py (the R0 forecast of each case study and start, the
2016-2019 training hindcasts and the ERA5 truth cache), different correction - see
extremes/tailcorr.py:
  QM          full quantile mapping (the whole distribution of the hindcasts is
              matched to ERA5's)
  TailQM      the same mapping applied only in the tails (beyond the 85th / 15th
              percentile, fully beyond the 95th / 5th); typical days stay as R0
  TailQM-reg  TailQM learned only from hindcasts that started in the same K-means
              weather regime (Phase 2) as the forecast, in the event's sector
              (falls back to all hindcasts if fewer than --min-regime are left,
              and for events outside the regime sectors)
2 m temperature: both tails (heat and cold records); 10 m wind: speed corrected in the
upper tail, u and v rescaled to it (direction unchanged).

Writes results/phase4/forecasts_{QM,TailQM,TailQM-reg}/<event>_minus<d>d.nc (Phase 3
format) and results/phase4/harm_check_tail.csv (box-wide RMSE, all cells and times).
Then, as for PBC:
    for m in QM TailQM TailQM-reg; do python scripts/33_verify.py --label $m --forecasts results/phase4/forecasts_$m --no-maps; done
    python scripts/44_compare.py --labels R0 PBC QM TailQM TailQM-reg --tag _tail
    python scripts/45_robustness.py --labels R0 PBC QM TailQM TailQM-reg --tag _tail

    python scripts/71_tail_correction.py
    EXTREMES_SET=ext python scripts/71_tail_correction.py        # the extended event set
"""
import argparse

import numpy as np
import pandas as pd
import xarray

from extremes import paths, phase3, phase4, tailcorr

T2M, U, V = "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"
METHODS = ("QM", "TailQM", "TailQM-reg")


def quantity(ds, which: str, q: str) -> xarray.DataArray:
    if q == "t2m":
        return ds[f"{which}_{T2M}"]
    return np.hypot(ds[f"{which}_{U}"], ds[f"{which}_{V}"])


def rmse(a, b) -> float:
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_regimes():
    f = paths.RESULTS_DIR / "phase2" / "regimes_daily.csv"
    if not f.exists():
        print("(no results/phase2/regimes_daily.csv - TailQM-reg falls back to TailQM)")
        return None
    r = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    r.index = r.index.normalize()
    return r


def regime_on(reg, sector: str, t) -> int | None:
    if reg is None or sector not in reg.columns:
        return None
    day = pd.Timestamp(t).normalize()
    return int(reg.at[day, sector]) if day in reg.index else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hindcasts", type=int, default=20)
    ap.add_argument("--min-regime", type=int, default=15, help="fewest same-regime hindcasts for TailQM-reg")
    ap.add_argument("--radius", type=int, default=1, help="neighbouring cells pooled in the fit (1 = 3 x 3)")
    a = ap.parse_args()
    cases = phase3.case_studies()
    reg = load_regimes()
    for m in METHODS:
        (phase4.OUT / f"forecasts_{m}").mkdir(parents=True, exist_ok=True)
    harm = []

    for c in cases.itertuples():
        tf = phase4.truth_path(c.event_id)
        if not tf.exists():
            print(f"{c.event_id}: no truth cache yet - skipped"); continue
        T = xarray.load_dataset(tf)
        for d in phase3.LEADS_DAYS:
            fp = phase3.forecast_path(c.event_id, d)
            if not fp.exists():
                print(f"{fp.name}: no R0 forecast - skipped"); continue
            F = xarray.load_dataset(fp)
            t_e = pd.Timestamp(F.attrs["init"])
            inits = [t for t in phase4.training_inits(t_e) if phase4.hindcast_path(c.event_id, t).exists()]
            if len(inits) < a.min_hindcasts:
                print(f"{c.event_id} -{d} d: only {len(inits)} hindcasts - skipped"); continue
            r_now = regime_on(reg, c.sector, t_e)
            same = [t for t in inits if r_now is not None and regime_on(reg, c.sector, t) == r_now]
            use_reg = len(same) >= a.min_regime
            H = {t: xarray.load_dataset(phase4.hindcast_path(c.event_id, t)) for t in inits}
            leads = pd.to_timedelta(F.time.values - np.datetime64(t_e))

            out = {m: F.copy(deep=True) for m in METHODS}
            for q in ("t2m", "wind"):
                Fq = quantity(F, "forecast", q).values                                        # (K, Y, X)
                truth_q = quantity(F, "truth", q).values

                def stack(ts):
                    Hq = np.stack([quantity(H[t], "forecast", q).values for t in ts])         # (J, K, Y, X)
                    Oq = np.stack([quantity(T.sel(time=t + leads), "truth", q).values for t in ts])
                    return Hq, Oq
                tails = dict(upper=True, lower=(q == "t2m"))
                Hq, Oq = stack(inits)
                res = tailcorr.correct(Fq, Hq, Oq, radius=a.radius, **tails)
                if use_reg:
                    res["TailQM-reg"] = tailcorr.correct(Fq, *stack(same), radius=a.radius, **tails)["TailQM"]
                else:
                    res["TailQM-reg"] = res["TailQM"]
                row = dict(event_id=c.event_id, kind=c.kind, start_days=d, quantity=q, n_hindcasts=len(inits),
                           n_same_regime=len(same), regime_used=use_reg, rmse_R0=rmse(Fq, truth_q))
                for m in METHODS:
                    X = res[m]
                    row[f"rmse_{m}"] = rmse(X, truth_q)
                    if q == "t2m":
                        out[m][f"forecast_{T2M}"].values[:] = X
                    else:
                        speed = np.maximum(X, 0.0)
                        scale = np.where(Fq > 0.1, speed / np.maximum(Fq, 0.1), 1.0)
                        out[m][f"forecast_{U}"].values[:] = F[f"forecast_{U}"].values * scale
                        out[m][f"forecast_{V}"].values[:] = F[f"forecast_{V}"].values * scale
                harm.append(row)
            for m in METHODS:
                o = out[m]
                o.attrs.update(model=f"GraphCast_small (R0) + {m}", n_hindcasts=len(inits))
                o.to_netcdf(phase4.OUT / f"forecasts_{m}" / fp.name)
            t2, wd = harm[-2], harm[-1]
            print(f"{c.event_id} -{d} d: {len(inits)} hindcasts ({len(same)} same regime"
                  f"{'' if use_reg else ', too few - all used'}); box RMSE t2m R0 {t2['rmse_R0']:.2f} -> "
                  f"TailQM {t2['rmse_TailQM']:.2f} K, wind {wd['rmse_R0']:.2f} -> {wd['rmse_TailQM']:.2f} m/s",
                  flush=True)

    h = pd.DataFrame(harm)
    h.to_csv(phase4.OUT / "harm_check_tail.csv", index=False)
    if len(h):
        print("\nbox-wide RMSE (all cells and times), mean over forecasts:")
        print(h.groupby("quantity")[[f"rmse_{m}" for m in ("R0",) + METHODS]].mean().round(3).to_string())
    print("\nnext, on the Mac:")
    for m in METHODS:
        print(f"  python scripts/33_verify.py --label {m} --forecasts {phase4.OUT / ('forecasts_' + m)} --no-maps")
    print("  python scripts/44_compare.py --labels R0 PBC QM TailQM TailQM-reg --tag _tail")
    print("  python scripts/45_robustness.py --labels R0 PBC QM TailQM TailQM-reg --tag _tail")


if __name__ == "__main__":
    main()
