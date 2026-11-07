"""Phase 4, step 4 (Mac): learn the corrections from the hindcasts and apply them.

For each case study and start (-1/-3/-5/-7 d): load the R0 forecast (Phase 3), the
training hindcasts (starts within +-7 days of its calendar date, 2016-2019) and the
ERA5 truth cache, then correct 2 m temperature and 10 m wind speed per cell and lead
step with Debias, Persistence and their average (PBC). Wind: the speed is corrected
and u, v are rescaled to it (direction unchanged). Sea-level pressure is left as is.

Writes results/phase4/forecasts_{Debias,Persistence,PBC}/<event>_minus<d>d.nc in the
Phase 3 format, so they are verified exactly like R0:
    python scripts/33_verify.py --label PBC --forecasts results/phase4/forecasts_PBC --no-maps
and results/phase4/harm_check.csv: RMSE over ALL cells and times of each box (not
only record cells) - a correction must not make ordinary weather in the box worse.

    python scripts/43_apply_pbc.py
"""
import argparse

import numpy as np
import pandas as pd
import xarray

from extremes import pbc, phase3, phase4

T2M, U, V = "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"
METHODS = ("Debias", "Persistence", "PBC")


def quantity(ds, which: str, q: str) -> xarray.DataArray:
    if q == "t2m":
        return ds[f"{which}_{T2M}"]
    return np.hypot(ds[f"{which}_{U}"], ds[f"{which}_{V}"])


def rmse(a, b) -> float:
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hindcasts", type=int, default=20,
                    help="skip a forecast if fewer training hindcasts exist (yet)")
    a = ap.parse_args()
    cases = phase3.case_studies()
    for m in METHODS:
        (phase4.OUT / f"forecasts_{m}").mkdir(parents=True, exist_ok=True)
    harm, counts = [], []

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
            counts.append(dict(event_id=c.event_id, start_days=d, n_hindcasts=len(inits)))
            if len(inits) < a.min_hindcasts:
                print(f"{c.event_id} -{d} d: only {len(inits)} hindcasts - skipped"); continue
            H = [xarray.load_dataset(phase4.hindcast_path(c.event_id, t)) for t in inits]
            leads = pd.to_timedelta(F.time.values - np.datetime64(t_e))

            out = {m: F.copy(deep=True) for m in METHODS}
            for q in ("t2m", "wind"):
                Fq = quantity(F, "forecast", q).values                                # (K, Y, X)
                Hq = np.stack([quantity(h, "forecast", q).values for h in H])         # (J, K, Y, X)
                Oq = np.stack([quantity(T.sel(time=t + leads), "truth", q).values for t in inits])
                H0 = np.stack([quantity(T.sel(time=[t]), "truth", q).values[0] for t in inits])
                O0 = quantity(T.sel(time=[t_e]), "truth", q).values[0]
                res = pbc.correct(Fq, O0, Hq, Oq, H0)
                truth_q = quantity(F, "truth", q).values
                row = dict(event_id=c.event_id, kind=c.kind, start_days=d, quantity=q,
                           n_hindcasts=len(inits), rmse_R0=rmse(Fq, truth_q))
                for m, X in res.items():
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
            print(f"{c.event_id} -{d} d: {len(inits)} hindcasts; t2m box RMSE R0 "
                  f"{harm[-2]['rmse_R0']:.2f} -> PBC {harm[-2]['rmse_PBC']:.2f} K; wind "
                  f"{harm[-1]['rmse_R0']:.2f} -> {harm[-1]['rmse_PBC']:.2f} m/s", flush=True)

    pd.DataFrame(counts).to_csv(phase4.OUT / "training_counts.csv", index=False)
    h = pd.DataFrame(harm)
    h.to_csv(phase4.OUT / "harm_check.csv", index=False)
    if len(h):
        print("\nbox-wide RMSE (all cells and times), mean over forecasts:")
        print(h.groupby("quantity")[[f"rmse_{m}" for m in ("R0",) + METHODS]].mean().round(3).to_string())
    print("\nnext, on the Mac:")
    for m in METHODS:
        print(f"  python scripts/33_verify.py --label {m} --forecasts results/phase4/forecasts_{m} --no-maps")
    print("  python scripts/44_compare.py")


if __name__ == "__main__":
    main()
