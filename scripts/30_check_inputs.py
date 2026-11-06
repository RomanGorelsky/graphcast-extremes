"""Phase 3, step 1 (Colab): do our ERA5 inputs equal the model's own data?

Rebuilds the authors' 1-degree example batch (2022-01-01) from the WeatherBench2
0.25-degree store and compares, variable by variable:
  * all surface and pressure-level variables at the two input times,
  * top-of-atmosphere solar radiation at all times,
  * the static fields.
Every variable must match (max |difference| tiny relative to its spread).
Then runs ONE forecast step from both batches and compares the predictions.
Writes results/phase3/input_check.json.

    python scripts/30_check_inputs.py
"""
import json

import numpy as np

from extremes import era5, inputs, model, paths

OUT = paths.RESULTS_DIR / "phase3"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref, name = model.load_example(min_steps=1)
    t = ref["datetime"].values[0]
    init = t[1]                                         # second time = last input
    n_steps = len(t) - 2
    print(f"example {name}: init {str(init)[:16]}, {n_steps} steps", flush=True)

    ds = era5.open_store(era5.WB2_ERA5)
    missing = [v for v in inputs.SURFACE + inputs.ATMOS + [inputs.TISR] + inputs.STATIC if v not in ds]
    print("variables missing in the store:", missing or "none", flush=True)
    ours = inputs.build_batch(ds, init, n_steps)

    report, ok = {"example": name, "variables": {}}, True
    for var in inputs.SURFACE + inputs.ATMOS + [inputs.TISR] + inputs.STATIC:
        if var not in ours:
            report["variables"][var] = "missing in our batch (model computes it)" if var == inputs.TISR else "MISSING"
            ok &= var == inputs.TISR
            print(f"  {var:32s} {report['variables'][var]}")
            continue
        a, b = ours[var], ref[var]
        if "time" in a.dims and var != inputs.TISR:      # only the two input times
            a, b = a.isel(time=[0, 1]), b.isel(time=[0, 1])
        if "level" in b.dims:
            b = b.sel(level=inputs.LEVELS)
        a, b = a.transpose(*b.dims).values.astype(np.float64), b.values.astype(np.float64)
        d = np.abs(a - b)
        spread = float(np.nanstd(b)) or 1.0
        rel = float(np.nanmax(d)) / spread
        match = rel < 1e-3
        ok &= match
        report["variables"][var] = {"max_abs_diff": float(np.nanmax(d)), "rmse": float(np.sqrt(np.nanmean(d ** 2))),
                                    "max_diff_over_std": rel, "match": bool(match)}
        print(f"  {var:32s} max|diff| {np.nanmax(d):.4g}  ({rel:.1e} of its std)  {'OK' if match else 'MISMATCH'}",
              flush=True)

    # one forecast step from both batches
    m = model.load_model()
    preds, forcings = [], []
    for name_, batch in (("example", ref), ("ours", ours)):
        i, tg, f = m.extract(batch, 1)          # first 3 times; computes solar radiation if missing
        forcings.append(f)
        p = m.forecast(i, tg, f)
        preds.append(p)
        n_in = {v: int(np.isnan(i[v].values).sum()) for v in i.data_vars if np.isnan(i[v].values).any()}
        n_f = {v: int(np.isnan(f[v].values).sum()) for v in f.data_vars if np.isnan(f[v].values).any()}
        n_p = int(sum(np.isnan(p[v].values).sum() for v in p.data_vars))
        print(f"  {name_:8s} NaN in model inputs: {n_in or 'none'} | in forcings: {n_f or 'none'} | "
              f"in the forecast: {n_p:,}", flush=True)
        report[f"nan_{name_}"] = {"inputs": n_in, "forcings": n_f, "forecast": n_p}

    # solar radiation: example (ERA5) vs what we feed / the library computes
    a = forcings[0][inputs.TISR].values.astype(np.float64)
    b = forcings[1][inputs.TISR].values.astype(np.float64)
    d = np.abs(a - b)
    report["tisr_vs_example"] = {"max_abs_diff": float(np.nanmax(d)), "rmse": float(np.sqrt(np.nanmean(d ** 2))),
                                 "max_diff_over_std": float(np.nanmax(d) / np.nanstd(a)),
                                 "nan_ours": int(np.isnan(b).sum())}
    print("  solar radiation, ours vs example:", report["tisr_vs_example"], flush=True)
    same_grid = all(np.array_equal(preds[0][c].values, preds[1][c].values) for c in ("lat", "lon", "time"))
    dmax = {v: float(np.nanmax(np.abs(preds[0][v].values - preds[1][v].values)))
            for v in ("2m_temperature", "10m_u_component_of_wind")}
    report["one_step_prediction_max_diff"] = dmax
    report["same_output_grid"] = bool(same_grid)
    report["all_match"] = bool(ok)
    print("one-step forecast, max |difference| between the two batches (ignoring NaN):", dmax,
          "| same output grid:", same_grid)
    (OUT / "input_check.json").write_text(json.dumps(report, indent=1))
    print("\nALL INPUTS MATCH" if ok else "\nMISMATCH - send this output to Claude before running forecasts")


if __name__ == "__main__":
    main()
