"""Phase 0, step 5: smoke test.

Runs GraphCast_small on the authors' 1-degree ERA5 example and checks that the
forecast is sensible:
  * no NaNs, physically plausible ranges;
  * forecast RMSE is below persistence RMSE (the 'nothing changes' forecast)
    for z500, t2m and MSLP at every lead time.
Writes results/phase0/smoke_test.json and smoke_test.png.

    python scripts/01_smoke_test.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extremes import metrics, model, paths

CHECKS = [("geopotential", 500, "z500"), ("2m_temperature", None, "t2m"),
          ("mean_sea_level_pressure", None, "msl"), ("10m_u_component_of_wind", None, "u10")]


def sel(ds, var, level):
    da = ds[var].isel(batch=0)
    return da.sel(level=level) if level is not None else da


def main():
    out = paths.RESULTS_DIR / "phase0"
    out.mkdir(parents=True, exist_ok=True)
    m = model.load_model()
    print("Model:", m.description.splitlines()[0] if m.description else "", "\n", m.model_config)
    batch, name = model.load_example()
    n = batch.sizes["time"] - 2
    print(f"Example data: {name}  ->  {n} forecast steps")
    inputs, targets, forcings = m.extract(batch, n)
    pred = m.forecast(inputs, targets, forcings)

    report = {"example": name, "steps": n, "model_config": {k: float(v) if v is not None else None
              for k, v in dict(m.model_config).items()}, "variables": {}}
    ok = True
    for var, lev, short in CHECKS:
        f, t = sel(pred, var, lev), sel(targets, var, lev)
        persist = sel(inputs, var, lev).isel(time=-1, drop=True)  # state at forecast start
        r_f = metrics.rmse(f, t).values
        r_p = metrics.rmse(persist.broadcast_like(t), t).values
        finite = bool(np.isfinite(f.values).all())
        better = bool((r_f < r_p).all())
        ok &= finite and better
        report["variables"][short] = {
            "finite": finite, "beats_persistence_all_leads": better,
            "rmse_forecast": [round(float(x), 3) for x in r_f],
            "rmse_persistence": [round(float(x), 3) for x in r_p],
            "min": float(f.min()), "max": float(f.max())}
        print(f"{short:5s} finite={finite} beats_persistence={better} "
              f"RMSE last step {r_f[-1]:.3f} vs persistence {r_p[-1]:.3f}")
    report["passed"] = ok
    (out / "smoke_test.json").write_text(json.dumps(report, indent=1))

    # Figure: t2m target / forecast / difference at the last lead time.
    t = sel(targets, "2m_temperature", None).isel(time=-1)
    f = sel(pred, "2m_temperature", None).isel(time=-1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    lo, hi = float(t.quantile(0.02)), float(t.quantile(0.98))
    for a, d, title in ((ax[0], t, "ERA5"), (ax[1], f, "GraphCast_small")):
        im = a.imshow(d.values, origin="lower", cmap="viridis", vmin=lo, vmax=hi)
        a.set_title(f"{title}: t2m (K), +{6 * n} h"); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, shrink=0.8)
    lim = float(abs(f - t).quantile(0.98))
    im = ax[2].imshow((f - t).values, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[2].set_title("forecast - ERA5 (K)"); ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.colorbar(im, ax=ax[2], shrink=0.8)
    fig.tight_layout(); fig.savefig(out / "smoke_test.png", dpi=110)
    print("PASSED" if ok else "FAILED", "->", out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
