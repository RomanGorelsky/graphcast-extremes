"""Phase 1, step 1: which 0.25 -> 1 degree conversion matches GraphCast_small's data?

The authors' 1-degree example file (2022-01-01) was made from ERA5 the same way
as the model's training data. We take the same times from the 0.25-degree
WeatherBench2 store, convert them both ways ('subsample', 'conservative') and
compare. The method with (near-)zero difference is written to
results/phase1/regrid_check.json and used by all later Phase 1 scripts.

    python scripts/10_check_regridding.py --list     # only show what is in the bucket
    python scripts/10_check_regridding.py
"""
import argparse
import json

import numpy as np

from extremes import era5, model, paths

VARS = (era5.T2M, era5.U10, era5.V10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=era5.WB2_ERA5)
    ap.add_argument("--list", action="store_true", help="list ERA5 stores in the bucket and exit")
    a = ap.parse_args()
    if a.list:
        for p in era5.list_bucket():
            print(p)
        return

    example, name = model.load_example(min_steps=1)
    example = example.isel(batch=0)
    times = example["datetime"].values
    print("example:", name, "| times:", [str(t)[:16] for t in times])

    ds = era5.open_store(a.store)
    print("store:", a.store, "| grid:", ds.sizes.get("lat"), "x", ds.sizes.get("lon"),
          "| time range:", str(ds.time.values[0])[:10], "->", str(ds.time.values[-1])[:10])

    report = {"store": a.store, "example": name, "variables": {}}
    for var in VARS:
        ref = example[var].values                                     # (time, 181, 360)
        report["variables"][var] = {}
        for method in ("subsample", "conservative"):
            ours = era5.load_1deg(ds, var, times, method).values
            d = ours - ref
            r = {"rmse": float(np.sqrt(np.mean(d ** 2))), "max_abs": float(np.abs(d).max())}
            report["variables"][var][method] = r
            print(f"  {var:28s} {method:12s} RMSE {r['rmse']:.4f}  max|diff| {r['max_abs']:.4f}")

    score = {m: np.mean([report["variables"][v][m]["rmse"] for v in VARS])
             for m in ("subsample", "conservative")}
    best = min(score, key=score.get)
    report["mean_rmse"] = score
    report["chosen_method"] = best
    exact = score[best] < 0.01
    report["exact_match"] = bool(exact)
    out = paths.RESULTS_DIR / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "regrid_check.json").write_text(json.dumps(report, indent=1))
    print(f"\nchosen method: {best}  (mean RMSE {score[best]:.4f} vs {max(score.values()):.4f})")
    print("exact match - same processing as the training data" if exact else
          "WARNING: no exact match; the closer method is used - tell Claude these numbers")


if __name__ == "__main__":
    main()
