"""Phase 1, step 1: which ERA5 source on the 1-degree grid matches GraphCast_small's data?

The authors' 1-degree example file (2022-01-01) was made from ERA5 the same way
as the model's training data. We take the same times and compare it with three
candidates:

  wb2-1deg      WeatherBench2's ready-made 1-degree store (conservative regridding)
  subsample     the 0.25-degree store, every 4th point
  conservative  the 0.25-degree store, our own area average over each 1-degree cell

The best one (near-zero difference) is written to results/phase1/regrid_check.json
and used by all later Phase 1 scripts.

    python scripts/10_check_regridding.py --list     # only show what is in the bucket
    python scripts/10_check_regridding.py
"""
import argparse
import json

import numpy as np

from extremes import era5, model, paths

VARS = (era5.T2M, era5.U10, era5.V10)
CANDIDATES = {                       # name -> (store, method)
    "wb2-1deg": (era5.WB2_ERA5_1DEG, "as-stored"),
    "subsample": (era5.WB2_ERA5, "subsample"),
    "conservative": (era5.WB2_ERA5, "conservative"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list ERA5 stores in the bucket and exit")
    a = ap.parse_args()
    if a.list:
        for p in era5.list_bucket():
            print(p)
        return

    example, name = model.load_example(min_steps=1)
    example = example.isel(batch=0)
    times = example["datetime"].values
    print("example:", name, "| times:", str(times[0])[:16], "...", str(times[-1])[:16], f"({len(times)})")

    report = {"example": name, "candidates": {}}
    stores = {}
    for cand, (store, method) in CANDIDATES.items():
        if store not in stores:
            stores[store] = era5.open_store(store)
            d = stores[store]
            print(f"store {store.split('/')[-1]}: {d.sizes['lat']} x {d.sizes['lon']}, "
                  f"{str(d.time.values[0])[:10]} -> {str(d.time.values[-1])[:10]}")
        res = {}
        for var in VARS:
            ours = era5.load_1deg(stores[store], var, times, method).values
            ref = example[var].transpose("time", "lat", "lon").values
            dif = ours - ref
            res[var] = {"rmse": float(np.sqrt(np.mean(dif ** 2))), "max_abs": float(np.abs(dif).max())}
            print(f"  {cand:12s} {var:26s} RMSE {res[var]['rmse']:.5f}  max|diff| {res[var]['max_abs']:.5f}")
        report["candidates"][cand] = {"store": store, "method": method, "variables": res,
                                      "mean_rmse": float(np.mean([r["rmse"] for r in res.values()]))}

    best = min(report["candidates"], key=lambda c: report["candidates"][c]["mean_rmse"])
    b = report["candidates"][best]
    report.update(chosen=best, chosen_store=b["store"], chosen_method=b["method"],
                  exact_match=bool(b["mean_rmse"] < 0.01))
    out = paths.RESULTS_DIR / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "regrid_check.json").write_text(json.dumps(report, indent=1))
    print(f"\nchosen: {best}  (mean RMSE {b['mean_rmse']:.5f}; others: " +
          ", ".join(f"{c} {v['mean_rmse']:.4f}" for c, v in report["candidates"].items() if c != best) + ")")
    print("exact match - same processing as the training data" if report["exact_match"] else
          "WARNING: no exact match; the closest source is used - tell Claude these numbers")


if __name__ == "__main__":
    main()
