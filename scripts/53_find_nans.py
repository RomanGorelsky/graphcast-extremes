"""Phase 5 diagnosis (cluster, CPU job, ~10-20 min): where are missing values (NaN) in
the staged data?

For every staged month and every flux variable (plus precipitation and 2 m
temperature as controls): how many time steps contain NaN, how many cells are NaN at
those times, and the first few affected time stamps. Writes results/phase5/nan_scan.csv.

    python scripts/53_find_nans.py
"""
import numpy as np
import pandas as pd
import xarray

from extremes import phase5, physics

VARS = list(physics.FLUX_VARS) + [physics.PRECIP, "2m_temperature"]


def main():
    files = sorted(phase5.STAGED.glob("????-??.zarr"))
    rows = []
    for f in files:
        ds = xarray.open_zarr(f)
        for v in VARS:
            x = ds[v].values                                         # (time, lat, lon)
            bad = np.isnan(x).reshape(len(x), -1)
            per_time = bad.sum(1)
            if per_time.any():
                t = pd.DatetimeIndex(ds.time.values)[per_time > 0]
                rows.append(dict(month=f.name[:7], var=v, times_with_nan=int((per_time > 0).sum()),
                                 times_in_month=len(x), max_cells_nan=int(per_time.max()),
                                 cells=bad.shape[1], first_times=" ".join(str(s)[:13] for s in t[:4])))
        print(f.name[:7], "checked", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(phase5.OUT / "nan_scan.csv", index=False)
    if out.empty:
        print("\nno missing values in any staged month")
        return
    print("\nmonths with missing values:")
    print(out.to_string(index=False))
    print("\nper variable: months affected, time steps affected")
    print(out.groupby("var").agg(months=("month", "count"), times=("times_with_nan", "sum")).to_string())


if __name__ == "__main__":
    main()
