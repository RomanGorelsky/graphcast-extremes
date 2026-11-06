"""Phase 2, step 0 (Colab): daily 500 hPa geopotential, 1979-2020, for the weather regimes.

Weather regimes are large-scale patterns, so a coarse grid is enough: WeatherBench2's
2.8-degree ERA5 store (128 x 64), 00 UTC of every day. Read year by year (resumable)
and saved compactly (int16, 0.5 m2/s2 steps) to
results/phase2/z500_daily_1979-2020_2p8deg.nc (~100-200 MB).

    python scripts/20_fetch_z500.py
"""
import argparse
import time

import dask
import numpy as np
import xarray

from extremes import era5, paths

STORE = "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_conservative.zarr"
OUT = paths.RESULTS_DIR / "phase2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[1979, 2020])
    ap.add_argument("--io-threads", type=int, default=32)
    a = ap.parse_args()
    parts = OUT / "z500_parts"
    parts.mkdir(parents=True, exist_ok=True)
    ds = era5.open_store(STORE)
    z = ds["geopotential"].sel(level=500)
    print("store:", STORE.split("/")[-1], "| grid", ds.sizes["lat"], "x", ds.sizes["lon"], flush=True)
    for year in range(a.years[0], a.years[1] + 1):
        f = parts / f"{year}.nc"
        if f.exists():
            continue
        t0 = time.perf_counter()
        t = ds.time.sel(time=str(year)).values
        t = t[(t.astype("datetime64[h]").astype(int) % 24) == 0]          # 00 UTC only
        with dask.config.set(scheduler="threads", num_workers=a.io_threads):
            y = z.sel(time=t).astype(np.float32).transpose("time", "lat", "lon").load()
        y.drop_vars([c for c in y.coords if c not in ("time", "lat", "lon")]).to_netcdf(f)
        print(f"{year}: {y.sizes['time']} days in {time.perf_counter() - t0:.0f} s", flush=True)

    files = [parts / f"{y}.nc" for y in range(a.years[0], a.years[1] + 1)]
    allz = xarray.concat([xarray.load_dataarray(p) for p in files], "time").rename("z500")
    out = OUT / f"z500_daily_{a.years[0]}-{a.years[1]}_2p8deg.nc"
    allz.to_netcdf(out, encoding={"z500": {"dtype": "int16", "scale_factor": 0.5, "add_offset": 53000.0,
                                           "_FillValue": -32768, "zlib": True, "complevel": 4}})
    print(f"written {out} ({out.stat().st_size / 1e6:.0f} MB): {allz.sizes['time']} days, "
          f"range {float(allz.min()):.0f} .. {float(allz.max()):.0f} m2/s2")


if __name__ == "__main__":
    main()
