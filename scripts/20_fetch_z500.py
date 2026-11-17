"""Phase 2, step 0 (Colab): daily 500 hPa geopotential, 1979-2020, for the weather regimes.

Weather regimes are large-scale patterns, so a coarse grid is enough: WeatherBench2's
2.8-degree ERA5 store (128 x 64), 00 UTC of every day. Read year by year (resumable)
and saved compactly (int16, 0.5 m2/s2 steps) to
results/phase2/z500_daily_1979-2020_2p8deg.nc (~100-200 MB).

    python scripts/20_fetch_z500.py
    python scripts/20_fetch_z500.py --years 1979 2022      # Phase 7

Years the 2.8-degree store does not cover completely (it ends before 2022) are made
from the 0.25-degree WeatherBench2 store (to 2023-01-10) by conservative (area-
weighted) regridding onto exactly the same 128 x 64 grid. A year present in both
stores is regridded once as a check and the difference is printed (should be small,
a few m2/s2 against values of ~55,000).
"""
import argparse
import time

import dask
import numpy as np
import xarray

from extremes import era5, paths

STORE = "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_conservative.zarr"
OUT = paths.RESULTS_DIR / "phase2"


def _edges(c: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Cell edges of a regular grid of centres c (clipped to [lo, hi] for latitude)."""
    d = np.diff(c).mean()
    e = np.r_[c - d / 2, c[-1] + d / 2]
    return np.clip(e, lo, hi) if lo is not None else e


def _overlap(src_e: np.ndarray, dst_e: np.ndarray, period: float | None = None) -> np.ndarray:
    """W[i, j] = length of the overlap of destination cell i with source cell j (periodic
    in longitude), rows normalised to 1 -> conservative averaging matrix."""
    shifts = (-period, 0.0, period) if period else (0.0,)
    W = np.zeros((dst_e.size - 1, src_e.size - 1))
    for s in shifts:
        lo = np.maximum(dst_e[:-1, None], src_e[None, :-1] + s)
        hi = np.minimum(dst_e[1:, None], src_e[None, 1:] + s)
        W += np.clip(hi - lo, 0, None)
    return W / W.sum(1, keepdims=True)


def regrid_matrices(src_lat, src_lon, dst_lat, dst_lon):
    """Separable conservative regridding: area ~ d(sin lat) x d(lon)."""
    s = lambda e: np.sin(np.deg2rad(e))
    wlat = _overlap(s(_edges(np.asarray(src_lat), -90, 90)), s(_edges(np.asarray(dst_lat), -90, 90)))
    wlon = _overlap(_edges(np.asarray(src_lon)), _edges(np.asarray(dst_lon)), period=360.0)
    return wlat, wlon


def from_fine_store(period: str, dst_lat, dst_lon, io_threads: int, chunk: int = 31) -> xarray.DataArray:
    """00 UTC z500 of a period ("2021", "2020-01") from the 0.25-deg store, regridded to (dst_lat, dst_lon)."""
    fine = era5.open_store(era5.WB2_ERA5)
    zf = fine["geopotential"].sel(level=500)
    t = zf.time.sel(time=period).values
    t = t[(t.astype("datetime64[h]").astype(int) % 24) == 0]
    wlat, wlon = regrid_matrices(zf.lat.values, zf.lon.values, dst_lat, dst_lon)
    out = []
    for k in range(0, t.size, chunk):
        with dask.config.set(scheduler="threads", num_workers=io_threads):
            y = zf.sel(time=t[k:k + chunk]).transpose("time", "lat", "lon").astype(np.float32).values
        out.append(np.einsum("ia,tab,jb->tij", wlat, y, wlon, optimize=True).astype(np.float32))
    return xarray.DataArray(np.concatenate(out), dims=("time", "lat", "lon"),
                            coords=dict(time=t, lat=np.asarray(dst_lat), lon=np.asarray(dst_lon)))


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
    last = np.datetime64(ds.time.values[-1], "D")
    print(f"2.8-degree store ends {last}", flush=True)
    checked = False
    for year in range(a.years[0], a.years[1] + 1):
        f = parts / f"{year}.nc"
        if f.exists():
            continue
        t0 = time.perf_counter()
        if last >= np.datetime64(f"{year}-12-31"):
            t = ds.time.sel(time=str(year)).values
            t = t[(t.astype("datetime64[h]").astype(int) % 24) == 0]          # 00 UTC only
            with dask.config.set(scheduler="threads", num_workers=a.io_threads):
                y = z.sel(time=t).astype(np.float32).transpose("time", "lat", "lon").load()
            src = "2.8-degree store"
        else:
            if not checked:                                   # once: regridding vs the store itself
                ref_year = int(str(last)[:4]) - 1
                ref = z.sel(time=f"{ref_year}-01").isel(time=slice(0, None, 4)).astype(np.float32).load()
                mine = from_fine_store(f"{ref_year}-01", ds.lat.values, ds.lon.values, a.io_threads)
                d = mine.sel(time=ref.time.values).values - ref.transpose("time", "lat", "lon").values
                print(f"regridding check, January {ref_year}: mean difference {d.mean():+.1f}, "
                      f"RMS {np.sqrt((d ** 2).mean()):.1f} m2/s2 (values ~55,000)", flush=True)
                checked = True
            y = from_fine_store(str(year), ds.lat.values, ds.lon.values, a.io_threads)
            src = "0.25-degree store, regridded"
        y.drop_vars([c for c in y.coords if c not in ("time", "lat", "lon")]).to_netcdf(f)
        print(f"{year}: {y.sizes['time']} days in {time.perf_counter() - t0:.0f} s ({src})", flush=True)

    files = [parts / f"{y}.nc" for y in range(a.years[0], a.years[1] + 1)]
    allz = xarray.concat([xarray.load_dataarray(p) for p in files], "time").rename("z500")
    out = OUT / f"z500_daily_{a.years[0]}-{a.years[1]}_2p8deg.nc"
    allz.to_netcdf(out, encoding={"z500": {"dtype": "int16", "scale_factor": 0.5, "add_offset": 53000.0,
                                           "_FillValue": -32768, "zlib": True, "complevel": 4}})
    print(f"written {out} ({out.stat().st_size / 1e6:.0f} MB): {allz.sizes['time']} days, "
          f"range {float(allz.min()):.0f} .. {float(allz.max()):.0f} m2/s2")


if __name__ == "__main__":
    main()
