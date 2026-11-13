"""Phase 5, step 1 (cluster, CPU job): stage the training data on the 1-degree grid.

For each month of the requested years: the 5 surface and 6 x 13 pressure-level
variables of GraphCast_small from WeatherBench2 ERA5 (0.25 deg, subsampled to 1 deg
exactly as in Phases 1-4) and the 7 flux variables (6-h accumulations made from
hourly ERA5, see extremes/phase5.py), written to
EXTREMES_DATA/phase5/era5_1deg/<YYYY-MM>.zarr. Resumable: finished months are skipped
(a half-written month is redone). ~34 GB per year.

    python scripts/50_stage_data.py --year 2014            # one year (the cluster job runs one per year)
    python scripts/50_stage_data.py --statics              # terrain height + land-sea mask, once
    python scripts/50_stage_data.py --stats --years 2010 2011 2012 2013 2014   # flux normalisation
"""
import argparse
import shutil
import time

import numpy as np
import pandas as pd
import xarray

from extremes import era5, inputs, phase5, physics

DAYS_PER_WRITE = 2


def load_fluxes(arco: xarray.Dataset, t: pd.DatetimeIndex, io_threads: int) -> dict:
    hourly = phase5.hours_before(t)
    out = {}
    for ours, name in phase5.FLUX_SOURCE.items():
        x = era5.load_1deg(arco, name, hourly, "subsample", io_threads)            # (time*6, lat, lon)
        a = np.asarray(x.values, np.float64).reshape(len(t), 6, x.sizes["lat"], x.sizes["lon"]).sum(1)
        out[ours] = xarray.DataArray(a.astype(np.float32), dims=("time", "lat", "lon"),
                                     coords={"time": t.values, "lat": x.lat.values, "lon": x.lon.values})
    return out


def stage_month(wb2, arco, year: int, month: int, io_threads: int) -> None:
    final = phase5.month_path(year, month)
    if final.exists():
        return
    tmp = final.with_suffix(".part.zarr")
    if tmp.exists():
        shutil.rmtree(tmp)
    times = pd.DatetimeIndex(era5.month_times(wb2, year, month))
    t0 = time.perf_counter()
    step = 4 * DAYS_PER_WRITE
    for i in range(0, len(times), step):
        t = times[i:i + step]
        parts = {v: era5.load_1deg(wb2, v, t.values, "subsample", io_threads) for v in inputs.SURFACE}
        lev = wb2.sel(level=inputs.LEVELS)
        parts.update({v: era5.load_1deg(lev, v, t.values, "subsample", io_threads) for v in inputs.ATMOS})
        parts.update(load_fluxes(arco, t, io_threads))
        ds = xarray.Dataset({v: da.drop_vars([c for c in da.coords if c not in ("time", "level", "lat", "lon")])
                             for v, da in parts.items()}).astype(np.float32)
        # drop the storage settings inherited from the source stores (coordinates
        # included): their old-format compression is refused by the format we write
        ds = ds.drop_encoding().chunk({"time": 1})
        ds.to_zarr(tmp, mode="w") if i == 0 else ds.to_zarr(tmp, append_dim="time")
        if i == 0:
            print(f"  {year}-{month:02d}: first {len(t)} times in {time.perf_counter() - t0:.0f} s "
                  f"(month: {len(times)})", flush=True)
    tmp.rename(final)
    print(f"{year}-{month:02d}: {len(times)} times in {time.perf_counter() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    ap.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    ap.add_argument("--statics", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--years", type=int, nargs="+", help="training years for --stats")
    ap.add_argument("--io-threads", type=int, default=32)
    a = ap.parse_args()
    phase5.STAGED.mkdir(parents=True, exist_ok=True)
    phase5.OUT.mkdir(parents=True, exist_ok=True)

    if a.statics:
        wb2 = era5.open_store(era5.WB2_ERA5)
        st = xarray.Dataset(inputs.load_statics(wb2, "subsample"))
        st.to_zarr(phase5.STAGED / "statics.zarr", mode="w")
        print("written:", phase5.STAGED / "statics.zarr")
    if a.year is not None:
        wb2 = era5.open_store(era5.WB2_ERA5)
        arco = phase5.open_arco()
        for m in a.months:
            stage_month(wb2, arco, a.year, m, a.io_threads)
        print(f"{a.year} done", flush=True)
    if a.stats:
        ds = phase5.open_staged(a.years)
        mean, std = phase5.flux_stats(ds)
        xarray.Dataset({**{f"mean_{v}": mean[v] for v in physics.FLUX_VARS},
                        **{f"std_{v}": std[v] for v in physics.FLUX_VARS}}).to_netcdf(phase5.FLUX_STATS)
        for v in physics.FLUX_VARS:
            print(f"  {v:36s} mean {float(mean[v]):12.4g}  std {float(std[v]):12.4g}")
        print("written:", phase5.FLUX_STATS)


if __name__ == "__main__":
    main()
