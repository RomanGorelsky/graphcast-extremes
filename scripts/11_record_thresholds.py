"""Phase 1, step 2: record thresholds from ERA5 1979-2015 (GraphCast_small's training years).

For every 1-degree cell and calendar month: highest and lowest 2 m temperature
and highest 10 m wind speed over all 6-hourly values of 1979-2015.

Streams ERA5 month by month (nothing large is stored). Each finished year is
saved to results/phase1/yearly/<year>.nc, so the script can be stopped and
restarted at any time - it continues with the first missing year.
At the end it writes results/phase1/thresholds_1979-2015_1deg.nc.

    python scripts/11_record_thresholds.py --years 1979 1979   # test: one year
    python scripts/11_record_thresholds.py                     # all 37 years
"""
import argparse
import time

import xarray

from extremes import era5, phase1, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=None, help="default: the store chosen by 10_check_regridding.py")
    ap.add_argument("--years", nargs=2, type=int, default=[phase1.REF_START, phase1.REF_END])
    ap.add_argument("--io-threads", type=int, default=32)
    a = ap.parse_args()
    store, method = phase1.chosen()
    a.store = a.store or store
    y0, y1 = a.years
    phase1.YEARLY.mkdir(parents=True, exist_ok=True)
    ds = era5.open_store(a.store)
    print(f"method={method}  years {y0}-{y1}  -> {phase1.YEARLY}", flush=True)

    t_start = time.perf_counter()
    for year in range(y0, y1 + 1):
        f = phase1.YEARLY / f"{year}.nc"
        if f.exists():
            print(f"{year}: done earlier, skipping", flush=True)
            continue
        t0 = time.perf_counter()
        per_month = {}
        for month in range(1, 13):
            t2m, wsp = era5.load_surface_month(ds, year, month, method, a.io_threads)
            per_month[month] = records.month_extremes(t2m, wsp)
        yearly = records.stack_months(per_month)
        tmp = f.with_suffix(".part.nc")
        yearly.to_netcdf(tmp)
        tmp.rename(f)
        print(f"{year}: {int(yearly.n_times.sum())} time steps in {time.perf_counter() - t0:.0f} s"
              f"  (total {(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)

    files = [phase1.YEARLY / f"{y}.nc" for y in range(phase1.REF_START, phase1.REF_END + 1)]
    missing = [p.stem for p in files if not p.exists()]
    if missing:
        print(f"{len(missing)} reference years still missing ({missing[0]}...); "
              "thresholds are written once all 37 years exist.")
        return
    th = records.combine_years([xarray.load_dataset(p) for p in files])
    lsm, mask = phase1.land_mask()
    th["land_sea_mask"], th["analysis_mask"] = lsm, mask
    th.attrs.update(reference_years=f"{phase1.REF_START}-{phase1.REF_END}", regrid_method=method,
                    source=a.store, definition="per cell and calendar month, 00/06/12/18 UTC values")
    out = phase1.thresholds_path()
    th.to_netcdf(out)
    print(f"thresholds written: {out}  ({out.stat().st_size / 1e6:.1f} MB), "
          f"{int(th.n_times.sum())} time steps from {int(th.n_years)} years")


if __name__ == "__main__":
    main()
