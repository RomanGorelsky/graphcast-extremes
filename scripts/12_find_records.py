"""Phase 1, step 3: every record-breaking cell and time in the test year (default 2020).

Compares each 6-hourly 2 m temperature and 10 m wind speed of the year with the
1979-2015 thresholds of its calendar month (land only, north of 60 S).
Writes, in results/phase1/:
  records_<year>.csv.gz          one row per record: kind, time, lat, lon, value, threshold, exceedance
  records_<year>_summary.json    counts per kind and month, largest exceedances
Resumable by month (months already done are kept in results/phase1/records_<year>_parts/).

    python scripts/12_find_records.py              # 2020
    python scripts/12_find_records.py --year 2018
"""
import argparse
import json

import pandas as pd
import xarray

from extremes import era5, phase1, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--store", default=era5.WB2_ERA5)
    ap.add_argument("--io-threads", type=int, default=32)
    a = ap.parse_args()
    method = phase1.chosen_method()
    th = xarray.load_dataset(phase1.thresholds_path())
    if th.attrs.get("regrid_method") != method:
        raise RuntimeError("thresholds were made with a different regridding method")
    mask = th["analysis_mask"]
    parts = phase1.OUT / f"records_{a.year}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    ds = era5.open_store(a.store)

    for month in range(1, 13):
        f = parts / f"{month:02d}.csv.gz"
        if f.exists():
            continue
        t2m, wsp = era5.load_surface_month(ds, a.year, month, method, a.io_threads)
        df = records.find_records(t2m, wsp, th, mask)
        df.to_csv(f, index=False)
        counts = df.kind.value_counts().to_dict()
        print(f"{a.year}-{month:02d}: " + ", ".join(f"{k} {counts.get(k, 0):,}" for k in records.KINDS),
              flush=True)

    df = pd.concat([pd.read_csv(parts / f"{m:02d}.csv.gz", parse_dates=["time"]) for m in range(1, 13)],
                   ignore_index=True)
    out = phase1.OUT / f"records_{a.year}.csv.gz"
    df.to_csv(out, index=False)

    df["month"] = df.time.dt.month
    summary = {"year": a.year, "regrid_method": method,
               "n_records": {k: int((df.kind == k).sum()) for k in records.KINDS},
               "n_cells_with_record": {k: int(df[df.kind == k][["lat", "lon"]].drop_duplicates().shape[0])
                                       for k in records.KINDS},
               "per_month": {k: {int(m): int(n) for m, n in df[df.kind == k].month.value_counts().sort_index().items()}
                             for k in records.KINDS},
               "largest_exceedance": {}}
    for k in records.KINDS:
        top = df[df.kind == k].nlargest(5, "exceedance")
        summary["largest_exceedance"][k] = [
            {"time": str(r.time)[:16], "lat": float(r.lat), "lon": float(r.lon),
             "exceedance": round(float(r.exceedance), 2), "unit": records.UNITS[k]} for r in top.itertuples()]
    (phase1.OUT / f"records_{a.year}_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n" + json.dumps(summary["n_records"]), "->", out)


if __name__ == "__main__":
    main()
