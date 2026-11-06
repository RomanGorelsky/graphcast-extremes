"""Phase 3, step 2 (Colab GPU): 10-day GraphCast_small forecasts of every case study.

For each event in results/phase2/case_studies.csv and each start 1, 3, 5, 7 days
before its peak: build the inputs from ERA5, run 40 steps (10 days), read the ERA5
truth at the same valid times, cut both to the event's box and save
results/phase3/forecasts/<event>_minus<d>d.nc
  forecast_<var>, truth_<var>   (time = valid time, lat, lon) for
  2 m temperature, 10 m wind u/v, sea-level pressure.
Resumable: existing files are skipped.

    python scripts/31_run_case_forecasts.py
    python scripts/31_run_case_forecasts.py --only heat-004 wind-004     # a subset
"""
import argparse
import gc
import time

import pandas as pd
import xarray

from extremes import era5, inputs, model, phase3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="event_ids to run (default: all)")
    ap.add_argument("--io-threads", type=int, default=32)
    a = ap.parse_args()
    cases = phase3.case_studies()
    if a.only:
        cases = cases[cases.event_id.isin(a.only)]
    phase3.FORECASTS.mkdir(parents=True, exist_ok=True)

    ds = era5.open_store(era5.WB2_ERA5)
    statics = inputs.load_statics(ds)
    m = model.load_model()
    todo = [(c, d) for c in cases.itertuples() for d in phase3.LEADS_DAYS
            if not phase3.forecast_path(c.event_id, d).exists()]
    print(f"{len(cases)} events, {len(todo)} forecasts to run "
          f"({len(cases) * len(phase3.LEADS_DAYS) - len(todo)} done earlier)", flush=True)

    t_start = time.perf_counter()
    for n, (c, d) in enumerate(todo, 1):
        t0 = time.perf_counter()
        init = pd.Timestamp(getattr(c, f"init_minus_{d}d"))
        batch = inputs.build_batch(ds, init, phase3.N_STEPS, statics=statics, io_threads=a.io_threads)
        i, tg, f = m.extract(batch, phase3.N_STEPS)
        del batch
        gc.collect()
        pred = m.forecast(i, tg, f, keep={v: None for v in phase3.VERIFY_VARS}).isel(batch=0)
        valid = init + pd.to_timedelta(pred.time.values)
        out = xarray.Dataset()
        for var in phase3.VERIFY_VARS:
            fc = pred[var].assign_coords(time=valid)
            truth = era5.load_1deg(ds, var, valid.values, "subsample", a.io_threads)
            out[f"forecast_{var}"] = phase3.crop(fc, *phase3.box_of(c)).astype("float32")
            out[f"truth_{var}"] = phase3.crop(truth, *phase3.box_of(c)).astype("float32")
        out = out.drop_vars([v for v in out.coords if v not in ("time", "lat", "lon")])
        out.attrs.update(event_id=c.event_id, kind=c.kind, init=str(init), lead_days_before_peak=d,
                         peak_time=str(c.peak_time), model="GraphCast_small (R0)")
        p = phase3.forecast_path(c.event_id, d)
        out.to_netcdf(p.with_suffix(".part.nc"))
        p.with_suffix(".part.nc").rename(p)
        del i, tg, f, pred, out
        gc.collect()                                   # keep memory flat between forecasts
        print(f"[{n}/{len(todo)}] {c.event_id} start -{d} d ({str(init)[:13]}): "
              f"{time.perf_counter() - t0:.0f} s  (total {(time.perf_counter() - t_start) / 60:.1f} min)",
              flush=True)
    print("done:", phase3.FORECASTS)


if __name__ == "__main__":
    main()
