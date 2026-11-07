"""Phase 4, step 2 (Colab GPU, background): GraphCast_small hindcasts for PBC training.

For each start date in results/phase4/hindcast_plan.csv (newest year first): build
the inputs from ERA5, forecast 10 days, keep 2 m temperature and 10 m wind, cut to
the box of every event the date serves and save
results/phase4/hindcasts/<event>/<YYYYMMDDHH>.nc.
Only inputs are read here; the ERA5 truth comes from 42_truth_cache.py.
Resumable (finished files are skipped). Run it in the background like Phase 3:

    nohup python scripts/41_run_hindcasts.py > results/phase4/run41.log 2>&1 &
    python scripts/41_run_hindcasts.py --max-runs 20      # a fixed batch, then exit
"""
import argparse
import gc
import time

import pandas as pd

from extremes import era5, inputs, model, phase3, phase4

VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--io-threads", type=int, default=32)
    a = ap.parse_args()
    plan = pd.read_csv(phase4.OUT / "hindcast_plan.csv", parse_dates=["init"])
    cases = phase3.case_studies().set_index("event_id")
    todo = []
    for r in plan.itertuples():
        missing = [e for e in r.event_ids.split() if not phase4.hindcast_path(e, r.init).exists()]
        if missing:
            todo.append((r.init, missing))
    if a.max_runs:
        todo = todo[:a.max_runs]
    print(f"{len(plan)} planned runs, {len(todo)} to do now", flush=True)
    if not todo:
        return

    ds = era5.open_store(era5.WB2_ERA5)
    statics = inputs.load_statics(ds)
    m = model.load_model()
    t_start = time.perf_counter()
    for n, (init, events) in enumerate(todo, 1):
        t0 = time.perf_counter()
        batch = inputs.build_batch(ds, init, phase3.N_STEPS, statics=statics, io_threads=a.io_threads)
        i, tg, f = m.extract(batch, phase3.N_STEPS)
        del batch
        gc.collect()
        pred = m.forecast(i, tg, f, keep={v: None for v in VARS}).isel(batch=0)
        pred = pred.assign_coords(time=init + pd.to_timedelta(pred.time.values))
        for e in events:
            c = cases.loc[e]
            box = (c.box_lat_min, c.box_lat_max, c.box_lon_min, c.box_lon_max)
            out = pred.map(lambda da: phase3.crop(da, *box)).astype("float32")
            out = out.rename({v: f"forecast_{v}" for v in VARS})
            out = out.drop_vars([v for v in out.coords if v not in ("time", "lat", "lon")])
            out.attrs.update(event_id=e, init=str(init), model="GraphCast_small (R0) hindcast")
            p = phase4.hindcast_path(e, init)
            p.parent.mkdir(parents=True, exist_ok=True)
            out.to_netcdf(p.with_suffix(".part.nc"))
            p.with_suffix(".part.nc").rename(p)
        del i, tg, f, pred
        gc.collect()
        print(f"[{n}/{len(todo)}] {init:%Y-%m-%d} -> {len(events)} events: {time.perf_counter() - t0:.0f} s "
              f"(total {(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
