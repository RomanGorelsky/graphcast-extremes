"""Phase 4, step 3 (Colab, CPU is enough): ERA5 truth for the hindcasts, read once.

For each event: every 6-hourly time from each of its hindcast starts to 10 days
later (the start time itself included - Persistence uses the weather observed at
the start), 2 m temperature and 10 m wind, cut to the event box, saved as
results/phase4/truth/<event>.nc. Resumable per event; can run at the same time as
41_run_hindcasts.py (e.g. in a second Colab tab with a CPU runtime).

    python scripts/42_truth_cache.py
"""
import argparse
import time

import numpy as np
import pandas as pd
import xarray

from extremes import era5, phase3, phase4

VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
BLOCK = 40                                     # time steps read per request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io-threads", type=int, default=32)
    a = ap.parse_args()
    plan = pd.read_csv(phase4.OUT / "hindcast_plan.csv", parse_dates=["init"])
    cases = phase3.case_studies().set_index("event_id")
    phase4.TRUTH.mkdir(parents=True, exist_ok=True)
    ds = era5.open_store(era5.WB2_ERA5)
    steps = pd.to_timedelta(np.arange(0, phase3.N_STEPS + 1) * 6, unit="h")

    for e, c in cases.iterrows():
        f = phase4.truth_path(e)
        if f.exists():
            continue
        t0 = time.perf_counter()
        inits = plan[plan.event_ids.str.split().apply(lambda s: e in s)].init
        # + the 2020 start times of the event's own forecasts (Persistence needs the
        #   weather observed at the start; Phase 3 files begin 6 h after it)
        own = {pd.Timestamp(getattr(c, f"init_minus_{d}d")) for d in phase3.LEADS_DAYS}
        times = pd.DatetimeIndex(sorted({t + s for t in inits for s in steps} | own))
        box = (c.box_lat_min, c.box_lat_max, c.box_lon_min, c.box_lon_max)
        parts = {v: [] for v in VARS}
        for k in range(0, len(times), BLOCK):
            tt = times[k:k + BLOCK].values
            for v in VARS:
                parts[v].append(phase3.crop(era5.load_1deg(ds, v, tt, "subsample", a.io_threads), *box))
        out = xarray.Dataset({f"truth_{v}": xarray.concat(parts[v], "time").astype("float32") for v in VARS})
        out = out.drop_vars([v for v in out.coords if v not in ("time", "lat", "lon")])
        out.attrs.update(event_id=e, n_inits=int(len(inits)))
        out.to_netcdf(f.with_suffix(".part.nc"))
        f.with_suffix(".part.nc").rename(f)
        print(f"{e}: {len(times)} time steps for {len(inits)} hindcast starts in "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    print("done:", phase4.TRUTH)


if __name__ == "__main__":
    main()
