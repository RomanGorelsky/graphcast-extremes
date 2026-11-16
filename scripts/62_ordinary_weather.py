"""Phase 6 extension (cluster, 1 GPU): do the methods help ORDINARY forecasts, as the
papers claim? 52 weekly 10-day forecasts of 2020 (every 7 days from 1 January, 00 UTC)
with one model, scored against ERA5 at lead days 1-10:

  rmse, bias   latitude-weighted, for 2 m temperature, 10 m u/v, sea-level pressure,
               500 hPa height (z500) and 850 hPa temperature (t850)
  acc          anomaly correlation against the WeatherBench2 ERA5 climatology
               (1990-2019); skipped if that store cannot be opened
  activity     spatial std of the forecast anomaly / of the ERA5 anomaly (sharpness:
               < 1 = smoother than reality)
  budgets      global dry-air mass, water vapour and total energy of the forecast on
               the 13 levels (physics.py) - compared with ERA5's own in
               64_ordinary_summary.py (ERA5 values from --truth-budgets)

R1-R4 run with their conservation schemes at every step (as in training).
Writes results/phase6/ordinary/<model>.csv (resumable: finished dates are skipped).

    python scripts/62_ordinary_weather.py --model R0
    python scripts/62_ordinary_weather.py --model R4
    python scripts/62_ordinary_weather.py --truth-budgets      # ERA5's budgets on the same days (CPU)
"""
import argparse
import gc
import time

import numpy as np
import pandas as pd
import xarray

from extremes import era5, finetune, inputs, metrics, model, paths, physics

OUT = paths.RESULTS_DIR / "phase6" / "ordinary"
DATES = pd.date_range("2020-01-01T00", periods=52, freq="7D")
LEAD_DAYS = range(1, 11)
N_STEPS = 40
CLIMATOLOGY = "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
SURF = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"]
UPPER = {"z500": ("geopotential", 500), "t850": ("temperature", 850)}
BUDGET_VARS = ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"]


def open_climatology():
    try:
        c = era5.standardise(xarray.open_zarr(CLIMATOLOGY, storage_options={"token": "anon"}, chunks=None))
        print("climatology: WeatherBench2 ERA5 1990-2019", flush=True)
        return c
    except Exception as e:                                   # ACC is optional
        print(f"climatology not available ({type(e).__name__}: {e}) - ACC skipped", flush=True)
        return None


def clim_field(clim, var, valid: pd.Timestamp, level=None):
    """Climatology at the valid time's day of year and hour, on the 1-degree grid."""
    c = clim[var]
    if level is not None:
        c = c.sel(level=level)
    c = c.sel(dayofyear=min(valid.day_of_year, 366), hour=valid.hour)
    return c.isel(lat=slice(None, None, 4), lon=slice(None, None, 4)).astype(np.float32).load()


def acc(f, o, c) -> float:
    w = metrics.lat_weights(o["lat"])
    fa, oa = f - c, o - c
    num = (fa * oa).weighted(w).mean(("lat", "lon"))
    den = np.sqrt((fa ** 2).weighted(w).mean(("lat", "lon")) * (oa ** 2).weighted(w).mean(("lat", "lon")))
    return float(num / den)


def activity(f, o, c) -> float:
    w = metrics.lat_weights(o["lat"])
    sf = np.sqrt(((f - c) ** 2).weighted(w).mean(("lat", "lon")))
    so = np.sqrt(((o - c) ** 2).weighted(w).mean(("lat", "lon")))
    return float(sf / so)


def budgets(state: xarray.Dataset, phi_s) -> dict:
    area, wlev = physics.cell_area(state.lat, state.lon), physics.level_weights(state.level)
    q = state["specific_humidity"]
    return dict(dry_mass=float(physics.dry_air_mass(q, area, wlev)),
                vapour=float(physics.water_vapour(q, area, wlev)),
                energy=float(physics.total_energy(state["temperature"], q, state["u_component_of_wind"],
                                                  state["v_component_of_wind"], phi_s, area, wlev)))


def truth_budgets(ds, statics, io_threads):
    f = OUT / "truth_budgets.csv"
    done = set(pd.read_csv(f, parse_dates=["valid"]).valid) if f.exists() else set()
    times = sorted({d + pd.Timedelta(days=k) for d in DATES for k in [0, *LEAD_DAYS]} - done)
    lev = ds.sel(level=inputs.LEVELS)
    for n, t in enumerate(times, 1):
        st = xarray.Dataset({v: era5.load_1deg(lev, v, [t], "subsample", io_threads).isel(time=0)
                             for v in BUDGET_VARS})
        row = dict(valid=t, **budgets(st, statics["geopotential_at_surface"]))
        pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)
        if n % 25 == 0:
            print(f"truth budgets {n}/{len(times)}", flush=True)
    print("written:", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="R0", help="R0 (GraphCast_small) or a fine-tuned run R1-R4")
    ap.add_argument("--truth-budgets", action="store_true", help="only ERA5's budgets on the forecast days")
    ap.add_argument("--io-threads", type=int, default=32)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = era5.open_store(era5.WB2_ERA5)
    statics = inputs.load_statics(ds)
    if a.truth_budgets:
        truth_budgets(ds, statics, a.io_threads)
        return

    f = OUT / f"{a.model}.csv"
    done = set(pd.read_csv(f, parse_dates=["init"]).init) if f.exists() else set()
    m = model.load_model() if a.model == "R0" else finetune.load_run(a.model)
    extra = tuple(v for v in m.task_config.target_variables if v not in inputs.SURFACE and v not in inputs.ATMOS)
    clim = open_climatology()
    keep = {**{v: None for v in SURF}, **{v: None for v in BUDGET_VARS}, "geopotential": [500]}
    phi = statics["geopotential_at_surface"]
    lev = ds.sel(level=inputs.LEVELS)

    for init in DATES:
        if init in done:
            continue
        t0 = time.perf_counter()
        batch = inputs.build_batch(ds, init, N_STEPS, statics=statics, io_threads=a.io_threads, extra=extra)
        i, tg, fo = m.extract(batch, N_STEPS)
        del batch
        gc.collect()
        pred = m.forecast(i, tg, fo, keep=keep).isel(batch=0)
        rows = []
        for d in LEAD_DAYS:
            k = 4 * d - 1
            valid = init + pd.Timedelta(days=d)
            step = pred.isel(time=k)
            fields = {v: (step[v], era5.load_1deg(ds, v, [valid], "subsample", a.io_threads).isel(time=0), v, None)
                      for v in SURF}
            for name, (var, lv) in UPPER.items():
                o = era5.load_1deg(ds.sel(level=[lv]), var, [valid], "subsample", a.io_threads).isel(time=0, level=0)
                fields[name] = (step[var].sel(level=lv), o, var, lv)
            for name, (fc, ob, var, lv) in fields.items():
                r = dict(init=init, lead_days=d, var=name, rmse=float(metrics.rmse(fc, ob)),
                         bias=float((fc - ob).weighted(metrics.lat_weights(ob["lat"])).mean(("lat", "lon"))))
                if clim is not None:
                    c = clim_field(clim, var, valid, lv)
                    r.update(acc=acc(fc, ob, c), activity=activity(fc, ob, c))
                rows.append(r)
            b = budgets(xarray.Dataset({v: step[v] for v in BUDGET_VARS}), phi)
            rows += [dict(init=init, lead_days=d, var=f"budget_{n}", value=v) for n, v in b.items()]
        b0 = budgets(xarray.Dataset({v: era5.load_1deg(lev, v, [init], "subsample", a.io_threads).isel(time=0)
                                     for v in BUDGET_VARS}), phi)
        rows += [dict(init=init, lead_days=0, var=f"budget_{n}", value=v) for n, v in b0.items()]
        pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
        del i, tg, fo, pred
        gc.collect()
        z = [r for r in rows if r["var"] == "z500" and r["lead_days"] == 5][0]
        print(f"{a.model} {init:%Y-%m-%d}: z500 day-5 RMSE {z['rmse']:.0f}"
              + (f", ACC {z['acc']:.3f}" if "acc" in z else "") + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
    print("written:", f)


if __name__ == "__main__":
    main()
