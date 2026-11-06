"""Phase 3, step 3 (Colab GPU): 'do no harm' reference on ordinary days.

12 forecasts of 10 days, one per month of 2020 (15th, 00 UTC). For each, the global
latitude-weighted RMSE against ERA5 at lead days 1, 3, 5, 10 for 2 m temperature,
sea-level pressure, 10 m wind (u, v) and 500 hPa geopotential.
Rows are appended to results/phase3/noharm.csv with a --label, so every later model
(PBC, R1-R4) is scored on exactly the same dates. Resumable.

    python scripts/32_run_noharm.py --label R0
"""
import argparse
import gc
import time

import pandas as pd

from extremes import era5, inputs, metrics, model, phase3

LEADS = (1, 3, 5, 10)
SURF = ["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="R0")
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--io-threads", type=int, default=32)
    a = ap.parse_args()
    phase3.OUT.mkdir(parents=True, exist_ok=True)
    f = phase3.OUT / "noharm.csv"
    done = pd.read_csv(f, parse_dates=["init"]) if f.exists() else pd.DataFrame(columns=["label", "init"])
    done = set(zip(done.label, pd.to_datetime(done.init)))

    ds = era5.open_store(era5.WB2_ERA5)
    statics = inputs.load_statics(ds)
    m = model.load_model()
    for month in range(1, 13):
        init = pd.Timestamp(f"{a.year}-{month:02d}-15T00")
        if (a.label, init) in done:
            continue
        t0 = time.perf_counter()
        batch = inputs.build_batch(ds, init, phase3.N_STEPS, statics=statics, io_threads=a.io_threads)
        i, tg, fo = m.extract(batch, phase3.N_STEPS)
        pred = m.forecast(i, tg, fo).isel(batch=0)
        rows = []
        for lead in LEADS:
            k = 4 * lead - 1                                   # index of the 6-h step at `lead` days
            valid = init + pd.Timedelta(days=lead)
            for var in SURF:
                truth = era5.load_1deg(ds, var, [valid], "subsample", a.io_threads).isel(time=0)
                rows.append((var, lead, float(metrics.rmse(pred[var].isel(time=k), truth))))
            z = era5.load_1deg(ds.sel(level=[500]), "geopotential", [valid], "subsample", a.io_threads)
            rows.append(("z500", lead, float(metrics.rmse(pred["geopotential"].sel(level=500).isel(time=k),
                                                          z.isel(time=0, level=0)))))
        del batch, i, tg, fo, pred
        gc.collect()
        new = pd.DataFrame([dict(label=a.label, init=init, var=v, lead_days=l, rmse=r) for v, l, r in rows])
        new.to_csv(f, mode="a", header=not f.exists(), index=False)
        print(f"{a.label} {init:%Y-%m-%d}: " + ", ".join(
            f"{v.split('_')[0]} d{l} {r:.2f}" for v, l, r in rows if l in (1, 5)) +
              f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
    print("written:", f)


if __name__ == "__main__":
    main()
