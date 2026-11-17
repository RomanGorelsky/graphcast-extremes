"""Phase 7, prospect 4 (cluster GPU): perturbed-start ensembles of GraphCast_small (R0)
for every case study and start (-1/-3/-5/-7 d). See extremes/ensemble.py.

Member 0 is the unperturbed R0 forecast of Phase 3 (not run again); members 1..N are
started from R0's starting state + alpha x (difference of two ERA5 states of other years
at the same time of year), in +/- pairs. The inputs of a case and start are built once
and reused for all members, so a member costs only the 40 model steps (seconds on an H100).

Writes results/phase7/ensemble_a<alpha>/m<NN>/<event>_minus<d>d.nc - forecast fields only
(2 m temperature, 10 m wind, sea-level pressure) in the event box; the truth is in the
Phase 3 file. Resumable per member file. Jobs split the events: --shard k --num-shards n.

    python scripts/72_ensemble_forecasts.py                          # 20 members, alpha 0.1
    python scripts/72_ensemble_forecasts.py --alpha 0.2 --members 10
    EXTREMES_SET=ext python scripts/72_ensemble_forecasts.py         # the extended event set
"""
import argparse
import gc
import time

import pandas as pd
import xarray

from extremes import ensemble, era5, inputs, model, paths, phase3


def member_path(alpha: float, j: int, event_id: str, d: int):
    return paths.set_dir("phase7") / f"ensemble_a{alpha:g}" / f"m{j:02d}" / phase3.forecast_path(event_id, d).name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=20, help="perturbed members (even; + the R0 control)")
    ap.add_argument("--alpha", type=float, default=0.1, help="size of the perturbation (x state difference)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--io-threads", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    cases = phase3.case_studies()
    if a.only:
        cases = cases[cases.event_id.isin(a.only)]
    cases = cases.iloc[[k for k in range(len(cases)) if k % a.num_shards == a.shard]]   # split by event
    pairs = ensemble.member_pairs(len(ensemble.pool_dates("2020-01-01")), a.members, a.seed)

    todo = [(c, d) for c in cases.itertuples() for d in phase3.LEADS_DAYS
            if not all(member_path(a.alpha, j, c.event_id, d).exists() for j in range(1, a.members + 1))]
    share = f" (share {a.shard + 1} of {a.num_shards})" if a.num_shards > 1 else ""
    print(f"{len(cases)} events{share}, {a.members} members at alpha {a.alpha:g}: {len(todo)} case starts to run",
          flush=True)
    if not todo:
        return
    ds = era5.open_store(era5.WB2_ERA5)
    statics = inputs.load_statics(ds)
    m = model.load_model()
    pool, pool_event = None, None
    t_start = time.perf_counter()
    for n, (c, d) in enumerate(todo, 1):
        t0 = time.perf_counter()
        if pool_event != c.event_id:                         # the pool depends only on the event's date
            pool = [ensemble.load_state(ds, t, statics, a.io_threads) for t in ensemble.pool_dates(c.peak_time)]
            pool_event = c.event_id
        init = pd.Timestamp(getattr(c, f"init_minus_{d}d"))
        batch = inputs.build_batch(ds, init, phase3.N_STEPS, statics=statics, io_threads=a.io_threads)
        i, tg, f = m.extract(batch, phase3.N_STEPS)
        del batch
        for j, (pa, pb, sign) in enumerate(pairs, 1):
            p = member_path(a.alpha, j, c.event_id, d)
            if p.exists():
                continue
            ij = ensemble.perturb(i, ensemble.perturbation(pool, pa, pb, sign, a.alpha))
            pred = m.forecast(ij, tg, f, keep={v: None for v in phase3.VERIFY_VARS}).isel(batch=0)
            valid = init + pd.to_timedelta(pred.time.values)
            out = xarray.Dataset({f"forecast_{v}": phase3.crop(pred[v].assign_coords(time=valid), *phase3.box_of(c))
                                  .astype("float32") for v in phase3.VERIFY_VARS})
            out = out.drop_vars([v for v in out.coords if v not in ("time", "lat", "lon")])
            out.attrs.update(event_id=c.event_id, kind=c.kind, init=str(init), lead_days_before_peak=d,
                             member=j, alpha=a.alpha, pool_pair=f"{pa},{pb}", sign=sign,
                             model="GraphCast_small (R0), perturbed start")
            p.parent.mkdir(parents=True, exist_ok=True)
            out.to_netcdf(p.with_suffix(".part.nc"))
            p.with_suffix(".part.nc").rename(p)
            del ij, pred, out
        del i, tg, f
        gc.collect()
        print(f"[{n}/{len(todo)}] {c.event_id} start -{d} d: {a.members} members in {time.perf_counter() - t0:.0f} s "
              f"(total {(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)
    print("done:", member_path(a.alpha, 1, "x", 1).parent.parent)


if __name__ == "__main__":
    main()
