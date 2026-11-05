"""Phase 0, step 6: timing benchmark.

Measures (1) the one-off JAX compile time and (2) seconds per 6-h step, and
projects the time of one 10-day forecast (40 steps). Run it on each machine
(Mac, Colab GPU, later the cluster); results are appended to
results/phase0/benchmark.jsonl with the machine's name.

    python scripts/02_benchmark.py --label mac-cpu
"""
import argparse
import json
import platform
import time

import jax

from extremes import model, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="e.g. mac-cpu, colab-t4, cluster-a100")
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()

    m = model.load_model()
    batch, name = model.load_example()
    n = batch.sizes["time"] - 2
    inputs, targets, forcings = m.extract(batch, n)

    t0 = time.perf_counter()
    m.forecast(inputs, targets.isel(time=slice(0, 1)), forcings.isel(time=slice(0, 1)))
    compile_s = time.perf_counter() - t0          # first call = compile + 1 step

    runs = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        m.forecast(inputs, targets, forcings)
        runs.append(time.perf_counter() - t0)
    per_step = min(runs) / n
    rec = {"label": a.label, "devices": [str(d) for d in jax.devices()],
           "platform": platform.platform(), "example": name, "steps_timed": n,
           "compile_plus_first_step_s": round(compile_s, 1),
           "seconds_per_step": round(per_step, 3),
           "projected_10day_forecast_s": round(40 * per_step, 1)}
    out = paths.RESULTS_DIR / "phase0"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "benchmark.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
