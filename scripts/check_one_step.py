"""Quick check: does ONE jitted 6-h GraphCast_small step give finite numbers?

Prints the JAX version, any XLA_FLAGS in use, the number of NaN values in the
forecast and the time taken. Use it to compare environment fixes quickly.

    python scripts/check_one_step.py
"""
import os
import time

import jax

from extremes import model

if __name__ == "__main__":
    print("jax", jax.__version__, "| XLA_FLAGS =", os.environ.get("XLA_FLAGS", "(none)"), flush=True)
    m = model.load_model(use_bf16=False)
    batch, _ = model.load_example()
    inputs, targets, forcings = m.extract(batch, 1)
    t0 = time.perf_counter()
    pred = m.forecast(inputs, targets, forcings)
    n_nan = int(sum(int(pred[v].isnull().sum()) for v in pred.data_vars))
    print(f"NaN values: {n_nan:,}  ({'BAD' if n_nan else 'OK'})  time: {time.perf_counter() - t0:.0f} s")
    if not n_nan:
        t = pred["2m_temperature"]
        print(f"2 m temperature range: {float(t.min()):.1f} .. {float(t.max()):.1f} K (plausible: ~200 .. 330)")
