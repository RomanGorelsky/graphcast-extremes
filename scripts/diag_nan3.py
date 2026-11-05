"""Third NaN diagnostic: find the exact operation that first produces a NaN.

Run with JAX's NaN checker switched on; it stops with a FloatingPointError at the
first NaN and prints the line of library code responsible:

    JAX_DEBUG_NANS=True python scripts/diag_nan3.py > results/phase0/debug_nans.txt 2>&1

One 6-h step, real weights, float32. With the checker on, JAX re-runs the
failing step operation by operation, so this can take several minutes.
"""
import jax

from extremes import model

if __name__ == "__main__":
    print("jax", jax.__version__, "| debug_nans =", jax.config.jax_debug_nans, flush=True)
    m = model.load_model(use_bf16=False)
    batch, _ = model.load_example()
    inputs, targets, forcings = m.extract(batch, 1)
    pred = m.forecast(inputs, targets, forcings)
    n_nan = int(sum(int(pred[v].isnull().sum()) for v in pred.data_vars))
    print("finished without the NaN checker stopping; NaN values in forecast:", n_nan)
