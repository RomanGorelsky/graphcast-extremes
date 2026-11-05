"""Diagnostic: where do the NaNs in the smoke test come from?

Checks, in order:
  1. NaNs in the example data (inputs / targets / forcings), per variable;
  2. NaNs or zero spreads in the normalisation statistics;
  3. NaNs in the model weights;
  4. ONE 6-h forecast step, in bfloat16 (authors' default) and in float32,
     reporting which output variables contain NaN.
Takes about as long as one compile + one step, twice (a few minutes on a CPU).

    python scripts/diag_nan.py
"""
import jax
import numpy as np

from extremes import model


def nan_report(ds, title):
    print(f"\n[{title}]")
    bad = 0
    for v in ds.data_vars:
        x = np.asarray(ds[v].values, dtype=np.float64)
        n = int(np.isnan(x).sum())
        if n:
            bad += 1
            where = {d: int(np.isnan(x).any(axis=tuple(j for j in range(x.ndim) if j != i)).sum())
                     for i, d in enumerate(ds[v].dims)}
            print(f"  {v:40s} NaN {n} of {x.size}  (slices with NaN per dim: {where})")
    print("  no NaNs" if not bad else f"  -> {bad} variables contain NaN")
    return bad


def main():
    m = model.load_model()
    batch, name = model.load_example()
    print("example:", name, "| devices:", jax.devices())
    inputs, targets, forcings = m.extract(batch, 1)

    nan_report(inputs, "1. inputs")
    nan_report(forcings, "1. forcings")
    nan_report(targets, "1. targets (only matter for scoring)")

    print("\n[2. normalisation statistics]")
    for k, ds in m.stats.items():
        for v in ds.data_vars:
            x = np.asarray(ds[v].values, dtype=np.float64)
            if np.isnan(x).any() or ("stddev" in k and (x <= 0).any()):
                print(f"  {k}/{v}: NaN={int(np.isnan(x).sum())}  min={np.nanmin(x):.3g}")
    needed = set(inputs.data_vars) | set(targets.data_vars) | set(forcings.data_vars)
    for k, ds in m.stats.items():
        missing = sorted(v for v in needed if v not in ds.data_vars
                         and not (k == "diffs_stddev_by_level" and v not in inputs.data_vars))
        if missing:
            print(f"  {k}: missing {missing}")
    print("  (anything listed above is a problem; nothing listed = OK)")

    leaves = jax.tree_util.tree_leaves(m.params)
    n_nan = sum(int(np.isnan(np.asarray(x, dtype=np.float32)).sum()) for x in leaves)
    print(f"\n[3. weights] {len(leaves)} arrays, {sum(x.size for x in leaves):,} numbers, NaN: {n_nan}")

    for bf16 in (True, False):
        mm = model.load_model(use_bf16=bf16)
        pred = mm.forecast(inputs, targets, forcings)
        print(f"\n[4. one step, {'bfloat16' if bf16 else 'float32'}]")
        nan_report(pred, "forecast")


if __name__ == "__main__":
    main()
