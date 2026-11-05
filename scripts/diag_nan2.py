"""Second NaN diagnostic: is it the graph geometry, the weights, or the libraries?

  A. library versions (numerical libraries that build the graph);
  B. normalised inputs, computed by hand: any NaN/inf?
  C. the model's fixed graph (grid/mesh node and edge features), built from the
     real 1-degree grid: any NaN/inf?
  D. one 6-h step with RANDOM weights but the real architecture and real data:
     NaN or not? (NaN here = not the checkpoint; finite here = the checkpoint
     or how it is loaded)

    python scripts/diag_nan2.py
"""
import importlib

import haiku as hk
import jax
import numpy as np

from extremes import model
from weathernext.utils import normalization
from weathernext.weathernext1_graph import graphcast


def bad(x):
    x = np.asarray(x, dtype=np.float64)
    return int(np.isnan(x).sum()), int(np.isinf(x).sum()), x.size


def main():
    print("[A. versions]")
    for name in ("jax", "jaxlib", "numpy", "scipy", "xarray", "trimesh", "rtree", "jraph", "haiku"):
        try:
            print(f"  {name:8s}", importlib.import_module(name).__version__)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:8s} not importable ({e.__class__.__name__})")

    m = model.load_model(use_bf16=False)
    batch, _ = model.load_example()
    inputs, targets, forcings = m.extract(batch, 1)

    print("\n[B. normalised inputs and forcings]")
    for title, ds in (("inputs", inputs), ("forcings", forcings)):
        norm = normalization.normalize(ds, m.stats["stddev_by_level"], m.stats["mean_by_level"])
        hits = {v: bad(norm[v].values)[:2] for v in norm.data_vars if any(bad(norm[v].values)[:2])}
        span = {v: (float(norm[v].min()), float(norm[v].max())) for v in norm.data_vars}
        widest = sorted(span.items(), key=lambda kv: -max(abs(kv[1][0]), abs(kv[1][1])))[:3]
        print(f"  {title}: NaN/inf in {hits or 'none'}; largest |values|: "
              + ", ".join(f"{v} [{a:.1f}, {b:.1f}]" for v, (a, b) in widest))

    print("\n[C. graph features from the real 1-degree grid]")

    def graphs(sample):
        gc = graphcast.GraphCast(m.model_config, m.task_config)
        gc._maybe_init(sample)  # builds the three graphs; creates no weights
        out = {}
        for gname, g in (("grid2mesh", gc._grid2mesh_graph_structure),
                         ("mesh", gc._mesh_graph_structure),
                         ("mesh2grid", gc._mesh2grid_graph_structure)):
            for k, ns in g.nodes.items():
                out[f"{gname}/node/{k}"] = ns.features
            for k, es in g.edges.items():
                out[f"{gname}/edge/{k.name}"] = es.features
                out[f"{gname}/n_edges"] = np.asarray(es.n_edge)
        return out

    feats = hk.without_apply_rng(hk.transform(graphs)).apply({}, inputs)
    for k, v in feats.items():
        if k.endswith("n_edges"):
            print(f"  {k}: {int(np.asarray(v).sum()):,}")
        else:
            n_nan, n_inf, size = bad(v)
            print(f"  {k:32s} shape {tuple(np.shape(v))}  NaN {n_nan}  inf {n_inf}")

    print("\n[D. one step, real architecture, RANDOM weights]")
    r = model.Model(params=None, state={}, model_config=m.model_config,
                    task_config=m.task_config, stats=m.stats, use_bf16=False)
    r.init_random_params(inputs, targets, forcings)
    pred = r.forecast(inputs, targets, forcings)
    n_nan = sum(bad(pred[v].values)[0] for v in pred.data_vars)
    print(f"  NaN values in forecast: {n_nan:,}  ->  "
          + ("NaN even with random weights: geometry / libraries" if n_nan
             else "finite with random weights: points at the checkpoint or how it is loaded"))

    print("\n[E. checkpoint vs. fresh weights: same names and shapes?]")
    def flat(tree):
        leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
        return {jax.tree_util.keystr(path): leaf for path, leaf in leaves}

    fl, ll = flat(r.params), flat(m.params)
    only_f, only_l = sorted(set(fl) - set(ll)), sorted(set(ll) - set(fl))
    shape_diff = [k for k in set(fl) & set(ll) if np.shape(fl[k]) != np.shape(ll[k])]
    print(f"  {len(fl)} fresh, {len(ll)} loaded; only in fresh: {only_f[:5]}; "
          f"only in checkpoint: {only_l[:5]}; shape mismatches: {shape_diff[:5]}")
    big = sorted(((float(np.abs(np.asarray(v, np.float32)).max()), k) for k, v in ll.items()),
                 reverse=True)[:3]
    print("  largest |weight|:", ", ".join(f"{k} {x:.3g}" for x, k in big))

if __name__ == "__main__":
    main()
