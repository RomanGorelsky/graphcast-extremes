"""Offline test of the full forecast pipeline with a tiny random-weight
GraphCast on synthetic data. Needs no network, runs in ~1-3 min on a CPU.

    python -m pytest -q tests
"""
import numpy as np

from extremes import metrics, model, synthetic
from weathernext.weathernext1_graph import graphcast


def _tiny_model(stats):
    mc = graphcast.ModelConfig(resolution=0, mesh_size=2, latent_size=16, gnn_msg_steps=2,
                               hidden_layers=1, radius_query_fraction_edge_length=0.6)
    return model.Model(params=None, state={}, model_config=mc,
                       task_config=graphcast.TASK_13, stats=stats)


def test_rollout_shapes_and_finite():
    ds = synthetic.make_dataset(n_times=6)
    m = _tiny_model(synthetic.make_stats(ds))
    n = ds.sizes["time"] - 2
    inputs, targets, forcings = m.extract(ds, n)
    m.init_random_params(inputs, targets, forcings)
    pred = m.forecast(inputs, targets, forcings)
    assert set(pred.data_vars) == set(graphcast.TASK_13.target_variables)
    assert pred.sizes["time"] == n
    for v in pred.data_vars:
        assert pred[v].shape == targets[v].shape, v
        assert np.isfinite(pred[v].values).all(), v


def test_rmse_weights():
    ds = synthetic.make_dataset(n_times=3)
    t = ds["2m_temperature"]
    assert float(metrics.rmse(t, t).max()) == 0.0
    assert np.allclose(float(metrics.rmse(t + 2.0, t).mean()), 2.0, atol=1e-4)
