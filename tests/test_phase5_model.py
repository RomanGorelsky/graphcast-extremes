"""Offline tests of the Phase 5 model and training loop with a tiny random GraphCast on
synthetic data (no network; a few minutes on a CPU).

    python -m pytest -q tests
"""
import dataclasses

import haiku as hk
import jax
import numpy as np
import pandas as pd
import pytest
import xarray

from extremes import finetune, physics, synthetic, train
from weathernext.utils import data_utils
from weathernext.weathernext1_graph import graphcast

OLD_TASK = graphcast.TASK_13_PRECIP_OUT          # as GraphCast_small
NEW_TASK = finetune.extended_task(OLD_TASK)
MC = graphcast.ModelConfig(resolution=0, mesh_size=2, latent_size=16, gnn_msg_steps=2,
                           hidden_layers=1, radius_query_fraction_edge_length=0.6)
SIX_H = 6 * 3600.0


def _with_fluxes(ds, seed=1):
    """Add the 7 flux variables (ERA5 sign conventions, 6-h accumulations)."""
    rng = np.random.default_rng(seed)
    shp = (ds.sizes["batch"], ds.sizes["time"], ds.sizes["lat"], ds.sizes["lon"])
    vals = {physics.EVAP: -np.abs(rng.normal(0.0015, 0.0007, shp)),
            physics.TOA_SOLAR: rng.uniform(0, 600, shp) * SIX_H,
            physics.TOA_THERMAL: -rng.uniform(200, 280, shp) * SIX_H,
            physics.SFC_SOLAR: rng.uniform(0, 300, shp) * SIX_H,
            physics.SFC_THERMAL: -rng.uniform(30, 90, shp) * SIX_H,
            physics.SFC_SENSIBLE: rng.normal(-20, 20, shp) * SIX_H,
            physics.SFC_LATENT: rng.normal(-80, 30, shp) * SIX_H}
    for k, a in vals.items():
        ds[k] = (("batch", "time", "lat", "lon"), a.astype(np.float32))
    # realistic precipitation: about as large as evaporation (ERA5: +0.74 vs -0.73 mm / 6 h)
    ds["total_precipitation_6hr"] = (("batch", "time", "lat", "lon"),
                                     np.abs(rng.normal(0.0015, 0.0007, shp)).astype(np.float32))
    ds["specific_humidity"] = np.abs(ds["specific_humidity"]) * 1e-2
    return ds


@pytest.fixture(scope="module")
def data():
    ds = _with_fluxes(synthetic.make_dataset(n_times=8))
    stats = synthetic.make_stats(ds)
    return ds, stats


def _split(ds, task, n):
    return data_utils.extract_inputs_targets_forcings(
        ds.isel(time=slice(0, 2 + n)), target_lead_times=slice("6h", f"{n * 6}h"), **dataclasses.asdict(task))


def _forward(task, stats, schemes):
    @hk.transform_with_state
    def f(inp, tmpl, frc):
        return finetune.construct(MC, task, stats, schemes, use_bf16=False)(inp, tmpl, frc)
    return f


def _old_params(ds, stats):
    inp, tgt, frc = _split(ds, OLD_TASK, 1)
    params, _ = _forward(OLD_TASK, stats, ()).init(jax.random.PRNGKey(0), inp, tgt, frc)
    return params


def test_channel_layout_counts():
    lay = finetune.channel_layout(NEW_TASK)
    assert sum(n for _, n in lay.values()) == 83 + 7
    assert all(lay[v][1] == 1 for v in physics.FLUX_VARS)
    assert lay["temperature"][1] == 13


def test_grown_model_reproduces_the_old_one(data):
    ds, stats = data
    p_old = _old_params(ds, stats)
    p_new = finetune.grow_params(p_old, OLD_TASK, NEW_TASK)
    inp, tgt_old, frc = _split(ds, OLD_TASK, 2)
    _, tgt_new, _ = _split(ds, NEW_TASK, 2)
    old = jax.device_get(_forward(OLD_TASK, stats, ()).apply(p_old, {}, None, inp, tgt_old * np.nan, frc)[0])
    new = jax.device_get(_forward(NEW_TASK, stats, ()).apply(p_new, {}, None, inp, tgt_new * np.nan, frc)[0])
    for v in OLD_TASK.target_variables:
        np.testing.assert_allclose(new[v].values, old[v].values, rtol=1e-5, atol=1e-5, err_msg=v)
    for v in physics.FLUX_VARS:                     # zero weights -> the climatological mean
        np.testing.assert_allclose(new[v].values, float(stats["mean_by_level"][v]), rtol=1e-4)


def test_full_schemes_close_the_budgets_in_a_rollout(data):
    ds, stats = data
    params = finetune.grow_params(_old_params(ds, stats), OLD_TASK, NEW_TASK)
    inp, tgt, frc = _split(ds, NEW_TASK, 3)
    pred = jax.device_get(_forward(NEW_TASK, stats, finetune.RUNS["R4"]).apply(
        params, {}, None, inp, tgt * np.nan, frc)[0])
    prev, phi = inp.isel(time=-1), inp["geopotential_at_surface"]
    for k in range(3):
        step = pred.isel(time=[k])
        r = physics.residuals(prev, step, phi)
        assert abs(float(r["mass"].max())) < 1e-5, k
        assert abs(float(r["moisture"].max())) < 1e-4, k
        assert abs(float(r["energy"].max())) < 1e-5, k
        prev = step.isel(time=0)


def _store(tmp_path, n_times=16):
    """The synthetic data in the staged-store layout (absolute time, no batch)."""
    ds = _with_fluxes(synthetic.make_dataset(n_times=n_times, start="2020-06-01T00"))
    t = ds["datetime"].values[0]
    st = ds.isel(batch=0).drop_vars("datetime").assign_coords(time=t)
    path = tmp_path / "store.zarr"
    st.to_zarr(path, mode="w")
    return path, synthetic.make_stats(ds)


def test_training_runs_lowers_nothing_to_nan_and_resumes(tmp_path):
    path, stats = _store(tmp_path)
    sampler = train.Sampler(path, NEW_TASK, years=[2020], io_threads=2)
    assert len(sampler.inits(2)) == 16 - 2 - 1
    inp, tgt, frc = sampler.batch(sampler.inits(1)[:2], 1)
    assert inp.sizes["batch"] == 2 and set(physics.FLUX_VARS) <= set(tgt.data_vars)
    assert np.isfinite(tgt[physics.EVAP].values).all()

    ds = _with_fluxes(synthetic.make_dataset(n_times=8))
    init = finetune.grow_params(_old_params(ds, stats), OLD_TASK, NEW_TASK)
    cfg = train.TrainConfig(run="R4", batch_size=2, stage1_steps=3, stage2_steps=2, rollout=2,
                            warmup=1, eval_every=5, n_val=2, ckpt_every=2, use_bf16=False)
    out = tmp_path / "run"
    train.train(cfg, out, sampler, sampler, MC, NEW_TASK, stats, init, OLD_TASK.target_variables,
                log=lambda *a: None, max_steps=3)
    assert (out / "checkpoint.pkl").exists()
    params = train.train(cfg, out, sampler, sampler, MC, NEW_TASK, stats, init, OLD_TASK.target_variables,
                         log=lambda *a: None)
    lines = pd.read_json(out / "train.jsonl", lines=True)
    val = pd.read_json(out / "validation.jsonl", lines=True)
    assert np.isfinite(lines.loss).all() and val.step.max() == 5
    assert train.load(out / "checkpoint.pkl")["step"] == 5
    res = train.residual_check(params, MC, NEW_TASK, stats, finetune.RUNS["R4"], sampler,
                               sampler.inits(2)[:1], n_steps=2, use_bf16=False)
    assert (res[["mass", "energy"]] < 1e-5).all().all()
