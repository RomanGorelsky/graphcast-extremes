"""Offline tests of the Phase 7 prospects (no network, a minute on a CPU).

    python -m pytest -q tests/test_phase7.py
"""
import dataclasses

import jax
import numpy as np
import pandas as pd
import pytest
import xarray

from extremes import ensemble, extended, finetune, synthetic, tailcorr, train
from weathernext.utils import data_utils, losses
from weathernext.weathernext1_graph import graphcast


# ------------------------------------------------------------------ prospect 1: quantile mapping
def test_interp_along_identity_and_shift_extrapolation():
    q = np.linspace(-2, 2, tailcorr.LEVELS.size)[:, None, None] * np.ones((1, 2, 3))
    x = np.array([[[-1.0, 0.0, 1.5], [3.0, -5.0, 0.2]]])                 # (K=1, Y=2, X=3)
    np.testing.assert_allclose(tailcorr.interp_along(x, q, q), x)          # same curves -> unchanged
    y = tailcorr.interp_along(x, q, q + 1.0)                              # truth 1 higher everywhere
    np.testing.assert_allclose(y, x + 1.0)                                 # incl. beyond the ends (3, -5)
    lev = np.broadcast_to(tailcorr.LEVELS[:, None, None], q.shape)
    p = tailcorr.interp_along(x, q, lev, "clip")
    assert p[0, 1, 0] == tailcorr.LEVELS[-1] and p[0, 1, 1] == tailcorr.LEVELS[0]


def test_pool_neighbours_shape_and_edges():
    a = np.arange(12, dtype=float).reshape(1, 3, 4)
    p = tailcorr.pool_neighbours(a, 1)
    assert p.shape == (9, 3, 4)
    assert np.isfinite(p[:, 1, 1]).all()                                    # interior: all 9
    assert np.isfinite(p[:, 0, 0]).sum() == 4                               # corner: 4 of 9


def test_quantile_mapping_widens_a_too_smooth_model():
    rng = np.random.default_rng(0)
    J, K, Y, X = 60, 8, 4, 5
    O = rng.normal(290, 3, (J, K, Y, X))                                    # truth: spread 3 K
    H = 290 + 0.5 * (O - 290) + rng.normal(0, 0.5, O.shape)                # model: too smooth
    F = np.full((K, Y, X), 290.0)
    F[:, 0, 0] = 293.0                                                      # a warm forecast (model's tail)
    F[:, 1, 1] = 287.0                                                      # a cold one
    r = tailcorr.correct(F, H, O)
    assert np.all(r["QM"][:, 0, 0] > 294.5)                                 # ~ +2 sigma of the model -> of truth
    assert np.all(r["QM"][:, 1, 1] < 285.5)
    np.testing.assert_allclose(r["TailQM"][:, 2, 2], 290.0, atol=0.3)       # typical value untouched
    assert np.all(r["TailQM"][:, 0, 0] > 293.5)                             # tail widened
    up = tailcorr.correct(F, H, O, lower=False)
    np.testing.assert_allclose(up["TailQM"][:, 1, 1], 287.0)                # wind: upper tail only


def test_tail_weight():
    p = np.array([0.5, 0.85, 0.9, 0.95, 0.99, 0.1, 0.05])
    np.testing.assert_allclose(tailcorr.tail_weight(p), [0, 0, 0.5, 1, 1, 0.5, 1])


# ------------------------------------------------------------------ prospect 2: extended set
def _events():
    return pd.DataFrame(dict(
        event_id=["heat-001", "heat-002", "heat-003", "cold-001", "wind-001"], kind=["heat", "heat", "heat", "cold", "wind"],
        event=[0, 1, 2, 0, 0], n_cells=[50, 3, 40, 20, 30], coastal_fraction=[0.1, 0.0, 0.9, 0.2, 0.0],
        centre_lat=[50, 40, 45, 80, 30], centre_lon=[10, 20, 30, 250, 100],
        record_shattering=[True, True, True, True, False], weighted_exceedance=[5, 9, 8, 3, 7],
        peak_time=pd.to_datetime(["2021-07-01 12:00"] * 5), start=pd.to_datetime(["2021-06-29"] * 5),
        end=pd.to_datetime(["2021-07-02"] * 5), peak_exceedance=[4, 6, 5, 3, 2], regime=["none"] * 5,
        sector=["other"] * 5))


def test_extended_choose_applies_the_phase2_rules():
    sel = extended.choose(_events(), caps={"heat": 5, "cold": 5, "wind": 5})
    assert list(sel.event_id) == ["heat-001"]          # too small, coastal, Arctic, not shattering: out
    sel = extended.choose(_events(), caps=None, only_shattering=False, max_lat=90)
    assert set(sel.event_id) == {"heat-001", "cold-001", "wind-001"}


def test_extended_case_row_ids_box_and_starts():
    e = _events().iloc[0]
    cells = pd.DataFrame(dict(lat=[48, 52], lon=[358, 3]))                  # crosses 0 deg
    r = extended.case_row(e, cells, 2021)
    assert r["event_id"] == "2021-heat-001" and r["source_event_id"] == "heat-001" and r["year"] == 2021
    assert r["box_lon_min"] == 353 and r["box_lon_max"] == 8
    assert r["init_minus_7d"] == pd.Timestamp("2021-06-24") and extended.fits_in_data(r)
    late = dict(r, init_minus_1d=pd.Timestamp("2023-01-05"))
    assert not extended.fits_in_data(late)


# ------------------------------------------------------------------ prospect 4: ensemble
def test_member_pairs_are_symmetric_and_distinct():
    pairs = ensemble.member_pairs(8, 20, seed=1)
    assert len(pairs) == 20
    assert all(pairs[2 * k][:2] == pairs[2 * k + 1][:2] and pairs[2 * k][2] == -pairs[2 * k + 1][2]
               for k in range(10))
    assert len({p[:2] for p in pairs}) == 10
    with pytest.raises(ValueError):
        ensemble.member_pairs(8, 7)


def test_pool_dates_and_perturb():
    d = ensemble.pool_dates("2020-02-29 12:00")
    assert len(d) == 8 and d[0] == pd.Timestamp("2016-02-22")
    lat, lon = np.linspace(-90, 90, 5), np.arange(0, 360, 90.0)
    inp = xarray.Dataset({
        "specific_humidity": (("batch", "time", "level", "lat", "lon"), np.full((1, 2, 2, 5, 4), 1e-3)),
        "2m_temperature": (("batch", "time", "lat", "lon"), np.full((1, 2, 5, 4), 280.0)),
        "toa_incident_solar_radiation": (("batch", "time", "lat", "lon"), np.ones((1, 2, 5, 4)))},
        coords=dict(lat=lat, lon=lon, level=[500, 850]))
    delta = xarray.Dataset({
        "specific_humidity": (("batch", "level", "lat", "lon"), np.full((1, 2, 5, 4), -5e-3)),
        "2m_temperature": (("batch", "lat", "lon"), np.full((1, 5, 4), 1.5))},
        coords=dict(lat=lat, lon=lon, level=[500, 850]))
    out = ensemble.perturb(inp, delta)
    assert (out["specific_humidity"].values == 0).all()                    # clipped at zero
    np.testing.assert_allclose(out["2m_temperature"].values, 281.5)        # both input times
    assert out["2m_temperature"].dims == inp["2m_temperature"].dims
    np.testing.assert_array_equal(out["toa_incident_solar_radiation"], inp["toa_incident_solar_radiation"])


# ------------------------------------------------------------------ prospect 3: AMSE loss (R5)
def _field_ds(a):
    lat, lon = np.linspace(-90, 90, 19), np.arange(0, 360, 10.0)
    return xarray.Dataset({"2m_temperature": (("batch", "lat", "lon"), a)}, coords=dict(lat=lat, lon=lon))


def test_amse_equals_mse_for_equal_spread_and_penalises_smoothing():
    rng = np.random.default_rng(0)
    t = rng.normal(0, 1, (2, 19, 36))
    tgt = _field_ds(t)
    shifted = _field_ds(t + 0.7)                                            # same spread, same pattern
    a, _ = finetune.amse_per_level(shifted, tgt)
    m, _ = losses.weighted_mse_per_level(shifted, tgt)
    np.testing.assert_allclose(a.values, m.values, rtol=1e-4)
    # an uncertain forecast (correlation ~0.7 with the truth): with the truth's spread
    # ("sharp") or shrunk toward the mean ("smooth", what squared-error training learns)
    noisy = t + rng.normal(0, 1, t.shape)
    sharp, smooth = noisy / np.sqrt(2), 0.5 * noisy
    m_sharp, m_smooth = (losses.weighted_mse_per_level(_field_ds(x), tgt)[0].values.mean() for x in (sharp, smooth))
    a_sharp, a_smooth = (finetune.amse_per_level(_field_ds(x), tgt, gradient_weight=0)[0].values.mean()
                         for x in (sharp, smooth))
    assert m_smooth < m_sharp            # plain MSE rewards the smoothed forecast ...
    assert a_sharp < a_smooth            # ... AMSE rewards the one with realistic spread


def test_amse_training_loss_has_finite_gradients():
    task = graphcast.TASK_13_PRECIP_OUT
    mc = graphcast.ModelConfig(resolution=0, mesh_size=2, latent_size=16, gnn_msg_steps=2,
                               hidden_layers=1, radius_query_fraction_edge_length=0.6)
    ds = synthetic.make_dataset(n_times=4)
    stats = synthetic.make_stats(ds)
    inp, tgt, frc = data_utils.extract_inputs_targets_forcings(
        ds.isel(time=slice(0, 3)), target_lead_times=slice("6h", "6h"), **dataclasses.asdict(task))
    fn_a = train.make_loss(mc, task, stats, (), use_bf16=False, loss_kind="amse")
    fn_m = train.make_loss(mc, task, stats, (), use_bf16=False, loss_kind="mse")
    params, state = fn_a.init(jax.random.PRNGKey(0), inp, tgt, frc)
    (la, _), _ = fn_a.apply(params, state, None, inp, tgt, frc)
    (lm, _), _ = fn_m.apply(params, state, None, inp, tgt, frc)
    assert np.isfinite(la) and float(la) >= float(lm) - 1e-6                # AMSE >= MSE
    g = jax.grad(lambda p: fn_a.apply(p, state, None, inp, tgt, frc)[0][0])(params)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(g))


def test_r5_is_registered_with_the_amse_loss():
    assert finetune.RUNS["R5"] == () and finetune.RUN_LOSS["R5"] == "amse"
    assert train.TrainConfig(run="R5").loss_kind() == "amse"
    assert train.TrainConfig(run="R1").loss_kind() == "mse"
