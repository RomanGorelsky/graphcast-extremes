"""Offline tests of the Phase 4 logic (training windows, Debias, Persistence, PBC).

    python -m pytest -q tests
"""
import numpy as np
import pandas as pd

from extremes import pbc, phase4


def test_training_window_sizes_and_years():
    t = phase4.training_inits(pd.Timestamp("2020-05-21T00"))
    assert len(t) == 15 * 4
    assert {x.year for x in t} == {2016, 2017, 2018, 2019}
    assert min(x for x in t if x.year == 2019) == pd.Timestamp("2019-05-14")
    assert max(x for x in t if x.year == 2019) == pd.Timestamp("2019-05-28")
    assert all(x.hour == 0 for x in t)


def test_leap_day_maps_to_28_february():
    assert phase4.same_date_in(2019, pd.Timestamp("2020-02-29T00")) == pd.Timestamp("2019-02-28")


def test_plan_has_one_row_per_init_event_start():
    cases = pd.DataFrame(dict(event_id=["a", "b"], **{f"init_minus_{d}d": [pd.Timestamp("2020-06-10"),
                                                                         pd.Timestamp("2020-06-12")]
                                                    for d in (1, 3, 5, 7)}))
    p = phase4.plan(cases)
    assert not p.duplicated(["init", "event_id", "start_days"]).any()
    # the two events' windows overlap -> shared start dates exist
    assert len(set(p[p.event_id == "a"].init) & set(p[p.event_id == "b"].init)) > 0


def _training(rng, J=60, K=4, Y=3, X=2):
    O0 = rng.normal(280, 3, (J, Y, X))                      # weather at each start
    H = rng.normal(280, 3, (J, K, Y, X))
    return H, O0


def test_debias_removes_a_constant_error():
    rng = np.random.default_rng(0)
    H, _ = _training(rng)
    O = H - 2.0                                             # model always 2 K too warm
    F = np.full(H.shape[1:], 290.0)
    assert np.allclose(pbc.debias(F, H, O), 288.0)


def test_persistence_recovers_the_regression():
    rng = np.random.default_rng(1)
    H, H0 = _training(rng)
    O = 10 + 0.6 * H + 0.35 * H0[:, None] + rng.normal(0, 0.01, H.shape)
    F = np.full(H.shape[1:], 285.0)
    O0 = np.full(H.shape[2:], 279.0)
    expect = 10 + 0.6 * 285 + 0.35 * 279
    assert np.allclose(pbc.persistence(F, O0, H, O, H0), expect, atol=0.05)


def test_too_few_samples_leaves_the_forecast_unchanged():
    rng = np.random.default_rng(2)
    H, H0 = _training(rng, J=5)
    O = H - 3.0
    F = np.full(H.shape[1:], 290.0)
    out = pbc.correct(F, np.full(H.shape[2:], 280.0), H, O, H0)
    assert all(np.allclose(v, 290.0) for v in out.values())


def test_pbc_is_the_average_and_handles_missing_hindcasts():
    rng = np.random.default_rng(3)
    H, H0 = _training(rng)
    H[:5] = np.nan                                          # five hindcasts missing
    O = H - 1.0
    F = np.full(H.shape[1:], 290.0)
    out = pbc.correct(F, np.full(H.shape[2:], 280.0), H, O, H0)
    assert np.allclose(out["PBC"], 0.5 * (out["Debias"] + out["Persistence"]))
    assert np.allclose(out["Debias"], 289.0)
    assert np.isfinite(out["Persistence"]).all()
