"""Offline tests of the Phase 4 checks (robustness across events, record probabilities).

    python -m pytest -q tests
"""
import numpy as np
import pandas as pd

from extremes import pbc, probrec, robust


# ---------------- check 1: robustness

def _errors(label_shift: dict, events=("e1", "e2", "e3"), n=20, seed=0):
    rng = np.random.default_rng(seed)
    base = []
    for e in events:
        for i in range(n):
            base.append(dict(event_id=e, kind="heat", start_days=5, time=f"2020-06-{1 + i % 20:02d}T00",
                             lat=float(i), lon=1.0, intensity_error=rng.normal(-3, 1),
                             hit=False, record_shattering=False, obs_exceedance=1.0))
    base = pd.DataFrame(base)
    out = {}
    for lab, shift in label_shift.items():
        e = base.copy()
        e["intensity_error"] = e.intensity_error + e.event_id.map(shift).fillna(0.0)
        e["hit"] = e.intensity_error > -2
        out[lab] = e
    return out


def test_paired_delta_recovers_a_uniform_shift():
    errs = _errors({"R0": {}, "PBC": {"e1": -1.0, "e2": -1.0, "e3": -1.0}})
    p = robust.paired(errs)
    assert len(p) == 60
    r = robust.bootstrap_delta(p, "PBC", B=500)
    assert np.isclose(r["delta"], -1.0) and np.isclose(r["ci_low"], -1.0) and np.isclose(r["ci_high"], -1.0)
    assert r["events_worse"] == 3 and r["events_better"] == 0


def test_one_event_driving_the_result_shows_in_leave_one_out():
    errs = _errors({"R0": {}, "PBC": {"e1": -6.0}})
    r = robust.bootstrap_delta(robust.paired(errs), "PBC", B=2000)
    assert np.isclose(r["delta"], -2.0)
    assert np.isclose(r["loo_max"], 0.0)                 # without e1 nothing changes
    assert r["largest_effect_event"] == "e1"
    assert r["ci_high"] >= 0.0                           # resampling can miss e1 -> range reaches 0


def test_paired_keeps_only_cells_present_for_every_label():
    errs = _errors({"R0": {}, "Debias": {}})
    errs["Debias"] = errs["Debias"][errs["Debias"].event_id != "e3"]
    assert set(robust.paired(errs).event_id) == {"e1", "e2"}


# ---------------- check 2: record probabilities

def test_prob_exceed_basic_values_and_signs():
    assert np.isclose(probrec.prob_exceed(300.0, 2.0, 300.0, +1), 0.5)
    assert probrec.prob_exceed(304.0, 2.0, 300.0, +1) > 0.97
    assert probrec.prob_exceed(304.0, 2.0, 300.0, -1) < 0.03          # cold: below the threshold
    assert probrec.prob_exceed(301.0, np.nan, 300.0, +1) == 1.0       # no spread -> yes/no
    assert probrec.prob_exceed(299.0, 0.0, 300.0, +1) == 0.0


def test_scores_of_a_perfect_and_a_useless_forecast():
    y = np.array([1, 0, 0, 1], bool)
    s = probrec.scores(y.astype(float), y)
    assert s["brier"] == 0 and s["pod_0.5"] == 1 and s["precision_0.5"] == 1
    s = probrec.scores(np.full(4, 0.5), y)
    assert np.isclose(s["brier"], 0.25) and s["mean_p_on_records"] == 0.5


def test_spreads_match_the_training_errors():
    rng = np.random.default_rng(4)
    J, K = 400, 2
    H0 = rng.normal(280, 3, (J, 1, 1))
    H = rng.normal(280, 3, (J, K, 1, 1))
    O = 5 + 0.5 * H + 0.4 * H0[:, None] + rng.normal(0, 1.5, H.shape)
    F = np.full(H.shape[1:], 281.0)
    _, sd = pbc.persistence(F, np.full((1, 1), 280.0), H, O, H0, return_spread=True)
    assert np.allclose(sd, 1.5, atol=0.15)
    assert np.allclose(pbc.debias_spread(H, O), np.nanstd(H - O, 0, ddof=1))


def test_spread_option_does_not_change_the_correction():
    rng = np.random.default_rng(5)
    H0 = rng.normal(280, 3, (60, 3, 2))
    H = rng.normal(280, 3, (60, 4, 3, 2))
    O = H - 1 + rng.normal(0, 1, H.shape)
    F = np.full(H.shape[1:], 285.0)
    O0 = np.full(H.shape[2:], 279.0)
    a = pbc.persistence(F, O0, H, O, H0)
    b, _ = pbc.persistence(F, O0, H, O, H0, return_spread=True)
    assert np.array_equal(a, b)
