"""Phase 4 check 2: a light probabilistic version of the corrections.

Each corrected forecast becomes a normal distribution: centre = the corrected value,
spread = how far that correction typically was from ERA5 on the training hindcasts.
From it: the probability that a cell breaks its record. Guan et al.'s PBC is
probabilistic; this approximates it without running an ensemble.

  R0          deterministic: probability 1 if the forecast breaks the record, else 0
  R0+spread   centre = raw forecast,  spread = RMS of the hindcast errors
  Debias      centre = debiased,      spread = standard deviation of the hindcast errors
  Persistence centre = regression,    spread = regression's error on the training cases
  PBC         average of the Debias and Persistence probabilities (equal mixture)
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr          # standard normal cumulative distribution

PROB_LEVELS = (0.2, 0.5)                # "warn" if the probability reaches this


def prob_exceed(centre, spread, threshold, sign: float) -> np.ndarray:
    """P(value beyond the record threshold): above it for sign=+1, below for sign=-1.
    Where the spread is unknown (NaN) or zero, falls back to 0/1."""
    centre, spread, threshold = np.broadcast_arrays(*(np.asarray(x, float) for x in (centre, spread, threshold)))
    beyond = sign * (centre - threshold)
    ok = np.isfinite(spread) & (spread > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = ndtr(beyond / np.where(ok, spread, 1.0))
    return np.where(ok, p, (beyond > 0).astype(float))


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    """Brier score and simple warning scores for probabilities p of events y (0/1)."""
    p, y = np.asarray(p, float), np.asarray(y, bool)
    out = dict(n=int(y.size), n_records=int(y.sum()), base_rate=float(y.mean()) if y.size else np.nan,
               brier=float(np.mean((p - y) ** 2)) if y.size else np.nan,
               mean_p_on_records=float(p[y].mean()) if y.any() else np.nan,
               mean_p_elsewhere=float(p[~y].mean()) if (~y).any() else np.nan)
    for lev in PROB_LEVELS:
        warn = p >= lev
        hits = int((warn & y).sum())
        out[f"pod_{lev}"] = hits / y.sum() if y.any() else np.nan            # share of records warned
        out[f"precision_{lev}"] = hits / warn.sum() if warn.any() else np.nan  # share of warnings right
    return out


def reliability(p: np.ndarray, y: np.ndarray, bins=(0, .05, .15, .3, .5, .7, .9, 1.0001)):
    """Mean forecast probability vs observed frequency per probability bin."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    k = np.digitize(p, bins) - 1
    rows = []
    for i in range(len(bins) - 1):
        m = k == i
        if m.sum():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows
