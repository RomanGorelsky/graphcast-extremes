"""Deterministic version of Guan et al.'s (2026) probabilistic bias correction (Phase 4).

For one forecast (start t, lead steps k = 1..40) and one quantity X (2 m temperature
or 10 m wind speed), separately for every grid cell and lead step, learned from the
training hindcasts j (same model, starts within +-7 days of t's calendar date in
2016-2019):

  Debias       X_D = F - mean_j( H_j - O_j )                 (remove the mean error)
  Persistence  X_P = a + b F + c O(t)                        (regression, fitted on j:
                                                              O_j = a + b H_j + c O_j(start))
  PBC          X_PBC = (X_D + X_P) / 2                       (as in the paper: average)

F = forecast to correct, H = hindcast, O = ERA5 truth, O(t) = weather observed at the
start. In the paper the regression also has a climatology term; within a +-7-day
window climatology is nearly constant per cell, so it is absorbed by the intercept a.
"""
from __future__ import annotations

import numpy as np

MIN_SAMPLES = 10        # fewer valid training samples -> leave the forecast unchanged
RIDGE = 1e-3            # tiny ridge on b, c (relative to their variance) for stability


def debias(F: np.ndarray, H: np.ndarray, O: np.ndarray) -> np.ndarray:
    """F (K, ...), H and O (J, K, ...) -> corrected F."""
    err = H - O
    n = np.sum(np.isfinite(err), 0)
    bias = np.where(n >= MIN_SAMPLES, np.nanmean(err, 0), 0.0)
    return F - bias


def persistence(F: np.ndarray, O0: np.ndarray, H: np.ndarray, O: np.ndarray, H0: np.ndarray) -> np.ndarray:
    """Regression O = a + b H + c O0 per lead and cell, applied to (F, O0).

    F (K, ...); O0 (...) = observed at the forecast's start;
    H, O (J, K, ...); H0 (J, ...) = observed at each training hindcast's start.
    """
    J, K = H.shape[:2]
    x1 = H
    x2 = np.broadcast_to(H0[:, None], H.shape)
    ok = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(O)
    n = ok.sum(0)
    w = ok.astype(float)
    # centre the predictors (per lead and cell) so the intercept separates cleanly
    def mean(x):
        return np.where(n > 0, np.nansum(np.where(ok, x, 0) * w, 0) / np.maximum(n, 1), 0.0)
    m1, m2, my = mean(x1), mean(x2), mean(O)
    d1 = np.where(ok, x1 - m1, 0.0)
    d2 = np.where(ok, x2 - m2, 0.0)
    dy = np.where(ok, O - my, 0.0)
    s11, s22, s12 = (d1 * d1).sum(0), (d2 * d2).sum(0), (d1 * d2).sum(0)
    s1y, s2y = (d1 * dy).sum(0), (d2 * dy).sum(0)
    s11r, s22r = s11 * (1 + RIDGE) + 1e-12, s22 * (1 + RIDGE) + 1e-12
    det = s11r * s22r - s12 * s12
    b = (s22r * s1y - s12 * s2y) / det
    c = (s11r * s2y - s12 * s1y) / det
    O0b = np.broadcast_to(O0, F.shape)
    pred = my + b * (F - m1) + c * (O0b - m2)
    return np.where(n >= MIN_SAMPLES, pred, F)


def correct(F, O0, H, O, H0) -> dict:
    d = debias(F, H, O)
    p = persistence(F, O0, H, O, H0)
    return {"Debias": d, "Persistence": p, "PBC": 0.5 * (d + p)}
