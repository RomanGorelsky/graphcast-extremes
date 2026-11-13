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


def _fit(O0, H, O, H0):
    """Least-squares fit of O = a + b H + c O0 per lead and cell (centred predictors)."""
    x1 = H
    x2 = np.broadcast_to(H0[:, None], H.shape)
    ok = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(O)
    n = ok.sum(0)

    def mean(x):
        return np.where(n > 0, np.nansum(np.where(ok, x, 0), 0) / np.maximum(n, 1), 0.0)
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
    resid = np.where(ok, dy - b * d1 - c * d2, 0.0)
    sigma = np.sqrt((resid ** 2).sum(0) / np.maximum(n - 3, 1))      # 3 fitted numbers: a, b, c
    return dict(n=n, m1=m1, m2=m2, my=my, b=b, c=c, sigma=sigma)


def persistence(F: np.ndarray, O0: np.ndarray, H: np.ndarray, O: np.ndarray, H0: np.ndarray,
                return_spread: bool = False):
    """Regression O = a + b H + c O0 per lead and cell, applied to (F, O0).

    F (K, ...); O0 (...) = observed at the forecast's start;
    H, O (J, K, ...); H0 (J, ...) = observed at each training hindcast's start.
    With return_spread=True also returns the typical size (standard deviation) of the
    regression's error on the training cases - NaN where there are too few samples.
    """
    f = _fit(O0, H, O, H0)
    O0b = np.broadcast_to(O0, F.shape)
    pred = f["my"] + f["b"] * (F - f["m1"]) + f["c"] * (O0b - f["m2"])
    enough = f["n"] >= MIN_SAMPLES
    out = np.where(enough, pred, F)
    if return_spread:
        return out, np.where(enough, f["sigma"], np.nan)
    return out


def debias_spread(H: np.ndarray, O: np.ndarray) -> np.ndarray:
    """Typical size (standard deviation) of the hindcast errors H - O, per lead and cell."""
    err = H - O
    n = np.sum(np.isfinite(err), 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.nanstd(err, 0, ddof=1)
    return np.where(n >= MIN_SAMPLES, sd, np.nan)


def correct(F, O0, H, O, H0) -> dict:
    d = debias(F, H, O)
    p = persistence(F, O0, H, O, H0)
    return {"Debias": d, "Persistence": p, "PBC": 0.5 * (d + p)}
