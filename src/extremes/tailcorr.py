"""Phase 7, prospect 1: tail-targeted statistical correction (quantile mapping).

Why: the Phase 4 corrections (Debias, Persistence/PBC) are regressions fitted to
mostly ordinary days by minimising the average error. For an uncertain forecast that
pulls the value toward the usual conditions - the wrong direction for a record.
Quantile mapping instead matches the whole DISTRIBUTION: "a forecast as warm as the
model's 97th percentile becomes ERA5's 97th percentile". If the model's forecasts are
too smooth (too narrow a distribution), the mapping widens the tails rather than
shrinking them.

For one forecast F(k, y, x) (lead step k, grid cell), learned from the same training
hindcasts H and ERA5 truth O as Phase 4 (starts within +-7 days of F's calendar date,
2016-2019), per grid cell and per LEAD DAY (the 4 steps of a day are pooled), with the
3 x 3 neighbouring cells pooled too, so a fit has ~60 x 4 x 9 ~ 2,000 values:

  Qh(p), Qo(p)   quantiles of hindcasts and of ERA5 at levels p (LEVELS)
  QM(F)          = Qo(Ph(F)): F's quantile level among the hindcasts, read off ERA5's
                   distribution; beyond the outermost levels the last shift is kept
                   (constant extrapolation of Qo - Qh), so a record-breaking forecast
                   can move beyond anything seen in training.
  TailQM(F)      = F + w(Ph(F)) (QM(F) - F): only in the tails. w = 0 between the 15th
                   and 85th percentile, rising linearly to 1 at the 5th / 95th. For 2 m
                   temperature both tails are corrected (heat and cold records), for wind
                   only the upper tail.
"""
from __future__ import annotations

import warnings

import numpy as np

LEVELS = np.array([0.001, 0.005, 0.01, 0.02, 0.05] + list(np.round(np.arange(0.1, 0.91, 0.05), 2))
                  + [0.95, 0.98, 0.99, 0.995, 0.999])
UPPER = (0.85, 0.95)        # weight 0 below 0.85, 1 above 0.95
LOWER = (0.15, 0.05)        # weight 0 above 0.15, 1 below 0.05
MIN_SAMPLES = 100           # fewer valid training values in a cell -> leave F unchanged


def interp_along(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, extrapolate: str = "shift") -> np.ndarray:
    """Piecewise-linear interpolation of x on the curve (xp, fp), separately for every
    grid cell. xp, fp: (P, ...) increasing along axis 0; x: (...) or (K, ...).
    extrapolate="shift": beyond the ends keep the end offset fp - xp (quantile mapping);
    "clip": hold the end value (for quantile levels)."""
    x = np.asarray(x, float)
    lead = x.ndim - (xp.ndim - 1)                       # extra leading axes of x (e.g. K)
    xp_b = xp.reshape(xp.shape[:1] + (1,) * lead + xp.shape[1:])
    fp_b = fp.reshape(fp.shape[:1] + (1,) * lead + fp.shape[1:])
    P = xp.shape[0]
    i = np.clip((xp_b <= x[None]).sum(0), 1, P - 1)     # x lies between xp[i-1] and xp[i]
    x0 = np.take_along_axis(np.broadcast_to(xp_b, (P,) + x.shape), (i - 1)[None], 0)[0]
    x1 = np.take_along_axis(np.broadcast_to(xp_b, (P,) + x.shape), i[None], 0)[0]
    y0 = np.take_along_axis(np.broadcast_to(fp_b, (P,) + x.shape), (i - 1)[None], 0)[0]
    y1 = np.take_along_axis(np.broadcast_to(fp_b, (P,) + x.shape), i[None], 0)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(x1 > x0, (x - x0) / (x1 - x0), 0.0)
    y = y0 + np.clip(t, 0, 1) * (y1 - y0)
    lo, hi = xp_b[0], xp_b[-1]
    if extrapolate == "shift":
        y = np.where(x < lo, x + (fp_b[0] - lo), y)
        y = np.where(x > hi, x + (fp_b[-1] - hi), y)
    else:
        y = np.where(x < lo, fp_b[0], y)
        y = np.where(x > hi, fp_b[-1], y)
    return y


def pool_neighbours(a: np.ndarray, radius: int = 1) -> np.ndarray:
    """(N, Y, X) -> (N * (2r+1)^2, Y, X): each cell's values plus those of its
    neighbours (NaN beyond the box edge)."""
    if radius == 0:
        return a
    N, Y, X = a.shape
    pad = np.full((N, Y + 2 * radius, X + 2 * radius), np.nan, a.dtype)
    pad[:, radius:radius + Y, radius:radius + X] = a
    parts = [pad[:, radius + dy:radius + dy + Y, radius + dx:radius + dx + X]
             for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    return np.concatenate(parts, 0)


def tail_weight(p: np.ndarray, upper: bool = True, lower: bool = True) -> np.ndarray:
    w = np.zeros_like(p, dtype=float)
    if upper:
        w = np.maximum(w, np.clip((p - UPPER[0]) / (UPPER[1] - UPPER[0]), 0, 1))
    if lower:
        w = np.maximum(w, np.clip((LOWER[0] - p) / (LOWER[0] - LOWER[1]), 0, 1))
    return w


def fit(H: np.ndarray, O: np.ndarray, radius: int = 1):
    """Quantiles of hindcasts and truth for one lead day. H, O: (J, k, Y, X) (k steps of
    the day, J hindcasts). Returns Qh, Qo (P, Y, X) and the number of valid pairs."""
    ok = np.isfinite(H) & np.isfinite(O)
    h = pool_neighbours(np.where(ok, H, np.nan).reshape(-1, *H.shape[2:]), radius)
    o = pool_neighbours(np.where(ok, O, np.nan).reshape(-1, *O.shape[2:]), radius)
    n = np.isfinite(h).sum(0)
    with warnings.catch_warnings():                       # all-NaN cells: NaN quantiles, masked later
        warnings.simplefilter("ignore", RuntimeWarning)
        Qh = np.nanquantile(h, LEVELS, axis=0)
        Qo = np.nanquantile(o, LEVELS, axis=0)
    return Qh, Qo, n


def correct(F: np.ndarray, H: np.ndarray, O: np.ndarray, steps_per_day: int = 4, radius: int = 1,
            upper: bool = True, lower: bool = True) -> dict:
    """F (K, Y, X); H, O (J, K, Y, X) -> {"QM": ..., "TailQM": ...}, each (K, Y, X).
    Lead steps are grouped by day (steps 1-4 = day 1, ...)."""
    K = F.shape[0]
    qm, tail = np.array(F, float), np.array(F, float)
    for s in range(0, K, steps_per_day):
        sl = slice(s, min(s + steps_per_day, K))
        Qh, Qo, n = fit(H[:, sl], O[:, sl], radius)
        good = n >= MIN_SAMPLES
        f = F[sl]
        mapped = interp_along(f, Qh, Qo, "shift")
        level = interp_along(f, Qh, np.broadcast_to(LEVELS.reshape(-1, *(1,) * (Qh.ndim - 1)), Qh.shape),
                             "clip")
        w = tail_weight(level, upper, lower)
        qm[sl] = np.where(good[None], mapped, f)
        tail[sl] = np.where(good[None], f + w * (mapped - f), f)
    return {"QM": qm, "TailQM": tail}
