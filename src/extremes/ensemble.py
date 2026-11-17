"""Phase 7, prospect 4 (in-scope part): a small GraphCast_small ensemble from perturbed
starting states.

GraphCast_small is deterministic, so differences between ensemble members can only come
from the starting state. We use "random field perturbations" (Magnusson et al. 2009):

    delta = alpha * (S_a - S_b)

where S_a, S_b are two real ERA5 states from other years at about the same time of year
(a pool of 8: the event's calendar date -6 and +6 days in 2016-2019). A difference of
two real states is physically balanced (winds fit the pressure field), unlike random
noise, and has realistic spatial patterns; alpha scales it to a plausible
analysis uncertainty. Members come in pairs +delta / -delta, so the perturbations
average to zero. The same delta is added to both input times (init - 6 h and init) of
every state variable; humidity and precipitation are kept non-negative. Forcings
(solar radiation) and static fields are not touched.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import xarray

from . import inputs

PERTURBED = list(inputs.SURFACE) + list(inputs.ATMOS)
NON_NEGATIVE = ("specific_humidity", "total_precipitation_6hr")
POOL_YEARS = (2016, 2017, 2018, 2019)
POOL_OFFSETS = (-6, 6)                     # days around the event's calendar date


def pool_dates(peak, years=POOL_YEARS, offsets=POOL_OFFSETS) -> list[pd.Timestamp]:
    """00 UTC dates of other years around the calendar date of `peak` (29 Feb -> 28 Feb)."""
    p = pd.Timestamp(peak).normalize()
    out = []
    for y in years:
        base = pd.Timestamp(year=y, month=p.month, day=28 if (p.month == 2 and p.day == 29) else p.day)
        out += [base + pd.Timedelta(days=o) for o in offsets]
    return out


def load_state(ds: xarray.Dataset, t, statics: dict, io_threads: int = 32) -> xarray.Dataset:
    """The ERA5 state (all perturbed variables) at time t on the 1-degree grid,
    dims (batch=1, [level,] lat, lon) - read the same way as the model's inputs."""
    b = inputs.build_batch(ds, pd.Timestamp(t), 1, statics=statics, io_threads=io_threads)
    return b[PERTURBED].isel(time=1).drop_vars(["time", "datetime"], errors="ignore").load()


def member_pairs(n_pool: int, n_members: int, seed: int = 0) -> list[tuple[int, int, int]]:
    """(a, b, sign) for each perturbed member: n_members // 2 distinct pool pairs, each
    used with +1 and -1. n_members must be even."""
    if n_members % 2:
        raise ValueError("the number of perturbed members must be even (+delta / -delta pairs)")
    combos = list(itertools.combinations(range(n_pool), 2))
    if n_members // 2 > len(combos):
        raise ValueError(f"{n_members} members need {n_members // 2} pairs, the pool gives only {len(combos)}")
    order = np.random.default_rng(seed).permutation(len(combos))[: n_members // 2]
    out = []
    for k in order:
        a, b = combos[int(k)]
        out += [(a, b, +1), (a, b, -1)]
    return out


def perturbation(pool: list[xarray.Dataset], a: int, b: int, sign: int, alpha: float) -> xarray.Dataset:
    return (sign * alpha) * (pool[a] - pool[b])


def perturb(inp: xarray.Dataset, delta: xarray.Dataset) -> xarray.Dataset:
    """Model inputs (from Model.extract) + delta at every input time, state variables only."""
    out = inp.copy()
    for v in delta.data_vars:
        if v not in inp:
            continue
        x = inp[v] + delta[v].astype(inp[v].dtype)
        if v in NON_NEGATIVE:
            x = x.clip(min=0)
        out[v] = x.transpose(*inp[v].dims)
    return out
