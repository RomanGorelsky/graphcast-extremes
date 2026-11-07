"""Shared settings of Phase 4 (probabilistic bias correction, deterministic version).

Training data = GraphCast_small "hindcasts": the same model, started at 00 UTC on
dates of out-of-sample years (2016-2019; 2020 is the test year) that lie within
+-HALF_WINDOW days of the calendar date of each event forecast's start - the
Debias++ idea of Guan et al. (2026) of learning from past forecasts issued at the
same time of year.
"""
from __future__ import annotations

import pandas as pd

from . import paths, phase3

OUT = paths.RESULTS_DIR / "phase4"
HINDCASTS = OUT / "hindcasts"
TRUTH = OUT / "truth"
CORRECTED = OUT / "forecasts_PBC"
YEARS = (2019, 2018, 2017, 2016)        # newest first: if time runs out, recent years exist
HALF_WINDOW = 7                          # days


def same_date_in(year: int, t: pd.Timestamp) -> pd.Timestamp:
    """The calendar date of t in another year (29 Feb -> 28 Feb), at 00 UTC."""
    t = pd.Timestamp(t).normalize()
    day = 28 if (t.month == 2 and t.day == 29) else t.day
    return pd.Timestamp(year=year, month=t.month, day=day)


def training_inits(event_init, years=YEARS, half_window: int = HALF_WINDOW) -> list[pd.Timestamp]:
    """Hindcast start dates used to correct a forecast started at `event_init`."""
    out = []
    for y in years:
        c = same_date_in(y, event_init)
        out += [c + pd.Timedelta(days=k) for k in range(-half_window, half_window + 1)]
    lo, hi = pd.Timestamp(f"{min(years)}-01-01"), pd.Timestamp(f"{max(years)}-12-31")
    return sorted(t for t in out if lo <= t <= hi)


def plan(cases: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (hindcast start, event) that needs it."""
    cases = phase3.case_studies() if cases is None else cases
    rows = []
    for c in cases.itertuples():
        for d in phase3.LEADS_DAYS:
            for t in training_inits(getattr(c, f"init_minus_{d}d")):
                rows.append(dict(init=t, event_id=c.event_id, start_days=d))
    p = pd.DataFrame(rows).drop_duplicates(["init", "event_id", "start_days"])
    return p


def hindcast_path(event_id: str, init):
    return HINDCASTS / event_id / f"{pd.Timestamp(init):%Y%m%d%H}.nc"


def truth_path(event_id: str):
    return TRUTH / f"{event_id}.nc"
