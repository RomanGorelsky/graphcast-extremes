"""Phase 7, prospect 2: a larger set of case studies (record-shattering events of
2020-2022) built with exactly the same Phase 2 rules as the 14 original case studies.

Event ids are made unique across years: "<year>-<event_id>", e.g. 2021-heat-004
(event_id restarts at 001 every year). `source_event_id` keeps the id used in
results/phase2/events_<year>.csv, so verification finds the event's record cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEADS_DAYS = (1, 3, 5, 7)
DATA_END = pd.Timestamp("2023-01-10 18:00")      # last time in the WeatherBench2 ERA5 store
CAPS = {"heat": 15, "cold": 8, "wind": 10}       # per kind and year (largest first); --all lifts them


def lon_range(lons: np.ndarray) -> tuple[float, float]:
    """Smallest longitude interval (0..360, may wrap) containing all lons (as in script 24)."""
    u = np.sort(np.unique(np.asarray(lons) % 360))
    gaps = np.diff(np.r_[u, u[0] + 360])
    i = int(np.argmax(gaps))
    return float(u[(i + 1) % u.size]), float(u[i])


def eligible(ev: pd.DataFrame, min_cells: int = 5, max_coastal: float = 0.5, max_lat: float = 75.0) -> pd.Series:
    """The Phase 2 (script 24) rules: not tiny, not mostly coastal, not in the high Arctic."""
    return ((ev.n_cells >= min_cells) & (ev.coastal_fraction.fillna(0) <= max_coastal)
            & (ev.centre_lat <= max_lat))


def choose(ev: pd.DataFrame, caps: dict | None, only_shattering: bool = True, **rules) -> pd.DataFrame:
    """Eligible (record-shattering) events of one year, largest area-weighted exceedance
    first, at most caps[kind] per kind (caps=None: all)."""
    ok = eligible(ev, **rules)
    if only_shattering:
        ok &= ev.record_shattering.astype(bool)
    out = []
    for kind, g in ev[ok].groupby("kind"):
        g = g.sort_values("weighted_exceedance", ascending=False)
        out.append(g if caps is None else g.head(caps.get(kind, len(g))))
    return pd.concat(out) if out else ev.iloc[:0]


def case_row(e, cells: pd.DataFrame, year: int, margin: float = 5.0, reason: str = "") -> dict:
    """One case-study row in the format of results/phase2/case_studies.csv (+ year, source_event_id)."""
    lo0, lo1 = lon_range(cells.lon.values)
    peak_day = pd.Timestamp(e.peak_time).normalize()
    return dict(
        event_id=f"{year}-{e.event_id}", source_event_id=e.event_id, year=int(year), kind=e.kind,
        peak_time=e.peak_time, start=e.start, end=e.end,
        centre_lat=round(float(e.centre_lat), 1), centre_lon=round(float(e.centre_lon), 1), n_cells=int(e.n_cells),
        peak_exceedance=round(float(e.peak_exceedance), 2), record_shattering=bool(e.record_shattering),
        regime=e.regime, sector=e.sector,
        box_lat_min=max(float(cells.lat.min()) - margin, -90), box_lat_max=min(float(cells.lat.max()) + margin, 90),
        box_lon_min=(lo0 - margin) % 360, box_lon_max=(lo1 + margin) % 360,
        **{f"init_minus_{d}d": peak_day - pd.Timedelta(days=d) for d in LEADS_DAYS},
        reason=reason)


def fits_in_data(row: dict, n_days: int = 10) -> bool:
    """Every forecast (start -1 d is the latest) and its 10 days of truth lie in the store."""
    return pd.Timestamp(row["init_minus_1d"]) + pd.Timedelta(days=n_days) <= DATA_END
