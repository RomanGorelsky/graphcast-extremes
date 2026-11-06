"""Offline tests of the Phase 2 logic (DBSCAN events, features, Isolation Forest, regimes).

    python -m pytest -q tests
"""
import numpy as np
import pandas as pd
import xarray

from extremes import events, regimes


def _blob(lat, lon, t0, n_lat=3, n_lon=3, n_t=4, exc=1.0, kind="heat"):
    rows = []
    for i in range(n_lat):
        for j in range(n_lon):
            for k in range(n_t):
                rows.append(dict(kind=kind, lat=lat + i, lon=(lon + j) % 360,
                                 time=pd.Timestamp(t0) + pd.Timedelta(hours=6 * k), exceedance=exc))
    return pd.DataFrame(rows)


def test_dbscan_separates_events_and_noise():
    a = _blob(60, 80, "2020-05-20")                       # Siberia
    b = _blob(60, 80, "2020-07-20")                       # same place, two months later
    c = _blob(30, 260, "2020-08-27")                      # elsewhere
    noise = pd.DataFrame([dict(kind="heat", lat=-30, lon=20, time=pd.Timestamp("2020-03-01"), exceedance=1)])
    df = pd.concat([a, b, c, noise], ignore_index=True)
    lab = events.cluster_records(df)
    assert len(set(lab[:len(a)])) == 1 and len(set(lab[len(a):2 * len(a)])) == 1
    assert lab[0] != lab[len(a)] != lab[2 * len(a)]       # three different events
    assert lab[-1] == -1                                  # isolated record = noise


def test_dbscan_joins_across_the_zero_meridian():
    df = _blob(50, 358, "2020-02-15", n_lon=4)            # lons 358, 359, 0, 1
    assert len(set(events.cluster_records(df))) == 1


def test_event_features():
    df = _blob(0, 10, "2020-06-01", n_lat=2, n_lon=2, n_t=3, exc=2.0)
    df.loc[5, "exceedance"] = 5.0
    df["event"] = 0
    lsm = xarray.DataArray(np.ones((181, 360)), dims=("lat", "lon"),
                           coords={"lat": np.arange(-90, 91.0), "lon": np.arange(360.0)})
    ev = events.event_features(df, lsm)
    r = ev.iloc[0]
    assert r.n_cells == 4 and r.n_records == 12
    assert np.isclose(r.duration_h, 18)                   # 3 steps of 6 h
    assert np.isclose(r.peak_exceedance, 5.0)
    assert 4 * 12_000 < r.area_km2 < 4 * 12_400           # four ~1x1 deg cells near the equator
    assert r.coastal_fraction == 0 and r.event_id == "heat-001"


def test_isolation_forest_flags_the_planted_monster():
    rng = np.random.default_rng(0)
    n = 60
    ev = pd.DataFrame(dict(kind="heat", area_km2=10 ** rng.normal(4.5, 0.3, n),
                           duration_h=10 ** rng.normal(1.5, 0.2, n), peak_exceedance=rng.gamma(2, 0.5, n),
                           mean_exceedance=rng.gamma(2, 0.25, n), n_records=(10 ** rng.normal(1.5, 0.3, n)).astype(int) + 5))
    ev.loc[0, ["area_km2", "duration_h", "peak_exceedance", "mean_exceedance", "n_records"]] = [3e6, 400, 9.0, 4.0, 5000]
    out = events.isolation_flags(ev, contamination=0.05, n_seeds=5)
    assert bool(out.loc[0, "record_shattering"])
    assert out.record_shattering.sum() <= 6


def test_sector_of():
    assert events.sector_of(70, 100) == "asia_siberia"
    assert events.sector_of(50, 355) == "euro_atlantic"   # wraps through 0 deg
    assert events.sector_of(40, 260) == "north_america"
    assert events.sector_of(-15, 300) == "south_america"
    assert events.sector_of(-25, 135) == "other"          # Australia


def test_regimes_recover_planted_patterns():
    # daily fields = seasonal cycle + trend + one of 3 fixed anomaly patterns + noise
    t = pd.date_range("1979-01-01", "2020-12-31", freq="D")
    lat, lon = np.arange(30.0, 81, 5), np.arange(190.0, 311, 5)
    rng = np.random.default_rng(0)
    pats = rng.normal(0, 400, (3, lat.size, lon.size))
    which = rng.integers(0, 3, t.size)
    doy = t.dayofyear.values
    base = 55000 + 800 * np.cos(2 * np.pi * doy / 365.25)[:, None, None] + 2.0 * (t.year.values - 1979)[:, None, None]
    z = base + pats[which] + rng.normal(0, 60, (t.size, lat.size, lon.size))
    z = xarray.DataArray(z, dims=("time", "lat", "lon"), coords={"time": t, "lat": lat, "lon": lon})
    r = regimes.fit_sector(regimes.anomalies(z), "north_america", n_pcs=5, k_range=range(2, 6))
    assert r["k"] == 3
    lab = r["labels"].values
    # each planted pattern maps to one regime (up to renaming)
    purity = sum(np.bincount(lab[which == p]).max() for p in range(3)) / t.size
    assert purity > 0.99
