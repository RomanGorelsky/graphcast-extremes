"""Offline tests of the Phase 1 logic (regridding, thresholds, record detection).

    python -m pytest -q tests
"""
import numpy as np
import pandas as pd
import xarray

from extremes import era5, records

LAT_Q = np.linspace(-90, 90, 721)
LON_Q = np.arange(1440) * 0.25


def _quarter(field: np.ndarray, n_times: int = 1) -> xarray.DataArray:
    t = pd.date_range("2020-06-01", periods=n_times, freq="6h")
    return xarray.DataArray(np.broadcast_to(field, (n_times, 721, 1440)).astype(np.float32),
                            dims=("time", "lat", "lon"), coords={"time": t, "lat": LAT_Q, "lon": LON_Q})


def test_regrid_constant_field_is_unchanged():
    for method in ("subsample", "conservative"):
        out = era5.regrid(_quarter(np.full((721, 1440), 280.0)), method)
        assert out.shape == (1, 181, 360)
        assert np.allclose(out.values, 280.0, atol=1e-3), method
        assert np.allclose(out.lat, era5.LAT_1DEG) and np.allclose(out.lon, era5.LON_1DEG)


def test_subsample_takes_the_1deg_points():
    f = np.random.default_rng(0).standard_normal((721, 1440))
    out = era5.regrid(_quarter(f), "subsample").values[0]
    assert np.array_equal(out, f[::4, ::4].astype(np.float32))


def test_conservative_is_a_local_average():
    # a single hot 0.25-deg point inside a cell raises that cell by 1/16 of the spike
    # (interior point, weight 1 of 4 in each direction), and no other cell
    f = np.zeros((721, 1440))
    f[4 * 100 + 1, 4 * 50 + 1] = 16.0               # inside cell (lat index 100, lon index 50)
    out = era5.regrid(_quarter(f), "conservative").values[0]
    assert np.isclose(out[100, 50], 1.0, rtol=0.01)
    out[100, 50] = 0
    assert np.abs(out).max() < 1e-6
    # longitude wraps around: a point just west of 0 deg belongs to cell lon=0
    f = np.zeros((721, 1440)); f[400, 1439] = 16.0
    out = era5.regrid(_quarter(f), "conservative").values[0]
    assert out[100, 0] > 0.9


def _one_deg(values: np.ndarray, start="2020-07-01") -> xarray.DataArray:
    t = pd.date_range(start, periods=values.shape[0], freq="6h")
    return xarray.DataArray(values.astype(np.float32), dims=("time", "lat", "lon"),
                            coords={"time": t, "lat": era5.LAT_1DEG, "lon": era5.LON_1DEG})


def test_thresholds_and_records():
    rng = np.random.default_rng(1)
    # two "years" of July, 8 time steps each
    years = []
    for _ in range(2):
        t2m = _one_deg(290 + rng.random((8, 181, 360)))     # 290 .. 291
        wsp = _one_deg(5 * rng.random((8, 181, 360)))       # 0 .. 5
        years.append(records.stack_months({7: records.month_extremes(t2m, wsp)}))
    th = records.combine_years(years)
    assert int(th.n_years) == 2 and int(th.n_times.sel(month=7)) == 16
    assert float(th.t2m_max.max()) <= 291 and float(th.t2m_min.min()) >= 290

    # test year: all within range except three planted records
    mid = ((th.t2m_max + th.t2m_min) / 2).sel(month=7).values     # safely inside the range
    t2m = _one_deg(np.broadcast_to(mid, (4, 181, 360)).copy())
    wsp = _one_deg(np.full((4, 181, 360), 1.0))
    i_lat, i_lon = 150, 30                        # 60N, 30E
    t2m[1, i_lat, i_lon] = float(th.t2m_max.sel(month=7)[i_lat, i_lon]) + 2.0     # heat, +2 K
    t2m[2, i_lat, i_lon] = float(th.t2m_min.sel(month=7)[i_lat, i_lon]) - 3.0     # cold, -3 K
    wsp[3, i_lat, i_lon] = float(th.wind_max.sel(month=7)[i_lat, i_lon]) + 4.0    # wind, +4 m/s
    t2m[1, 10, 30] = 400.0                        # 80S: outside the analysis area
    lsm = xarray.zeros_like(th.t2m_max.sel(month=7)) + 1.0   # all land
    mask = records.analysis_mask(lsm)
    df = records.find_records(t2m, wsp, th, mask)
    got = {r.kind: (r.lat, r.lon, round(float(r.exceedance), 3)) for r in df.itertuples()}
    assert len(df) == 3, df
    assert got["heat"] == (60.0, 30.0, 2.0)
    assert got["cold"] == (60.0, 30.0, 3.0)
    assert got["wind"] == (60.0, 30.0, 4.0)


def test_sea_cells_are_excluded():
    th = records.combine_years([records.stack_months({7: records.month_extremes(
        _one_deg(np.full((2, 181, 360), 290.0)), _one_deg(np.full((2, 181, 360), 3.0)))})])
    t2m = _one_deg(np.full((1, 181, 360), 300.0))           # +10 K everywhere
    wsp = _one_deg(np.full((1, 181, 360), 1.0))
    lsm = xarray.zeros_like(th.t2m_max.sel(month=7))         # all sea
    assert len(records.find_records(t2m, wsp, th, records.analysis_mask(lsm))) == 0
