"""Offline tests of the Phase 3 logic (input builder, cropping, verification metrics).

    python -m pytest -q tests
"""
import numpy as np
import pandas as pd
import xarray

from extremes import era5, inputs, model, phase3, synthetic, verify
from weathernext.weathernext1_graph import graphcast


def _fake_store(start="2020-06-01T00", n_times=6) -> xarray.Dataset:
    """A 1-degree 'WeatherBench2-like' store: latitude/longitude names, (time, level,
    longitude, latitude) axis order and descending latitude - all of which the
    reader must undo."""
    rng = np.random.default_rng(0)
    t = pd.date_range(start, periods=n_times, freq="6h")
    lat, lon = era5.LAT_1DEG[::-1], era5.LON_1DEG
    shp_s, shp_a = (n_times, lon.size, lat.size), (n_times, len(inputs.LEVELS), lon.size, lat.size)
    v = {}
    for var in inputs.SURFACE:
        v[var] = (("time", "longitude", "latitude"), (280 + rng.standard_normal(shp_s)).astype(np.float32))
    for var in inputs.ATMOS:
        v[var] = (("time", "level", "longitude", "latitude"), (250 + rng.standard_normal(shp_a)).astype(np.float32))
    v[inputs.TISR] = (("time", "longitude", "latitude"), np.abs(rng.standard_normal(shp_s)).astype(np.float32) * 1e6)
    v["geopotential_at_surface"] = (("longitude", "latitude"), rng.random((lon.size, lat.size)).astype(np.float32))
    v["land_sea_mask"] = (("longitude", "latitude"), rng.random((lon.size, lat.size)).astype(np.float32))
    ds = xarray.Dataset(v, coords={"time": t, "level": inputs.LEVELS, "latitude": lat, "longitude": lon})
    return era5.standardise(ds)


def test_build_batch_layout_and_rollout():
    ds = _fake_store()
    init = pd.Timestamp("2020-06-01T06")
    b = inputs.build_batch(ds, init, n_steps=2, io_threads=2)
    assert b.sizes == {"batch": 1, "time": 4, "level": 13, "lat": 181, "lon": 360}
    assert list(pd.to_timedelta(b.time.values)) == [pd.Timedelta(h, "h") for h in (0, 6, 12, 18)]
    assert pd.Timestamp(b.datetime.values[0, 1]) == init
    # input values are the store's values (lat flipped back to ascending)
    src = ds["2m_temperature"].sel(time=init).transpose("lat", "lon").values
    assert np.array_equal(b["2m_temperature"].isel(batch=0, time=1).values, src)
    # forecast-time placeholders are NaN, solar radiation is real at all times
    assert np.isnan(b["temperature"].isel(time=2)).all()
    assert np.isfinite(b[inputs.TISR]).all()
    # the batch goes through the model unchanged
    stats = synthetic.make_stats(synthetic.make_dataset(n_times=6, res=1.0))
    mc = graphcast.ModelConfig(resolution=0, mesh_size=2, latent_size=16, gnn_msg_steps=2,
                               hidden_layers=1, radius_query_fraction_edge_length=0.6)
    m = model.Model(params=None, state={}, model_config=mc, task_config=graphcast.TASK_13, stats=stats)
    i, tg, f = m.extract(b, 2)
    m.init_random_params(i, tg, f)
    pred = m.forecast(i, tg, f)
    assert all(np.isfinite(pred[v].values).all() for v in pred.data_vars)


def test_crop_across_zero_meridian():
    da = xarray.DataArray(np.zeros((181, 360)), dims=("lat", "lon"),
                          coords={"lat": era5.LAT_1DEG, "lon": era5.LON_1DEG})
    c = phase3.crop(da, 40, 50, 350, 10)
    assert c.sizes == {"lat": 11, "lon": 21}
    assert set(c.lon.values) == set(range(350, 360)) | set(range(0, 11))


def _fc(kind_value_truth, kind_value_fc, times, lat=[50.0], lon=[10.0]):
    shp = (len(times), len(lat), len(lon))
    ds = xarray.Dataset(coords={"time": times, "lat": lat, "lon": lon})
    for which, val in (("truth", kind_value_truth), ("forecast", kind_value_fc)):
        ds[f"{which}_2m_temperature"] = (("time", "lat", "lon"), np.full(shp, val, dtype=float))
        ds[f"{which}_10m_u_component_of_wind"] = (("time", "lat", "lon"), np.full(shp, 3.0))
        ds[f"{which}_10m_v_component_of_wind"] = (("time", "lat", "lon"), np.full(shp, 4.0))
        ds[f"{which}_mean_sea_level_pressure"] = (("time", "lat", "lon"), np.full(shp, 1e5))
    return ds


def test_cold_sign_convention_and_hits():
    times = pd.date_range("2020-10-27T00", periods=4, freq="6h")
    fc = _fc(250.0, 253.0, times)                     # observed 250 K, forecast 3 K too warm
    recs = pd.DataFrame(dict(time=[times[2]], lat=[50.0], lon=[10.0], value=[250.0], threshold=[252.0]))
    e = verify.record_errors(fc, recs, "cold", init=times[0])
    r = e.iloc[0]
    assert np.isclose(r.raw_bias, 3.0) and np.isclose(r.intensity_error, -3.0)   # too mild
    assert np.isclose(r.obs_exceedance, 2.0) and np.isclose(r.fc_exceedance, -1.0)
    assert not r.hit and np.isclose(r.lead_h, 12) and r.truth_check == 0
    s = verify.case_summary(e)
    assert np.isclose(s["capture_ratio"], -0.5) and s["hit_rate"] == 0


def test_heat_hit_and_records_before_init_are_skipped():
    times = pd.date_range("2020-05-22T00", periods=4, freq="6h")
    fc = _fc(300.0, 299.0, times)                      # forecast 1 K too cold, still above 298 threshold
    recs = pd.DataFrame(dict(time=[times[0], times[3]], lat=[50.0, 50.0], lon=[10.0, 10.0],
                             value=[300.0, 300.0], threshold=[298.0, 298.0]))
    e = verify.record_errors(fc, recs, "heat", init=times[0])
    assert len(e) == 1                                 # the record AT the start time is not a forecast
    assert bool(e.iloc[0].hit) and np.isclose(e.iloc[0].intensity_error, -1.0)


def test_wind_speed_and_precision():
    times = pd.date_range("2020-03-12T00", periods=2, freq="6h")
    fc = _fc(280.0, 280.0, times, lat=[25.0, 26.0], lon=[30.0])
    fc["truth_10m_u_component_of_wind"][:] = 0.0                   # truth speed 4, forecast speed 5
    th = xarray.Dataset({"wind_max": (("month", "lat", "lon"), np.full((12, 181, 360), 4.5))},
                        coords={"month": range(1, 13), "lat": era5.LAT_1DEG, "lon": era5.LON_1DEG})
    mask = xarray.DataArray(np.ones((181, 360), bool), dims=("lat", "lon"),
                            coords={"lat": era5.LAT_1DEG, "lon": era5.LON_1DEG})
    b = verify.box_exceedances(fc, th, mask, "wind", times[0], times[-1])
    assert b["n_fc_records"] == 4 and b["n_obs_records"] == 0 and b["precision"] == 0
