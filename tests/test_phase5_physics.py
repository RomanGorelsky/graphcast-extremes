"""Offline tests of the conservation schemes (Phase 5): each budget closes after its
scheme, nothing else changes, and gradients pass through.

    python -m pytest -q tests
"""
import numpy as np
import pytest
import xarray

from extremes import physics as ph

LEVELS = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])


def _state(rng, lat, lon, time=False):
    """A plausible atmosphere on a coarse global grid (poles included)."""
    nl, ny, nx = len(LEVELS), len(lat), len(lon)
    shape3 = (1, nl, ny, nx)
    frac = (LEVELS / 1000.0)[None, :, None, None]
    coords = {"level": LEVELS, "lat": lat, "lon": lon}
    dims3 = ("batch", "level", "lat", "lon")
    ds = xarray.Dataset({
        "specific_humidity": (dims3, (0.015 * frac ** 3 * rng.uniform(0.3, 1.2, shape3)).astype(np.float32)),
        "temperature": (dims3, (210 + 80 * frac + rng.normal(0, 3, shape3)).astype(np.float32)),
        "u_component_of_wind": (dims3, rng.normal(5, 10, shape3).astype(np.float32)),
        "v_component_of_wind": (dims3, rng.normal(0, 8, shape3).astype(np.float32)),
    }, coords=coords)
    if time:
        ds = ds.expand_dims(time=[np.timedelta64(6, "h")], axis=1)
    return ds


def _forecast(rng, prev):
    """prev + small random changes + 6-h accumulated fluxes (ERA5 sign conventions)."""
    fc = (prev + 0).expand_dims(time=[np.timedelta64(6, "h")], axis=1)
    for v, s in (("specific_humidity", 3e-4), ("temperature", 0.8), ("u_component_of_wind", 1.0),
                 ("v_component_of_wind", 1.0)):
        fc[v] = (fc[v] + rng.normal(0, s, fc[v].shape)).astype(np.float32)
    fc["specific_humidity"] = fc.specific_humidity.where(fc.specific_humidity > 0, 1e-6)
    dims2 = ("batch", "time", "lat", "lon")
    shp = (1, 1, prev.sizes["lat"], prev.sizes["lon"])
    six_h = 6 * 3600.0
    flux = {ph.PRECIP: np.abs(rng.normal(0.0015, 0.001, shp)),
            ph.EVAP: -np.abs(rng.normal(0.0015, 0.0007, shp)),
            ph.TOA_SOLAR: rng.uniform(0, 600, shp) * six_h,
            ph.TOA_THERMAL: -rng.uniform(200, 280, shp) * six_h,
            ph.SFC_SOLAR: rng.uniform(0, 300, shp) * six_h,
            ph.SFC_THERMAL: -rng.uniform(30, 90, shp) * six_h,
            ph.SFC_SENSIBLE: rng.normal(-20, 20, shp) * six_h,
            ph.SFC_LATENT: rng.normal(-80, 30, shp) * six_h}
    for k, a in flux.items():
        fc[k] = (dims2, a.astype(np.float32))
    return fc


@pytest.fixture
def case():
    rng = np.random.default_rng(0)
    lat = np.linspace(-90, 90, 19)                       # 10-degree grid incl. poles
    lon = np.arange(0, 360, 10.0)
    prev = _state(rng, lat, lon)
    fc = _forecast(rng, prev)
    phi_s = xarray.DataArray(rng.uniform(0, 3e4, (19, 36)).astype(np.float32), dims=("lat", "lon"),
                             coords={"lat": lat, "lon": lon})
    return prev, fc, phi_s


def test_weights_integrate_the_sphere_and_the_column():
    lat = xarray.DataArray(np.linspace(-90, 90, 181), dims="lat")
    lon = xarray.DataArray(np.arange(0, 360, 1.0), dims="lon")
    a = ph.cell_area(lat, lon)
    assert np.isclose(float(a.sum()), 4 * np.pi * ph.R_EARTH ** 2, rtol=1e-5)
    w = ph.level_weights(xarray.DataArray(LEVELS, dims="level", coords={"level": LEVELS}))
    assert np.isclose(float(w.sum()), (1000 - 50) * 100 / ph.G, rtol=1e-6)


def test_random_forecast_violates_all_budgets(case):
    prev, fc, phi_s = case
    r = ph.residuals(prev, fc, phi_s)
    assert all(abs(float(r[k].max())) > 1e-5 for k in ("mass", "moisture", "energy"))


def test_dry_air_mass_closes_and_upper_levels_stay(case):
    prev, fc, phi_s = case
    out = ph.fix_dry_air_mass(prev, fc)
    assert abs(float(ph.residuals(prev, out, phi_s)["mass"].max())) < 1e-6
    upper = LEVELS < ph.P_SPLIT_HPA
    np.testing.assert_array_equal(out.specific_humidity.sel(level=LEVELS[upper]).values,
                                  fc.specific_humidity.sel(level=LEVELS[upper]).values)
    for v in fc.data_vars:
        if v != "specific_humidity":
            np.testing.assert_array_equal(out[v].values, fc[v].values)


def test_moisture_budget_closes_by_changing_precipitation_only(case):
    prev, fc, phi_s = case
    out = ph.fix_moisture(prev, fc)
    assert abs(float(ph.residuals(prev, out, phi_s)["moisture"].max())) < 1e-5
    for v in fc.data_vars:
        if v != ph.PRECIP:
            np.testing.assert_array_equal(out[v].values, fc[v].values)
    ratio = (out[ph.PRECIP] / fc[ph.PRECIP]).values
    assert np.allclose(ratio, ratio.flat[0], rtol=1e-5)          # one global ratio


def test_precipitation_ratio_is_capped_for_a_drifting_forecast(case):
    prev, fc, phi_s = case
    fc2 = fc.copy()
    fc2[ph.PRECIP] = fc[ph.PRECIP] * 1e-3                 # far too little rain to close the budget
    out = ph.fix_moisture(prev, fc2)
    np.testing.assert_allclose((out[ph.PRECIP] / fc2[ph.PRECIP]).values, 10.0, rtol=1e-5)
    fc3 = fc.copy()
    fc3[ph.EVAP] = -fc[ph.EVAP] * 50                      # budget would need negative rain
    out = ph.fix_moisture(prev, fc3)
    assert float(out[ph.PRECIP].min()) == 0.0 and float(out[ph.PRECIP].max()) == 0.0


def test_energy_closes_by_changing_temperature_only(case):
    prev, fc, phi_s = case
    out = ph.fix_energy(prev, fc, phi_s)
    assert abs(float(ph.residuals(prev, out, phi_s)["energy"].max())) < 1e-5
    for v in fc.data_vars:
        if v != "temperature":
            np.testing.assert_array_equal(out[v].values, fc[v].values)
    assert float(abs(out.temperature - fc.temperature).max()) < 20     # a correction, not a new field


def test_all_schemes_in_order_close_all_budgets(case):
    prev, fc, phi_s = case
    out = ph.apply(prev, fc, phi_s)
    r = ph.residuals(prev, out, phi_s)
    assert abs(float(r["mass"].max())) < 1e-6
    assert abs(float(r["moisture"].max())) < 1e-5
    assert abs(float(r["energy"].max())) < 1e-5


def test_control_run_is_untouched_and_negatives_are_clipped(case):
    prev, fc, phi_s = case
    fc2 = fc.copy()
    fc2["specific_humidity"] = fc.specific_humidity - 0.01                  # many negatives
    assert ph.apply(prev, fc2, phi_s, ()) is fc2
    out = ph.non_negative(fc2)
    assert float(out.specific_humidity.min()) >= 0
    with pytest.raises(ValueError):
        ph.apply(prev, fc, phi_s, ("momentum",))


def test_gradients_pass_through_the_schemes(case):
    jax = pytest.importorskip("jax")
    xarray_jax = pytest.importorskip("xarray_jax")
    jnp = jax.numpy
    prev, fc, phi_s = case

    def loss(t):                       # xarray accepts JAX arrays directly
        f = xarray.Dataset({v: (fc[v].dims, jnp.asarray(fc[v].values)) for v in fc.data_vars},
                           coords=fc.coords)
        f["temperature"] = (fc.temperature.dims, t)
        out = ph.apply(prev, f, phi_s)
        return jnp.sum(xarray_jax.unwrap_data(out["temperature"]) ** 2) * 1e-6

    g = jax.grad(loss)(jnp.asarray(fc.temperature.values))
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0
