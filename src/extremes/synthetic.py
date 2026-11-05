"""Synthetic ERA5-like datasets for offline tests (no network needed).

The layout copies the authors' example files: dims (batch, time, level, lat, lon),
`time` = timedelta from the first input, `datetime` = (batch, time) coordinate.
"""
import numpy as np
import pandas as pd
import xarray

from weathernext.utils import data_utils
from weathernext.weathernext1_graph import graphcast

LEVELS = list(graphcast.TASK_13.pressure_levels)
_SURF = {"2m_temperature": (288.0, 15.0), "mean_sea_level_pressure": (101325.0, 1000.0),
         "10m_u_component_of_wind": (0.0, 5.0), "10m_v_component_of_wind": (0.0, 5.0),
         "total_precipitation_6hr": (0.0005, 0.001)}
_ATM = {"temperature": (250.0, 20.0), "geopotential": (50000.0, 5000.0),
        "u_component_of_wind": (5.0, 10.0), "v_component_of_wind": (0.0, 8.0),
        "vertical_velocity": (0.0, 0.2), "specific_humidity": (0.003, 0.003)}


def make_dataset(n_times: int = 6, res: float = 10.0, start: str = "2020-06-01T00",
                 seed: int = 0) -> xarray.Dataset:
    rng = np.random.default_rng(seed)
    lat = np.arange(-90, 90 + res / 2, res)
    lon = np.arange(0, 360, res)
    td = pd.to_timedelta(np.arange(n_times) * 6, unit="h")
    dt = pd.Timestamp(start) + td
    sh_s, sh_a = (1, n_times, lat.size, lon.size), (1, n_times, len(LEVELS), lat.size, lon.size)
    v = {}
    for k, (m, s) in _SURF.items():
        x = m + s * rng.standard_normal(sh_s)
        v[k] = (("batch", "time", "lat", "lon"), np.abs(x) if "precip" in k else x)
    for k, (m, s) in _ATM.items():
        x = m + s * rng.standard_normal(sh_a)
        v[k] = (("batch", "time", "level", "lat", "lon"), np.abs(x) if "humidity" in k else x)
    v["geopotential_at_surface"] = (("lat", "lon"), 1000 * np.abs(rng.standard_normal((lat.size, lon.size))))
    v["land_sea_mask"] = (("lat", "lon"), (rng.random((lat.size, lon.size)) > 0.7).astype(float))
    ds = xarray.Dataset({k: (d, a.astype(np.float32)) for k, (d, a) in v.items()},
                        coords={"lat": lat, "lon": lon, "level": LEVELS, "time": td,
                                "datetime": (("batch", "time"), dt.values[None])})
    return ds


def make_stats(ds: xarray.Dataset) -> dict:
    """Normalisation stats in the authors' format, computed from `ds` itself."""
    d = ds.copy()
    data_utils.add_derived_vars(d)
    data_utils.add_tisr_var(d)
    d = d.drop_vars("datetime")
    dims = [x for x in ("batch", "time", "lat", "lon") if x in d.dims]
    mean = d.mean(dims)
    std = d.std(dims) + 1e-6
    diffs = d.diff("time").std(dims) + 1e-6
    return {"mean_by_level": mean, "stddev_by_level": std, "diffs_stddev_by_level": diffs}
