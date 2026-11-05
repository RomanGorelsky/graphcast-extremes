"""Basic verification metrics on regular lat-lon grids."""
import numpy as np
import xarray


def lat_weights(lat: xarray.DataArray) -> xarray.DataArray:
    """Area weights proportional to cos(latitude), normalised to mean 1."""
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()


def rmse(forecast: xarray.DataArray, truth: xarray.DataArray, over=("lat", "lon")) -> xarray.DataArray:
    """Latitude-weighted RMSE over the given dims (other dims, e.g. time, are kept)."""
    err2 = (forecast - truth) ** 2
    return np.sqrt(err2.weighted(lat_weights(truth["lat"])).mean(over))
