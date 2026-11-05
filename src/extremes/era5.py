"""Reading ERA5 from the public WeatherBench2 bucket and putting it on the 1-degree grid.

GraphCast_small works on a 1-degree grid (181 latitudes from -90 to 90, 360
longitudes from 0 to 359). WeatherBench2 publishes ERA5 at 0.25 degrees
(721 x 1440). Two ways to get to 1 degree are implemented:

  'subsample'     keep every 4th point (the 1-degree points are a subset of
                  the 0.25-degree points)
  'conservative'  area-weighted average over each 1-degree cell
                  (+-0.5 deg around the point; edge points count half)

`scripts/10_check_regridding.py` decides which one matches the data the model
was trained on, by comparing with the authors' 1-degree example file.
The chosen method is then used for EVERYTHING (thresholds, records, truth).

All reads are anonymous: no Google account needed.
"""
from __future__ import annotations

import numpy as np
import xarray

# WeatherBench2 ERA5, 0.25 deg, 6-hourly (00/06/12/18 UTC), 1959 - 2023-01-10.
# `scripts/10_check_regridding.py --list` prints what is in the bucket, in case
# the name has changed; any store can be passed with --store.
WB2_ERA5 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

T2M, U10, V10 = "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"
LSM = "land_sea_mask"

LAT_1DEG = np.arange(-90.0, 90.5, 1.0)       # 181
LON_1DEG = np.arange(0.0, 360.0, 1.0)        # 360


def open_store(store: str = WB2_ERA5) -> xarray.Dataset:
    """Lazily open a zarr store (nothing is downloaded until values are used).

    chunks={} keeps the store's own chunking as dask chunks, so that a read of
    many time steps fetches them in parallel (see `load_1deg`)."""
    ds = xarray.open_zarr(store, storage_options={"token": "anon"}, chunks={})
    return standardise(ds)


def list_bucket(prefix: str = "weatherbench2/datasets/era5/") -> list[str]:
    import gcsfs
    return sorted(gcsfs.GCSFileSystem(token="anon").ls(prefix))


def standardise(ds: xarray.Dataset) -> xarray.Dataset:
    """Names lat/lon, latitude ascending (-90..90), longitude 0..360."""
    ren = {k: v for k, v in (("latitude", "lat"), ("longitude", "lon")) if k in ds.dims}
    ds = ds.rename(ren)
    if ds.lat[0] > ds.lat[-1]:
        ds = ds.isel(lat=slice(None, None, -1))
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=ds.lon % 360).sortby("lon")
    return ds


# --------------------------------------------------------------------------- regridding
def _check_quarter_grid(lat: np.ndarray, lon: np.ndarray) -> None:
    if lat.size != 721 or lon.size != 1440 or not np.isclose(lat[0], -90) or not np.isclose(lon[0], 0):
        raise ValueError(f"expected an ascending 0.25-deg grid (721 x 1440 from -90/0), got {lat.size} x {lon.size}")


def _conservative_weights() -> tuple[np.ndarray, np.ndarray]:
    """Offsets (in 0.25-deg steps) and weights covering one 1-deg cell."""
    off = np.arange(-2, 3)                        # -0.5 .. +0.5 deg
    w = np.array([0.5, 1.0, 1.0, 1.0, 0.5])       # edge points shared with the neighbour cell
    return off, w / w.sum()


def regrid_array(x: np.ndarray, method: str) -> np.ndarray:
    """(..., 721, 1440) array on the 0.25-deg grid -> (..., 181, 360) on the 1-deg grid."""
    if method == "subsample":
        return x[..., ::4, ::4]
    if method != "conservative":
        raise ValueError(method)
    lat_q = np.linspace(-90, 90, 721)
    off, w = _conservative_weights()
    rows = 4 * np.arange(181)[:, None] + off[None, :]          # (181, 5) latitude indices
    rows = np.clip(rows, 0, 720)                                # poles: cell is clipped at the pole
    cols = (4 * np.arange(360)[:, None] + off[None, :]) % 1440  # (360, 5) periodic in longitude
    # latitude weights: trapezoid weight x cos(latitude) (area), renormalised per cell;
    # at the poles, the duplicated (clipped) rows get zero weight.
    wl = w[None, :] * np.cos(np.deg2rad(lat_q[rows]))
    dup = (4 * np.arange(181)[:, None] + off[None, :] != rows)
    wl = np.where(dup, 0.0, wl)
    pole = (np.arange(181) == 0) | (np.arange(181) == 180)
    wl[pole] = np.where(dup[pole], 0.0, w[None, :])             # cos=0 at the pole itself
    wl = wl / wl.sum(1, keepdims=True)
    # sum over the 5x5 stencil: first longitudes, then latitudes
    xl = np.tensordot(x[..., cols], w, axes=([-1], [0]))       # (..., 721, 360)
    xr = xl[..., rows, :]                                       # (..., 181, 5, 360)
    return np.einsum("...ijk,ij->...ik", xr, wl).astype(x.dtype)


def regrid(da: xarray.DataArray, method: str) -> xarray.DataArray:
    """DataArray with (lat, lon) as its last two dims -> same on the 1-deg grid."""
    _check_quarter_grid(da.lat.values, da.lon.values)
    out = regrid_array(np.asarray(da.values), method)
    dims = da.dims
    coords = {d: da[d] for d in dims[:-2] if d in da.coords}
    return xarray.DataArray(out, dims=dims, coords={**coords, "lat": LAT_1DEG, "lon": LON_1DEG},
                            name=da.name, attrs=da.attrs)


def load_1deg(ds: xarray.Dataset, var: str, times, method: str,
              io_threads: int = 32) -> xarray.DataArray:
    """Download `var` at `times` (0.25 deg) and return it on the 1-deg grid (float32).

    Downloading is network-bound, so many threads fetch chunks at once even on
    Colab's 2 CPUs."""
    import dask
    da = ds[var].sel(time=times).astype(np.float32)
    with dask.config.set(scheduler="threads", num_workers=io_threads):
        da = da.load()
    if da.sizes["lat"] == 181:                                 # already 1 degree
        return da
    return regrid(da, method)


def month_times(ds: xarray.Dataset, year: int, month: int) -> np.ndarray:
    """All 6-hourly time stamps of one calendar month in the store."""
    t = ds.time.sel(time=f"{year}-{month:02d}").values
    if t.size == 0:
        raise ValueError(f"no data for {year}-{month:02d} in the store")
    return t


def load_surface_month(ds: xarray.Dataset, year: int, month: int, method: str,
                       io_threads: int = 32):
    """2 m temperature and 10 m wind speed of one month on the 1-deg grid."""
    t = month_times(ds, year, month)
    t2m = load_1deg(ds, T2M, t, method, io_threads)
    u = load_1deg(ds, U10, t, method, io_threads)
    v = load_1deg(ds, V10, t, method, io_threads)
    return t2m, np.hypot(u, v).rename("10m_wind_speed")
