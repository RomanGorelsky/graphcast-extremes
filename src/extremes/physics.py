"""Global conservation schemes of Sha et al. (2025, JAMES; arXiv:2501.05648, Appendix A),
adapted to GraphCast_small (13 pressure levels, 50-1000 hPa, specific humidity).

Applied to every forecast step, after the model output and before the step is fed
back as input, so that training sees (and its gradients pass through) the corrected
fields. Order as in the paper:

  0. non-negativity     specific humidity >= 0, precipitation >= 0
  1. dry-air mass       global dry-air mass stays as at the previous step; only
                        humidity at levels at or below 600 hPa (p >= 600) is changed,
                        by one multiplicative ratio on (1 - q)                  (A.2)
  2. moisture budget    change of global water vapour = evaporation - precipitation
                        over the 6 h; precipitation is scaled by one ratio       (A.3)
  3. energy             change of global total energy = net flux in at the top of
                        the atmosphere - net flux into the surface over the 6 h;
                        temperature is corrected with one ratio                  (A.4)

Global sums: SUM(X) = sum over cells of X * R^2 * d(sin lat) * d(lon) (A.1); vertical
integral (1/g) * integral of X dp between the lowest and highest dataset level
(trapezoid rule) - as in the paper, the column below 1000 hPa / above 50 hPa is ignored.

Units (ERA5 conventions, all fluxes positive DOWNWARD, accumulated over 6 h):
  total_precipitation_6hr, evaporation_6hr        m of water (evaporation mostly < 0)
  the five radiation / heat flux variables         J m-2
Differences from the paper (state in the report): specific humidity instead of specific
total water; top level 50 hPa instead of 1 hPa; constants below.

A consequence of fixed pressure levels (GraphCast has no surface-pressure output): the
column mass sum(dp)/g is the same at every step, so dry-air mass = column mass - water
vapour, and conserving dry-air mass is the same as keeping GLOBAL WATER VAPOUR constant.
With the moisture scheme on top, global precipitation then equals global evaporation in
every 6-h step. ERA5 itself deviates from this by ~0.2 % of the vapour (~7 % of the 6-h
precipitation; 52_check_staged.py). In Sha et al.'s model levels the column mass follows
surface pressure, so there the two schemes are distinct. Report R2 as "global vapour fixed".

Works on xarray Datasets whose arrays are numpy (tests) or JAX (training).
"""
from __future__ import annotations

import numpy as np
import xarray

G = 9.80665              # m s-2
R_EARTH = 6_371_000.0    # m
RHO_WATER = 1000.0       # kg m-3
CP_DRY = 1004.64         # J kg-1 K-1
CP_VAPOUR = 1846.0       # J kg-1 K-1
L_V = 2.501e6            # J kg-1
P_SPLIT_HPA = 600.0      # dry-air mass: correct humidity only at p >= this level

PRECIP = "total_precipitation_6hr"
EVAP = "evaporation_6hr"
TOA_SOLAR = "top_net_solar_radiation_6hr"          # R_T = TOA_SOLAR + TOA_THERMAL
TOA_THERMAL = "top_net_thermal_radiation_6hr"      # (ERA5 'ttr' = minus outgoing longwave)
SFC_SOLAR = "surface_net_solar_radiation_6hr"      # F_S = sum of the four surface terms
SFC_THERMAL = "surface_net_thermal_radiation_6hr"
SFC_SENSIBLE = "surface_sensible_heat_flux_6hr"
SFC_LATENT = "surface_latent_heat_flux_6hr"
FLUX_VARS = (EVAP, TOA_SOLAR, TOA_THERMAL, SFC_SOLAR, SFC_THERMAL, SFC_SENSIBLE, SFC_LATENT)
SCHEMES = ("mass", "moisture", "energy")


# ------------------------------------------------------------------ weights
def cell_area(lat: xarray.DataArray, lon: xarray.DataArray) -> xarray.DataArray:
    """Area (m2) of each grid cell of a regular lat-lon grid that may include the poles."""
    la = np.asarray(lat, float)
    dlat = abs(la[1] - la[0])
    lo_edge = np.clip(la - dlat / 2, -90, 90)
    hi_edge = np.clip(la + dlat / 2, -90, 90)
    band = np.abs(np.sin(np.deg2rad(hi_edge)) - np.sin(np.deg2rad(lo_edge)))
    dlon = np.deg2rad(360.0 / len(np.asarray(lon)))
    a = R_EARTH ** 2 * band[:, None] * dlon * np.ones((1, len(np.asarray(lon))))
    return xarray.DataArray(a.astype(np.float32), dims=("lat", "lon"),
                            coords={"lat": lat.values, "lon": lon.values})


def level_weights(level: xarray.DataArray) -> xarray.DataArray:
    """Trapezoid weights dp/g (kg m-2 per unit of X) for integrating over pressure levels."""
    p = np.asarray(level, float) * 100.0                     # hPa -> Pa
    order = np.argsort(p)
    ps = p[order]
    w_sorted = np.zeros_like(ps)
    w_sorted[1:] += np.diff(ps) / 2
    w_sorted[:-1] += np.diff(ps) / 2
    w = np.empty_like(w_sorted)
    w[order] = w_sorted
    return xarray.DataArray((w / G).astype(np.float32), dims=("level",), coords={"level": level.values})


def _gsum(x: xarray.DataArray, area: xarray.DataArray) -> xarray.DataArray:
    """Global area-weighted sum over lat and lon (keeps batch / time)."""
    return (x.astype(np.float32) * area).sum(("lat", "lon"))


def _column(x: xarray.DataArray, wlev: xarray.DataArray) -> xarray.DataArray:
    return (x.astype(np.float32) * wlev).sum("level")


# ------------------------------------------------------------------ budgets (diagnostics)
def dry_air_mass(q, area, wlev):
    return _gsum(_column(1.0 - q, wlev), area)                        # kg


def water_vapour(q, area, wlev):
    return _gsum(_column(q, wlev), area)                              # kg


def heat_capacity(q):
    return CP_DRY * (1.0 - q) + CP_VAPOUR * q


def total_energy(t, q, u, v, phi_s, area, wlev):
    """A = SUM (1/g) int (Cp T + Lv q + phi_s + k) dp            (J)."""
    k = 0.5 * (u * u + v * v)
    return _gsum(_column(heat_capacity(q) * t + L_V * q + phi_s + k, wlev), area)


def net_energy_in(fc, area):
    """6-h net energy gained by the atmosphere, R_T - F_S, globally (J)."""
    r_top = fc[TOA_SOLAR] + fc[TOA_THERMAL]
    f_sfc = fc[SFC_SOLAR] + fc[SFC_THERMAL] + fc[SFC_SENSIBLE] + fc[SFC_LATENT]
    return _gsum(r_top - f_sfc, area)


def residuals(prev: xarray.Dataset, fc: xarray.Dataset, phi_s: xarray.DataArray) -> dict:
    """Relative violations of the three budgets between the previous state and the
    forecast step (0 = closed). prev/fc: one time each (no 'time' dim or size 1)."""
    area, wlev = cell_area(fc.lat, fc.lon), level_weights(fc.level)
    q0, q1 = prev["specific_humidity"], fc["specific_humidity"]
    md0, md1 = dry_air_mass(q0, area, wlev), dry_air_mass(q1, area, wlev)
    mv0, mv1 = water_vapour(q0, area, wlev), water_vapour(q1, area, wlev)
    p = _gsum(fc[PRECIP], area) * RHO_WATER
    e = _gsum(fc[EVAP], area) * RHO_WATER
    a0 = total_energy(prev["temperature"], q0, prev["u_component_of_wind"], prev["v_component_of_wind"],
                      phi_s, area, wlev)
    a1 = total_energy(fc["temperature"], q1, fc["u_component_of_wind"], fc["v_component_of_wind"],
                      phi_s, area, wlev)
    return {"mass": (md1 - md0) / md0,
            "moisture": ((mv1 - mv0) + e + p) / mv0,          # dMv = -(E + P), both downward
            "energy": ((a1 - a0) - net_energy_in(fc, area)) / a0}


# ------------------------------------------------------------------ schemes
def non_negative(fc: xarray.Dataset) -> xarray.Dataset:
    out = fc.copy()
    for v in ("specific_humidity", PRECIP):
        if v in out:
            out[v] = out[v].where(out[v] > 0, 0.0)            # differentiable, works on JAX arrays
    return out


def fix_dry_air_mass(prev: xarray.Dataset, fc: xarray.Dataset) -> xarray.Dataset:
    """A.2 'improved': q*(p >= 600 hPa) = 1 - (1 - q) * r, with r chosen so that the global
    dry-air mass equals the previous step's; levels above 600 hPa are unchanged."""
    area, wlev = cell_area(fc.lat, fc.lon), level_weights(fc.level)
    q0, q1 = prev["specific_humidity"], fc["specific_humidity"]
    low = xarray.DataArray(np.asarray(fc.level) >= P_SPLIT_HPA, dims="level", coords={"level": fc.level.values})
    target = dry_air_mass(q0, area, wlev)
    above = dry_air_mass(q1, area, wlev.where(~low, 0.0))
    below = dry_air_mass(q1, area, wlev.where(low, 0.0))
    r = (target - above) / below
    q_new = (1.0 - (1.0 - q1) * r).where(low, q1)
    out = fc.copy()
    out["specific_humidity"] = q_new.astype(q1.dtype).transpose(*q1.dims)
    return out


def fix_moisture(prev: xarray.Dataset, fc: xarray.Dataset) -> xarray.Dataset:
    """A.3: P* = P * ratio, so that global dMv/dt = -(E + P*)  (E, P downward-positive)."""
    area, wlev = cell_area(fc.lat, fc.lon), level_weights(fc.level)
    dmv = water_vapour(fc["specific_humidity"], area, wlev) - water_vapour(prev["specific_humidity"], area, wlev)
    p_target = -dmv / RHO_WATER - _gsum(fc[EVAP], area)               # m3 of water, global
    p_now = _gsum(fc[PRECIP], area)
    ratio = p_target / p_now
    # safeguard (not in the paper): a badly drifting forecast could ask for negative or
    # exploding precipitation; keep the ratio in [0, 10] (real data: ~0.9-1.1)
    ratio = ratio.where(ratio > 0.0, 0.0)
    ratio = ratio.where(ratio < 10.0, 10.0)
    out = fc.copy()
    out[PRECIP] = (fc[PRECIP] * ratio).astype(fc[PRECIP].dtype).transpose(*fc[PRECIP].dims)
    return out


def fix_energy(prev: xarray.Dataset, fc: xarray.Dataset, phi_s: xarray.DataArray) -> xarray.Dataset:
    """A.4: gamma = A*/A(t1) with A* = A(t0) + (R_T - F_S); then
    T* = gamma T + (gamma - 1) / Cp * (Lv q + phi_s + k), which makes A(T*) = A* exactly."""
    area, wlev = cell_area(fc.lat, fc.lon), level_weights(fc.level)
    t, q, u, v = (fc[x] for x in ("temperature", "specific_humidity", "u_component_of_wind",
                                  "v_component_of_wind"))
    a0 = total_energy(prev["temperature"], prev["specific_humidity"], prev["u_component_of_wind"],
                      prev["v_component_of_wind"], phi_s, area, wlev)
    a1 = total_energy(t, q, u, v, phi_s, area, wlev)
    gamma = (a0 + net_energy_in(fc, area)) / a1
    k = 0.5 * (u * u + v * v)
    t_new = gamma * t + (gamma - 1.0) / heat_capacity(q) * (L_V * q + phi_s + k)
    out = fc.copy()
    out["temperature"] = t_new.astype(t.dtype).transpose(*t.dims)
    return out


def apply(prev: xarray.Dataset, fc: xarray.Dataset, phi_s: xarray.DataArray,
          schemes: tuple = SCHEMES) -> xarray.Dataset:
    """Non-negativity, then the requested schemes in the paper's order.
    prev = the last input step (the state the forecast step starts from).
    No schemes (the control run R1) = the forecast unchanged, without the clipping too."""
    if not schemes:
        return fc
    unknown = set(schemes) - set(SCHEMES)
    if unknown:
        raise ValueError(f"unknown schemes {unknown}; choose from {SCHEMES}")
    out = non_negative(fc)
    if "mass" in schemes:
        out = fix_dry_air_mass(prev, out)
    if "moisture" in schemes:
        out = fix_moisture(prev, out)
    if "energy" in schemes:
        out = fix_energy(prev, out, phi_s)
    return out
