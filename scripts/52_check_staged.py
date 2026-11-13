"""Phase 5 sanity check (cluster, CPU job): do the staged data look like the real Earth?

For one staged month:
  1. global mean of each flux, in W m-2 (mm per 6 h for water), next to the known
     climatological values (Trenberth et al. 2009 / ERA5): wrong units or signs show here;
  2. how well ERA5 ITSELF closes the three budgets on our 13 pressure levels
     (physics.residuals between consecutive 6-h times) - the floor that the schemes
     will force to zero; reported in the Phase 5 results.

    python scripts/52_check_staged.py --month 2015-01
"""
import argparse

import numpy as np
import xarray

from extremes import phase5, physics

SIX_H = 6 * 3600.0
EXPECTED = {   # global annual means, ERA5 sign convention (downward positive)
    physics.TOA_SOLAR: "+240 W m-2", physics.TOA_THERMAL: "-240 W m-2",
    physics.SFC_SOLAR: "+160 W m-2", physics.SFC_THERMAL: "-55 W m-2",
    physics.SFC_SENSIBLE: "-20 W m-2", physics.SFC_LATENT: "-85 W m-2",
    physics.EVAP: "-0.75 mm / 6 h", physics.PRECIP: "+0.75 mm / 6 h",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2015-01")
    a = ap.parse_args()
    y, m = map(int, a.month.split("-"))
    ds = xarray.open_zarr(phase5.month_path(y, m)).merge(xarray.open_zarr(phase5.STAGED / "statics.zarr"))
    ds = ds.load()
    area = physics.cell_area(ds.lat, ds.lon)
    w = area / area.sum()

    print(f"1. global means, {a.month} ({ds.sizes['time']} times)   [expected, annual]")
    for v, exp in EXPECTED.items():
        g = float((ds[v] * w).sum(("lat", "lon")).mean())
        shown = f"{g * 1000:+8.2f} mm / 6 h" if v in (physics.EVAP, physics.PRECIP) else f"{g / SIX_H:+8.1f} W m-2"
        print(f"   {v:36s} {shown}   [{exp}]")

    print("\n2. ERA5's own budget residuals on 13 levels (relative; the schemes force these to 0)")
    phi = ds["geopotential_at_surface"]
    rows = {k: [] for k in ("mass", "moisture", "energy")}
    for i in range(1, ds.sizes["time"]):
        r = physics.residuals(ds.isel(time=i - 1), ds.isel(time=[i]), phi)
        for k in rows:
            rows[k].append(float(np.abs(r[k]).max()))
    for k, v in rows.items():
        v = np.asarray(v)
        print(f"   {k:8s} median {np.median(v):.2e}   95% {np.percentile(v, 95):.2e}   max {v.max():.2e}")


if __name__ == "__main__":
    main()
