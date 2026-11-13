"""Phase 5 repair (cluster, CPU job, ~10-20 min): fill the few isolated blank (NaN) cells
of the staged flux variables in place, with the mean of their neighbours
(phase5.fill_isolated_nans). Only the affected time steps are rewritten.

The scan (53_find_nans.py) found 33 single cells in 6 years, all in the sensible and
latent heat fluxes - gaps in the ARCO hourly archive. Months staged after this fix
are filled while staging; this script repairs those staged before.

    python scripts/54_fill_nans.py
"""
import numpy as np
import zarr

from extremes import phase5, physics


def main():
    total = 0
    for f in sorted(phase5.STAGED.glob("????-??.zarr")):
        g = zarr.open_group(str(f), mode="r+")
        for v in list(physics.FLUX_VARS) + [physics.PRECIP]:
            arr = g[v]
            a = arr[:]                                               # (time, lat, lon)
            for k in np.where(np.isnan(a).any(axis=(1, 2)))[0]:
                n = int(np.isnan(a[k]).sum())
                arr[k] = phase5.fill_isolated_nans(a[k])
                total += n
                print(f"{f.name[:7]} {v}: time {k}, {n} cell(s) filled", flush=True)
    print(f"\nfilled {total} cells")


if __name__ == "__main__":
    main()
