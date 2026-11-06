"""Phase 2: redraw the regime composites figure from saved results (no recomputation).

Reads results/phase2/regimes_composite_<sector>.nc and regimes_summary.json (both
written by 23_regimes.py) and writes results/phase2/regimes_composites.png.
If the .nc files do not exist yet, rerun 23_regimes.py once (it saves them).

    python scripts/23b_plot_regimes.py
"""
import json

import xarray

from extremes import events, paths, phase1, plots

OUT = paths.RESULTS_DIR / "phase2"


def main():
    summary = json.loads((OUT / "regimes_summary.json").read_text())
    files = {s: OUT / f"regimes_composite_{s}.nc" for s in events.SECTORS}
    missing = [s for s, f in files.items() if not f.exists()]
    if missing:
        raise FileNotFoundError(f"no saved composites for {missing}: run scripts/23_regimes.py once")
    comps = {s: xarray.load_dataarray(f) for s, f in files.items()}
    freqs = {s: summary[s]["frequency"] for s in comps}
    lsm = xarray.load_dataset(phase1.thresholds_path())["land_sea_mask"]
    out = OUT / "regimes_composites.png"
    plots.regime_composites(comps, freqs, lsm, out)
    print("written:", out)


if __name__ == "__main__":
    main()
