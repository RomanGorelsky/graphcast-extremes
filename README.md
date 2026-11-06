# graphcast-extremes

Course project: **Can bias correction and physics constraints improve AI forecasts of
record-breaking extremes?** A case study with GraphCast_small (1°, 13 levels, trained on
ERA5 1979–2015).

The full research plan (phases 0–6) lives in the project doc. This README only covers
how to run what exists.

## Layout

| Path | What |
| --- | --- |
| `src/extremes/` | Reusable code (`import extremes`): model loading, metrics, paths |
| `scripts/` | Numbered entry points, run from the repo root |
| `notebooks/` | Colab entry points |
| `tests/` | Offline tests (synthetic data, no network) |
| `data/` | Downloaded files, **not in git** (`EXTREMES_DATA` overrides the location) |
| `results/` | Small outputs (`EXTREMES_RESULTS` overrides the location) |

## Phase 0 — setup

### Mac (Terminal)

```bash
cd ~/Documents/GitHub/graphcast-extremes
bash scripts/setup_mac.sh            # once: conda env "extremes", Python 3.12
conda activate extremes              # in every new Terminal
export JAX_PLATFORMS=cpu             # JAX on the Mac uses the CPU

python scripts/00_download_assets.py # model + stats + 1° example data (~0.4 GB)
python -m pytest -q tests            # offline test, should say "2 passed"
python scripts/01_smoke_test.py      # must end with PASSED
python scripts/02_benchmark.py --label mac-cpu
```

### Colab

1. Push this repository to GitHub (see below).
2. Open `notebooks/00_colab_setup.ipynb` in Colab (File → Open notebook → GitHub).
3. Runtime → Change runtime type → T4 GPU, edit `REPO_URL` in step 2, run all cells.

### Put the repository on GitHub (once)

Easiest with **GitHub Desktop**: File -> Add Local Repository -> this folder -> "create a
repository" (Git ignore: None, the project has its own) -> Create Repository -> Publish
repository. The repository is https://github.com/RomanGorelsky/graphcast-extremes.

## Phase 1 — record thresholds and 2020 records (Colab, CPU runtime)

Open `notebooks/01_colab_phase1.ipynb` in Colab and run it top to bottom:

| Step | Script | Output in `results/phase1/` |
| --- | --- | --- |
| Which 0.25°→1° conversion matches the model's data | `10_check_regridding.py` | `regrid_check.json` |
| Highest/lowest t2m and highest 10 m wind per 1° cell and month, 1979–2015 | `11_record_thresholds.py` | `thresholds_1979-2015_1deg.nc` (+ `yearly/`) |
| Every 2020 record (land, north of 60°S) | `12_find_records.py` | `records_2020.csv.gz`, `records_2020_summary.json` |
| Sanity maps (compare Zhang et al. Fig. 1A–B) | `13_sanity_maps.py` | `records_2020_maps.png` |

Data source: WeatherBench2 ERA5 (`gs://weatherbench2/...-wb13-6h-1440x721...zarr`), read
anonymously. `11_` is resumable (one file per finished year). The `.nc` / `.csv.gz` outputs are
not in git; download them from Drive into `results/phase1/` on the Mac.

## Phase 2 — events, record-shattering outliers, weather regimes (Mac + one Colab step)

1. Colab: `notebooks/02_colab_phase2_z500.ipynb` (daily 500 hPa heights, ~10 min), then download
   `z500_daily_1979-2020_2p8deg.nc` into `results/phase2/`.
2. Mac (`conda activate extremes`), in order:

| Script | Method | Output in `results/phase2/` |
| --- | --- | --- |
| `21_detect_events.py` | DBSCAN on record cells (space-time) | `events_2020.csv`, `kdistance.png`, `events_2020_map.png` |
| `22_outliers.py` | Isolation Forest (20 seeds) | record-shattering flags, `events_2020_outliers.png` |
| `23_regimes.py` | PCA + K-means on z500 anomalies per sector | `regimes_*.{csv,json,png}`, regime per event |
| `24_select_events.py` | rule-based selection | `case_studies.csv`, `candidates_<kind>.csv` |

## Phase 3 — baseline forecasts of the case studies (Colab GPU + Mac)

1. Colab (T4 GPU): `notebooks/03_colab_phase3.ipynb`

| Script | What | Output in `results/phase3/` |
| --- | --- | --- |
| `30_check_inputs.py` | our ERA5 inputs vs the authors' example, variable by variable | `input_check.json` |
| `31_run_case_forecasts.py` | 14 events x starts -1/-3/-5/-7 d, 10-day forecasts, cut to event boxes | `forecasts/*.nc` |
| `32_run_noharm.py --label R0` | 12 ordinary dates, global RMSE | `noharm.csv` |

2. Mac: download `forecasts/`, `noharm.csv`, `input_check.json`, then `python scripts/33_verify.py`
   -> `R0_case_metrics.csv`, `R0_summary.json`, `R0_bias_vs_exceedance.png`, `R0_by_start.png`,
   `event_maps/`.

## What "done" means for Phase 0

* `pytest` passes on the Mac and in Colab.
* `01_smoke_test.py` prints **PASSED**: no NaNs, and GraphCast_small beats persistence
  (the "weather stays the same" forecast) for z500, t2m, MSLP and 10 m wind at every lead
  time. The figure `results/phase0/smoke_test.png` shows ERA5, the forecast and their difference.
* `results/phase0/benchmark.jsonl` has one line for `mac-cpu` and one for `colab-<gpu>`;
  `projected_10day_forecast_s` sizes Phases 3–4.

## Notes

* The GraphCast code is the authors' library (`weathernext`, from
  github.com/google-deepmind/graphcast), pinned to commit `f2f2c51`. It needs Python ≥ 3.12.
* Model, statistics and example data come from the authors' public bucket `gs://dm_graphcast`
  (read anonymously, no account needed).
