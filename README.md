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

```bash
cd ~/Documents/GitHub/graphcast-extremes
git init && git add . && git commit -m "Phase 0: setup, smoke test, benchmark"
# create an empty repo named graphcast-extremes on github.com, then:
git remote add origin https://github.com/<your-username>/graphcast-extremes.git
git branch -M main && git push -u origin main
```

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
