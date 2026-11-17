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

## Phase 4 — deterministic probabilistic bias correction (Colab GPU + Mac)

1. Colab (T4 GPU, several sessions): `notebooks/04_colab_phase4.ipynb`

| Script | What | Output in `results/phase4/` |
| --- | --- | --- |
| `40_hindcast_plan.py` | start dates 2016-2019, +-7 d around each event start | `hindcast_plan.csv` |
| `41_run_hindcasts.py` | ~780 10-day hindcasts, cut to event boxes (background, resumable) | `hindcasts/<event>/*.nc` |
| `42_truth_cache.py` | ERA5 truth for all hindcast times, read once | `truth/<event>.nc` |

2. Mac: download `hindcasts/`, `truth/`, `hindcast_plan.csv`, then

```bash
python scripts/43_apply_pbc.py                       # Debias, Persistence, PBC -> corrected forecasts
python scripts/33_verify.py --label Debias --forecasts results/phase4/forecasts_Debias --no-maps
python scripts/33_verify.py --label Persistence --forecasts results/phase4/forecasts_Persistence --no-maps
python scripts/33_verify.py --label PBC --forecasts results/phase4/forecasts_PBC --no-maps
python scripts/44_compare.py                         # R0 vs Debias vs Persistence vs PBC
```

3. Mac: two checks of how reliable the Phase 4 result is (no GPU, minutes)

| Script | Question | Output in `results/phase4/` |
| --- | --- | --- |
| `45_robustness.py` | Do the changes vs R0 hold across events, or does one event drive them? (paired, bootstrap over events, leave-one-event-out) | `robustness.csv`, `robustness_events.csv`, `robustness_events.png` |
| `46_record_probability.py` | Does a probabilistic version (spread learned from the hindcasts) give useful record probabilities? (Brier score, warnings, reliability) | `record_probability.csv`, `record_probability_cases.csv`, `record_probability.png` |

## Phase 5 — physics-constrained fine-tuning (cluster)

GraphCast_small gets 7 extra outputs (6-h evaporation and radiation / heat fluxes) and Sha et
al.'s (2025) conservation schemes inside the forecast loop (`src/extremes/physics.py`,
`finetune.py`, `train.py`). Four runs with identical data, steps and learning rate:
R1 control (new outputs, no schemes) | R2 + dry-air mass | R3 + moisture budget | R4 + energy.

Mac first (offline tests, include a tiny training run): `python -m pytest -q tests`.
On the cluster (`git pull`, then once `pip install optax` in the environment):

```bash
sbatch slurm/phase5_stage.sbatch        # 6 CPU jobs: 2010-2014 training, 2015 validation (~34 GB/year)
sbatch slurm/phase5_benchmark.sbatch    # after staging: flux stats, R0 score, 100-step speed test
sbatch slurm/phase5_train.sbatch        # R1-R4 on 4 GPUs; resubmit to resume
```

| Script | What | Output |
| --- | --- | --- |
| `50_stage_data.py` | 1-degree training data, month by month (resumable) | `data/phase5/era5_1deg/` |
| `51_train.py` | one run (`--run R1..R4`), or `--baseline` for R0 | `results/phase5/runs/<run>/` (train/validation logs, checkpoint, `residuals.csv`) |

## Phase 6 — evaluation of R1-R4 on the 2020 events (cluster + Mac)

1. Cluster (1 GPU per run, 2 at a time, ~1-2 h per run):
   `sbatch slurm/phase6_forecasts.sbatch` -> the 14 case studies x 4 starts and the 12
   ordinary dates with each fine-tuned run (`31_run_case_forecasts.py --model R<k>`,
   `32_run_noharm.py --model R<k>`), into `results/phase6/`.
2. Mac: copy `results/phase6/` over (`aws s3 sync ...`), then

```bash
for r in R1 R2 R3 R4; do python scripts/33_verify.py --label $r --forecasts results/phase6/forecasts_$r --no-maps; done
python scripts/44_compare.py --labels R0 R1 R2 R3 R4 --tag _physics       # extremes: R0 vs R1-R4
python scripts/45_robustness.py --labels R1 R2 R3 R4 --base R1 --tag _physics   # physics effect vs control
python scripts/61_phase6_summary.py                                       # ordinary weather + sharpness
```

### Phase 6b — ordinary weather, 52 dates (do the methods help everyday forecasts?)

52 weekly 10-day forecasts of 2020 with R0-R4, scored at lead days 1-10: RMSE and bias
(t2m, u10, v10, msl, z500, t850), anomaly correlation and activity against the
WeatherBench2 1990-2019 climatology (skipped if it cannot be opened), and the global
dry-air mass / water vapour / energy of each forecast vs ERA5's.

1. Cluster (1 GPU per model, 2 at a time, ~1 h per model): `sbatch slurm/phase6_ordinary.sbatch`
   -> `results/phase6/ordinary/R0..R4.csv` and `truth_budgets.csv` (resumable).
2. Mac: copy `results/phase6/ordinary/` over, then `python scripts/64_ordinary_summary.py`
   -> tables, `ordinary_change_vs_R0.csv`, `budget_drift.csv`, `ordinary_skill.png`.

## Phase 7 — the in-scope prospects (cluster + Mac)

Four follow-up experiments from the report's "Prospects". Tests: `python -m pytest -q tests/test_phase7.py`.

| Prospect | What | Code | Where |
| --- | --- | --- | --- |
| 1 Tail-targeted post-processing | quantile mapping of R0 learned from the 2016-2019 hindcasts, applied everywhere (QM), only in the tails (TailQM), or only from hindcasts in the same K-means regime (TailQM-reg) | `extremes/tailcorr.py`, `71_tail_correction.py` | Mac |
| 2 More events and years | records, DBSCAN, Isolation Forest and regimes for 2021 and 2022 (the WeatherBench2 ERA5 store ends 2023-01-10, so not 2023); record-shattering events of 2020-2022 + the 14 originals | `extremes/extended.py`, `70_select_extended.py`, `EXTREMES_SET=ext` | cluster + Mac |
| 3 Sharpness-preserving training (R5) | R1 retrained with an "adjusted MSE" loss (after Subich et al. 2025, on the grid: field + its east-west and north-south gradients) that does not reward smoothing | `finetune.amse_per_level`, `51_train.py --run R5` | cluster |
| 4 Perturbed-start ensemble | 20 members + R0 control, starts perturbed by alpha x (difference of two ERA5 states of other years at the same season), in +/- pairs; record probabilities and spread | `extremes/ensemble.py`, `72_ensemble_forecasts.py`, `73_ensemble_verify.py` | cluster + Mac |

**The event set switch.** With `EXTREMES_SET=ext` every Phase 3/4/6 script (31, 33, 40-46,
61, 71, 72, 73) works on `results/phase2/case_studies_ext.csv` and reads/writes
`results/sets/ext/phase3`, `phase4`, `phase6`, `phase7` instead of `results/phase3` ... -
the original 14-event results are never touched. Unset = the 14 original case studies.

### Once, on the Mac (the cluster needs two Phase 1-2 files that git normally skips)

```bash
git add -f results/phase1/thresholds_1979-2015_1deg.nc results/phase2/records_2020_labeled.csv.gz
git commit -m "Phase 7 inputs" && git push        # (or commit/push in GitHub Desktop)
```

### Cluster (`git pull` first; at most 3 GPUs at a time, at most 5 jobs submitted)

```bash
sbatch slurm/phase7_events.sbatch          # CPU, ~4-6 h: 2021-2022 records, events -> case_studies_ext.csv
sbatch slurm/phase7_train_r5.sbatch        # GPU 1: R5 (resumable, resubmit if stopped)
sbatch slurm/phase7_ensemble.sbatch        # GPU 2: ensembles of the 14 case studies, alpha 0.1 (~3-4 h)
sbatch --export=ALL,ALPHA=0.2 slurm/phase7_ensemble.sbatch     # GPU 3: the same with alpha 0.2
# when phase7_events is done (and the ensembles have freed their GPUs):
EXTREMES_SET=ext python scripts/40_hindcast_plan.py --years 2019 2018     # login node, seconds
sbatch slurm/phase7_ext_truth.sbatch       # CPU
sbatch --array=0-2%2 slurm/phase7_ext_forecasts.sbatch    # R0, R1, R4 on the extended set (2 GPUs + R5's = 3)
# when R5 is trained:
sbatch slurm/phase7_r5_eval.sbatch         # R5 on the 14 case studies + 52 ordinary dates
sbatch --array=3 slurm/phase7_ext_forecasts.sbatch        # R5 on the extended set
# GPUs free again:
sbatch --array=0-2 slurm/phase7_ext_hindcasts.sbatch      # 3 GPUs, hindcasts for PBC/QM on the extended set
```

Then copy the results to S3 from the cluster (`aws s3 sync results s3://.../results --region
ap-northeast-1`) and down to the Mac (the same command with the two places swapped).

### Mac

```bash
# 1. tail post-processing (14 case studies; hindcasts and truth from Phase 4 are already here)
python scripts/71_tail_correction.py
for m in QM TailQM TailQM-reg; do python scripts/33_verify.py --label $m --forecasts results/phase4/forecasts_$m --no-maps; done
python scripts/44_compare.py --labels R0 PBC QM TailQM TailQM-reg --tag _tail
python scripts/45_robustness.py --labels R0 PBC QM TailQM TailQM-reg --tag _tail

# 3. R5 on the 14 case studies and on ordinary weather
python scripts/33_verify.py --label R5 --forecasts results/phase6/forecasts_R5 --no-maps
python scripts/44_compare.py --labels R0 R1 R5 --tag _sharp
python scripts/45_robustness.py --labels R1 R5 --base R1 --tag _sharp
python scripts/61_phase6_summary.py --labels R0 R1 R5         # sharpness
python scripts/64_ordinary_summary.py --labels R0 R1 R4 R5

# 4. ensembles
python scripts/73_ensemble_verify.py
python scripts/33_verify.py --label ENSmean --forecasts results/phase7/forecasts_ENSmean_a0.1 --no-maps
python scripts/44_compare.py --labels R0 ENSmean --tag _ens

# 2. the extended set (after its forecasts; corrections after its hindcasts and truth)
export EXTREMES_SET=ext
python scripts/33_verify.py --label R0 --no-maps
for r in R1 R4 R5; do python scripts/33_verify.py --label $r --forecasts results/sets/ext/phase6/forecasts_$r --no-maps; done
python scripts/44_compare.py --labels R0 R1 R4 R5 --tag _physics
python scripts/45_robustness.py --labels R1 R4 R5 --base R1 --tag _physics
python scripts/43_apply_pbc.py && python scripts/71_tail_correction.py
for m in PBC TailQM; do python scripts/33_verify.py --label $m --forecasts results/sets/ext/phase4/forecasts_$m --no-maps; done
python scripts/45_robustness.py --labels R0 PBC TailQM --tag _corrections
unset EXTREMES_SET
```

## Cluster (MBZUAI Shared-Cluster-v3, H100)

Connect with `ssh shared-cluster-v3` (see the cluster user guide). Heavy work runs only as
jobs (`sbatch`), never directly on the login node. Once, on the login node:

```bash
git clone https://github.com/RomanGorelsky/graphcast-extremes.git ~/graphcast-extremes
cd ~/graphcast-extremes
bash scripts/setup_cluster.sh           # ~10 min: Python 3.12 env ~/venvs/extremes + model
sbatch slurm/phase0_check.sbatch        # GPU check, tests, smoke test, speed
squeue --me                             # wait until the job is gone, then:
cat slurm/logs/phase0_*.out
```

Phase 4 on the cluster (`git pull` first to get the latest code):

```bash
python scripts/40_hindcast_plan.py      # light, fine on the login node
sbatch slurm/phase4_hindcasts.sbatch    # 4 GPUs share the ~780 hindcasts; resubmit to resume
sbatch slurm/phase4_truth.sbatch        # ERA5 truth, no GPU, may run at the same time
```

| Job file | What | Log |
| --- | --- | --- |
| `slurm/phase0_check.sbatch` | 1 GPU, ~10 min | `slurm/logs/phase0_<job>.out` |
| `slurm/phase4_hindcasts.sbatch` | 4 jobs x 1 GPU, up to 12 h each | `slurm/logs/hind_<job>_<0-3>.out` |
| `slurm/phase4_truth.sbatch` | CPU only, 128 GB | `slurm/logs/truth_<job>.out` |

Results land in `~/graphcast-extremes/results` on the cluster, which is copied to S3
automatically. To bring them to the Mac (in Terminal, from the repository folder; replace
`<user>` with your cluster username):

```bash
aws s3 sync s3://shared-cluster-november/homes/<user>/graphcast-extremes/results/phase4 results/phase4 --region ap-northeast-1
```

Hindcasts already made in Colab can be uploaded instead of recomputed (they are skipped):
`aws s3 sync results/phase4/hindcasts s3://shared-cluster-november/homes/<user>/graphcast-extremes/results/phase4/hindcasts --region ap-northeast-1`.
Deleting a file on the cluster also deletes it from S3 - there is no recycle bin.

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
