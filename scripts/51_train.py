"""Phase 5, step 2 (cluster, 1 GPU per run): fine-tune GraphCast_small as R1-R4.

    python scripts/51_train.py --run R4                        # full run (resumes if stopped)
    python scripts/51_train.py --run R4 --max-steps 100 --out results/phase5/benchmark
                                                               # benchmark: speed and memory
    python scripts/51_train.py --baseline                      # R0's validation loss, for comparison

Starts from the GraphCast_small checkpoint with the decoder grown by the 7 flux outputs
(finetune.grow_params). Needs the staged data (50_stage_data.py) and the flux stats.
Writes results/phase5/runs/<run>/: config.json, train.jsonl, validation.jsonl,
checkpoint.pkl (resume state), validation_long.jsonl (1-day and 5-day forecasts),
params_stage<k>.pkl, and at the end params_<run>.ckpt and params_<run>_stage<k>.ckpt for the
earlier stage ends (the authors' checkpoint format, with the extended task) and residuals.csv (budget
violations of 4-step forecasts from 8 validation starts).
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import xarray

from extremes import finetune, model, phase5, physics, train
from weathernext.utils import checkpoint
from weathernext.weathernext1_graph import graphcast


def flux_stats():
    f = xarray.load_dataset(phase5.FLUX_STATS)
    mean = xarray.Dataset({v: f[f"mean_{v}"].astype("float32") for v in physics.FLUX_VARS})
    std = xarray.Dataset({v: f[f"std_{v}"].astype("float32") for v in physics.FLUX_VARS})
    return mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(finetune.RUNS), default="R1")
    ap.add_argument("--baseline", action="store_true", help="only score R0 on the validation set")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (benchmark)")
    ap.add_argument("--one-stage", action="store_true",
                    help="train only up to the end of the current stage, then exit (the cluster job "
                         "starts a fresh process for every stage: GPU memory left over from one stage "
                         "made the next one fail)")
    for f in dataclasses.fields(train.TrainConfig):
        if f.name not in ("run",) and f.type in ("int", "float"):
            ap.add_argument("--" + f.name.replace("_", "-"), type=eval(f.type), default=None)
    ap.add_argument("--train-years", type=int, nargs="+", default=None)
    ap.add_argument("--init-params", default=None,
                    help="start from these weights (a params_stage<k>.pkl) instead of GraphCast_small")
    ap.add_argument("--skip-stages", type=int, default=0, help="drop the first N stages of the schedule")
    ap.add_argument("--only-stages", type=int, default=None, help="keep only N stages (after skipping)")
    a = ap.parse_args()

    cfg = train.TrainConfig(run=a.run)
    stages = cfg.stages[a.skip_stages:]
    cfg.stages = stages[:a.only_stages] if a.only_stages else stages
    for f in dataclasses.fields(cfg):
        v = getattr(a, f.name, None)
        if v is not None and f.name != "run":
            setattr(cfg, f.name, tuple(v) if isinstance(v, list) else v)

    m = model.load_model(use_bf16=cfg.use_bf16)
    old_task = m.task_config
    mean, std = flux_stats()
    stats = finetune.extend_stats(m.stats, mean, std)

    if a.baseline:
        task, params, schemes, run = old_task, m.params, (), "R0"
    else:
        task = finetune.extended_task(old_task)
        params, schemes, run = finetune.grow_params(m.params, old_task, task), finetune.RUNS[a.run], a.run
        if a.init_params:
            params = train.load(Path(a.init_params))["params"]
            print(f"{run}: starting from {a.init_params}", flush=True)
    out = Path(a.out) if a.out else phase5.run_dir(run)
    out.mkdir(parents=True, exist_ok=True)

    statics = phase5.STAGED / "statics.zarr"
    val = train.Sampler(phase5.month_files([cfg.val_year]), task, [cfg.val_year], statics_store=statics)
    if a.baseline:
        eval_fn = train.make_eval(train.make_loss(m.model_config, task, stats, schemes, cfg.use_bf16))
        r = train.score(eval_fn, params, {}, val, 1, cfg.n_val, cfg.batch_size, old_task.target_variables)
        rec = dict(step=0, val_loss=r["loss"], val_core_loss=r["core"], val_per_variable=r["per_variable"])
        (out / "validation.jsonl").write_text(json.dumps(rec) + "\n")
        (out / "validation_long.jsonl").unlink(missing_ok=True)
        cfg.run = "R0"
        print(f"R0 validation loss {rec['val_loss']:.4f}, core {rec['val_core_loss']:.4f}")
        train.score_long(eval_fn, params, {}, val, cfg, old_task.target_variables, 0, out)
        print("written:", out)
        return

    if a.one_stage:
        ck_file = out / "checkpoint.pkl"
        done = train.load(ck_file)["step"] if ck_file.exists() else 0
        nxt = next((e for e in cfg.stage_ends() if e > done), None)
        if nxt is not None:
            a.max_steps = int(nxt - done)
            print(f"{run}: step {done} -> {nxt} (stage {cfg.stage_ends().index(nxt) + 1} of {len(cfg.stages)})",
                  flush=True)

    sampler = train.Sampler(phase5.month_files(cfg.train_years), task, cfg.train_years, statics_store=statics)
    print(f"{run}: schemes {schemes or 'none'}; {len(sampler.times):,} training times "
          f"({cfg.train_years}), validation year {cfg.val_year}; {cfg.total_steps()} steps", flush=True)
    params = train.train(cfg, out, sampler, val, m.model_config, task, stats, params,
                         old_task.target_variables, max_steps=a.max_steps, old_task=old_task)

    state = train.load(out / "checkpoint.pkl")
    if state["step"] >= cfg.total_steps():
        params = state["params"]
        ck = graphcast.CheckPoint(params=params, model_config=m.model_config, task_config=task,
                                  description=f"GraphCast_small fine-tuned as {run} (schemes: {schemes})",
                                  license="derived from GraphCast_small, CC BY-NC-SA 4.0")
        with open(out / f"params_{run}.ckpt", "wb") as f:
            checkpoint.dump(f, ck)
        for k in range(1, len(cfg.stages)):                     # the weights at earlier stage ends too
            sk = out / f"params_stage{k}.pkl"
            if sk.exists():
                ckk = graphcast.CheckPoint(params=train.load(sk)["params"], model_config=m.model_config,
                                           task_config=task, license=ck.license,
                                           description=f"GraphCast_small fine-tuned as {run}, after stage {k}")
                with open(out / f"params_{run}_stage{k}.ckpt", "wb") as f:
                    checkpoint.dump(f, ckk)
        starts = val.inits(4)[::max(1, len(val.inits(4)) // 8)][:8]
        res = train.residual_check(params, m.model_config, task, stats, schemes, val, starts, 4, cfg.use_bf16)
        res.to_csv(out / "residuals.csv", index=False)
        print("largest relative budget violations (4-step forecasts):")
        print(res[["mass", "moisture", "energy"]].abs().max().to_string())
        print("written:", out)


if __name__ == "__main__":
    main()
