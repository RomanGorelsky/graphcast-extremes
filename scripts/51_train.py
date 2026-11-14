"""Phase 5, step 2 (cluster, 1 GPU per run): fine-tune GraphCast_small as R1-R4.

    python scripts/51_train.py --run R4                        # full run (resumes if stopped)
    python scripts/51_train.py --run R4 --max-steps 100 --out results/phase5/benchmark
                                                               # benchmark: speed and memory
    python scripts/51_train.py --baseline                      # R0's validation loss, for comparison

Starts from the GraphCast_small checkpoint with the decoder grown by the 7 flux outputs
(finetune.grow_params). Needs the staged data (50_stage_data.py) and the flux stats.
Writes results/phase5/runs/<run>/: config.json, train.jsonl, validation.jsonl,
checkpoint.pkl (resume state), and at the end params_<run>.ckpt (the authors'
checkpoint format, with the extended task) and residuals.csv (budget violations of
4-step forecasts from 8 validation starts).
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
    for f in dataclasses.fields(train.TrainConfig):
        if f.name not in ("run",) and f.type in ("int", "float"):
            ap.add_argument("--" + f.name.replace("_", "-"), type=eval(f.type), default=None)
    ap.add_argument("--train-years", type=int, nargs="+", default=None)
    a = ap.parse_args()

    cfg = train.TrainConfig(run=a.run)
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
    out = Path(a.out) if a.out else phase5.run_dir(run)
    out.mkdir(parents=True, exist_ok=True)

    statics = phase5.STAGED / "statics.zarr"
    val = train.Sampler(phase5.month_files([cfg.val_year]), task, [cfg.val_year], statics_store=statics)
    if a.baseline:
        loss_fn = train.make_loss(m.model_config, task, stats, schemes, cfg.use_bf16)
        eval_fn = train.make_eval(loss_fn)
        inits = val.inits(1)
        inits = inits[np.linspace(0, len(inits) - 1, cfg.n_val).astype(int)]
        tot, diag_sum, n = 0.0, {}, 0
        for i in range(0, len(inits), cfg.batch_size):
            loss, diag = eval_fn(params, {}, *val.batch(inits[i:i + cfg.batch_size], 1))
            tot += float(loss); n += 1
            for k, v in diag.items():
                diag_sum[k] = diag_sum.get(k, 0.0) + float(v)
        diag_mean = {k: v / n for k, v in diag_sum.items()}
        rec = dict(step=0, val_loss=tot / n, val_core_loss=train.core_loss(diag_mean, old_task.target_variables),
                   val_per_variable=diag_mean)
        (out / "validation.jsonl").write_text(json.dumps(rec) + "\n")
        print(f"R0 validation loss {rec['val_loss']:.4f}, core {rec['val_core_loss']:.4f} -> {out}")
        return

    sampler = train.Sampler(phase5.month_files(cfg.train_years), task, cfg.train_years, statics_store=statics)
    print(f"{run}: schemes {schemes or 'none'}; {len(sampler.times):,} training times "
          f"({cfg.train_years}), validation year {cfg.val_year}; {cfg.total_steps()} steps", flush=True)
    params = train.train(cfg, out, sampler, val, m.model_config, task, stats, params,
                         old_task.target_variables, max_steps=a.max_steps)

    state = train.load(out / "checkpoint.pkl")
    if state["step"] >= cfg.total_steps():
        params = state["params"]
        ck = graphcast.CheckPoint(params=params, model_config=m.model_config, task_config=task,
                                  description=f"GraphCast_small fine-tuned as {run} (schemes: {schemes})",
                                  license="derived from GraphCast_small, CC BY-NC-SA 4.0")
        with open(out / f"params_{run}.ckpt", "wb") as f:
            checkpoint.dump(f, ck)
        starts = val.inits(4)[::max(1, len(val.inits(4)) // 8)][:8]
        res = train.residual_check(params, m.model_config, task, stats, schemes, val, starts, 4, cfg.use_bf16)
        res.to_csv(out / "residuals.csv", index=False)
        print("largest relative budget violations (4-step forecasts):")
        print(res[["mass", "moisture", "energy"]].abs().max().to_string())
        print("written:", out)


if __name__ == "__main__":
    main()
