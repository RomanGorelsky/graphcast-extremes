"""Phase 5 fine-tuning loop (one run = one of R1-R4, see finetune.RUNS).

Two stages, as GraphCast's own training: (1) one-step forecasts with a warm-up then
cosine-decaying learning rate; (2) a short multi-step stage (the model's own outputs
are fed back for `rollout` steps) at a small constant learning rate, so the schemes
act on what the model produces itself. AdamW as in GraphCast (b2 = 0.95, weight decay
0.1). Every `eval_every` steps a fixed validation set (held-out year) is scored:
  loss       - full loss (incl. the new fluxes)
  core_loss  - only the variables R0 also predicts: comparable across R0-R4
Checkpoints every `ckpt_every` steps; a job that is stopped resumes where it was.

Data: training samples are read from the staged 1-degree store (scripts/50_stage_data.py),
B random start times per step, loaded in a background thread while the GPU trains.
"""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import json
import math
import pickle
import time
from pathlib import Path

import haiku as hk
import jax
import numpy as np
import optax
import pandas as pd
import xarray
import xarray_jax

from weathernext.utils import data_utils

from . import finetune, inputs, physics

STEP = pd.Timedelta(hours=6)


@dataclasses.dataclass
class TrainConfig:
    run: str = "R1"
    train_years: tuple = (2010, 2011, 2012, 2013, 2014)
    val_year: int = 2015
    batch_size: int = 4
    stage1_steps: int = 6000
    stage2_steps: int = 1000
    rollout: int = 4                     # stage-2 forecast steps (6 h each)
    peak_lr: float = 1e-4
    final_lr: float = 3e-6
    warmup: int = 300
    stage2_lr: float = 3e-6
    weight_decay: float = 0.1
    eval_every: int = 250
    n_val: int = 32
    ckpt_every: int = 250
    seed: int = 0
    use_bf16: bool = True

    def total_steps(self) -> int:
        return self.stage1_steps + self.stage2_steps

    def lr(self, step: int) -> float:
        if step >= self.stage1_steps:
            return self.stage2_lr
        if step < self.warmup:
            return self.peak_lr * (step + 1) / self.warmup
        x = (step - self.warmup) / max(1, self.stage1_steps - self.warmup)
        return self.final_lr + 0.5 * (self.peak_lr - self.final_lr) * (1 + math.cos(math.pi * x))

    def n_steps(self, step: int) -> int:
        return 1 if step < self.stage1_steps else self.rollout


# ------------------------------------------------------------------ data
class Sampler:
    """Random training batches from the staged 1-degree store."""

    def __init__(self, store, task, years, io_threads: int = 16):
        """store: a path to one zarr store, or an opened dataset (phase5.open_staged)."""
        self.ds = store if isinstance(store, xarray.Dataset) else xarray.open_zarr(store, chunks={})
        self.task = task
        t = pd.DatetimeIndex(self.ds.time.values)
        self.times = t[t.year.isin(list(years))]
        self._have = set(self.times)
        self.statics = {v: self.ds[v].astype(np.float32).load() for v in inputs.STATIC}
        self.io_threads = io_threads
        self.extra = tuple(v for v in task.target_variables
                           if v not in inputs.SURFACE and v not in inputs.ATMOS)

    def inits(self, n_steps: int) -> pd.DatetimeIndex:
        """Start times whose 2 inputs and n_steps targets are all in the store."""
        ok = [t for t in self.times
              if (t - STEP) in self._have and all((t + STEP * k) in self._have for k in range(1, n_steps + 1))]
        return pd.DatetimeIndex(ok)

    def batch(self, inits, n_steps: int):
        # one sample at a time: the library computes solar radiation only for batch size 1
        split = [data_utils.extract_inputs_targets_forcings(
            inputs.build_batch(self.ds, t, n_steps, method="native", statics=self.statics,
                               io_threads=self.io_threads, truth=True, extra=self.extra),
            target_lead_times=slice("6h", f"{n_steps * 6}h"), **dataclasses.asdict(self.task))
            for t in inits]
        if len(split) == 1:
            return split[0]
        # static fields (no batch dimension) are identical in every sample: keep them once
        return tuple(xarray.concat([s[i] for s in split], "batch", data_vars="minimal", coords="minimal",
                                   compat="override") for i in range(3))


# ------------------------------------------------------------------ model functions
def make_loss(model_config, task_config, stats, schemes, use_bf16=True):
    @hk.transform_with_state
    def loss_fn(inp, tgt, frc):
        p = finetune.construct(model_config, task_config, stats, schemes, use_bf16)
        loss, diag = p.loss(inp, tgt, frc)
        unwrap = lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True)   # noqa: E731
        return unwrap(loss), {k: unwrap(v) for k, v in diag.items()}
    return loss_fn


def core_loss(diag: dict, core_vars) -> float:
    """Weighted loss over the variables R0 predicts (comparable between runs)."""
    return float(sum(finetune.LOSS_WEIGHTS.get(v, 1.0) * float(diag[v]) for v in core_vars if v in diag))


def make_step(loss_fn, optimizer):
    def step(params, state, opt_state, inp, tgt, frc):
        def f(p):
            (loss, diag), new_state = loss_fn.apply(p, state, None, inp, tgt, frc)
            return loss, (diag, new_state)
        (loss, (diag, new_state)), grads = jax.value_and_grad(f, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_state, opt_state, loss, diag, optax.global_norm(grads)
    return jax.jit(step)


def make_eval(loss_fn):
    return jax.jit(lambda params, state, inp, tgt, frc: loss_fn.apply(params, state, None, inp, tgt, frc)[0])


def optimizer(cfg: TrainConfig):
    return optax.inject_hyperparams(optax.adamw)(learning_rate=cfg.lr(0), b1=0.9, b2=0.95,
                                                 weight_decay=cfg.weight_decay)


# ------------------------------------------------------------------ checkpoints
def save(path: Path, obj) -> None:
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        pickle.dump(jax.device_get(obj), f)
    tmp.rename(path)


def load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ loop
def train(cfg: TrainConfig, out: Path, sampler: Sampler, val_sampler: Sampler, model_config, task_config,
          stats: dict, init_params: dict, core_vars, log=print, max_steps: int | None = None):
    """Run (or resume) one fine-tuning run; returns the final params."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    schemes = finetune.RUNS[cfg.run]
    loss_fn = make_loss(model_config, task_config, stats, schemes, cfg.use_bf16)
    opt = optimizer(cfg)
    step_fn, eval_fn = make_step(loss_fn, opt), make_eval(loss_fn)

    ck = out / "checkpoint.pkl"
    if ck.exists():
        s = load(ck)
        params, state, opt_state, start, rng_state = s["params"], s["state"], s["opt_state"], s["step"], s["rng"]
        rng = np.random.default_rng(); rng.bit_generator.state = rng_state
        log(f"resumed {cfg.run} at step {start}")
    else:
        params, state, opt_state, start = init_params, {}, opt.init(init_params), 0
        rng = np.random.default_rng(cfg.seed)

    val_inits = val_sampler.inits(1)
    val_inits = val_inits[np.linspace(0, len(val_inits) - 1, cfg.n_val).astype(int)]
    val_batches = [val_sampler.batch(val_inits[i:i + cfg.batch_size], 1)
                   for i in range(0, len(val_inits), cfg.batch_size)]

    def validate(step):
        tot, diag_sum = 0.0, {}
        for b in val_batches:
            loss, diag = eval_fn(params, state, *b)
            tot += float(loss)
            for k, v in diag.items():
                diag_sum[k] = diag_sum.get(k, 0.0) + float(v)
        n = len(val_batches)
        diag_mean = {k: v / n for k, v in diag_sum.items()}
        rec = dict(step=step, val_loss=tot / n, val_core_loss=core_loss(diag_mean, core_vars),
                   val_per_variable={k: round(v, 6) for k, v in diag_mean.items()})
        with open(out / "validation.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"[{cfg.run}] step {step}: validation loss {rec['val_loss']:.4f}, core {rec['val_core_loss']:.4f}")
        return rec

    if start == 0:
        validate(0)
    pool = cf.ThreadPoolExecutor(max_workers=1)
    inits_cache = {}

    def draw(step):
        n = cfg.n_steps(step)
        if n not in inits_cache:
            inits_cache[n] = sampler.inits(n)
        pick = inits_cache[n][rng.choice(len(inits_cache[n]), cfg.batch_size, replace=False)]
        return pool.submit(sampler.batch, pick, n)

    end = cfg.total_steps() if max_steps is None else min(cfg.total_steps(), start + max_steps)
    nxt = draw(start) if start < end else None
    t0 = time.perf_counter()
    for step in range(start, end):
        inp, tgt, frc = nxt.result()
        nxt = draw(step + 1) if step + 1 < end else None
        opt_state.hyperparams["learning_rate"] = jax.numpy.asarray(cfg.lr(step), jax.numpy.float32)
        params, state, opt_state, loss, diag, gnorm = step_fn(params, state, opt_state, inp, tgt, frc)
        if (step + 1) % 25 == 0 or step == start:
            rec = dict(step=step + 1, loss=float(loss), grad_norm=float(gnorm), lr=cfg.lr(step),
                       rollout=cfg.n_steps(step), s_per_step=(time.perf_counter() - t0) / (step + 1 - start))
            if not np.isfinite(rec["loss"]):
                raise FloatingPointError(f"loss is {rec['loss']} at step {step + 1} - stopping")
            with open(out / "train.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"[{cfg.run}] step {step + 1}/{cfg.total_steps()} loss {rec['loss']:.4f} "
                f"lr {rec['lr']:.1e} rollout {rec['rollout']} ({rec['s_per_step']:.2f} s/step)")
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.total_steps():
            validate(step + 1)
        if (step + 1) % cfg.ckpt_every == 0 or step + 1 == end:
            save(ck, dict(params=params, state=state, opt_state=opt_state, step=step + 1,
                          rng=rng.bit_generator.state))
    pool.shutdown()
    return params


def residual_check(params, model_config, task_config, stats, schemes, sampler: Sampler, inits, n_steps=4,
                   use_bf16=True) -> pd.DataFrame:
    """Forecast `n_steps` from each start and report the relative budget violations
    (physics.residuals) of every step - near zero for the schemes a run uses."""
    @hk.transform_with_state
    def fwd(inp, tmpl, frc):
        return finetune.construct(model_config, task_config, stats, schemes, use_bf16)(inp, tmpl, frc)
    apply = jax.jit(fwd.apply)
    rows = []
    for t in inits:
        inp, tgt, frc = sampler.batch([t], n_steps)
        pred = jax.device_get(apply(params, {}, None, inp, tgt * np.nan, frc)[0])
        prev = inp.isel(time=-1)
        phi = inp["geopotential_at_surface"]
        for k in range(n_steps):
            step = pred.isel(time=[k])
            r = physics.residuals(prev, step, phi)
            rows.append(dict(init=t, step=k + 1, **{n: float(np.abs(v).max()) for n, v in r.items()}))
            prev = step.isel(time=0)
    return pd.DataFrame(rows)
