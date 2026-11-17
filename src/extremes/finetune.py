"""Phase 5: GraphCast_small with 7 extra flux outputs and Sha et al.'s conservation
schemes inside the forecast loop, ready for fine-tuning.

Model stack (outermost first):
  autoregressive.Predictor      multi-step rollout; the training loss averages the steps
  PhysicsPredictor              applies the schemes to each step (physical units) and
                                computes the loss AFTER them, so gradients pass through
  normalization.InputsAndResiduals, casting.Bfloat16Cast, graphcast.GraphCast  (as R0)

Runs (identical data, steps and learning rate):
  R1 control (new outputs, no schemes) | R2 + dry-air mass | R3 + moisture | R4 + energy
  R5 (Phase 7): R1 trained with the sharpness-preserving loss `amse_per_level` instead of MSE

The 7 new outputs are target-only (like precipitation in GraphCast_small): predicted,
never fed back. The decoder's last layer is grown by their channels; the existing
weights are copied unchanged and the new ones start at zero, so before training the
model predicts exactly what R0 predicts, plus the climatological mean flux.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import xarray

from weathernext.utils import autoregressive, casting, losses, normalization, predictor_base, variables, \
    xarray_tree
from weathernext.weathernext1_graph import graphcast

from . import physics

RUNS = {"R1": (), "R2": ("mass",), "R3": ("mass", "moisture"), "R4": ("mass", "moisture", "energy"),
        "R5": ()}                          # Phase 7, prospect 3: R1 with a sharpness-preserving loss
RUN_LOSS = {"R5": "amse"}                  # training loss per run (default "mse"); validation is always MSE
FLUX_LOSS_WEIGHT = 0.1                     # as GraphCast weights precipitation (Sha et al.: 0.1)
LOSS_WEIGHTS = {                           # graphcast.GraphCast.loss_and_predictions, + the fluxes
    "2m_temperature": 1.0, "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1, "total_precipitation_6hr": 0.1,
    **{v: FLUX_LOSS_WEIGHT for v in physics.FLUX_VARS}}


# ------------------------------------------------------------------ task and parameters
def extended_task(task):
    """Same task with the 7 flux variables added as target-only outputs."""
    new = tuple(v for v in physics.FLUX_VARS if v not in task.target_variables)
    return dataclasses.replace(task, target_variables=tuple(task.target_variables) + new)


def channel_layout(task) -> dict:
    """{variable: (first channel, number of channels)} of the decoder output, in the
    library's order (variables sorted by name, pressure levels inside)."""
    layout, start = {}, 0
    for v in sorted(task.target_variables):
        n = len(task.pressure_levels) if v in variables.ALL_ATMOSPHERIC_VARS else 1
        layout[v] = (start, n)
        start += n
    return layout


def _decoder_output_key(params: dict, n_out: int) -> str:
    keys = [k for k, p in params.items()
            if "mesh2grid_gnn" in k and "decoder_nodes_grid_nodes" in k
            and "w" in p and np.shape(p["w"])[-1] == n_out]
    if len(keys) != 1:
        raise RuntimeError(f"expected one decoder output layer with {n_out} outputs, found {keys}")
    return keys[0]


def grow_params(params: dict, old_task, new_task) -> dict:
    """Copy of `params` whose decoder output layer has the new task's channels: old
    channels moved to their new positions unchanged, new channels zero."""
    old, new = channel_layout(old_task), channel_layout(new_task)
    n_old = sum(n for _, n in old.values())
    n_new = sum(n for _, n in new.values())
    key = _decoder_output_key(params, n_old)
    w, b = np.asarray(params[key]["w"]), np.asarray(params[key]["b"])
    w2 = np.zeros(w.shape[:-1] + (n_new,), w.dtype)
    b2 = np.zeros((n_new,), b.dtype)
    for v, (s_old, n) in old.items():
        s_new, n2 = new[v]
        assert n == n2, v
        w2[..., s_new:s_new + n] = w[..., s_old:s_old + n]
        b2[s_new:s_new + n] = b[s_old:s_old + n]
    out = {k: dict(p) for k, p in params.items()}
    out[key] = {**out[key], "w": w2, "b": b2}
    return out


def new_channel_mask(params: dict, old_task, new_task) -> tuple[str, np.ndarray]:
    """(decoder output layer key, 0/1 vector over its output channels): 1 only for the
    channels that are new in new_task (the flux outputs). Used to train only those."""
    old, new = channel_layout(old_task), channel_layout(new_task)
    n_new = sum(n for _, n in new.values())
    key = _decoder_output_key(params, n_new)
    cols = np.zeros(n_new, np.float32)
    for v, (s, n) in new.items():
        if v not in old:
            cols[s:s + n] = 1.0
    return key, cols


def extend_stats(stats: dict, flux_mean: xarray.Dataset, flux_std: xarray.Dataset) -> dict:
    """Add the flux variables' mean and standard deviation to the normalisation stats
    (target-only variables are normalised directly, so no difference stats are needed)."""
    return {**stats,
            "mean_by_level": xarray.merge([stats["mean_by_level"], flux_mean], compat="override"),
            "stddev_by_level": xarray.merge([stats["stddev_by_level"], flux_std], compat="override")}


# ------------------------------------------------------------------ predictor
class PhysicsPredictor(predictor_base.Predictor):
    """One-step predictor: inner model, then the conservation schemes; loss after them.

    The loss is the library's latitude- and level-weighted MSE in the same normalised
    space as GraphCast's own training (residuals for variables that are also inputs,
    values for target-only ones), with LOSS_WEIGHTS."""

    def __init__(self, predictor, stats: dict, schemes: tuple = (), loss_weights: dict | None = None,
                 loss_kind: str = "mse"):
        if loss_kind not in ("mse", "amse"):
            raise ValueError(f"unknown loss {loss_kind!r}")
        self._predictor = predictor
        self._schemes = tuple(schemes)
        self._scales = stats["stddev_by_level"]
        self._locations = stats["mean_by_level"]
        self._residual_scales = stats["diffs_stddev_by_level"]
        self._weights = dict(LOSS_WEIGHTS if loss_weights is None else loss_weights)
        self._loss_kind = loss_kind

    def __call__(self, inputs, targets_template, forcings, **kwargs):
        pred = self._predictor(inputs, targets_template, forcings=forcings, **kwargs)
        return physics.apply(inputs.isel(time=-1), pred, inputs["geopotential_at_surface"], self._schemes)

    def _normalise(self, inputs, x):
        if x.name in inputs:
            return normalization.normalize(x - inputs[x.name].isel(time=-1), self._residual_scales, None)
        return normalization.normalize(x, self._scales, self._locations)

    def loss_and_predictions(self, inputs, targets, forcings, **kwargs):
        pred = self(inputs, targets, forcings, is_training=True, **kwargs)
        norm_pred = xarray_tree.map_structure(lambda x: self._normalise(inputs, x), pred)
        norm_targ = xarray_tree.map_structure(lambda x: self._normalise(inputs, x), targets)
        weights = {k: w for k, w in self._weights.items() if k in targets}
        if self._loss_kind == "amse":
            loss = amse_per_level(norm_pred, norm_targ, per_variable_weights=weights)
        else:
            loss = losses.weighted_mse_per_level(norm_pred, norm_targ, per_variable_weights=weights)
        return loss, pred

    def loss(self, inputs, targets, forcings, **kwargs):
        return self.loss_and_predictions(inputs, targets, forcings, **kwargs)[0]


# ------------------------------------------------------------------ sharpness-preserving loss (R5)
AMSE_EPS = 1e-6


def _amse_band(f, t, w):
    """'Adjusted MSE' of one field pair over (lat, lon), after Subich et al. (2025).

    With latitude-weighted means m, standard deviations s and correlation r:
        MSE  = (m_f - m_t)^2 + (s_f - s_t)^2 + 2 s_f s_t (1 - r)
        AMSE = (m_f - m_t)^2 + (s_f - s_t)^2 + 2 max(s_f, s_t)^2 (1 - r)
    Under plain MSE an uncertain forecast (r < 1) lowers its loss by shrinking s_f -
    i.e. by smoothing. In AMSE the last term no longer shrinks with s_f, so smoothing
    only costs (through (s_f - s_t)^2). AMSE >= MSE, with equality when s_f = s_t."""
    def mean(x):
        return (x * w).mean(("lat", "lon"))
    mf, mt = mean(f), mean(t)
    df, dt = f - mf, t - mt
    vf, vt, cov = mean(df * df), mean(dt * dt), mean(df * dt)
    sf, st = (vf + AMSE_EPS) ** 0.5, (vt + AMSE_EPS) ** 0.5
    vmax = 0.5 * (vf + vt + abs(vf - vt))                 # max(vf, vt) without a branch
    return (mf - mt) ** 2 + (sf - st) ** 2 + 2 * vmax * (1 - cov / (sf * st))


def _gradients(x):
    """East-west (periodic) and north-south differences between neighbouring cells -
    the small-scale detail that smoothing removes first."""
    # slices + concat (not integer-array indexing), which works the same on numpy and JAX data
    east = xarray.concat([x.isel(lon=slice(1, None)), x.isel(lon=slice(0, 1))], "lon")
    gx = east.assign_coords(lon=x.lon) - x
    lower = x.isel(lat=slice(None, -1))
    gy = x.isel(lat=slice(1, None)).assign_coords(lat=lower.lat) - lower
    return gx, gy


def amse_per_level(predictions, targets, per_variable_weights=None, gradient_weight: float = 1.0):
    """Drop-in for losses.weighted_mse_per_level (same weights, same output form): per
    variable, AMSE of the field + gradient_weight x mean AMSE of its two gradient fields,
    pressure-level weighted, averaged over everything but the batch."""
    def one(p, t):
        w = losses.normalized_latitude_weights(t).astype(p.dtype)
        loss = _amse_band(p, t, w)
        if gradient_weight:
            (px, py), (tx, ty) = _gradients(p), _gradients(t)
            wy = w.isel(lat=slice(None, -1))
            wy = wy / wy.mean()
            loss = loss + gradient_weight * 0.5 * (_amse_band(px, tx, w) + _amse_band(py, ty, wy))
        if "level" in t.dims:
            loss = loss * losses.normalized_level_weights(t).astype(loss.dtype)
        return loss.mean([d for d in loss.dims if d != "batch"])
    per_var = xarray_tree.map_structure(one, predictions, targets)
    return losses.sum_per_variable_losses(per_var, per_variable_weights or {})


def loss_weights(flux_weight: float = FLUX_LOSS_WEIGHT) -> dict:
    """LOSS_WEIGHTS with the 7 flux variables at `flux_weight` (0 = fluxes do not count)."""
    return {**LOSS_WEIGHTS, **{v: flux_weight for v in physics.FLUX_VARS}}


def construct(model_config, task_config, stats: dict, schemes: tuple = (), use_bf16: bool = True,
              gradient_checkpointing: bool = True, weights: dict | None = None, loss_kind: str = "mse"):
    """The full model stack; call inside a haiku transform."""
    p = graphcast.GraphCast(model_config, task_config)
    p = casting.Bfloat16Cast(p, enabled=use_bf16)
    p = normalization.InputsAndResiduals(p, diffs_stddev_by_level=stats["diffs_stddev_by_level"],
                                         mean_by_level=stats["mean_by_level"],
                                         stddev_by_level=stats["stddev_by_level"])
    p = PhysicsPredictor(p, stats, schemes, weights, loss_kind)
    return autoregressive.Predictor(p, gradient_checkpointing=gradient_checkpointing)


# ------------------------------------------------------------------ forecasting (Phase 6)
def load_run(run: str, stage: int | None = None, use_bf16: bool = True):
    """A fine-tuned run (R1-R4) as a forecast model with the same interface as R0's
    (model.Model: extract / forecast); its conservation schemes act at every step.
    stage=None: final weights; stage=k: the weights at the end of stage k."""
    import haiku as hk
    import jax
    from weathernext.utils import checkpoint
    from . import model, phase5

    name = f"params_{run}.ckpt" if stage is None else f"params_{run}_stage{stage}.ckpt"
    with open(phase5.run_dir(run) / name, "rb") as f:
        ck = checkpoint.load(f, graphcast.CheckPoint)
    stats = extend_stats(model.load_stats(model.paths.ASSETS_DIR), *phase5.load_flux_stats())
    schemes = RUNS[run]

    class FineTuned(model.Model):
        def __post_init__(self):
            @hk.transform_with_state
            def run_forward(inputs, targets_template, forcings):
                return construct(self.model_config, self.task_config, self.stats, schemes, self.use_bf16,
                                 gradient_checkpointing=False)(inputs, targets_template, forcings)
            self._transformed = run_forward
            apply = jax.jit(run_forward.apply)
            self._forward = lambda **kw: apply(self.params, self.state, **kw)[0]

    return FineTuned(params=ck.params, state={}, model_config=ck.model_config, task_config=ck.task_config,
                     stats=stats, description=ck.description, use_bf16=use_bf16)

