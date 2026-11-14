"""Phase 5: GraphCast_small with 7 extra flux outputs and Sha et al.'s conservation
schemes inside the forecast loop, ready for fine-tuning.

Model stack (outermost first):
  autoregressive.Predictor      multi-step rollout; the training loss averages the steps
  PhysicsPredictor              applies the schemes to each step (physical units) and
                                computes the loss AFTER them, so gradients pass through
  normalization.InputsAndResiduals, casting.Bfloat16Cast, graphcast.GraphCast  (as R0)

Runs (identical data, steps and learning rate):
  R1 control (new outputs, no schemes) | R2 + dry-air mass | R3 + moisture | R4 + energy

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

RUNS = {"R1": (), "R2": ("mass",), "R3": ("mass", "moisture"), "R4": ("mass", "moisture", "energy")}
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

    def __init__(self, predictor, stats: dict, schemes: tuple = (), loss_weights: dict | None = None):
        self._predictor = predictor
        self._schemes = tuple(schemes)
        self._scales = stats["stddev_by_level"]
        self._locations = stats["mean_by_level"]
        self._residual_scales = stats["diffs_stddev_by_level"]
        self._weights = dict(LOSS_WEIGHTS if loss_weights is None else loss_weights)

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
        loss = losses.weighted_mse_per_level(norm_pred, norm_targ, per_variable_weights=weights)
        return loss, pred

    def loss(self, inputs, targets, forcings, **kwargs):
        return self.loss_and_predictions(inputs, targets, forcings, **kwargs)[0]


def construct(model_config, task_config, stats: dict, schemes: tuple = (), use_bf16: bool = True,
              gradient_checkpointing: bool = True):
    """The full model stack; call inside a haiku transform."""
    p = graphcast.GraphCast(model_config, task_config)
    p = casting.Bfloat16Cast(p, enabled=use_bf16)
    p = normalization.InputsAndResiduals(p, diffs_stddev_by_level=stats["diffs_stddev_by_level"],
                                         mean_by_level=stats["mean_by_level"],
                                         stddev_by_level=stats["stddev_by_level"])
    p = PhysicsPredictor(p, stats, schemes)
    return autoregressive.Predictor(p, gradient_checkpointing=gradient_checkpointing)
