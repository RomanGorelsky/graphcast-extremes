"""Loading and running GraphCast_small.

Adapted from the authors' demo notebook (docs/weathernext1_graph/graphcast_demo.ipynb
in github.com/google-deepmind/graphcast), with the interactive widgets replaced
by plain functions:

  download_assets()   -> copies checkpoint, stats and example data to ASSETS_DIR
  load_model()        -> Model with jitted forward function
  Model.forecast()    -> autoregressive rollout, returns an xarray.Dataset
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import haiku as hk
import jax
import numpy as np
import xarray

from weathernext.utils import autoregressive
from weathernext.utils import casting
from weathernext.utils import checkpoint
from weathernext.utils import data_utils
from weathernext.utils import normalization
from weathernext.utils import rollout
from weathernext.weathernext1_graph import graphcast

from . import paths


# --------------------------------------------------------------------------- assets
def parse_file_parts(file_name: str) -> dict:
    """'source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc' -> dict."""
    return dict(part.split("-", 1) for part in file_name.removesuffix(".nc").split("_"))


def _bucket():
    from google.cloud import storage
    return storage.Client.create_anonymous_client().get_bucket(paths.GCS_BUCKET)


def _list(bucket, sub: str) -> list[str]:
    pre = paths.GCS_PREFIX + sub
    return [n for b in bucket.list_blobs(prefix=pre) if (n := b.name.removeprefix(pre))]


def _fetch(bucket, sub: str, name: str, dest_dir: Path) -> Path:
    dest = dest_dir / name
    if not dest.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        bucket.blob(paths.GCS_PREFIX + sub + name).download_to_filename(str(tmp))
        tmp.rename(dest)
        print(f"downloaded {sub}{name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def find_params_name(names: list[str]) -> str:
    """Pick the GraphCast_small checkpoint from the bucket listing."""
    hits = [n for n in names if n.startswith(paths.MODEL_KEY + " ")
            and f"resolution {paths.MODEL_RESOLUTION}" in n
            and f"pressure levels {paths.MODEL_LEVELS}" in n]
    if len(hits) != 1:
        raise RuntimeError(f"expected one {paths.MODEL_KEY} checkpoint, found {hits} in {names}")
    return hits[0]


def example_dataset_names(names: list[str]) -> list[str]:
    """ERA5 example files matching the model (1 deg, 13 levels), fewest steps first."""
    ok = []
    for n in names:
        p = parse_file_parts(n)
        if (p.get("source") == "era5" and float(p.get("res", 0)) == paths.MODEL_RESOLUTION
                and int(p.get("levels", 0)) == paths.MODEL_LEVELS):
            ok.append(n)
    return sorted(ok, key=lambda n: int(parse_file_parts(n)["steps"]))


def download_assets(dest: Optional[Path] = None, max_example_steps: int = 12) -> Path:
    """Download checkpoint, normalisation stats and 1-deg ERA5 example files
    with at most `max_example_steps` forecast steps (12 steps = 3 days, ~0.3 GB)."""
    dest = Path(dest or paths.ASSETS_DIR)
    b = _bucket()
    _fetch(b, "params/", find_params_name(_list(b, "params/")), dest / "params")
    for s in paths.STATS_FILES:
        _fetch(b, "stats/", s, dest / "stats")
    for n in example_dataset_names(_list(b, "dataset/")):
        if int(parse_file_parts(n)["steps"]) <= max_example_steps:
            _fetch(b, "dataset/", n, dest / "dataset")
    return dest


# --------------------------------------------------------------------------- model
@dataclasses.dataclass
class Model:
    params: dict
    state: dict
    model_config: graphcast.ModelConfig
    task_config: graphcast.TaskConfig
    stats: dict
    description: str = ""
    use_bf16: bool = True     # authors' default; False = full float32 precision

    def __post_init__(self):
        mc, tc, st = self.model_config, self.task_config, self.stats

        def construct():
            p = graphcast.GraphCast(mc, tc)
            p = casting.Bfloat16Cast(p, enabled=self.use_bf16)
            p = normalization.InputsAndResiduals(
                p, diffs_stddev_by_level=st["diffs_stddev_by_level"],
                mean_by_level=st["mean_by_level"], stddev_by_level=st["stddev_by_level"])
            return autoregressive.Predictor(p, gradient_checkpointing=True)

        @hk.transform_with_state
        def run_forward(inputs, targets_template, forcings):
            return construct()(inputs, targets_template=targets_template, forcings=forcings)

        self._transformed = run_forward
        apply = jax.jit(run_forward.apply)
        self._forward = lambda **kw: apply(self.params, self.state, **kw)[0]

    def init_random_params(self, inputs, targets_template, forcings, seed: int = 0) -> None:
        """Replace params with random weights (for tests / architecture changes)."""
        init = jax.jit(self._transformed.init)
        self.params, self.state = init(jax.random.PRNGKey(seed), inputs=inputs,
                                       targets_template=targets_template, forcings=forcings)

    def extract(self, batch: xarray.Dataset, n_steps: int):
        """Split a dataset (2 input times + n_steps target times) into model arrays.

        The library counts the input times backwards from the END of the batch, so a
        longer batch is first cut to its first 2 + n_steps times - otherwise later
        times would silently be used as inputs (found in Phase 3).
        """
        if batch.sizes["time"] < 2 + n_steps:
            raise ValueError(f"batch has {batch.sizes['time']} times, needs 2 + {n_steps}")
        batch = batch.isel(time=slice(0, 2 + n_steps))
        return data_utils.extract_inputs_targets_forcings(
            batch, target_lead_times=slice("6h", f"{n_steps * 6}h"),
            **dataclasses.asdict(self.task_config))

    def forecast(self, inputs, targets_template, forcings, seed: int = 0,
                 keep: dict | None = None) -> xarray.Dataset:
        """Autoregressive rollout, one jitted 6-h step at a time.

        keep: optional {variable: None | [levels]} - only these variables (and levels)
        are kept from each step as it is produced. A full 10-day forecast of all 83
        fields is ~0.9 GB and is held twice while being joined; keeping only what is
        verified keeps Colab's 12 GB of RAM well clear (found in Phase 3).
        """
        template = targets_template if bool(np.isnan(targets_template[
            next(iter(targets_template.data_vars))].values.flat[0])) else targets_template * np.nan
        chunks = []
        for chunk in rollout.chunked_prediction_generator(
                predictor_fn=self._forward, rng=jax.random.PRNGKey(seed), inputs=inputs,
                targets_template=template, forcings=forcings):
            chunk = jax.device_get(chunk)
            if keep is not None:
                chunk = xarray.Dataset({v: chunk[v] if lev is None else chunk[v].sel(level=lev)
                                        for v, lev in keep.items()})
            chunks.append(chunk)
            del chunk
        pred = xarray.concat(chunks, dim="time")
        del chunks
        # concatenation moves time first; restore each variable's own dimension order
        # (surface and pressure-level variables differ) so forecasts and targets line up.
        return pred.assign({v: pred[v].transpose(*[d for d in targets_template[v].dims if d in pred[v].dims])
                            for v in pred.data_vars})


def load_stats(assets: Path) -> dict:
    return {s.removesuffix(".nc"): xarray.load_dataset(assets / "stats" / s).compute()
            for s in paths.STATS_FILES}


def load_model(assets: Optional[Path] = None, use_bf16: bool = True) -> Model:
    assets = Path(assets or paths.ASSETS_DIR)
    files = sorted((assets / "params").glob(paths.MODEL_KEY + " *"))
    if not files:
        raise FileNotFoundError(f"no {paths.MODEL_KEY} checkpoint in {assets/'params'}; run download_assets()")
    with open(files[0], "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    return Model(params=ckpt.params, state={}, model_config=ckpt.model_config,
                 task_config=ckpt.task_config, stats=load_stats(assets),
                 description=ckpt.description, use_bf16=use_bf16)


def load_example(assets: Optional[Path] = None, min_steps: int = 1) -> tuple[xarray.Dataset, str]:
    """Largest local 1-deg ERA5 example file with at least min_steps target steps."""
    assets = Path(assets or paths.ASSETS_DIR)
    names = example_dataset_names([p.name for p in (assets / "dataset").glob("*.nc")])
    names = [n for n in names if int(parse_file_parts(n)["steps"]) >= min_steps]
    if not names:
        raise FileNotFoundError("no suitable example dataset; run download_assets()")
    return xarray.load_dataset(assets / "dataset" / names[-1],
                               decode_timedelta=True).compute(), names[-1]
