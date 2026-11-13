"""Phase 4 check 1: are the differences between R0 and the corrections robust across events?

All comparisons are PAIRED: the same record cell-time, forecast by R0 and by a
correction. With only 4-5 events per kind, the uncertainty comes mostly from which
events happen to be in the sample, so it is estimated by resampling whole events
(bootstrap) and by leaving one event out at a time.

Sign convention as in verify.py: intensity_error < 0 = forecast too mild, so
delta = method - R0 < 0 means the correction made the underestimate WORSE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["event_id", "kind", "start_days", "time", "lat", "lon"]


def paired(errs: dict, base: str = "R0") -> pd.DataFrame:
    """errs: label -> record_errors table (33_verify.py). One row per record cell-time
    present for every label, with columns ie_<label> and hit_<label>."""
    out = None
    for lab, e in errs.items():
        e = e.copy()
        e["time"] = pd.to_datetime(e["time"])
        cols = {"intensity_error": f"ie_{lab}", "hit": f"hit_{lab}"}
        keep = KEY + list(cols)
        if lab == base:
            keep += ["record_shattering", "obs_exceedance"]
        e = e[keep].rename(columns=cols)
        out = e if out is None else out.merge(e, on=KEY, how="inner")
    return out


def event_table(p: pd.DataFrame, labels: list, base: str = "R0") -> pd.DataFrame:
    """Mean intensity error and hit rate per event and start, for every label."""
    agg = {"n": (f"ie_{base}", "size")}
    for lab in labels:
        agg[f"ie_{lab}"] = (f"ie_{lab}", "mean")
        agg[f"hit_{lab}"] = (f"hit_{lab}", "mean")
    t = p.groupby(["kind", "event_id", "start_days"]).agg(**agg).reset_index()
    for lab in labels:
        if lab != base:
            t[f"delta_{lab}"] = t[f"ie_{lab}"] - t[f"ie_{base}"]
    return t


def _event_sums(g: pd.DataFrame, col: str):
    s = g.groupby("event_id")[col].agg(["sum", "size"])
    return s["sum"].to_numpy(float), s["size"].to_numpy(float), list(s.index)


def bootstrap_delta(g: pd.DataFrame, lab: str, base: str = "R0", B: int = 5000, seed: int = 0,
                    what: str = "ie") -> dict:
    """Pooled (all record cell-times) mean of method - base over the events in g,
    with a 95% range from resampling whole events with replacement, the range when
    one event is left out, and how many events got worse / better."""
    g = g.assign(_d=g[f"{what}_{lab}"].astype(float) - g[f"{what}_{base}"].astype(float))
    S, N, ids = _event_sums(g, "_d")
    E = len(S)
    est = S.sum() / N.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, E, size=(B, E))
    boot = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    loo = [(S.sum() - S[i]) / (N.sum() - N[i]) for i in range(E)] if E > 1 else [np.nan]
    per_event = S / N
    return dict(n_events=E, n_cells=int(N.sum()), delta=float(est), ci_low=float(lo), ci_high=float(hi),
                loo_min=float(np.min(loo)), loo_max=float(np.max(loo)),
                events_worse=int((per_event < 0).sum()), events_better=int((per_event > 0).sum()),
                largest_effect_event=ids[int(np.argmax(np.abs(S)))])


def robustness_table(p: pd.DataFrame, labels: list, base: str = "R0", B: int = 5000) -> pd.DataFrame:
    rows = []
    for (kind, d), g in p.groupby(["kind", "start_days"]):
        for lab in labels:
            if lab == base:
                continue
            for what in ("ie", "hit"):
                rows.append(dict(kind=kind, start_days=d, method=lab, metric=what,
                                 **bootstrap_delta(g, lab, base, B, what=what)))
    return pd.DataFrame(rows)
