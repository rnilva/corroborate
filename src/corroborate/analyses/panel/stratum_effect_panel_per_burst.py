"""`stratum_effect_panel_per_burst` — per-(env, burst) Cohen's d
panel via independent-samples seed pooling.

The independent-samples counterpart to `paired_g_per_burst`: same
per-burst-source iteration over per-cell ndarray data, but
treatment and baseline seeds are pooled INDEPENDENTLY within each
(env, burst) stratum (no seed pairing). Cohen's d uses the
simple-mean-variance form, matching
`stratified_arm_diff_pooled` / `stratum_panel.cohen_d`.

Existence rationale (closes CLAUDE.md §"Methodology debt"). The
seed-paired form (`paired_g_per_burst`) pseudo-replicates seeds
inside each (env, burst) stratum on RL implementation — CLAUDE.md
flags it as off-limits in RL bridges (`feedback_paired_g_in_rl`).
Three bridges retained the paired form as a documented "principled
exception" because they test phase consistency (per-burst sign-
count) and the canonical migration target — this primitive — did
not yet exist. With the primitive in hand, those bridges can
migrate without redesigning their phase-consistency question
shape.

Shape:
  cells → PerBurstPanelDResult
  per (env, burst): independent-samples Cohen's d on the per-burst
  source's mean across the (treatment seeds, baseline seeds) pools.

Same `source: Measurable[Mapping, NDArray]` parameter as
`paired_g_per_burst`. Same per-cell ndarray shape contract
(`(n_bursts,)` per cell). Same burst-spacing-agnostic
contract.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import polars as pl

from corroborate._internals.polars import as_rows
from corroborate.analyses._cell_value import evaluate_per_burst_source
from corroborate.bridge.analysis import analysis
from corroborate.measurables import Measurable
from corroborate.measurables.reductions import from_key, reduce_axis


DEFAULT_PER_BURST_SOURCE: Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
] = reduce_axis(from_key('mc_return'), axis=-1, op='mean')


@dataclass(frozen=True, slots=True)
class PerBurstStratumD:
    """One (env, burst) stratum: independent-samples Cohen's d +
    its Hedges' SE + per-arm sample counts + per-arm means.

    `cohen_d` uses the simple-mean-variance form
    `(μ_t − μ_b) / sqrt((σ_t² + σ_b²) / 2)` — same as
    `stratum_panel.cohen_d` / `stratified_arm_diff_pooled`.
    `cohen_se` uses Hedges 1981's independent-samples SE
    `sqrt((n_t+n_b)/(n_t·n_b) + d²/(2(n_t+n_b−2)))`. NaN when
    pooled variance is zero or finite-n per arm < 2."""
    env_name: str
    burst_index: int
    cohen_d: float
    cohen_se: float
    n_treatment: int
    n_baseline: int
    mean_treatment: float
    mean_baseline: float


@dataclass(frozen=True, slots=True)
class PerBurstPanelDResult:
    """Output of `stratum_effect_panel_per_burst`: panel of
    per-(env, burst) independent-samples Cohen's d values plus
    the input shape parameters the bridge author can introspect.

    Mirror of `PerBurstResult` (the paired-form output) with field
    `cohen_d` / `cohen_se` instead of `g` / `se` to surface the
    different statistic at the type level — bridges that consume
    one shouldn't silently accept the other."""
    strata: tuple[PerBurstStratumD, ...]
    measurable: str
    treatment_arm: str
    baseline_arm: str

    @property
    def n_strata(self) -> int:
        return len(self.strata)


def _cohen_d_indep_samples(
    mean_t: float, mean_b: float,
    sd_t: float, sd_b: float,
    n_t: int, n_b: int,
) -> tuple[float, float]:
    """Independent-samples Cohen's d + Hedges 1981 SE. NaN-NaN
    when pooled variance is zero or finite-n per arm < 2.

    Matches `stratum_panel.cohen_d` / `.cohen_se` so a panel
    bridge that mixes per-burst and per-stratum analyses sees
    the same statistic shape on both."""
    if n_t < 2 or n_b < 2:
        return float('nan'), float('nan')
    if math.isnan(sd_t) or math.isnan(sd_b):
        return float('nan'), float('nan')
    pooled_var = (sd_t ** 2 + sd_b ** 2) / 2.0
    if pooled_var <= 0.0:
        return float('nan'), float('nan')
    d = (mean_t - mean_b) / math.sqrt(pooled_var)
    n_sum = n_t + n_b
    n_prod = n_t * n_b
    df_se = n_sum - 2
    if n_prod == 0 or df_se <= 0:
        return d, float('nan')
    se_sq = n_sum / n_prod + d ** 2 / (2.0 * df_se)
    return d, math.sqrt(max(se_sq, 0.0))


@analysis
def stratum_effect_panel_per_burst(
    cells: pl.DataFrame | Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = DEFAULT_PER_BURST_SOURCE,
    env_name: str | None = None,
    arm_field: str = 'arm_key',
) -> PerBurstPanelDResult:
    """Per-(env, burst) independent-samples Cohen's d panel.

    For each cell, evaluate `source` to get a per-burst vector
    (length `n_bursts`). Group cells by `(env_name, arm)`; at each
    (env, burst) compute Cohen's d via the simple-mean-variance
    form on the treatment-seed pool vs the baseline-seed pool.

    `source` is a typed Measurable returning a per-burst NDArray.
    Default: `reduce_axis(from_key('mc_return'), axis=-1, op='mean')`
    — the per-burst-mean of `mc_return`. Bridges that want a
    different per-burst quantity compose the same way they do for
    `paired_g_per_burst`.

    `env_name`, when supplied, restricts the analysis to one env
    (skips cells with `record['env_name'] != env_name`). When
    None, all envs participate.

    `pair_by` is accepted from the bridge framework's calling
    convention but unused in this primitive: independent-samples
    pooling doesn't pair seeds across arms. The parameter stays
    so the bridge framework's `pair_by`-defaulted dispatch wires
    a value without erroring; pass `()` to make the absence
    explicit.

    Bursts that don't reach length b on any cell don't contribute
    (matches `paired_g_per_burst`'s multi-regime walk: cells with
    shorter trajectories naturally drop out of the higher-index
    burst strata)."""
    cells = as_rows(cells)
    del pair_by  # unused: independent-samples pooling doesn't pair seeds

    by_env_arm: dict[
        tuple[str, str], list[npt.NDArray[np.floating]],
    ] = {}
    for cell in cells:
        env = cell.get('env_name')
        arm = cell.get(arm_field)
        if not isinstance(env, str) or not isinstance(arm, str):
            continue
        if env_name is not None and env != env_name:
            continue
        if arm not in (treatment_arm, baseline_arm):
            continue
        per_burst = evaluate_per_burst_source(source, cell)
        if per_burst.size == 0:
            continue
        by_env_arm.setdefault((env, arm), []).append(per_burst)

    strata: list[PerBurstStratumD] = []
    envs = {env for (env, _) in by_env_arm.keys()}
    for env in sorted(envs):
        treat_arrays = by_env_arm.get((env, treatment_arm), [])
        base_arrays = by_env_arm.get((env, baseline_arm), [])
        if not treat_arrays or not base_arrays:
            continue
        max_bursts = max(
            (arr.shape[0] for arr in (*treat_arrays, *base_arrays)),
            default=0,
        )
        for b in range(max_bursts):
            treat_vals = [
                float(arr[b]) for arr in treat_arrays if arr.shape[0] > b
            ]
            base_vals = [
                float(arr[b]) for arr in base_arrays if arr.shape[0] > b
            ]
            treat_vals = [v for v in treat_vals if not math.isnan(v)]
            base_vals = [v for v in base_vals if not math.isnan(v)]
            n_t = len(treat_vals)
            n_b = len(base_vals)
            mean_t = (
                float(np.mean(treat_vals)) if n_t > 0 else float('nan')
            )
            mean_b = (
                float(np.mean(base_vals)) if n_b > 0 else float('nan')
            )
            sd_t = (
                float(np.std(treat_vals, ddof=1)) if n_t >= 2
                else float('nan')
            )
            sd_b = (
                float(np.std(base_vals, ddof=1)) if n_b >= 2
                else float('nan')
            )
            d, se = _cohen_d_indep_samples(
                mean_t, mean_b, sd_t, sd_b, n_t, n_b,
            )
            strata.append(PerBurstStratumD(
                env_name=env, burst_index=b,
                cohen_d=d, cohen_se=se,
                n_treatment=n_t, n_baseline=n_b,
                mean_treatment=mean_t, mean_baseline=mean_b,
            ))

    return PerBurstPanelDResult(
        strata=tuple(strata),
        measurable=source.name,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
    )


def panel_for_env_d(
    result: PerBurstPanelDResult, env_name: str,
) -> tuple[PerBurstStratumD, ...]:
    """Convenience: filter strata to one env in burst order.
    Mirror of `paired_g_per_burst.panel_for_env` for the
    independent-samples shape."""
    return tuple(
        s for s in result.strata
        if s.env_name == env_name
    )


__all__ = [
    'DEFAULT_PER_BURST_SOURCE',
    'PerBurstPanelDResult', 'PerBurstStratumD',
    'panel_for_env_d',
    'stratum_effect_panel_per_burst',
]
