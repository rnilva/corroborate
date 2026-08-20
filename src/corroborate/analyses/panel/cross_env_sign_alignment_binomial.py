"""`cross_env_sign_alignment_binomial` — per-stratum bivariate
sign-test on per-stratum Cohen's d of TWO measurables.

Companion to `cross_env_consistency_binomial`:
  - `cross_env_consistency_binomial` asks "phenomenon X has the
    same sign at every env" (one measurable).
  - `cross_env_sign_alignment_binomial` asks "phenomenon X and
    phenomenon Y have aligned signs (same or opposite) at every
    env" (two measurables).

Use case: at γ=0.999 across canonical envs, DDQN's effect on
outcome (Δ_outcome) and on within-episode revisit rate
(Δ_rep_ea) align in OPPOSITE direction at every env — where
DDQN helps outcome, it reduces revisits; where it harms
(Asterix γ=0.999), it increases revisits. The "loop-reduction
channel" claim (REPORT_loop_hypothesis_synthesis.md §2.1) maps
cleanly onto this shape — sign of one tracks sign of the other
across envs.

At n=10:
  10/10 aligned → p = 0.0010 (one-tail)
   9/10 aligned → p = 0.0107
   8/10 aligned → p = 0.0547
   7/10 aligned → p = 0.1719

**Use this primitive when**:
1. The claim is "two measurables align in sign (same OR opposite)
   at every env" — channel-coupling style claim.
2. Within-env magnitudes differ (so a population correlation
   would be confounded by env-scale), but per-env directional
   coupling is the substantive question.

**Don't use this primitive when**:
1. You want to test "X causes Y across envs" — that's a
   population correlation question (`cross_stratum_arm_diff_slope`
   or partial-Spearman). Sign-alignment is necessary but not
   sufficient.
2. n_strata < 5 (insufficient binomial power even at full
   alignment).

Distinct from:
- `cross_env_consistency_binomial` — single measurable, asks
  "X holds at every env." This primitive is bivariate.
- `cross_stratum_arm_diff_slope` — Spearman of Δ_x vs Δ_y
  across envs. Asks "Δ_x SCALES with Δ_y" (magnitude
  question). Sign-alignment is the weaker shape (direction
  only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import binomtest as _binomtest  # type: ignore[attr-defined]

import polars as pl

from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    stratified_arm_diff_pooled,
)
from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class CrossEnvSignAlignmentBinomialResult:
    """Per-stratum bivariate sign-test for d_x vs d_y.

    `n_strata_total` — strata with finite d on BOTH measurables
    AND ≥ `min_seeds_per_arm` per arm AND |d| above respective
    null_floor on both.
    `n_strata_aligned` — count of strata where the signs match
    the `alignment` setting:
      `alignment='same'`     → sign(d_x) == sign(d_y)
      `alignment='opposite'` → sign(d_x) == −sign(d_y)
    `p_value` — one-tailed binomial p (Pr[X ≥ k | n, p=0.5])
    against random-alignment null.
    `cohen_d_x_per_stratum` / `cohen_d_y_per_stratum` —
    per-stratum d values (in stratum_ids order) for inspection.
    """
    n_strata_total: int
    n_strata_aligned: int
    p_value: float
    measurable_x: str
    measurable_y: str
    alignment: Literal['same', 'opposite']
    null_floor_x: float
    null_floor_y: float
    cohen_d_x_per_stratum: tuple[float, ...]
    cohen_d_y_per_stratum: tuple[float, ...]
    stratum_ids: tuple[tuple[object, ...], ...]


@analysis
def cross_env_sign_alignment_binomial(
    cells: pl.DataFrame,
    *,
    source_x: str,
    source_y: str,
    treatment_arm: str,
    baseline_arm: str,
    stratify_by: tuple[str, ...] = ('env_name',),
    arm_field: str = 'arm_key',
    alignment: Literal['same', 'opposite'] = 'opposite',
    null_floor_x: float = 0.0,
    null_floor_y: float = 0.0,
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
) -> CrossEnvSignAlignmentBinomialResult:
    """Compute per-stratum Cohen's d for two measurables, count
    sign-alignment (same/opposite per `alignment`), binomial
    sign-test.

    `alignment`:
      `'same'`     — predicts sign(d_x) == sign(d_y) per stratum
      `'opposite'` — predicts sign(d_x) == −sign(d_y) per stratum

    `null_floor_x` / `null_floor_y` (Cohen's d units):
      Strata with `|d_x| < null_floor_x` OR `|d_y| < null_floor_y`
      are dropped from the alignment count (treated as no-direction
      on at least one axis).

    Per-stratum d is produced via `stratified_arm_diff_pooled.fn`
    called twice (once per measurable). Both panels MUST use the
    same `stratify_by` for stratum_ids to align."""
    pooled_x = stratified_arm_diff_pooled.fn(
        cells,
        source=source_x,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        stratify_by=stratify_by,
        arm_field=arm_field,
        scope_predictor=scope_predictor,
        min_baseline_predictor=min_baseline_predictor,
        min_seeds_per_arm=min_seeds_per_arm,
    )
    pooled_y = stratified_arm_diff_pooled.fn(
        cells,
        source=source_y,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        stratify_by=stratify_by,
        arm_field=arm_field,
        scope_predictor=scope_predictor,
        min_baseline_predictor=min_baseline_predictor,
        min_seeds_per_arm=min_seeds_per_arm,
    )
    # Index x's per-stratum d by stratum_id for fast lookup
    x_by_id: dict[tuple[object, ...], float] = {
        s.stratum_id: s.cohen_d for s in pooled_x.per_stratum
    }
    y_by_id: dict[tuple[object, ...], float] = {
        s.stratum_id: s.cohen_d for s in pooled_y.per_stratum
    }
    # Take the intersection of stratum_ids (both must have finite d)
    common_ids = [
        sid for sid in x_by_id
        if sid in y_by_id
        and math.isfinite(x_by_id[sid])
        and math.isfinite(y_by_id[sid])
    ]
    # Apply per-axis null floors
    above_floor = [
        sid for sid in common_ids
        if abs(x_by_id[sid]) >= null_floor_x
        and abs(y_by_id[sid]) >= null_floor_y
    ]
    n_total = len(above_floor)
    if alignment == 'same':
        k = sum(
            1 for sid in above_floor
            if (x_by_id[sid] > 0) == (y_by_id[sid] > 0)
        )
    else:  # 'opposite'
        k = sum(
            1 for sid in above_floor
            if (x_by_id[sid] > 0) != (y_by_id[sid] > 0)
        )
    if n_total == 0:
        p = float('nan')
    else:
        p = float(_binomtest(k, n_total, p=0.5, alternative='greater').pvalue)
    # Preserve stratum_id ordering from x's panel
    ordered = [sid for sid in x_by_id if sid in y_by_id]
    return CrossEnvSignAlignmentBinomialResult(
        n_strata_total=n_total,
        n_strata_aligned=k,
        p_value=p,
        measurable_x=source_x,
        measurable_y=source_y,
        alignment=alignment,
        null_floor_x=null_floor_x,
        null_floor_y=null_floor_y,
        cohen_d_x_per_stratum=tuple(x_by_id[sid] for sid in ordered),
        cohen_d_y_per_stratum=tuple(y_by_id[sid] for sid in ordered),
        stratum_ids=tuple(ordered),
    )


__all__ = [
    'CrossEnvSignAlignmentBinomialResult',
    'cross_env_sign_alignment_binomial',
]
