"""`cross_env_consistency_binomial` — per-stratum directional
sign-test against the random-direction null.

Use case: cross-env claims at small n_strata (~10) where
population magnitude tests (Spearman ρ, meta-regression slope)
are structurally underpowered, but a directional consistency
claim — "DDQN reduces jens at every env" — can still be
distinguished from chance via a binomial test on per-stratum
directions.

At n=10, the binomial gates are:
  10/10 same direction → p = 0.0010 (one-tail)
   9/10 same direction → p = 0.0107
   8/10 same direction → p = 0.0547
   7/10 same direction → p = 0.1719

The framework's bridge-cluster convention (each bridge = one
stratum, composed_verdict aggregates) is a VERDICT-level
sign-alignment. This primitive is the VALUE-level sign-alignment
— it acts on per-stratum Cohen's d values directly, so the
binomial test sees magnitudes (drop near-zero |d| values via
`null_floor`) and the null is sharp.

**Use this primitive when**:
1. The claim is "phenomenon X holds at most/all envs"
2. n_strata ≈ 5-20 (population magnitude tests fail; consistency
   succeeds at high alignment)
3. Per-env magnitudes are noisy but directions are stable

**Don't use this primitive when**:
1. The claim is "phenomenon X SCALES with env-feature Y" — that's
   a population magnitude claim; use `cross_stratum_property_slope`
   or `meta_regression_unpaired_d`.
2. n_strata < 5 — binomial test lacks power (8/8 still only
   p=0.0039 one-tail).
3. The effect of interest is the magnitude itself (use DL-pooled
   `stratified_arm_diff_pooled`).

Distinct from:
- `stratified_arm_diff_pooled` — DL random-effects pooling on
  Cohen's d. Asks "what's the typical effect size, weighted by
  precision?" — same per-stratum panel, different question.
- `cross_stratum_property_slope` — Spearman of per-stratum d vs
  per-stratum env feature. Asks "does effect SCALE with env-
  feature?" Population magnitude question.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from scipy.stats import binomtest as _binomtest  # type: ignore[attr-defined]

from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    stratified_arm_diff_pooled,
)
from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class CrossEnvConsistencyBinomialResult:
    """Per-stratum directional sign-test against the random-
    direction null.

    `n_strata_total` is the count of strata that contributed
    Cohen's d (post min_seeds_per_arm filter).
    `n_strata_above_floor` drops strata with `|d| < null_floor`
    (treated as no-direction).
    `n_signed_predicted` is the count of strata whose `d` agrees
    with `predicted_direction` (among the above-floor set).
    `p_value` is the one-tailed binomial p (Pr[X ≥ k | n,
    p=0.5]) for `predicted_direction != 'either'`; two-tailed
    otherwise.
    `cohen_d_per_stratum` is the raw per-stratum d for inspection.
    """
    n_strata_total: int
    n_strata_above_floor: int
    n_signed_predicted: int
    p_value: float
    measurable: str
    predicted_direction: Literal['a_gt_b', 'a_lt_b', 'either']
    null_floor: float
    cohen_d_per_stratum: tuple[float, ...]
    stratum_ids: tuple[tuple[object, ...], ...]


@analysis
def cross_env_consistency_binomial(
    cells: Iterable[Mapping[str, object]],
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    stratify_by: tuple[str, ...] = ('env_name',),
    arm_field: str = 'arm_key',
    predicted_direction: Literal['a_gt_b', 'a_lt_b', 'either'] = 'either',
    null_floor: float = 0.0,
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
) -> CrossEnvConsistencyBinomialResult:
    """Compute the per-stratum Cohen's d panel, count signs
    against the predicted direction, run a binomial sign-test.

    `predicted_direction`:
      `a_gt_b` — predicts `d > +null_floor` per stratum (DDQN
        higher than vanilla on `source`).
      `a_lt_b` — predicts `d < -null_floor` per stratum.
      `either` — two-tailed test against random direction (drops
        |d| < null_floor, runs the test on the resulting count).

    `null_floor` (Cohen's d units, default 0.0):
      Strata with `|d| < null_floor` are treated as no-direction
      and dropped before the binomial count. Set to ≈0.1 to
      require a "meaningful" direction; default 0.0 counts every
      finite-d stratum.

    Per-stratum d is produced via `stratified_arm_diff_pooled.fn`
    (delegates to the same panel build the DL meta-analysis uses).
    Strata where Cohen's d is NaN (saturated outcome, n<2 in an
    arm, etc.) are dropped before counting."""
    cells_list = list(cells)
    pooled = stratified_arm_diff_pooled.fn(
        cells_list,
        source=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        stratify_by=stratify_by,
        arm_field=arm_field,
        scope_predictor=scope_predictor,
        min_baseline_predictor=min_baseline_predictor,
        min_seeds_per_arm=min_seeds_per_arm,
    )
    ds = [s.cohen_d for s in pooled.per_stratum]
    ids = [s.stratum_id for s in pooled.per_stratum]
    finite_pairs = [
        (d, sid) for d, sid in zip(ds, ids) if math.isfinite(d)
    ]
    n_total = len(finite_pairs)
    above_floor = [
        (d, sid) for d, sid in finite_pairs if abs(d) >= null_floor
    ]
    n_above = len(above_floor)
    if predicted_direction == 'a_gt_b':
        k = sum(1 for d, _ in above_floor if d > 0)
    elif predicted_direction == 'a_lt_b':
        k = sum(1 for d, _ in above_floor if d < 0)
    else:
        # 'either' — two-tailed test: how many in the majority direction?
        n_pos = sum(1 for d, _ in above_floor if d > 0)
        n_neg = n_above - n_pos
        k = max(n_pos, n_neg)
    if n_above == 0:
        p = float('nan')
    elif predicted_direction == 'either':
        # Two-tailed binomial against p=0.5
        p = float(_binomtest(k, n_above, p=0.5, alternative='two-sided').pvalue)
    else:
        # One-tailed (we predicted direction)
        p = float(_binomtest(k, n_above, p=0.5, alternative='greater').pvalue)
    return CrossEnvConsistencyBinomialResult(
        n_strata_total=n_total,
        n_strata_above_floor=n_above,
        n_signed_predicted=k,
        p_value=p,
        measurable=source,
        predicted_direction=predicted_direction,
        null_floor=null_floor,
        cohen_d_per_stratum=tuple(d for d, _ in finite_pairs),
        stratum_ids=tuple(ids),
    )


__all__ = [
    'CrossEnvConsistencyBinomialResult',
    'cross_env_consistency_binomial',
]
