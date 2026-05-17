"""Shared verdict helpers for the ddqn_three_conditions bridges.

Each helper takes an analysis-fixture result + threshold and
returns a `Verdict` (or `Verdict, RefutationClass | None`).
Bridges use these by importing the helper and calling it from
the bridge body — the framework's `random_effects_verdict`
doesn't cover all the asymmetric prediction shapes we need
(predicted-`a_lt_b` upper bound, meta-regression coefficient,
etc.), so this module hosts the substrate-specific verdict
logic.
"""
from __future__ import annotations

import math

from corroborate.analyses.meta_regression_unpaired_d import (
    MetaRegressionResult,
)
from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.analyses.stratified_partial_spearman import (
    StratifiedPartialSpearmanResult,
)
from corroborate.analyses.stratified_spearman import (
    StratifiedSpearmanResult,
)
from corroborate.bridge.verdict import RefutationClass, Verdict


def meta_regression_coef_verdict(
    result: MetaRegressionResult,
    coef_name: str,
    *,
    sign: int,
    threshold: float,
    min_strata: int,
) -> Verdict:
    """Verdict on a named coefficient from a meta-regression.

    `sign=-1`: HELD when `coef ≤ threshold` (a negative-valued
    threshold) AND significant. Significant in the opposite
    direction (`coef ≥ -threshold`) → NO_EFFECT (sign-flip).
    `sign=+1`: mirror of -1.
    `sign=0`: HELD when `|coef| < |threshold|` AND non-significant
    (null prediction confirmed); significant → NO_EFFECT.
    POWER_INSUFFICIENT when too few strata or coef is missing /
    NaN."""
    if result.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    coef = next(
        (c for c in result.coefficients if c.name == coef_name), None,
    )
    if coef is None or math.isnan(coef.coefficient):
        return Verdict.POWER_INSUFFICIENT
    if sign == -1:
        if coef.coefficient <= threshold and coef.is_significant:
            return Verdict.HELD
        if coef.is_significant and coef.coefficient >= -threshold:
            return Verdict.NO_EFFECT
        return Verdict.POWER_INSUFFICIENT
    if sign == 1:
        if coef.coefficient >= threshold and coef.is_significant:
            return Verdict.HELD
        if coef.is_significant and coef.coefficient <= -threshold:
            return Verdict.NO_EFFECT
        return Verdict.POWER_INSUFFICIENT
    if abs(coef.coefficient) < abs(threshold) and not coef.is_significant:
        return Verdict.HELD
    return Verdict.NO_EFFECT


def per_stratum_upper_bound_verdict(
    res: StratifiedArmDiffPooledResult,
    *,
    upper_bound: float,
    min_strata: int,
) -> tuple[Verdict, RefutationClass | None]:
    """Per-stratum upper-bound verdict for `predicted_direction=
    'a_lt_b'` asymmetric tests.

    HELD iff every stratum's Cohen's d ≤ `upper_bound`. Each
    stratum's CI is also inspected:
    - INVARIANT_VIOLATION if any CI fully > `upper_bound`
      (predicted upper bound exceeded — the `a_lt_b` direction
      is refuted in the wrong direction).
    - POWER_INSUFFICIENT if any CI straddles the bound (some d
      below, some CI extending above).
    - NO_EFFECT otherwise (some strata above the bound but no
      CI fully exceeds it)."""
    if res.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    all_below = True
    any_strong_above = False
    any_spans = False
    n_valid = 0
    for s in res.per_stratum:
        d = s.cohen_d
        se = s.cohen_se
        if math.isnan(d) or math.isnan(se):
            continue
        n_valid += 1
        ci_lo = d - 1.96 * se
        ci_hi = d + 1.96 * se
        if ci_lo > upper_bound:
            any_strong_above = True
        if d > upper_bound:
            all_below = False
        # Iteration-order-invariant spans check: a stratum with the
        # point estimate below the bound but the CI extending above
        # is "in the gray zone" regardless of what other strata
        # look like.
        if d <= upper_bound and ci_hi > upper_bound:
            any_spans = True
    if n_valid < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if any_strong_above:
        return Verdict.INVARIANT_VIOLATION, None
    if all_below:
        return Verdict.HELD, None
    if any_spans:
        return Verdict.POWER_INSUFFICIENT, None
    return Verdict.NO_EFFECT, None


def spearman_rho_verdict(
    result: StratifiedSpearmanResult | StratifiedPartialSpearmanResult,
    *,
    sign: int,
    threshold: float,
    min_strata: int = 1,
    alpha: float = 0.05,
) -> Verdict:
    """Verdict on a Fisher-z-pooled stratified Spearman ρ (marginal
    or partial).

    `sign=-1`: HELD iff `rho_pooled ≤ -|threshold|` AND `p < alpha`.
    `sign=+1`: mirror.
    `sign=0`: HELD iff `|rho_pooled| ≤ |threshold|` AND `p ≥ alpha`
    (null prediction confirmed); refuted if significant in either
    direction.

    POWER_INSUFFICIENT when too few strata or ρ is NaN."""
    if result.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = result.rho_pooled
    p = result.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT
    is_sig = p < alpha
    if sign == -1:
        if rho <= -abs(threshold) and is_sig:
            return Verdict.HELD
        if is_sig and rho >= abs(threshold):
            return Verdict.NO_EFFECT
        return Verdict.POWER_INSUFFICIENT
    if sign == 1:
        if rho >= abs(threshold) and is_sig:
            return Verdict.HELD
        if is_sig and rho <= -abs(threshold):
            return Verdict.NO_EFFECT
        return Verdict.POWER_INSUFFICIENT
    if abs(rho) <= abs(threshold) and not is_sig:
        return Verdict.HELD
    if is_sig:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


def per_stratum_d_threshold_verdict(
    res: StratifiedArmDiffPooledResult,
    *,
    threshold: float,
    sign: int,
    min_strata: int,
    wrong_sign_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """Per-stratum threshold verdict for "DDQN's effect is
    uniformly negative (or positive) across strata".

    `sign=-1`: HELD iff every per-stratum cohen_d ≤ `threshold`
    (a negative number). `wrong_sign_threshold` flags
    sign-flipped strata that cross +0.3 — refutation.
    `sign=+1`: mirror — every per-stratum cohen_d ≥ `threshold`.

    POWER_INSUFFICIENT when fewer than `min_strata` valid
    (non-NaN) strata contribute."""
    if res.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    all_pass = True
    any_wrong_sign = False
    n_valid = 0
    for s in res.per_stratum:
        d = s.cohen_d
        if math.isnan(d):
            continue
        n_valid += 1
        if sign == -1:
            if d > threshold:
                all_pass = False
            if d > wrong_sign_threshold:
                any_wrong_sign = True
        else:  # sign == 1
            if d < threshold:
                all_pass = False
            if d < -wrong_sign_threshold:
                any_wrong_sign = True
    if n_valid < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if any_wrong_sign:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if all_pass:
        return Verdict.HELD, None
    return Verdict.NO_EFFECT, None
