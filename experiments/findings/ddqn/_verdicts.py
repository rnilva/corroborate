"""CI-vs-threshold verdict helpers + theory-derived rescue threshold.

The `_native_diff_*_verdict` helpers widen the CI when
`paired_g.assumption_violations` flags heavy-tail / skew — the
framework's own SE under-covers ~10-25% on those distributions
(reviewer-3 catch); the verdict propagates that knowledge."""
from __future__ import annotations

import math

from corroborate.bridge.verdict import Verdict


def rescue_threshold(
    *,
    failure_baseline: float = 0.1,
    optimal_ceiling: float = 0.8,
    rescue_fraction: float = 0.5,
) -> float:
    """`rescue_fraction × (optimal_ceiling − failure_baseline)`
    = 0.5 × (0.8 − 0.1) = 0.35. Empirically calibrated:
    failure_baseline ≈ vanilla DQN floor at rs=0.1 on FR; ceiling
    ≈ empirical RL convergence; fraction = "≥half headroom" bound."""
    return rescue_fraction * (optimal_ceiling - failure_baseline)


def has_heavy_tail_violation(
    assumption_violations: tuple[str, ...],
) -> bool:
    """True when paired_g flagged heavy-tail/skew bias that makes
    the Gaussian ±1.96×se CI anti-conservative."""
    return any(
        'heavy_tail' in v or 'skew' in v for v in assumption_violations
    )


def native_diff_ci_verdict(
    md: float, se: float, threshold: float,
    *,
    assumption_violations: tuple[str, ...] = (),
    ci_widening_factor: float = 1.25,
) -> Verdict:
    """HELD when widened-95%-CI lower bound ≥ threshold;
    NO_EFFECT when upper bound < threshold; else POW_INSUF.
    Widening factor (default 1.25) compensates for paired_g's
    flagged heavy-tail / skew miscalibration."""
    if math.isnan(md) or math.isnan(se):
        return Verdict.POWER_INSUFFICIENT
    se_eff = (
        se * ci_widening_factor
        if has_heavy_tail_violation(assumption_violations)
        else se
    )
    ci_lo = md - 1.96 * se_eff
    ci_hi = md + 1.96 * se_eff
    if ci_lo >= threshold:
        return Verdict.HELD
    if ci_hi < threshold:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


def native_diff_null_verdict(
    md: float, se: float, null_ceiling: float,
    *,
    assumption_violations: tuple[str, ...] = (),
    ci_widening_factor: float = 1.25,
) -> Verdict:
    """Null-prediction sibling of `native_diff_ci_verdict`. HELD
    when widened CI fully inside ±null_ceiling."""
    if math.isnan(md) or math.isnan(se):
        return Verdict.POWER_INSUFFICIENT
    se_eff = (
        se * ci_widening_factor
        if has_heavy_tail_violation(assumption_violations)
        else se
    )
    ci_lo = md - 1.96 * se_eff
    ci_hi = md + 1.96 * se_eff
    if ci_lo >= -null_ceiling and ci_hi <= null_ceiling:
        return Verdict.HELD
    if ci_lo > null_ceiling or ci_hi < -null_ceiling:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT
