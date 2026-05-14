"""CI-vs-threshold verdict helpers + theory-derived rescue threshold.

The `_native_diff_*_verdict` helpers widen the CI when
`paired_g.assumption_violations` flags heavy-tail / skew — the
framework's own SE under-covers ~10-25% on those distributions
(reviewer-3 catch); the verdict propagates that knowledge.

DoWhy + partial-Spearman verdict deciders factor out the 6-bridge
DoWhy refutation-trio pattern (bias_correction.py) and the
4-bridge partial-Spearman shadow/mediator pattern (mediation.py)
into the substrate's shared verdict-logic layer."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

import scipy.stats as stats

from corroborate.bridge.verdict import RefutationClass, Verdict


# ============ Typed protocols for analysis-result shapes ============


class _BackdoorResult(Protocol):
    @property
    def identified(self) -> bool: ...
    @property
    def ate(self) -> float: ...


class _RefutationResult(Protocol):
    @property
    def real_ate(self) -> float: ...
    @property
    def refuted_ate(self) -> float: ...


class _PartialSpearmanResult(Protocol):
    @property
    def n_strata(self) -> int: ...
    @property
    def rho_pooled(self) -> float: ...


# ============ DoWhy result deciders ============


def dowhy_backdoor_verdict(
    b: _BackdoorResult,
    *,
    ate_threshold: float,
    sign: Literal[-1, 1] = -1,
    zero_guard: bool = False,
) -> Verdict:
    """DoWhy backdoor ATE verdict.

    `sign=-1` (default, predicted-negative direction): HELD when
    identified ∧ ATE ≤ `ate_threshold` (typically a negative
    ceiling like -0.1); POW_INSUF when ATE in (ceiling, 0);
    NO_EFFECT when ATE ≥ 0.

    `sign=+1` (predicted-positive direction): HELD when
    identified ∧ ATE ≥ `ate_threshold` (typically a positive
    floor like +0.1); POW_INSUF when ATE in (0, floor);
    NO_EFFECT when ATE ≤ 0.

    `zero_guard=True` treats |ATE| < 1e-6 as POW_INSUF — DoWhy
    machine-epsilon signs are RNG-dependent and shouldn't flip
    verdicts."""
    if not b.identified:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(b.ate):
        return Verdict.POWER_INSUFFICIENT
    if zero_guard and abs(b.ate) < 1e-6:
        return Verdict.POWER_INSUFFICIENT
    if sign < 0:
        if b.ate <= ate_threshold:
            return Verdict.HELD
        if b.ate < 0.0:
            return Verdict.POWER_INSUFFICIENT
        return Verdict.NO_EFFECT
    if b.ate >= ate_threshold:
        return Verdict.HELD
    if b.ate > 0.0:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


def dowhy_placebo_verdict(
    p: _RefutationResult,
    *,
    max_ratio: float,
) -> Verdict:
    """Placebo refutation verdict. HELD when |placebo/real| <
    `max_ratio` (random treatment shrinks ATE to ≪ real).
    NaN/zero real → POW_INSUF."""
    real, placebo = p.real_ate, p.refuted_ate
    if math.isnan(real) or math.isnan(placebo) or abs(real) < 1e-9:
        return Verdict.POWER_INSUFFICIENT
    ratio = abs(placebo / real)
    if ratio < max_ratio:
        return Verdict.HELD
    if ratio < max_ratio * 2:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


def dowhy_rcc_verdict(
    r: _RefutationResult,
    *,
    max_drift_ratio: float,
) -> Verdict:
    """Random-common-cause refutation verdict. HELD when
    |refuted-real|/|real| < `max_drift_ratio` (synthetic
    confounder leaves ATE near-stable). NaN/zero real → POW_INSUF."""
    real, refuted = r.real_ate, r.refuted_ate
    if math.isnan(real) or math.isnan(refuted) or abs(real) < 1e-9:
        return Verdict.POWER_INSUFFICIENT
    drift_ratio = abs(refuted - real) / abs(real)
    if drift_ratio < max_drift_ratio:
        return Verdict.HELD
    if drift_ratio < max_drift_ratio * 2:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


# ============ Partial-Spearman deciders ============


def partial_spearman_null_verdict(
    result: _PartialSpearmanResult,
    *,
    max_abs_rho: float,
    min_strata: int,
) -> Verdict:
    """Null-form partial-Spearman verdict (used when the bridge's
    predicted_direction is 'null'). HELD when |ρ| < `max_abs_rho`
    across ≥ `min_strata` strata."""
    if result.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = result.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if abs(rho) < max_abs_rho:
        return Verdict.HELD
    return Verdict.NO_EFFECT


def partial_spearman_signed_verdict(
    result: _PartialSpearmanResult,
    *,
    threshold: float,
    sign: Literal[-1, 1],
    min_strata: int,
) -> Verdict:
    """Signed-direction partial-Spearman verdict.
    sign=-1: HELD when ρ ≤ -threshold (predicted negative);
    sign=+1: HELD when ρ ≥ +threshold (predicted positive)."""
    if result.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = result.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if sign < 0 and rho <= -threshold:
        return Verdict.HELD
    if sign > 0 and rho >= threshold:
        return Verdict.HELD
    return Verdict.NO_EFFECT


class _StratumDiffLike(Protocol):
    """Read-only shape of `StratumDiff` — per-stratum Cohen's d
    panel returned by `stratified_arm_diff_pooled`."""
    @property
    def stratum_id(self) -> tuple[object, ...]: ...
    @property
    def cohen_d(self) -> float: ...
    @property
    def cohen_se(self) -> float: ...


def _verdict_from_pearson(
    r: float, p: float, *,
    sign: Literal[-1, 0, 1], threshold: float, alpha: float,
) -> Verdict:
    """Shared decision logic for Pearson-r-based bridge verdicts.

    `sign=+1`: HELD when r ≥ threshold AND p < alpha; sign-flipped
    significant → NO_EFFECT.
    `sign=-1`: mirror of +1.
    `sign=0`: HELD when |r| < threshold AND non-significant
    (null confirmed); significant slope refutes the null →
    NO_EFFECT."""
    if math.isnan(r):
        return Verdict.POWER_INSUFFICIENT
    if sign == 0:
        if abs(r) < threshold and p >= alpha:
            return Verdict.HELD
        if p < alpha:
            return Verdict.NO_EFFECT
        return Verdict.POWER_INSUFFICIENT
    if sign > 0:
        if r >= threshold and p < alpha:
            return Verdict.HELD
        if r <= -threshold and p < alpha:
            return Verdict.NO_EFFECT
        return Verdict.POWER_INSUFFICIENT
    if r <= -threshold and p < alpha:
        return Verdict.HELD
    if r >= threshold and p < alpha:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


def env_covariate_pearson_verdict(
    per_stratum: Sequence[_StratumDiffLike],
    env_covariate: Mapping[str, float],
    *,
    sign: Literal[-1, 0, 1],
    threshold: float,
    min_envs: int = 3,
    alpha: float = 0.05,
) -> Verdict:
    """Pearson r between per-env Cohen's d and env-level covariate
    (e.g. effective_horizon, argmax_entropy_late). Each per-stratum
    row contributes (env_covariate[env_name], cohen_d). Strata whose
    first stratum_id element isn't a known env_name (or whose
    cohen_d is NaN) are skipped. See `_verdict_from_pearson` for
    verdict semantics."""
    xs: list[float] = []
    ys: list[float] = []
    for s in per_stratum:
        env = s.stratum_id[0] if s.stratum_id else None
        if not isinstance(env, str):
            continue
        cov = env_covariate.get(env)
        if cov is None:
            continue
        if math.isnan(s.cohen_d):
            continue
        xs.append(cov)
        ys.append(s.cohen_d)
    if len(xs) < min_envs:
        return Verdict.POWER_INSUFFICIENT
    r_raw, p_raw = stats.pearsonr(xs, ys)
    return _verdict_from_pearson(
        float(r_raw), float(p_raw),
        sign=sign, threshold=threshold, alpha=alpha,
    )


class _MetaRegressionCoefficient(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def coefficient(self) -> float: ...
    @property
    def is_significant(self) -> bool: ...


class _MetaRegressionResultLike(Protocol):
    @property
    def n_strata(self) -> int: ...
    @property
    def coefficients(self) -> tuple[_MetaRegressionCoefficient, ...]: ...


def meta_regression_coefficient_verdict(
    result: _MetaRegressionResultLike,
    coef_name: str,
    *,
    sign: Literal[-1, 0, 1],
    threshold: float,
    min_strata: int = 3,
) -> Verdict:
    """Verdict on a named coefficient from a `MetaRegressionResult`.

    `sign=+1`: HELD when `coef ≥ threshold` AND significant;
    sign-flipped significant → NO_EFFECT.
    `sign=-1`: mirror of +1.
    `sign=0`: HELD when `|coef| < threshold` AND non-significant
    (null confirmed); significant slope → NO_EFFECT.

    POW_INSUF when n_strata < min_strata, coefficient missing /
    NaN, or signed prediction isn't significant in either
    direction."""
    if result.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    coef = next(
        (c for c in result.coefficients if c.name == coef_name),
        None,
    )
    if coef is None or math.isnan(coef.coefficient):
        return Verdict.POWER_INSUFFICIENT
    if sign == 0:
        if abs(coef.coefficient) < threshold and not coef.is_significant:
            return Verdict.HELD
        if coef.is_significant and abs(coef.coefficient) > threshold:
            return Verdict.NO_EFFECT
        return Verdict.POWER_INSUFFICIENT
    if sign > 0:
        if coef.coefficient >= threshold and coef.is_significant:
            return Verdict.HELD
        if coef.coefficient <= -threshold and coef.is_significant:
            return Verdict.NO_EFFECT
        return Verdict.POWER_INSUFFICIENT
    if coef.coefficient <= -threshold and coef.is_significant:
        return Verdict.HELD
    if coef.coefficient >= threshold and coef.is_significant:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


def stratum_id_scaling_verdict(
    per_stratum: Sequence[_StratumDiffLike],
    *,
    sign: Literal[-1, 0, 1],
    threshold: float,
    min_strata: int = 3,
    alpha: float = 0.05,
) -> Verdict:
    """Pearson r between per-stratum Cohen's d and the numeric
    `stratum_id[0]` value — for scaling tests where the
    `stratify_by` variable IS the scaling axis (γ, sync_period,
    action_duplicate_k, etc.). Sibling of
    `env_covariate_pearson_verdict`: that primitive looks up an
    env-name covariate; this one uses the stratum_id directly.

    Strata whose `stratum_id[0]` isn't numeric, or whose Cohen's
    d is NaN, are skipped. See `_verdict_from_pearson` for
    verdict semantics."""
    xs: list[float] = []
    ys: list[float] = []
    for s in per_stratum:
        if not s.stratum_id:
            continue
        x = s.stratum_id[0]
        if not isinstance(x, (int, float)):
            continue
        if math.isnan(s.cohen_d):
            continue
        xs.append(float(x))
        ys.append(s.cohen_d)
    if len(xs) < min_strata:
        return Verdict.POWER_INSUFFICIENT
    r_raw, p_raw = stats.pearsonr(xs, ys)
    return _verdict_from_pearson(
        float(r_raw), float(p_raw),
        sign=sign, threshold=threshold, alpha=alpha,
    )


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


class _SpearmanResult(Protocol):
    """Read-only shape of `CrossStratumPropertySlopeResult` and
    `CrossStratumArmDiffSlopeResult` — anything carrying a per-
    stratum Spearman ρ + p_value + n_strata."""
    @property
    def rho(self) -> float: ...
    @property
    def p_value(self) -> float: ...
    @property
    def n_strata(self) -> int: ...


def cross_stratum_signed_spearman_verdict(
    result: _SpearmanResult,
    *,
    sign: Literal[-1, 1],
    rho_threshold_held: float = 0.6,
    p_threshold: float = 0.05,
    null_threshold: float = 0.2,
    sign_flip_threshold: float = 0.5,
    min_strata: int = 10,
) -> tuple[Verdict, RefutationClass | None]:
    """Sign-aware Spearman verdict for cross-stratum slope claims.

    `sign=+1`: predicted positive (HELD when ρ ≥ +rho_threshold_held
    AND p ≤ p_threshold; SIGN_FLIP when ρ ≤ −sign_flip_threshold).
    `sign=-1`: predicted negative (mirrored).

    Verdict trichotomy:
      HELD                  : correct sign, |ρ| ≥ rho_threshold_held, p ≤ p_threshold
      NO_EFFECT (SIGN_FLIP) : wrong sign, |ρ| ≥ sign_flip_threshold (refutation by direction)
      NO_EFFECT (NULL_EFFECT) : |ρ| < null_threshold (clean null band, both directions)
      POWER_INSUFFICIENT    : in-between magnitudes, or n_strata < min_strata, or NaN

    Calibration. Two-sided critical |r| at p=0.05: n=10→0.648,
    n=8→0.707, n=6→0.829. Under H0, Spearman ρ has SD≈1/√(n−1) →
    ±0.45 at n=6, ±0.38 at n=8, ±0.33 at n=10. So:

    - `null_threshold=0.2` requires |ρ|<0.2 to fire NULL_EFFECT —
      below half the H0 SD at n=10. Calibrated for n_strata≥10;
      at smaller n the NULL band over-claims.
    - `rho_threshold_held=0.6` is decorative below n=10 — the
      p_threshold gate dominates (HELD requires |ρ|≥|r|_crit ≈ 0.65
      at n=10, ≥0.71 at n=8, ≥0.83 at n=6). Documented; do not
      "lower the held gate" — that would smuggle.
    - `min_strata=10` enforces both: below n=10, the verdict
      cannot resolve direction without overclaiming."""
    if result.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    rho = result.rho
    p = result.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if sign > 0:
        if rho >= rho_threshold_held and p <= p_threshold:
            return Verdict.HELD, None
        if rho <= -sign_flip_threshold:
            return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    else:
        if rho <= -rho_threshold_held and p <= p_threshold:
            return Verdict.HELD, None
        if rho >= sign_flip_threshold:
            return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(rho) < null_threshold:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None
