"""Statistics — paired-by-seed Hedges' g + MDE + verdict-decision tree.

Built on `scipy.stats` (distribution quantiles + cdf) and
`statsmodels.stats.power.TTestPower` (noncentral-t-aware MDE
calculations correct for small n). Hand-rolls only what the
libraries don't directly provide:

- **Hedges' g (paired)** — one-sample form on the Δ distribution
  with the Hedges 1981 small-sample correction `c_4 = 1 −
  3/(4n − 5)`. ~5 lines; no library has it directly without
  pulling pingouin (which drags matplotlib/seaborn/xarray —
  ~200 MiB for one function). v9-port.
- **Binary entropy `H_2(q)`** — 1 line; for the
  `delta_i_population = 1 − H_2(q)` information-gain reading.
- **Verdict-decision tree** — Popperian aggregation of
  (g, MDE, predicted_direction, adequately_powered) into the
  framework's `Verdict` + `RefutationClass`. Framework-specific,
  not statistical.

Power machinery uses `TTestPower.solve_power` — for n=10 paired
observations (the v0 sweep), the noncentral-t MDE diverges
meaningfully from the z-approximation; using the library is
correctness, not just convenience."""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import scipy.stats as ss
from statsmodels.stats.power import TTestPower

from corroborate.hypothesis import PredictedDirection
from corroborate.verdict import RefutationClass, Verdict


# ============ Effect-size + power primitives ============

def hedges_g_paired(
    deltas: Sequence[float],
) -> tuple[float, float]:
    """Hedges' g (one-sample form on paired Δ) + its SE.

    `deltas`: per-pair differences `treatment_i − baseline_i`.
    Returns `(g, se_g)` — both NaN if `n < 2` or `std(Δ) == 0`.

    Formula (Hedges 1981 small-sample correction; Borenstein 2009
    variance for one-sample g):

        d   = mean(Δ) / stdev(Δ)
        c_4 = 1 − 3 / (4n − 5)              # Hedges correction
        g   = d * c_4
        var = 1/n + g² / (2n)

    The c_4 factor reduces small-sample bias in d; for n=10 it
    shrinks |g| by ~3%. Don't drop it for honesty about n=10
    sample sizes."""
    n = len(deltas)
    if n < 2:
        return float('nan'), float('nan')
    s = statistics.stdev(deltas)
    if s == 0.0:
        return 0.0, float('nan')
    d = statistics.fmean(deltas) / s
    c4 = 1.0 - 3.0 / (4 * n - 5)
    g = d * c4
    var_g = 1.0 / n + g * g / (2 * n)
    return g, math.sqrt(var_g)


def mde_paired(
    n: int, *,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: Literal['larger', 'smaller', 'two-sided'] = 'larger',
) -> float:
    """Minimum detectable effect size for a one-sample paired
    t-test at the given α and power.

    Delegates to `statsmodels.stats.power.TTestPower.solve_power`.
    Noncentral-t aware — for small n (~10) the z-approximation
    `(z_α + z_β) * SE` underestimates by ~10-15%. The library
    handles this correctly."""
    if n < 2:
        return float('nan')
    return float(TTestPower().solve_power(
        effect_size=None, nobs=n, alpha=alpha, power=power,
        alternative=alternative,
    ))


def derived_q_from_g_se(g: float, se: float) -> float:
    """Φ(g / SE) — the probit of the standardized effect.

    Reads as 'the probability that the true effect is positive
    given the observed g + SE under a normal-approximation
    sampling distribution.' PAPER_NOTES.md axiom 19's information-
    gain formula expects a probability in (0, 1). NaN when g/se
    is undefined."""
    if math.isnan(g) or math.isnan(se) or se == 0.0:
        return float('nan')
    return float(ss.norm.cdf(g / se))


def delta_i_from_q(q: float) -> float:
    """Information gain from a Bernoulli at probability q:
    `1 − H_2(q)`, where `H_2(q) = −q log₂q − (1−q) log₂(1−q)`.

    q=0.5 (no signal) → ΔI=0. q→0 or q→1 (perfect signal) → ΔI=1.
    NaN propagates; returns 0.0 for q exactly at the boundary
    (treat boundary as no signal under the binary-entropy
    convention `0 log 0 = 0`)."""
    if math.isnan(q):
        return float('nan')
    if q <= 0.0 or q >= 1.0:
        return 0.0
    h = -q * math.log2(q) - (1 - q) * math.log2(1 - q)
    return 1.0 - h


def adequately_powered_paired(
    g: float, n: int, *,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: Literal['larger', 'smaller', 'two-sided'] = 'larger',
) -> bool:
    """True iff |g| ≥ MDE at the given α + power.

    Captures 'observed effect was at or above the smallest-
    detectable threshold for this n' — the v9-port criterion
    distinguishing POWER_INSUFFICIENT from NO_EFFECT."""
    if math.isnan(g) or n < 2:
        return False
    mde = mde_paired(n, alpha=alpha, power=power, alternative=alternative)
    if math.isnan(mde):
        return False
    return abs(g) >= mde


# ============ Verdict-decision tree (framework-specific) ============

def verdict_from_paired_stats(
    g: float, se: float, n: int,
    *,
    predicted_direction: PredictedDirection | None,
    alpha: float = 0.05,
    power: float = 0.8,
) -> tuple[Verdict, RefutationClass | None, bool]:
    """Popperian aggregation of (g, n, predicted_direction) into
    `(Verdict, RefutationClass | None, adequately_powered)`.

    Decision tree (v9-port, dialectic/hypothesis.py:305-309):

    1. `not adequately_powered` (|g| < MDE) → POWER_INSUFFICIENT,
       refutation_class=UNDERPOWERED.
    2. `adequately_powered AND |g| ≥ MDE`:
       a. `predicted_direction` is None: any sign is admissible
          (HELD when |g| ≥ MDE; the framework can't sign-check
          without an author commitment).
       b. `predicted_direction='a_gt_b'` (positive sign predicted):
          - `g > 0`: HELD.
          - `g < 0`: NO_EFFECT, refutation_class=SIGN_FLIP.
       c. `predicted_direction='a_lt_b'`: mirror of (b).
       d. `predicted_direction='two_sided'`: HELD on |g| ≥ MDE
          regardless of sign.

    Falls through to NO_EFFECT (refutation_class=NULL_EFFECT) if
    `adequately_powered` but |g| < MDE (shouldn't happen given the
    `adequately_powered` definition; defensive)."""
    if math.isnan(g) or n < 2:
        return (Verdict.POWER_INSUFFICIENT, RefutationClass.UNDERPOWERED, False)

    is_powered = adequately_powered_paired(g, n, alpha=alpha, power=power)
    if not is_powered:
        return (
            Verdict.POWER_INSUFFICIENT, RefutationClass.UNDERPOWERED, False,
        )

    # Adequately powered branch.
    if predicted_direction is None or predicted_direction == 'two_sided':
        return (Verdict.HELD, None, True)
    if predicted_direction == 'a_gt_b':
        if g > 0:
            return (Verdict.HELD, None, True)
        return (Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP, True)
    if predicted_direction == 'a_lt_b':
        if g < 0:
            return (Verdict.HELD, None, True)
        return (Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP, True)

    # Unreachable under the PredictedDirection Literal; defensive fallback.
    return (Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT, True)


# ============ Random-effects meta-analysis (DerSimonian-Laird) ============

@dataclass(frozen=True, slots=True)
class PooledStats:
    """Random-effects pooled effect across cells/envs.

    `pooled_g` — DerSimonian-Laird-pooled Hedges' g across cells.
    `se_pooled` — SE of the pooled estimate.
    `tau2` — between-cell heterogeneity variance (DerSimonian-
      Laird estimator).
    `I2` — fraction of total variance attributable to between-
      cell heterogeneity (Higgins).
    `Q` — Cochran's heterogeneity test statistic.
    `pi_lo`, `pi_hi` — 95% prediction interval (Higgins-Thompson-
      Spiegelhalter formula `pooled ± t_{n-2, 0.975} *
      sqrt(τ² + var_pooled)`). Empirical-min-g and empirical-
      max-g serve as honest backstops when the Gaussian-fit PI
      extrapolates past observed data.
    `empirical_min_g`, `empirical_max_g` — observed range.
    `n_cells` — number of valid cells used in pooling."""
    pooled_g: float
    se_pooled: float
    tau2: float
    I2: float
    Q: float
    pi_lo: float
    pi_hi: float
    empirical_min_g: float
    empirical_max_g: float
    n_cells: int


def random_effects_summary(
    g_se_pairs: Sequence[tuple[float, float]],
) -> PooledStats:
    """DerSimonian-Laird random-effects pool of (Hedges' g, SE)
    pairs across cells. Returns NaN-filled `PooledStats` when
    fewer than 2 valid cells are present (PI undefined).

    v9-port of
    `poc_v9/poc_v8/framework/reporting/aggregation.py:159-207` —
    same DL formula, same Higgins I², same HTS prediction
    interval.

    `g_se_pairs` — one (g, SE) per cell. NaN-bearing or
    zero-SE cells are filtered out (DL needs `var > 0`)."""
    valid: list[tuple[float, float]] = [
        (g, se) for g, se in g_se_pairs
        if not math.isnan(g) and not math.isnan(se) and se > 0.0
    ]
    n = len(valid)
    if n < 2:
        return PooledStats(
            pooled_g=float('nan'), se_pooled=float('nan'),
            tau2=float('nan'), I2=float('nan'), Q=float('nan'),
            pi_lo=float('nan'), pi_hi=float('nan'),
            empirical_min_g=float('nan'), empirical_max_g=float('nan'),
            n_cells=n,
        )
    gs = [g for g, _ in valid]
    vs = [se * se for _, se in valid]   # var = se²
    w_fixed = [1.0 / v for v in vs]
    sum_w = sum(w_fixed)
    g_fixed = sum(w * g for w, g in zip(w_fixed, gs)) / sum_w
    Q = sum(w * (g - g_fixed) ** 2 for w, g in zip(w_fixed, gs))
    df = n - 1
    sum_w_sq = sum(w * w for w in w_fixed)
    c_term = sum_w - sum_w_sq / sum_w
    tau2 = max(0.0, (Q - df) / c_term) if c_term > 0.0 else 0.0
    w_rand = [1.0 / (v + tau2) for v in vs]
    sum_w_rand = sum(w_rand)
    g_pooled = sum(w * g for w, g in zip(w_rand, gs)) / sum_w_rand
    var_pooled = 1.0 / sum_w_rand
    se_pooled = math.sqrt(var_pooled)
    I2 = max(0.0, (Q - df) / Q) if Q > 0.0 else 0.0
    t_crit = float(ss.t.ppf(0.975, df=df))
    pi_se = math.sqrt(tau2 + var_pooled)
    return PooledStats(
        pooled_g=g_pooled,
        se_pooled=se_pooled,
        tau2=tau2,
        I2=I2,
        Q=Q,
        pi_lo=g_pooled - t_crit * pi_se,
        pi_hi=g_pooled + t_crit * pi_se,
        empirical_min_g=min(gs),
        empirical_max_g=max(gs),
        n_cells=n,
    )


I2_THRESHOLD: float = 0.5
"""Random-effects I² above this threshold routes a corroboration
verdict (PI excludes zero in predicted direction) through
`HELD_WITH_SCOPE_FLAG` instead of plain `HELD`. v9's reframing
default; configurable per-study but holds the same role as
Higgins's "moderate-to-substantial" heterogeneity threshold."""


def _held_or_scope_flag(pooled: PooledStats) -> Verdict:
    """Return HELD_WITH_SCOPE_FLAG when I² ≥ threshold, else
    HELD. The held-conditions (PI excludes zero in predicted
    direction) are the caller's responsibility to verify before
    calling this helper."""
    if math.isnan(pooled.I2):
        return Verdict.HELD
    if pooled.I2 >= I2_THRESHOLD:
        return Verdict.HELD_WITH_SCOPE_FLAG
    return Verdict.HELD


def random_effects_verdict(
    pooled: PooledStats,
    *,
    predicted_direction: PredictedDirection | None,
) -> tuple[Verdict, RefutationClass | None]:
    """Apply Popperian aggregation to a random-effects pool.

    - n_cells < 3 → POWER_INSUFFICIENT (DL τ² estimation
      unreliable below 3 cells; PI fragile).
    - PI brackets zero → NO_EFFECT/NULL_EFFECT (population
      effect could be zero; not robustly directional).
    - PI excludes zero in predicted direction:
      - I² < threshold → HELD (uniform corroboration).
      - I² ≥ threshold → HELD_WITH_SCOPE_FLAG (corroborates at
        population level but heterogeneous across strata;
        meta-regression input).
    - PI strictly negative when predicted positive (or vice
      versa) → NO_EFFECT/SIGN_FLIP."""
    if math.isnan(pooled.pooled_g) or pooled.n_cells < 3:
        return (Verdict.POWER_INSUFFICIENT, RefutationClass.UNDERPOWERED)

    pi_excludes_zero = pooled.pi_lo > 0.0 or pooled.pi_hi < 0.0
    if not pi_excludes_zero:
        return (Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT)

    pi_positive = pooled.pi_lo > 0.0
    pi_negative = pooled.pi_hi < 0.0

    if predicted_direction is None or predicted_direction == 'two_sided':
        return (_held_or_scope_flag(pooled), None)
    if predicted_direction == 'a_gt_b':
        if pi_positive:
            return (_held_or_scope_flag(pooled), None)
        if pi_negative:
            return (Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP)
    if predicted_direction == 'a_lt_b':
        if pi_negative:
            return (_held_or_scope_flag(pooled), None)
        if pi_positive:
            return (Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP)

    return (Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT)


# ============ Power recommendation ============

def recommended_n_paired(
    observed_g: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: Literal['larger', 'smaller', 'two-sided'] = 'larger',
) -> float:
    """Given an observed effect size, return the n needed for
    adequately-powered detection at the given α and power.

    Wraps `statsmodels.stats.power.TTestPower.solve_power` —
    given (effect_size, alpha, power), it solves for nobs.

    Returns NaN for `observed_g == 0` (any n is insufficient to
    detect a true zero effect at non-trivial power)."""
    if math.isnan(observed_g) or observed_g == 0.0:
        return float('nan')
    return float(TTestPower().solve_power(
        effect_size=abs(observed_g),
        nobs=None, alpha=alpha, power=power,
        alternative=alternative,
    ))
