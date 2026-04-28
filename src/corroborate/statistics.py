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
from typing import Literal

import scipy.stats as ss
from statsmodels.stats.power import TTestPower

from corroborate.hypothesis import Direction
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
    predicted_direction: Direction | None,
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

    # Unreachable under the Direction Literal; defensive fallback.
    return (Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT, True)
