"""Directional inference for paired experimental units.

Unlike an observed-effect/MDE gate, this analysis keeps design power and
observed evidence separate:

* the bridge supplies its predicted direction and test configuration;
* a paired t statistic and interval describe the observed mean contrast; and
* two noncentral-t equivalence tests ask whether the standardised effect is
  contained in the declared ``[-sesoi_dz, +sesoi_dz]`` region.

The analysis makes no temporal claim about when those choices were authored.
For bridge evaluation, ``predicted_direction`` arrives from the bridge's
structural metadata; ``alpha``, ``sesoi_dz``, and ``minimum_pairs`` are normal
analysis parameters declared as defaults in the claim-test function.

The experimental unit is one matched key in ``pair_by``. Repeated episodes,
environments, transitions, or checkpoints within a run must be reduced before
this analysis; they do not increase ``n_pairs``.

As a member of the paired family this inherits the seed-pairing
restriction: off-limits in RL substrate bridges that pool across strata
(seed-pseudo-replication); valid for single-stratum designs and
synthetic SCM tests.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from scipy.optimize import brentq
from scipy.stats import nct, t

import polars as pl

from corroborate._internals.polars import as_rows
from corroborate.analyses.paired.paired_g import PairedGResult, paired_g
from corroborate.bridge.analysis import analysis
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.core.hypothesis import PredictedDirection


type DirectionalAlternative = Literal['a_gt_b', 'a_lt_b']


def _validate_test_configuration(
    predicted_direction: PredictedDirection,
    alpha: float,
    sesoi_dz: float,
    minimum_pairs: int,
) -> DirectionalAlternative:
    """Fail closed on analysis choices before reading any cells."""
    if predicted_direction == 'a_gt_b':
        direction: DirectionalAlternative = 'a_gt_b'
    elif predicted_direction == 'a_lt_b':
        direction = 'a_lt_b'
    else:
        raise ValueError(
            'paired_directional: predicted_direction must be '
            "'a_gt_b' or 'a_lt_b'",
        )
    if not 0.0 < alpha < 0.5:
        raise ValueError('paired_directional: alpha must be in (0, 0.5)')
    if not math.isfinite(sesoi_dz) or sesoi_dz <= 0.0:
        raise ValueError('paired_directional: sesoi_dz must be positive')
    if minimum_pairs < 2:
        raise ValueError(
            'paired_directional: minimum_pairs must be >= 2',
        )
    return direction


@dataclass(frozen=True, slots=True)
class PairedDirectionalResult:
    """Observed evidence for one configured directional contrast.

    ``mean_diff_ci`` and ``dz_ci`` are equal-tail two-sided intervals at
    confidence ``1 - 2 * alpha`` (90% at the default alpha=0.05)
    — the two-sided equivalent of the one-sided test, so the interval
    bound facing the predicted direction aligns with the test decision.
    """

    mean_diff: float
    mean_diff_se: float
    mean_diff_ci: tuple[float, float]
    dz: float
    dz_ci: tuple[float, float]
    hedges_g: float
    t_statistic: float
    p_value: float
    equivalence_p_lower: float
    equivalence_p_upper: float
    n_pairs: int
    n_treatment: int
    n_baseline: int
    pair_by: tuple[str, ...]
    measurable: str
    treatment_arm: str
    baseline_arm: str
    predicted_direction: DirectionalAlternative
    alpha: float
    sesoi_dz: float
    minimum_pairs: int
    assumption_violations: tuple[str, ...] = ()

    @property
    def degrees_of_freedom(self) -> int:
        """Paired t df; derived so the stored surface stays minimal."""
        return self.n_pairs - 1

    @property
    def opposite_p_value(self) -> float:
        """One-sided p against the predicted direction; the t CDF is
        continuous, so it is exactly the complement of ``p_value``."""
        return 1.0 - self.p_value

    @property
    def practically_equivalent(self) -> bool:
        """TOST decision at the configured alpha — derived, so it
        cannot drift from the stored equivalence p-values."""
        return (
            self.equivalence_p_lower <= self.alpha
            and self.equivalence_p_upper <= self.alpha
        )

    @property
    def minimum_pairs_met(self) -> bool:
        """Whether the achieved independent-pair count reaches the gate."""
        return self.n_pairs >= self.minimum_pairs


def _nan_interval() -> tuple[float, float]:
    return (float('nan'), float('nan'))


def _ncp_confidence_interval(
    statistic: float,
    degrees_of_freedom: int,
    *,
    confidence: float,
) -> tuple[float, float]:
    """Invert the noncentral-t CDF for an interval on its NCP.

    For observed statistic ``t_obs`` and noncentrality ``lambda``,
    ``F_nct(t_obs; lambda)`` is monotone decreasing in ``lambda``.
    The equal-tail interval solves ``F=1-alpha`` and ``F=alpha``.
    """
    if not math.isfinite(statistic) or degrees_of_freedom < 1:
        return _nan_interval()
    tail = (1.0 - confidence) / 2.0

    def solve(target: float) -> float:
        def objective(noncentrality: float) -> float:
            return float(nct.cdf(
                statistic,
                degrees_of_freedom,
                noncentrality,
            ) - target)

        lower = -1.0
        upper = 1.0
        while objective(lower) < 0.0 and abs(lower) < 1e6:
            lower *= 2.0
        while objective(upper) > 0.0 and abs(upper) < 1e6:
            upper *= 2.0
        if objective(lower) < 0.0 or objective(upper) > 0.0:
            return float('nan')
        return cast(float, brentq(objective, lower, upper, maxiter=200))

    return (solve(1.0 - tail), solve(tail))


@analysis
def paired_directional(
    cells: pl.DataFrame | Iterable[Mapping[str, object]],
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
    predicted_direction: PredictedDirection,
    alpha: float = 0.05,
    sesoi_dz: float = 0.5,
    minimum_pairs: int = 2,
) -> PairedDirectionalResult:
    """Evaluate a configured paired directional contrast.

    ``predicted_direction`` is required and accepts only ``a_gt_b``
    or ``a_lt_b``. During bridge evaluation it is injected from the
    bridge's structural metadata, avoiding a second direction setting
    in the analysis configuration.

    ``arm_field`` names the condition column; the default ``'arm_key'``
    (the fingerprint the runner stamps on each seeded run) covers every
    in-tree use — the parameter exists for run sets built outside the
    runner, matching the rest of the paired family.

    Duplicate ``(condition, pair_by)`` buckets always raise: the
    design fixes the experimental unit, so silently mean-aggregating
    duplicates (``paired_g``'s ``dedupe_strategy='mean'``) would be a
    post-hoc analytic choice. Tighten ``pair_by`` instead.

    The support test is the ordinary one-sided paired t-test for a mean
    contrast of zero. ``sesoi_dz`` is the smallest standardised
    effect of scientific interest used in design planning and in a
    TOST-style equivalence check. Equivalence uses the noncentral-t
    distribution at noncentralities ``±sesoi_dz * sqrt(n)``; it is not
    inferred from a non-significant support test.

    ``minimum_pairs`` is a declared admission threshold, not a
    function of the observed effect. A caller may set it to the
    power-derived minimum completed independent pairs.
    """
    cells = as_rows(cells)
    direction = _validate_test_configuration(
        predicted_direction, alpha, sesoi_dz, minimum_pairs,
    )
    paired: PairedGResult = paired_g.fn(
        cells,
        source=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        pair_by=pair_by,
        arm_field=arm_field,
        dedupe_strategy='raise',
    )
    n = paired.n_pairs
    df = n - 1
    valid = (
        n >= 2
        and math.isfinite(paired.mean_diff)
        and math.isfinite(paired.mean_diff_se)
        and paired.mean_diff_se > 0.0
    )
    if valid:
        statistic = paired.mean_diff / paired.mean_diff_se
        if direction == 'a_gt_b':
            support_p = float(t.sf(statistic, df=df))
        else:
            support_p = float(t.cdf(statistic, df=df))
        critical = float(t.ppf(1.0 - alpha, df=df))
        raw_ci = (
            paired.mean_diff - critical * paired.mean_diff_se,
            paired.mean_diff + critical * paired.mean_diff_se,
        )
        # mean_diff_se = s_delta / sqrt(n), hence dz = t / sqrt(n).
        root_n = math.sqrt(n)
        dz = statistic / root_n
        ncp_ci = _ncp_confidence_interval(
            statistic,
            df,
            confidence=1.0 - 2.0 * alpha,
        )
        dz_ci = (ncp_ci[0] / root_n, ncp_ci[1] / root_n)
        lower_ncp = -sesoi_dz * root_n
        upper_ncp = sesoi_dz * root_n
        equivalence_p_lower = float(nct.sf(statistic, df, lower_ncp))
        equivalence_p_upper = float(nct.cdf(statistic, df, upper_ncp))
    else:
        statistic = support_p = dz = float('nan')
        equivalence_p_lower = equivalence_p_upper = float('nan')
        raw_ci = dz_ci = _nan_interval()

    violations = list(paired.assumption_violations)
    if n < minimum_pairs:
        violations.append(
            f'minimum_pairs_not_met '
            f'(n_pairs={n} < minimum_pairs={minimum_pairs})',
        )
    return PairedDirectionalResult(
        mean_diff=paired.mean_diff,
        mean_diff_se=paired.mean_diff_se,
        mean_diff_ci=raw_ci,
        dz=dz,
        dz_ci=dz_ci,
        hedges_g=paired.g,
        t_statistic=statistic,
        p_value=support_p,
        equivalence_p_lower=equivalence_p_lower,
        equivalence_p_upper=equivalence_p_upper,
        n_pairs=n,
        n_treatment=paired.n_treatment,
        n_baseline=paired.n_baseline,
        pair_by=pair_by,
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        predicted_direction=direction,
        alpha=alpha,
        sesoi_dz=sesoi_dz,
        minimum_pairs=minimum_pairs,
        assumption_violations=tuple(violations),
    )


def paired_directional_verdict(
    result: PairedDirectionalResult,
) -> tuple[Verdict, RefutationClass | None]:
    """Map configured directional evidence to a verdict.

    Priority is scientifically conservative:

    1. an incomplete/degenerate design is inconclusive;
    2. a significant opposite-direction effect is a sign-flip refutation;
    3. TOST evidence that the effect lies within the SESOI region is a
       practical-null refutation;
    4. a significant effect in the predicted direction corroborates; and
    5. failure to establish any of those remains inconclusive.

    Therefore a non-significant p-value is never relabelled ``NO_EFFECT``.
    """
    alpha = result.alpha
    if not result.minimum_pairs_met or not math.isfinite(result.p_value):
        return Verdict.POWER_INSUFFICIENT, RefutationClass.UNDERPOWERED
    if result.opposite_p_value <= alpha:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if result.practically_equivalent:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    if result.p_value <= alpha:
        return Verdict.HELD, None
    return Verdict.POWER_INSUFFICIENT, RefutationClass.UNDERPOWERED


__all__ = [
    'DirectionalAlternative',
    'PairedDirectionalResult',
    'paired_directional',
    'paired_directional_verdict',
]
