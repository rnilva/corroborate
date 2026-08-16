"""Directional inference for paired experimental units under a frozen design.

Unlike an observed-effect/MDE gate, this analysis keeps design power and
observed evidence separate:

* a frozen ``DirectionalDesign`` describes the declared design;
* a paired t statistic and interval describe the observed mean contrast; and
* two noncentral-t equivalence tests ask whether the standardised effect is
  contained in the design's declared ``[-sesoi_dz, +sesoi_dz]`` region.

Whether the design was committed *before* the run set existed —
pre-registered — is provenance, not something this module can assert:
an external study records it by sealing a ``prospective_protocol`` in
its bundle, and the adapter receipt's ``protocol`` check reports the
register (verified-prospective vs admitted-retrospectively). A design
authored at analysis time is a retrospective declaration — equally
valid input, honestly labelled by the receipt that travels with the
panel.

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

from corroborate.analyses.paired.paired_g import PairedGResult, paired_g
from corroborate.bridge.analysis import analysis
from corroborate.bridge.verdict import RefutationClass, Verdict


type DirectionalAlternative = Literal['greater', 'less']


@dataclass(frozen=True, slots=True)
class DirectionalDesign:
    """One frozen directional design — the five knobs bound as a value.

    Bundling them makes the design something that can be authored
    once, passed whole, and echoed whole into the result — the
    analysis never re-derives design decisions from data.
    Construction validates the design, so an incoherent one fails
    closed before any cell is read.

    This value carries no claim about *when* it was authored.
    Pre-registration is provenance: a sealed
    ``prospective_protocol`` in an external bundle, reported by the
    adapter receipt's ``protocol`` check (see the module
    docstring). Declaring a design here and now — the exploratory
    register — is the ordinary case.
    """

    # The predicted sign of the contrast; committing it up front is what
    # makes the one-sided test honest.
    alternative: DirectionalAlternative = 'greater'
    # One-sided test level; also sets the (1 - 2*alpha) two-sided
    # confidence intervals and the TOST level.
    alpha: float = 0.05
    # Smallest standardised effect of scientific interest — the TOST
    # equivalence region is [-sesoi_dz, +sesoi_dz].
    sesoi_dz: float = 0.5
    # Declared admission gate on completed independent pairs; below
    # it the verdict is inconclusive no matter how large the effect.
    minimum_pairs: int = 2
    # The intended fixed design size, retained so the report shows how
    # far the achieved n fell short of the plan.
    planned_pairs: int = 2

    def __post_init__(self) -> None:
        if self.alternative not in ('greater', 'less'):
            raise ValueError(
                f'DirectionalDesign: unknown alternative '
                f'{self.alternative!r}',
            )
        if not 0.0 < self.alpha < 0.5:
            raise ValueError(
                'DirectionalDesign: alpha must be in (0, 0.5)',
            )
        if not math.isfinite(self.sesoi_dz) or self.sesoi_dz <= 0.0:
            raise ValueError(
                'DirectionalDesign: sesoi_dz must be positive',
            )
        if self.minimum_pairs < 2:
            raise ValueError(
                'DirectionalDesign: minimum_pairs must be >= 2',
            )
        if self.planned_pairs < self.minimum_pairs:
            raise ValueError(
                'DirectionalDesign: planned_pairs must be >= minimum_pairs',
            )


@dataclass(frozen=True, slots=True)
class PairedDirectionalResult:
    """Observed evidence for one contrast under a frozen design.

    ``mean_diff_ci`` and ``dz_ci`` are equal-tail two-sided intervals at
    confidence ``1 - 2 * design.alpha`` (90% at the default alpha=0.05)
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
    design: DirectionalDesign
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
        """TOST decision at the frozen design's alpha — derived, so it
        cannot drift from the stored equivalence p-values."""
        return (
            self.equivalence_p_lower <= self.design.alpha
            and self.equivalence_p_upper <= self.design.alpha
        )

    @property
    def design_complete(self) -> bool:
        """Whether the achieved independent-pair count reaches the gate."""
        return self.n_pairs >= self.design.minimum_pairs


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
    cells: Iterable[Mapping[str, object]],
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
    design: DirectionalDesign,
) -> PairedDirectionalResult:
    """Evaluate a paired directional contrast under a frozen design.

    ``design`` has no default: the design must be stated by the
    caller, even when it is just ``DirectionalDesign()`` — the
    verdict is only meaningful relative to a declared design.

    ``arm_field`` names the condition column; the default ``'arm_key'``
    (the fingerprint the runner stamps on each seeded run) covers every
    in-tree use — the parameter exists for run sets built outside the
    runner, matching the rest of the paired family.

    Duplicate ``(condition, pair_by)`` buckets always raise: the
    design fixes the experimental unit, so silently mean-aggregating
    duplicates (``paired_g``'s ``dedupe_strategy='mean'``) would be a
    post-hoc analytic choice. Tighten ``pair_by`` instead.

    The support test is the ordinary one-sided paired t-test for a mean
    contrast of zero. ``design.sesoi_dz`` is the smallest standardised
    effect of scientific interest used in design planning and in a
    TOST-style equivalence check. Equivalence uses the noncentral-t
    distribution at noncentralities ``±sesoi_dz * sqrt(n)``; it is not
    inferred from a non-significant support test.

    ``design.minimum_pairs`` is a declared admission threshold, not
    a function of the observed effect. A caller may set it to the
    power-derived minimum completed independent pairs and retain a
    larger ``planned_pairs`` for the intended fixed design.
    """
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
        if design.alternative == 'greater':
            support_p = float(t.sf(statistic, df=df))
        else:
            support_p = float(t.cdf(statistic, df=df))
        critical = float(t.ppf(1.0 - design.alpha, df=df))
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
            confidence=1.0 - 2.0 * design.alpha,
        )
        dz_ci = (ncp_ci[0] / root_n, ncp_ci[1] / root_n)
        lower_ncp = -design.sesoi_dz * root_n
        upper_ncp = design.sesoi_dz * root_n
        equivalence_p_lower = float(nct.sf(statistic, df, lower_ncp))
        equivalence_p_upper = float(nct.cdf(statistic, df, upper_ncp))
    else:
        statistic = support_p = dz = float('nan')
        equivalence_p_lower = equivalence_p_upper = float('nan')
        raw_ci = dz_ci = _nan_interval()

    violations = list(paired.assumption_violations)
    if n < design.minimum_pairs:
        violations.append(
            f'design_incomplete '
            f'(n_pairs={n} < minimum_pairs={design.minimum_pairs})',
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
        design=design,
        assumption_violations=tuple(violations),
    )


def paired_directional_verdict(
    result: PairedDirectionalResult,
) -> tuple[Verdict, RefutationClass | None]:
    """Map directional evidence under a frozen design to a verdict.

    Priority is scientifically conservative:

    1. an incomplete/degenerate design is inconclusive;
    2. a significant opposite-direction effect is a sign-flip refutation;
    3. TOST evidence that the effect lies within the SESOI region is a
       practical-null refutation;
    4. a significant effect in the predicted direction corroborates; and
    5. failure to establish any of those remains inconclusive.

    Therefore a non-significant p-value is never relabelled ``NO_EFFECT``.
    """
    alpha = result.design.alpha
    if not result.design_complete or not math.isfinite(result.p_value):
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
    'DirectionalDesign',
    'PairedDirectionalResult',
    'paired_directional',
    'paired_directional_verdict',
]
