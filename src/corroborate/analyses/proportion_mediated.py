"""`proportion_mediated` — linear-mediation decomposition for paired
intervention claims.

For a (treatment, baseline) contrast and a candidate `mediator`,
this analysis decomposes the per-pair Δ on `target` into
direct + indirect components and reports the proportion that's
*mediated* by the candidate. The standard Baron-Kenny / Sobel
form, adapted to the framework's per-pair-Δ idiom:

    Δ_Y_pair  = direct + indirect
    indirect  = β_YM · Δ_M_pair    (the part of Δ_Y "carried by" Δ_M)
    direct    = Δ_Y_pair − indirect

where β_YM is the slope of `target` on `mediator` fit across all
pairs (the linear-mediation assumption: M's effect on Y is a
single linear coefficient that doesn't depend on treatment level
or other covariates).

Aggregating to the population:

    proportion_mediated = β_YM · mean(Δ_M) / mean(Δ_Y)

The framework's existing `partial_spearman_rho` family expresses
the Spearman-rank-correlation form of the same decomposition; this
analysis returns the *raw* point estimate so bridges can author
threshold logic like `proportion_mediated > 0.5 → HELD` directly.

**Linear-mediation assumptions** (per ANALYSIS_RECIPE.md §3a):

1. No treatment × mediator interaction (single β_YM works for all
   cells).
2. Linear M → Y functional form (no saturation, threshold).
3. Mediator's distribution doesn't depend on treatment in
   nonlinear ways.

When these break, the linear proportion can land outside [0, 1] —
the diagnostic is captured on the result via `in_unit_interval`.
A bridge that gets `in_unit_interval=False` should escalate to
counterfactual mediation (Pearl NDE/NIE; deferred per
FUTURE_WORKS.md).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class ProportionMediatedResult:
    """Linear-mediation decomposition.

    `proportion`: the (indirect / total) share — the fraction of
    the treatment's effect on `target` that flows through
    `mediator`. NaN when total effect is too small to estimate
    against (|mean(Δ_Y)| < 1e-12), or when n_pairs < 3.

    `total`, `direct`, `indirect`: the underlying point estimates
    in the same units as `target`. `total = direct + indirect`
    by construction.

    `slope_y_on_m`: the β_YM coefficient (slope of Δ_Y on Δ_M).

    `in_unit_interval`: True iff `proportion ∈ [0, 1]`. False
    indicates linear-mediation assumptions have failed —
    interpret with caution, escalate to counterfactual
    decomposition.

    `n_pairs`: number of paired observations contributing.
    `target`, `mediator`, `treatment_arm`, `baseline_arm`,
    `pair_by`: provenance of the call."""
    proportion: float
    total: float
    direct: float
    indirect: float
    slope_y_on_m: float
    in_unit_interval: bool
    n_pairs: int
    target: str
    mediator: str
    treatment_arm: str
    baseline_arm: str
    pair_by: tuple[str, ...]


def _key_tuple(
    record: Mapping[str, object], pair_by: tuple[str, ...],
) -> tuple[object, ...]:
    return tuple(record[k] for k in pair_by)


def _resolve_value(record: Mapping[str, object], source: str) -> float:
    """Same resolution rule as `paired_g._resolve_value`: registry
    lookup first, then field-path on the record."""
    from corroborate.measurables import get_registered
    raw = record.get(source)
    if raw is not None:
        if isinstance(raw, bool):
            return float('nan')  # bool is not a numeric scalar
        if isinstance(raw, (int, float)):
            return float(raw)
        return float('nan')
    m = get_registered(source)
    if m is not None:
        computed = m.fn(record)
        if isinstance(computed, bool):
            return float('nan')
        if isinstance(computed, (int, float)):
            return float(computed)
    return float('nan')


@analysis
def proportion_mediated(
    cells: Iterable[Mapping[str, object]],
    *,
    target: str,
    mediator: str,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
) -> ProportionMediatedResult:
    """Linear proportion of the treatment effect on `target` that
    is mediated by `mediator`.

    Pairs cells on `pair_by` (typically `('seed',)`), computes
    per-pair `Δ_target` and `Δ_mediator`, fits the OLS slope of
    Δ_Y on Δ_M, then reports

        proportion = β_YM · mean(Δ_M) / mean(Δ_Y).

    `target` and `mediator` resolve through the measurable
    registry first, falling back to field-path reads on the cell
    record (same convention as `paired_g.source`).

    The result's `in_unit_interval` flag is the diagnostic for
    linear-mediation assumption failure — see ANALYSIS_RECIPE.md
    §3a for when to escalate to counterfactual mediation."""
    treatment_y: dict[tuple[object, ...], float] = {}
    treatment_m: dict[tuple[object, ...], float] = {}
    baseline_y: dict[tuple[object, ...], float] = {}
    baseline_m: dict[tuple[object, ...], float] = {}
    for cell in cells:
        arm = cell.get(arm_field)
        if arm not in (treatment_arm, baseline_arm):
            continue
        key = _key_tuple(cell, pair_by)
        y = _resolve_value(cell, target)
        m = _resolve_value(cell, mediator)
        if math.isnan(y) or math.isnan(m):
            continue
        if arm == treatment_arm:
            treatment_y[key] = y
            treatment_m[key] = m
        else:
            baseline_y[key] = y
            baseline_m[key] = m

    paired_keys = sorted(set(treatment_y) & set(baseline_y))
    n_pairs = len(paired_keys)
    if n_pairs < 3:
        return _nan_result(
            target=target, mediator=mediator,
            treatment_arm=treatment_arm, baseline_arm=baseline_arm,
            pair_by=pair_by, n_pairs=n_pairs,
        )

    delta_y = [treatment_y[k] - baseline_y[k] for k in paired_keys]
    delta_m = [treatment_m[k] - baseline_m[k] for k in paired_keys]

    # OLS slope of Δ_Y on Δ_M with intercept.
    n = float(n_pairs)
    mean_y = sum(delta_y) / n
    mean_m = sum(delta_m) / n
    cov_ym = sum(
        (y - mean_y) * (m - mean_m) for y, m in zip(delta_y, delta_m)
    )
    var_m = sum((m - mean_m) ** 2 for m in delta_m)
    if var_m < 1e-18:
        return _nan_result(
            target=target, mediator=mediator,
            treatment_arm=treatment_arm, baseline_arm=baseline_arm,
            pair_by=pair_by, n_pairs=n_pairs,
        )
    slope = cov_ym / var_m

    indirect = slope * mean_m
    direct = mean_y - indirect
    if abs(mean_y) < 1e-12:
        return ProportionMediatedResult(
            proportion=float('nan'),
            total=mean_y, direct=direct, indirect=indirect,
            slope_y_on_m=slope, in_unit_interval=False,
            n_pairs=n_pairs,
            target=target, mediator=mediator,
            treatment_arm=treatment_arm, baseline_arm=baseline_arm,
            pair_by=pair_by,
        )
    proportion = indirect / mean_y
    in_unit_interval = 0.0 <= proportion <= 1.0
    return ProportionMediatedResult(
        proportion=proportion,
        total=mean_y, direct=direct, indirect=indirect,
        slope_y_on_m=slope,
        in_unit_interval=in_unit_interval,
        n_pairs=n_pairs,
        target=target, mediator=mediator,
        treatment_arm=treatment_arm, baseline_arm=baseline_arm,
        pair_by=pair_by,
    )


def _nan_result(
    *, target: str, mediator: str,
    treatment_arm: str, baseline_arm: str,
    pair_by: tuple[str, ...], n_pairs: int,
) -> ProportionMediatedResult:
    return ProportionMediatedResult(
        proportion=float('nan'),
        total=float('nan'), direct=float('nan'), indirect=float('nan'),
        slope_y_on_m=float('nan'),
        in_unit_interval=False,
        n_pairs=n_pairs,
        target=target, mediator=mediator,
        treatment_arm=treatment_arm, baseline_arm=baseline_arm,
        pair_by=pair_by,
    )


__all__ = ['ProportionMediatedResult', 'proportion_mediated']
