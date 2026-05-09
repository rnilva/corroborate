"""Partial Spearman ρ on paired-Δ cells.

Bridge-callable wrapper around `corroborate.graph.discovery.
partial_spearman_rho`. Tests whether `target`'s correlation with
`mediator` survives conditioning on `conditioning_source` after
the per-pair Δ projection.

Use this to corroborate **shadow-mediator** claims: when X is
algebraically derived from Z (e.g. `q_divergence_score = jens /
Bellman_bound`) or co-varies tightly with Z through training
dynamics, the marginal correlation X ⟂ Y looks substantive but
the partial X ⟂ Y | Z collapses to near-zero — that's the
HELD-as-null verdict the bridge consumes.

Companion to `proportion_mediated` (linear-mediation form): this
returns the rank-based partial-correlation form, less sensitive
to outcome-scale differences across cells / envs."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from corroborate.bridge.analysis import analysis
from corroborate.analyses.paired_g import resolve_value as _resolve_value
from corroborate.graph.discovery import partial_spearman_rho


@dataclass(frozen=True, slots=True)
class PartialSpearmanPairedResult:
    """`rho` is the partial Spearman correlation of Δ_target on
    Δ_mediator, conditioning on Δ_conditioning, after seed-pairing.
    `p_value` is two-sided against rho=0. `marginal_rho` is the
    unconditioned ρ(Δ_target, Δ_mediator), reported alongside so
    bridges can quote the shrinkage (`marginal → partial`).
    `n_pairs` is the count of seed-pairs contributing.

    Bridge interpretation:
    - Large |partial| with significant p → independent direct path.
    - Small |partial| with marginal that was large → shadow / fully
      mediated by the conditioning variable.
    - Bridges with `predicted_direction='null'` HELD when the
      partial is consistent with zero (small magnitude AND p >
      alpha)."""
    rho: float
    p_value: float
    marginal_rho: float
    n_pairs: int
    target: str
    mediator: str
    conditioning: str
    treatment_arm: str
    baseline_arm: str
    pair_by: tuple[str, ...]


def _key_tuple(
    record: Mapping[str, object], pair_by: tuple[str, ...],
) -> tuple[object, ...]:
    return tuple(record[k] for k in pair_by)


def _safe_resolve(record: Mapping[str, object], source: str) -> float:
    try:
        return _resolve_value(record, source)
    except (KeyError, TypeError):
        return float('nan')


@analysis
def partial_spearman_paired_delta(
    cells: Iterable[Mapping[str, object]],
    *,
    target: str,
    mediator: str,
    conditioning: str,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
) -> PartialSpearmanPairedResult:
    """Partial Spearman ρ(Δ_target, Δ_mediator | Δ_conditioning)
    on cells paired by `pair_by`.

    Pairs cells with `arm_field == treatment_arm` against
    `arm_field == baseline_arm`, computes Δ for each of `target`,
    `mediator`, `conditioning`, then calls the closed-form partial
    Spearman from `corroborate.graph.discovery`.

    `target`, `mediator`, `conditioning` resolve through the
    measurable registry first (record-first fallback — same
    convention as `paired_g.source`).

    Returns NaN result on empty pairs / degenerate variance.
    `marginal_rho` is the unconditioned ρ(Δ_target, Δ_mediator)
    reported alongside so bridges can quote shrinkage."""
    cells_list = list(cells)
    by_key_arm: dict[
        tuple[object, ...],
        dict[str, tuple[float, float, float]],
    ] = {}
    for cell in cells_list:
        arm = cell.get(arm_field)
        if not isinstance(arm, str) or arm not in (treatment_arm, baseline_arm):
            continue
        key = _key_tuple(cell, pair_by)
        t = _safe_resolve(cell, target)
        m = _safe_resolve(cell, mediator)
        c = _safe_resolve(cell, conditioning)
        if any(np.isnan(v) for v in (t, m, c)):
            continue
        by_key_arm.setdefault(key, {})[arm] = (t, m, c)

    deltas_t: list[float] = []
    deltas_m: list[float] = []
    deltas_c: list[float] = []
    for key, by_arm in by_key_arm.items():
        if treatment_arm not in by_arm or baseline_arm not in by_arm:
            continue
        t_t, t_m, t_c = by_arm[treatment_arm]
        b_t, b_m, b_c = by_arm[baseline_arm]
        deltas_t.append(t_t - b_t)
        deltas_m.append(t_m - b_m)
        deltas_c.append(t_c - b_c)

    n_pairs = len(deltas_t)
    if n_pairs < 5:
        return PartialSpearmanPairedResult(
            rho=float('nan'),
            p_value=float('nan'),
            marginal_rho=float('nan'),
            n_pairs=n_pairs,
            target=target,
            mediator=mediator,
            conditioning=conditioning,
            treatment_arm=treatment_arm,
            baseline_arm=baseline_arm,
            pair_by=pair_by,
        )

    t_arr = np.asarray(deltas_t, dtype=np.float64)
    m_arr = np.asarray(deltas_m, dtype=np.float64)
    c_arr = np.asarray(deltas_c, dtype=np.float64)

    rho, p = partial_spearman_rho(m_arr, t_arr, c_arr)

    # Marginal Spearman ρ(Δ_target, Δ_mediator) for shrinkage quote
    from corroborate.graph.discovery import _spearman_marginal
    marg_rho, _ = _spearman_marginal(m_arr, t_arr)

    return PartialSpearmanPairedResult(
        rho=rho,
        p_value=p,
        marginal_rho=marg_rho,
        n_pairs=n_pairs,
        target=target,
        mediator=mediator,
        conditioning=conditioning,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        pair_by=pair_by,
    )


__all__ = [
    'PartialSpearmanPairedResult',
    'partial_spearman_paired_delta',
]
