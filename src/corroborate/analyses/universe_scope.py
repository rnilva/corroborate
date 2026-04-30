"""`universe_scope` — scope-conditioned effect on a paired-delta
dataset.

The shape DDQN closure consumes: the universal paired-delta
cells (one row per `(corpus, env, hp_signature, seed)` with both
arms present and ARM-DIFFERENCE columns precomputed). Each cell
already encodes the treatment-baseline contrast as scalars
(`delta_outcome_best`, `delta_jensen_gap`, …); the analysis's job
is to (a) filter by a scope predicate expressed via column
≥/≤ thresholds, (b) report helped fraction + paired Hedges' g
on the in-scope subset.

Distinct from `paired_g_pooled` — that analysis takes UNPAIRED
cells and pairs them inside; here cells arrive already paired
and we just need scope-conditional pooling. Distinct from
`paired_g_among_solvers` — that one gates pairs by an env-keyed
threshold mapping; here the gate is per-cell column thresholds.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from corroborate.analysis import analysis
from corroborate.statistics import hedges_g_paired


@dataclass(frozen=True, slots=True)
class UniverseScopeResult:
    """Scope-conditioned effect on the universal paired-delta
    dataset. `n_total` is the corpus size before filtering;
    `n_in_scope` is after. `helped_fraction` is the fraction of
    in-scope cells where the treatment delta is positive. `g`
    is Hedges' g of the delta column on the in-scope subset."""
    outcome_col: str
    n_total: int
    n_in_scope: int
    helped_fraction: float
    g_outcome: float
    se_outcome: float
    g_jensen_gap: float
    se_jensen_gap: float
    filter_min_pairs: tuple[tuple[str, float], ...]
    filter_max_pairs: tuple[tuple[str, float], ...]
    filter_eq_pairs: tuple[tuple[str, str], ...]


def _passes(
    cell: Mapping[str, object],
    min_pairs: tuple[tuple[str, float], ...],
    max_pairs: tuple[tuple[str, float], ...],
    eq_pairs: tuple[tuple[str, str], ...],
) -> bool:
    for col, thr in min_pairs:
        v = cell.get(col)
        if not isinstance(v, (int, float)):
            return False
        f = float(v)
        if math.isnan(f) or f < thr:
            return False
    for col, thr in max_pairs:
        v = cell.get(col)
        if not isinstance(v, (int, float)):
            return False
        f = float(v)
        if math.isnan(f) or f > thr:
            return False
    for col, val in eq_pairs:
        v = cell.get(col)
        if v != val:
            return False
    return True


@analysis
def universe_scope(
    cells: Iterable[Mapping[str, object]],
    *,
    outcome_col: str = 'delta_outcome_best',
    delta_jensen_col: str = 'delta_jensen_gap',
    filter_min_pairs: tuple[tuple[str, float], ...] = (),
    filter_max_pairs: tuple[tuple[str, float], ...] = (),
    filter_eq_pairs: tuple[tuple[str, str], ...] = (),
) -> UniverseScopeResult:
    """Filter `cells` by `filter_min_pairs` (column ≥ threshold)
    and `filter_max_pairs` (column ≤ threshold); compute helped
    fraction + paired g on the in-scope subset.

    `outcome_col` and `delta_jensen_col` are the precomputed
    delta columns. Defaults match the universal-paired-delta
    dataset's schema (`delta_outcome_best`,
    `delta_jensen_gap`)."""
    cells_list = [dict(c) for c in cells]
    in_scope = [
        c for c in cells_list
        if _passes(c, filter_min_pairs, filter_max_pairs, filter_eq_pairs)
    ]

    deltas_outcome: list[float] = []
    deltas_jensen: list[float] = []
    helped = 0
    for c in in_scope:
        v = c.get(outcome_col)
        if isinstance(v, (int, float)) and not math.isnan(float(v)):
            deltas_outcome.append(float(v))
            if float(v) > 0.0:
                helped += 1
        j = c.get(delta_jensen_col)
        if isinstance(j, (int, float)) and not math.isnan(float(j)):
            deltas_jensen.append(float(j))

    n_in = len(deltas_outcome)
    helped_frac = (helped / n_in) if n_in > 0 else float('nan')
    g_out, se_out = (
        hedges_g_paired(deltas_outcome) if n_in >= 2
        else (float('nan'), float('nan'))
    )
    g_jen, se_jen = (
        hedges_g_paired(deltas_jensen) if len(deltas_jensen) >= 2
        else (float('nan'), float('nan'))
    )

    return UniverseScopeResult(
        outcome_col=outcome_col,
        n_total=len(cells_list),
        n_in_scope=n_in,
        helped_fraction=helped_frac,
        g_outcome=g_out, se_outcome=se_out,
        g_jensen_gap=g_jen, se_jensen_gap=se_jen,
        filter_min_pairs=filter_min_pairs,
        filter_max_pairs=filter_max_pairs,
        filter_eq_pairs=filter_eq_pairs,
    )


__all__ = ['UniverseScopeResult', 'universe_scope']
