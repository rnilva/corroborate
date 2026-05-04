"""Stratified Spearman ρ between binary arm assignment and a
scalar outcome — the M2M-friendly companion to `paired_g`.

Where `paired_g` requires one cell per `(arm, pair_by)` (raises
or aggregates duplicates), `paired_arm_spearman` consumes ALL
cells. Each cell contributes one observation `(arm_indicator,
outcome)`; observations are stratified by `stratify_by` (e.g.
`('seed',)`) and pooled via Fisher z (`stratified_spearman_rho`
in `corroborate.causal_discovery`).

Within a single stratum the indicator has only two values, so the
per-stratum ρ is point-biserial-equivalent (sign of "treatment
mean − baseline mean" within that stratum). Fisher-z pooling
gives a rank-based test of whether the treatment systematically
out-ranks or under-ranks the baseline across strata, robust to
scale variation across corpora.

Used in bridges that previously consumed `paired_g` but where the
scope captured multiple cells per `(arm, pair_by)` — the M2M case
the deduped paired-g handles by averaging, this analysis handles
by ranking."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import stratified_spearman_rho


@dataclass(frozen=True, slots=True)
class PairedArmSpearmanResult:
    """`rho` is the Fisher-z-pooled Spearman correlation between
    the binary arm indicator (1 if treatment, 0 if baseline) and
    the per-cell `source` value, stratified by `stratify_by`.
    `p_value` is the two-sided test against rho=0. `n_obs` is the
    total number of cells (treatment + baseline). `n_strata` is
    the number of strata that contributed (i.e. had at least one
    cell from each arm — `min_stratum_size=4`)."""
    rho: float
    p_value: float
    n_obs: int
    n_strata: int
    treatment_arm: str
    baseline_arm: str
    source: str
    stratify_by: tuple[str, ...]


@analysis
def paired_arm_spearman(
    cells: Iterable[Mapping[str, object]],
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    stratify_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
) -> PairedArmSpearmanResult:
    """Compute stratified Spearman ρ(arm, source). See module
    docstring for semantics."""
    # Inlined: resolve a cell's `source` to a float, going through
    # the @measurable resolver if needed. Mirrors paired_g's
    # `_resolve_value`; private-cross-module imports break pyright,
    # so we duplicate the 6-line shape here.
    from corroborate.measurables import (
        evaluate_with_measurables, get_registered,
    )

    def _resolve_value(cell: Mapping[str, object], name: str) -> float:
        v = cell.get(name)
        if v is None:
            m = get_registered(name)
            if m is None:
                return float('nan')
            try:
                v = evaluate_with_measurables(m.fn, cell, cache={})
            except Exception:  # noqa: BLE001
                return float('nan')
        return float(v) if isinstance(v, (int, float)) else float('nan')

    arm_indicators: list[float] = []
    values: list[float] = []
    # `stratified_spearman_rho` does `np.asarray(strata, dtype=object)`
    # which mangles list-of-1-tuples into a 2-D array. Pack
    # multi-key strata as `'__'`-joined strings so the array stays
    # 1-D regardless of `stratify_by` arity.
    strata: list[str] = []
    for cell in cells:
        arm = cell.get(arm_field)
        if arm == treatment_arm:
            indicator = 1.0
        elif arm == baseline_arm:
            indicator = 0.0
        else:
            continue
        v = _resolve_value(cell, source)
        if math.isnan(v):
            continue
        try:
            stratum_key = '__'.join(str(cell[k]) for k in stratify_by)
        except KeyError:
            continue
        arm_indicators.append(indicator)
        values.append(v)
        strata.append(stratum_key)
    if not arm_indicators:
        return PairedArmSpearmanResult(
            rho=float('nan'), p_value=float('nan'),
            n_obs=0, n_strata=0,
            treatment_arm=treatment_arm, baseline_arm=baseline_arm,
            source=source, stratify_by=stratify_by,
        )
    x = np.asarray(arm_indicators, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    rho, p = stratified_spearman_rho(x, y, strata)
    # Count strata that have at least one of each arm.
    by_stratum: dict[str, set[float]] = {}
    for s, ind in zip(strata, arm_indicators):
        by_stratum.setdefault(s, set()).add(ind)
    distinct_strata = sum(1 for arms in by_stratum.values() if len(arms) >= 2)
    return PairedArmSpearmanResult(
        rho=rho, p_value=p,
        n_obs=len(arm_indicators), n_strata=distinct_strata,
        treatment_arm=treatment_arm, baseline_arm=baseline_arm,
        source=source, stratify_by=stratify_by,
    )


__all__ = [
    'PairedArmSpearmanResult',
    'paired_arm_spearman',
]
