"""`stratified_spearman` — per-cell JCI MARGINAL Spearman ρ(X, Y),
env-stratified, Fisher-z pooled.

Marginal sibling of `stratified_partial_spearman` (same module
shape, no conditioning variable). Use when the substantive
question is "does X correlate with Y within stratum?" without
needing to control for any third variable.

Wraps `corroborate.graph.discovery.stratified_spearman_rho`
under the @analysis decorator so bridges can consume it as a
fixture.

When to use this vs `stratified_partial_spearman`:
- Marginal Spearman (this): testing whether X and Y co-vary
  within stratum, no third-variable control needed.
- Partial Spearman: testing whether X→Y coupling SURVIVES
  conditioning on a third variable Z (mediator test, shadow
  test, residual-coupling test).

The two share the per-cell JCI-stratified Fisher-z-pool
discipline; the difference is whether to residualize on Z first.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from corroborate.analyses.paired_g import resolve_value
from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import stratified_spearman_rho


@dataclass(frozen=True, slots=True)
class StratifiedSpearmanResult:
    """JCI-stratified marginal Spearman ρ + Fisher-z-pooled p.

    `rho_pooled` is the tanh of the Fisher-z weighted average
    across strata; `p_value` is the two-sided test against ρ=0
    under the pooled z-statistic.

    Returns NaN ρ/p when no stratum reaches `min_stratum_size`.
    """
    x: str
    y: str
    stratify_by: str
    rho_pooled: float
    p_value: float
    n_obs_total: int
    n_strata: int


@analysis
def stratified_spearman(
    cells: Iterable[Mapping[str, object]],
    *,
    x: str,
    y: str,
    stratify_by: str = 'env_name',
    min_stratum_size: int = 4,
) -> StratifiedSpearmanResult:
    """JCI-stratified marginal Spearman ρ(X, Y).

    Each cell contributes one (x, y) observation. Strata with
    fewer than `min_stratum_size` complete pairs are dropped.
    Per-stratum ρ via `stats.spearmanr`; per-stratum Fisher-z
    values pooled by `(n_k − 3)` weight; returned ρ is tanh of
    pooled z. Two-sided p via pooled z-statistic.

    `x`, `y` resolve via `resolve_value` (registry-first;
    field-path fallback).
    """
    cells_list = list(cells)
    strata_keys: list[object] = []
    xs: list[float] = []
    ys: list[float] = []
    for cell in cells_list:
        try:
            xv = resolve_value(cell, x)
            yv = resolve_value(cell, y)
        except (KeyError, TypeError, ValueError):
            continue
        if xv != xv or yv != yv:  # NaN
            continue
        sk = cell.get(stratify_by)
        if sk is None:
            continue
        strata_keys.append(sk)
        xs.append(xv)
        ys.append(yv)

    if not xs:
        return StratifiedSpearmanResult(
            x=x, y=y, stratify_by=stratify_by,
            rho_pooled=float('nan'), p_value=float('nan'),
            n_obs_total=0, n_strata=0,
        )

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    rho, p = stratified_spearman_rho(
        x_arr, y_arr, strata_keys, min_stratum_size=min_stratum_size,
    )

    stratum_counts: dict[object, int] = {}
    for sk in strata_keys:
        stratum_counts[sk] = stratum_counts.get(sk, 0) + 1
    n_strata = sum(
        1 for c in stratum_counts.values() if c >= min_stratum_size
    )

    return StratifiedSpearmanResult(
        x=x, y=y, stratify_by=stratify_by,
        rho_pooled=float(rho), p_value=float(p),
        n_obs_total=len(xs), n_strata=n_strata,
    )
