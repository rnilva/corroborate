"""`stratified_partial_spearman_multi` — per-cell JCI partial
Spearman ρ(X, Y | Z₁, ..., Zₖ), env-stratified, Fisher-z pooled.

Multi-Z generalization of `stratified_partial_spearman`. Used
when a bridge tests whether MULTIPLE candidate mediators jointly
explain the X→Y coupling, e.g., "does conditioning on
{self_ref, q_late} together collapse γ→jens?"

Each cell contributes one observation (x, y, z₁, …, zₖ).
Per-stratum partial Spearman ρ is computed via OLS-residual
regression on rank-transformed variables
(`graph.discovery.partial_spearman_rho_multi`); strata are
pooled via Fisher z with weights `(n_k − 3 − k)` (df accounting
for the k conditioning variables).

Distinct from `stratified_partial_spearman` (single Z) — when
the residual partial ρ after one conditioning variable is
significant, the natural follow-up is "does conditioning on TWO
mediators jointly collapse it?" This primitive answers that
question.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from corroborate.analyses.paired_g import resolve_value
from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import stratified_partial_spearman_rho_multi


@dataclass(frozen=True, slots=True)
class StratifiedPartialSpearmanMultiResult:
    """JCI-stratified multi-Z partial Spearman ρ + Fisher-z-pooled p.

    `conditioning` records the tuple of conditioning column names
    in the order they were passed (matters for diagnostic /
    snapshot stability). `rho_pooled` is the tanh of the Fisher-z
    weighted average across strata; `p_value` is the two-sided
    test against `rho=0` under the pooled z-statistic.

    Returns NaN ρ/p when no stratum reaches `min_stratum_size` or
    has sufficient df after accounting for k conditioning vars.
    """
    x: str
    y: str
    conditioning: tuple[str, ...]
    stratify_by: str
    rho_pooled: float
    p_value: float
    n_obs_total: int
    n_strata: int


@analysis
def stratified_partial_spearman_multi(
    cells: Iterable[Mapping[str, object]],
    *,
    x: str,
    y: str,
    conditioning: tuple[str, ...],
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
) -> StratifiedPartialSpearmanMultiResult:
    """JCI-form multi-Z partial Spearman ρ(X, Y | Z₁, …, Zₖ),
    stratified by `stratify_by`, pooled via Fisher z.

    Each cell contributes one observation `(x, y, z₁, …, zₖ)`.
    Strata with fewer than `min_stratum_size` complete
    observations are dropped; strata where `n_k ≤ 3 + k` (df
    insufficient) are also dropped.

    `x`, `y`, and each entry of `conditioning` resolve via
    `resolve_value` (registry-first; field-path fallback).
    """
    if not conditioning:
        raise ValueError(
            'stratified_partial_spearman_multi: conditioning must be '
            'a non-empty tuple of column names; use '
            '`stratified_partial_spearman` for single-Z case.',
        )
    cells_list = list(cells)
    strata_keys: list[object] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[tuple[float, ...]] = []
    for cell in cells_list:
        try:
            xv = resolve_value(cell, x)
            yv = resolve_value(cell, y)
            z_vals = tuple(resolve_value(cell, z) for z in conditioning)
        except (KeyError, TypeError, ValueError):
            continue
        if xv != xv or yv != yv:
            continue
        if any(zv != zv for zv in z_vals):
            continue
        sk = cell.get(stratify_by)
        if sk is None:
            continue
        strata_keys.append(sk)
        xs.append(xv)
        ys.append(yv)
        zs.append(z_vals)

    if not xs:
        return StratifiedPartialSpearmanMultiResult(
            x=x, y=y, conditioning=conditioning,
            stratify_by=stratify_by,
            rho_pooled=float('nan'), p_value=float('nan'),
            n_obs_total=0, n_strata=0,
        )

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    z_arr = np.asarray(zs, dtype=np.float64)
    rho, p = stratified_partial_spearman_rho_multi(
        x_arr, y_arr, z_arr, strata_keys,
        min_stratum_size=min_stratum_size,
    )

    stratum_counts: dict[object, int] = {}
    for sk in strata_keys:
        stratum_counts[sk] = stratum_counts.get(sk, 0) + 1
    n_strata = sum(
        1 for c in stratum_counts.values() if c >= min_stratum_size
    )

    return StratifiedPartialSpearmanMultiResult(
        x=x, y=y, conditioning=conditioning,
        stratify_by=stratify_by,
        rho_pooled=float(rho), p_value=float(p),
        n_obs_total=len(xs),
        n_strata=n_strata,
    )
