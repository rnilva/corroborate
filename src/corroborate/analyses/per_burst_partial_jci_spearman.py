"""`per_burst_partial_jci_spearman` — per-(cell, burst) partial
Spearman ρ(x, y | z), env-stratified, Fisher-z pooled.

Sibling of `per_burst_jci_spearman` with a conditioning Measurable
`z`. Each cell unfolds to n_bursts rows of (x, y, z); per stratum
the closed-form partial Spearman ρ(x, y | z) is computed
(`graph.discovery.partial_spearman_rho`); strata are Fisher-z
pooled by `(n_k − 4)`.

Use case: surfaces partial-mediation Q-shape claims at per-burst
granularity. The marginal per-burst Spearman ρ(q_action_std, mc)
mixes substantive Q-shape mediation with the trivial Q-IS-MC
tautology on positive-return envs (Q estimates MC return). The
partial form ρ(q_action_std, mc | q_per_burst) partials the
Q-magnitude channel and isolates the Q-shape's substantive
contribution.

Result type mirrors `PerBurstJciSpearmanResult` so existing
verdict helpers (`partial_spearman_signed_verdict`,
`partial_spearman_null_verdict`) consume it without modification.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from corroborate.analyses.paired_g_per_burst import (
    evaluate_per_burst_source,
)
from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import stratified_partial_spearman_rho
from corroborate.measurables import Measurable


@dataclass(frozen=True, slots=True)
class PerBurstPartialJciSpearmanResult:
    """Per-burst-unfolded partial Spearman ρ(x, y | z) result.
    Field shape mirrors `PerBurstJciSpearmanResult` (and
    `StratifiedSpearmanResult`) so the existing verdict helpers
    consume it unchanged."""
    x: str
    y: str
    conditioning: str
    stratify_by: str
    rho_pooled: float
    p_value: float
    n_obs_total: int
    n_strata: int


@analysis
def per_burst_partial_jci_spearman(
    cells: Iterable[Mapping[str, object]],
    *,
    x: Measurable[Mapping[str, object], npt.NDArray[np.floating]],
    y: Measurable[Mapping[str, object], npt.NDArray[np.floating]],
    conditioning: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
) -> PerBurstPartialJciSpearmanResult:
    """Per-(cell, burst) partial Spearman ρ(x, y | z), stratified
    by `stratify_by` (default `env_name`), Fisher-z pooled.

    For each cell, `evaluate_per_burst_source(x|y|conditioning, cell)`
    yields per-burst arrays. Each cell contributes n_bursts rows;
    each row's stratum-key is `cell[stratify_by]`. Per-stratum
    closed-form partial Spearman ρ(x, y | z) computed via
    `graph.discovery.partial_spearman_rho`. Strata with fewer than
    `min_stratum_size` rows are dropped.

    The conditioning variable resolves to a per-burst array
    aligned with x and y (length-matched via element-wise min)."""
    cells_list = list(cells)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    strata_keys: list[object] = []
    for cell in cells_list:
        x_arr = evaluate_per_burst_source(x, cell)
        y_arr = evaluate_per_burst_source(y, cell)
        z_arr = evaluate_per_burst_source(conditioning, cell)
        n = min(x_arr.size, y_arr.size, z_arr.size)
        if n == 0:
            continue
        sk = cell.get(stratify_by)
        if sk is None:
            continue
        for i in range(n):
            xv = float(x_arr[i])
            yv = float(y_arr[i])
            zv = float(z_arr[i])
            if math.isnan(xv) or math.isnan(yv) or math.isnan(zv):
                continue
            xs.append(xv)
            ys.append(yv)
            zs.append(zv)
            strata_keys.append(sk)
    if not xs:
        return PerBurstPartialJciSpearmanResult(
            x=x.name, y=y.name, conditioning=conditioning.name,
            stratify_by=stratify_by,
            rho_pooled=float('nan'), p_value=float('nan'),
            n_obs_total=0, n_strata=0,
        )
    x_np = np.asarray(xs, dtype=np.float64)
    y_np = np.asarray(ys, dtype=np.float64)
    z_np = np.asarray(zs, dtype=np.float64)
    rho, p = stratified_partial_spearman_rho(
        x_np, y_np, z_np, strata_keys,
        min_stratum_size=min_stratum_size,
    )
    counts: dict[object, int] = {}
    for sk in strata_keys:
        counts[sk] = counts.get(sk, 0) + 1
    n_strata = sum(
        1 for c in counts.values() if c >= min_stratum_size
    )
    return PerBurstPartialJciSpearmanResult(
        x=x.name, y=y.name, conditioning=conditioning.name,
        stratify_by=stratify_by,
        rho_pooled=float(rho), p_value=float(p),
        n_obs_total=len(xs), n_strata=n_strata,
    )


__all__ = [
    'PerBurstPartialJciSpearmanResult',
    'per_burst_partial_jci_spearman',
]
