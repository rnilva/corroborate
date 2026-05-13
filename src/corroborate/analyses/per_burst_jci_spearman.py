"""`per_burst_jci_spearman` — per-(cell, burst) JCI Spearman ρ.

Takes two per-burst-array Measurables (`x`, `y`) and a
stratification column. For each cell, unfolds the per-burst
arrays into n_bursts rows; pools all rows; runs Spearman ρ(x, y)
with the stratification (Fisher-z pool) — the cell-level
`stratified_spearman` shape applied to per-burst-unfolded data.

Use case: when the relationship of interest varies by burst
(training phase), and cell-level aggregation (full-trajectory
mean, or late-50% window) averages away the per-burst dynamics
— surfacing a misleading aggregate signal or hiding a real
phase-specific one. Per-burst unfold preserves the within-
burst-across-cells correlation while still pooling for
inference.

The stratify_by column applies to the source CELL (not per-
burst); typically `env_name`. Each (cell, burst) row inherits
the cell's env_name. Pool is then over (env, burst) pairs
within each env."""
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
from corroborate.graph.discovery import stratified_spearman_rho
from corroborate.measurables import Measurable


@dataclass(frozen=True, slots=True)
class PerBurstJciSpearmanResult:
    """Per-burst-unfolded JCI Spearman result. Field shape mirrors
    `StratifiedSpearmanResult` so existing verdict helpers
    (`partial_spearman_signed_verdict`, `partial_spearman_null_verdict`)
    consume it without modification."""
    x: str
    y: str
    stratify_by: str
    rho_pooled: float
    p_value: float
    n_obs_total: int
    n_strata: int


@analysis
def per_burst_jci_spearman(
    cells: Iterable[Mapping[str, object]],
    *,
    x: Measurable[Mapping[str, object], npt.NDArray[np.floating]],
    y: Measurable[Mapping[str, object], npt.NDArray[np.floating]],
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
) -> PerBurstJciSpearmanResult:
    """Per-(cell, burst) JCI Spearman ρ(x, y), stratified by
    `stratify_by`.

    For each cell, `evaluate_per_burst_source(x, cell)` and
    `evaluate_per_burst_source(y, cell)` give per-burst arrays.
    Each cell contributes n_bursts rows; the row's stratum-key
    is `cell[stratify_by]`. Per-stratum Spearman ρ computed on
    the unfolded rows, Fisher-z-pooled across strata.

    Strata with fewer than `min_stratum_size` rows are dropped.

    Returns `PerBurstJciSpearmanResult` mirroring the existing
    Spearman result shape."""
    cells_list = list(cells)
    xs: list[float] = []
    ys: list[float] = []
    strata_keys: list[object] = []
    for cell in cells_list:
        x_arr = evaluate_per_burst_source(x, cell)
        y_arr = evaluate_per_burst_source(y, cell)
        n = min(x_arr.size, y_arr.size)
        if n == 0:
            continue
        sk = cell.get(stratify_by)
        if sk is None:
            continue
        for i in range(n):
            xv = float(x_arr[i])
            yv = float(y_arr[i])
            if math.isnan(xv) or math.isnan(yv):
                continue
            xs.append(xv)
            ys.append(yv)
            strata_keys.append(sk)
    if not xs:
        return PerBurstJciSpearmanResult(
            x=x.name, y=y.name, stratify_by=stratify_by,
            rho_pooled=float('nan'), p_value=float('nan'),
            n_obs_total=0, n_strata=0,
        )
    x_np = np.asarray(xs, dtype=np.float64)
    y_np = np.asarray(ys, dtype=np.float64)
    rho, p = stratified_spearman_rho(
        x_np, y_np, strata_keys, min_stratum_size=min_stratum_size,
    )
    # Count contributing strata
    counts: dict[object, int] = {}
    for sk in strata_keys:
        counts[sk] = counts.get(sk, 0) + 1
    n_strata = sum(1 for c in counts.values() if c >= min_stratum_size)
    return PerBurstJciSpearmanResult(
        x=x.name, y=y.name, stratify_by=stratify_by,
        rho_pooled=float(rho), p_value=float(p),
        n_obs_total=len(xs), n_strata=n_strata,
    )


__all__ = ['PerBurstJciSpearmanResult', 'per_burst_jci_spearman']
