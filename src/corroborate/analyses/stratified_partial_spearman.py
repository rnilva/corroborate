"""`stratified_partial_spearman` — per-cell JCI partial Spearman
ρ(X, Y | Z), env-stratified, Fisher-z pooled.

The cell-level conditional-independence test the framework
recommends for "is M a mediator of X → Y, after conditioning
on jens (or any upstream Z)?" — per
`corroborate.analyses.proportion_mediated`'s deprecation note.

Each cell is one observation; per-stratum partial Spearman ρ is
computed via the closed-form three-rank-correlation identity
(`graph.discovery.partial_spearman_rho`); strata are pooled via
Fisher-z weighted by `(n_k − 4)`. Stratum-level rather than
per-pair-Δ form: each stratum's cells contribute independent
samples of the (X, Y, Z) joint distribution; conditioning on Z
removes the variance Z explains; ρ_partial is the residual
non-Z-mediated coupling.

Distinct from:
- `partial_spearman_paired_delta` — per-pair-Δ form (treats Δs
  as samples; init-correlation enters the slope estimate).
- `proportion_mediated` — linear-mediation share (DEPRECATED;
  ratio explodes near zero total effect; doesn't handle
  Q-explosion / non-monotone M→Y).
- `stratified_partial_spearman_rho` (the bare function in
  `graph.discovery`) — this module's @analysis wrapper exposes
  it via the bridge fixture system.

Used when a bridge's question is "does conditioning on `z`
(typically jens) collapse the X→Y coupling?" with env as a
structural confounder. Null prediction (X is not an independent
mediator) HELDs when |ρ_partial| < null_max_abs_rho.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from corroborate.analyses.paired_g import resolve_value
from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import stratified_partial_spearman_rho


@dataclass(frozen=True, slots=True)
class StratifiedPartialSpearmanResult:
    """JCI-stratified partial Spearman ρ + Fisher-z-pooled p.

    `rho_pooled` is the tanh of the Fisher-z weighted average
    across strata; `p_value` is the two-sided test against
    `rho=0` under the pooled z-statistic.

    `n_obs_total` is the total number of cells with non-NaN
    values across all stratum sizes ≥ `min_stratum_size`.
    `n_strata` is the number of strata that contributed.

    Returns NaN ρ/p when no stratum reaches `min_stratum_size`.
    """
    x: str
    y: str
    conditioning: str
    stratify_by: str
    rho_pooled: float
    p_value: float
    n_obs_total: int
    n_strata: int


@analysis
def stratified_partial_spearman(
    cells: Iterable[Mapping[str, object]],
    *,
    x: str,
    y: str,
    conditioning: str,
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
) -> StratifiedPartialSpearmanResult:
    """JCI-form partial Spearman ρ(X, Y | Z), stratified by
    `stratify_by` (default `env_name`), pooled via Fisher z.

    Each cell contributes one observation triple `(x, y, z)`.
    Strata with fewer than `min_stratum_size` complete triples
    are dropped. Within each surviving stratum, the closed-form
    partial Spearman ρ(X, Y | Z) is computed via
    `corroborate.graph.discovery.partial_spearman_rho` (three-
    rank-correlation identity). Per-stratum ρs are converted to
    Fisher z, weighted by `(n_k − 4)`, averaged, and converted
    back to ρ. The two-sided p uses the pooled z-statistic
    normalised by `√(Σ (n_k − 4))`.

    `x`, `y`, `conditioning` resolve via `resolve_value`
    (registry-first; field-path fallback) — same discipline as
    other `@analysis` primitives.
    """
    cells_list = list(cells)
    strata_keys: list[object] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for cell in cells_list:
        try:
            xv = resolve_value(cell, x)
            yv = resolve_value(cell, y)
            zv = resolve_value(cell, conditioning)
        except (KeyError, TypeError, ValueError):
            continue
        if any(v != v for v in (xv, yv, zv)):  # NaN check
            continue
        sk = cell.get(stratify_by)
        if sk is None:
            continue
        strata_keys.append(sk)
        xs.append(xv)
        ys.append(yv)
        zs.append(zv)

    if not xs:
        return StratifiedPartialSpearmanResult(
            x=x, y=y, conditioning=conditioning,
            stratify_by=stratify_by,
            rho_pooled=float('nan'), p_value=float('nan'),
            n_obs_total=0, n_strata=0,
        )

    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    z_arr = np.asarray(zs, dtype=np.float64)
    rho, p = stratified_partial_spearman_rho(
        x_arr, y_arr, z_arr, strata_keys,
        min_stratum_size=min_stratum_size,
    )

    # Count strata that contributed (≥ min_stratum_size).
    stratum_counts: dict[object, int] = {}
    for sk in strata_keys:
        stratum_counts[sk] = stratum_counts.get(sk, 0) + 1
    n_strata = sum(
        1 for c in stratum_counts.values() if c >= min_stratum_size
    )

    return StratifiedPartialSpearmanResult(
        x=x, y=y, conditioning=conditioning,
        stratify_by=stratify_by,
        rho_pooled=float(rho), p_value=float(p),
        n_obs_total=len(xs),
        n_strata=n_strata,
    )
