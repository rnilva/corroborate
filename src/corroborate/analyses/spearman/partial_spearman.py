"""`partial_spearman` — unified JCI Spearman ρ(X, Y | Z₁, …, Zₖ),
env-stratified, Fisher-z pooled, per-cell OR per-burst.

Subsumes the five separately-named Spearman primitives:
  - `stratified_spearman`              → k=0, granularity=per_cell
  - `stratified_partial_spearman`      → k=1, granularity=per_cell
  - `stratified_partial_spearman_multi`→ k≥1, granularity=per_cell
  - `per_burst_jci_spearman`           → k=0, granularity=per_burst
  - `per_burst_partial_jci_spearman`   → k=1, granularity=per_burst

Granularity is detected from input types:
  - `x: str` (column name)  → per-cell observations
  - `x: Measurable[..., NDArray]` → per-burst observations

`y` and each entry of `conditioning` must match `x`'s shape;
mixed-mode inputs raise.

Conditioning is `tuple[str, ...] | tuple[Measurable, ...]`:
  - `()` empty → marginal Spearman ρ(X, Y)
  - `(z,)` single-Z → partial ρ(X, Y | Z)
  - `(z₁, …, zₖ)` multi-Z → partial ρ(X, Y | Z₁, …, Zₖ) via
    OLS-residual regression on rank-transformed variables

The dispatch internally:
  - k = 0 → `graph.discovery.stratified_spearman_rho`
  - k ≥ 1 → `graph.discovery.stratified_partial_spearman_rho_multi`

Both paths Fisher-z-pool per-stratum ρ_k with weight `(n_k − 3 − k)`.
The unified result type satisfies the same `_PartialSpearmanResult`
Protocol as the five legacy result types — verdict helpers
(`partial_spearman_null_verdict`, `partial_spearman_signed_verdict`)
consume it unchanged.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from corroborate.analyses._cell_value import (
    evaluate_per_burst_source, resolve_value,
)
from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import (
    stratified_partial_spearman_rho,
    stratified_partial_spearman_rho_multi,
    stratified_spearman_rho,
)
from corroborate.measurables import Measurable


type _PerBurstMeasurable = Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
]
type _ScalarOrPerBurst = str | _PerBurstMeasurable

type _Granularity = Literal['per_cell', 'per_burst']


@dataclass(frozen=True, slots=True)
class PartialSpearmanResult:
    """Unified JCI (partial) Spearman ρ + Fisher-z-pooled p.

    `x`, `y` are column names (for str inputs) or measurable
    names (for Measurable inputs — uses `.name`). `conditioning`
    is the tuple of conditioning column-or-measurable names in
    the order passed; empty tuple for marginal Spearman.
    `granularity` records the observation shape (`per_cell` or
    `per_burst`) for diagnostic / snapshot stability.

    `rho_pooled` is the tanh of the Fisher-z weighted average
    across strata; `p_value` is the two-sided test against ρ=0
    under the pooled z-statistic. NaN when no stratum reaches
    `min_stratum_size` (or `min_stratum_size > 3 + k` for
    multi-Z forms — df accounts for k conditioning vars).

    `n_obs_total` counts observations contributing to the pool
    (per-cell: one per cell; per-burst: one per (cell, burst)
    that has non-NaN x/y/z values). `n_strata` counts strata that
    met the size+df floor."""
    x: str
    y: str
    conditioning: tuple[str, ...]
    stratify_by: str
    granularity: _Granularity
    rho_pooled: float
    p_value: float
    n_obs_total: int
    n_strata: int


def _arg_name(arg: _ScalarOrPerBurst) -> str:
    return arg if isinstance(arg, str) else arg.name


def _detect_granularity(
    x: _ScalarOrPerBurst, y: _ScalarOrPerBurst,
    conditioning: tuple[_ScalarOrPerBurst, ...],
) -> _Granularity:
    """All inputs must agree on shape: all str → per_cell, all
    Measurable → per_burst. Mixing raises — silently coercing
    would mask a bridge-author bug (treating a per-burst array
    as a scalar or vice versa)."""
    all_args: tuple[_ScalarOrPerBurst, ...] = (x, y, *conditioning)
    str_count = sum(1 for a in all_args if isinstance(a, str))
    if str_count == len(all_args):
        return 'per_cell'
    if str_count == 0:
        return 'per_burst'
    raise TypeError(
        f'partial_spearman: x/y/conditioning must all be str '
        f'(per-cell) OR all Measurable (per-burst); got mix of '
        f'{str_count} str and {len(all_args) - str_count} '
        f'Measurable. Mixed types mask a bridge bug — coercion '
        f'would silently flatten or broadcast incorrectly.',
    )


def _collect_per_cell(
    cells: list[Mapping[str, object]],
    *, x: str, y: str, conditioning: tuple[str, ...],
    stratify_by: str,
) -> tuple[list[float], list[float], list[list[float]], list[object]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[list[float]] = [[] for _ in conditioning]
    strata: list[object] = []
    for cell in cells:
        try:
            xv = resolve_value(cell, x)
            yv = resolve_value(cell, y)
            z_vals = tuple(
                resolve_value(cell, z) for z in conditioning
            )
        except (KeyError, TypeError, ValueError):
            continue
        if math.isnan(xv) or math.isnan(yv):
            continue
        if any(math.isnan(zv) for zv in z_vals):
            continue
        sk = cell.get(stratify_by)
        if sk is None:
            continue
        xs.append(xv)
        ys.append(yv)
        for col, zv in zip(zs, z_vals):
            col.append(zv)
        strata.append(sk)
    return xs, ys, zs, strata


def _collect_per_burst(
    cells: list[Mapping[str, object]],
    *, x: _PerBurstMeasurable, y: _PerBurstMeasurable,
    conditioning: tuple[_PerBurstMeasurable, ...],
    stratify_by: str,
) -> tuple[list[float], list[float], list[list[float]], list[object]]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[list[float]] = [[] for _ in conditioning]
    strata: list[object] = []
    for cell in cells:
        x_arr = evaluate_per_burst_source(x, cell)
        y_arr = evaluate_per_burst_source(y, cell)
        z_arrs = tuple(
            evaluate_per_burst_source(z, cell) for z in conditioning
        )
        # Length-match via element-wise min across x, y, all z's —
        # multi-regime cells with mismatched burst counts contribute
        # the shorter prefix.
        n = min(
            (x_arr.size, y_arr.size, *(z.size for z in z_arrs)),
            default=0,
        )
        if n == 0:
            continue
        sk = cell.get(stratify_by)
        if sk is None:
            continue
        for i in range(n):
            xv = float(x_arr[i])
            yv = float(y_arr[i])
            z_vals = tuple(float(z[i]) for z in z_arrs)
            if math.isnan(xv) or math.isnan(yv):
                continue
            if any(math.isnan(zv) for zv in z_vals):
                continue
            xs.append(xv)
            ys.append(yv)
            for col, zv in zip(zs, z_vals):
                col.append(zv)
            strata.append(sk)
    return xs, ys, zs, strata


@analysis
def partial_spearman(
    cells: Iterable[Mapping[str, object]],
    *,
    x: _ScalarOrPerBurst,
    y: _ScalarOrPerBurst,
    conditioning: tuple[_ScalarOrPerBurst, ...] = (),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
) -> PartialSpearmanResult:
    """JCI-stratified Spearman ρ(X, Y | Z₁, …, Zₖ).

    Granularity is detected from input types: str → per-cell
    observations (one per cell); Measurable → per-burst
    observations (one per (cell, burst)). All of `x`, `y`,
    `conditioning` must agree on type.

    `conditioning=()` → marginal Spearman; one or more entries →
    partial Spearman (single-Z or multi-Z, dispatched internally).

    Strata with fewer than `min_stratum_size` complete
    observations are dropped. For multi-Z forms, strata where
    `n_k ≤ 3 + k` are also dropped (insufficient df).

    Returns NaN ρ/p when no stratum survives the size + df
    floors."""
    cells_list = list(cells)
    granularity = _detect_granularity(x, y, conditioning)
    if granularity == 'per_cell':
        # Type narrowing for pyright — _detect_granularity guarantees
        # all-str when granularity is per_cell.
        x_s = x if isinstance(x, str) else x.name
        y_s = y if isinstance(y, str) else y.name
        cond_s = tuple(
            z if isinstance(z, str) else z.name for z in conditioning
        )
        xs, ys, zs, strata = _collect_per_cell(
            cells_list, x=x_s, y=y_s, conditioning=cond_s,
            stratify_by=stratify_by,
        )
    else:
        # All-Measurable; narrow each.
        assert not isinstance(x, str)
        assert not isinstance(y, str)
        cond_m: tuple[_PerBurstMeasurable, ...] = tuple(
            z for z in conditioning if not isinstance(z, str)
        )
        assert len(cond_m) == len(conditioning), (
            'partial_spearman: granularity mismatch slipped through '
            'detection — should be unreachable'
        )
        xs, ys, zs, strata = _collect_per_burst(
            cells_list, x=x, y=y, conditioning=cond_m,
            stratify_by=stratify_by,
        )

    x_name = _arg_name(x)
    y_name = _arg_name(y)
    cond_names = tuple(_arg_name(z) for z in conditioning)
    if not xs:
        return PartialSpearmanResult(
            x=x_name, y=y_name, conditioning=cond_names,
            stratify_by=stratify_by, granularity=granularity,
            rho_pooled=float('nan'), p_value=float('nan'),
            n_obs_total=0, n_strata=0,
        )

    x_np = np.asarray(xs, dtype=np.float64)
    y_np = np.asarray(ys, dtype=np.float64)
    if len(conditioning) == 0:
        rho, p = stratified_spearman_rho(
            x_np, y_np, strata,
            min_stratum_size=min_stratum_size,
        )
    elif len(conditioning) == 1:
        # Closed-form first-order partial Spearman is numerically
        # distinct from the OLS-residual multi-Z form at k=1
        # (both correct under joint-normality; multi uses ranked-
        # OLS-residual which picks up tie-handling drift the
        # closed form doesn't). Dispatching k=1 to the closed-form
        # primitive preserves bit-exact compatibility with the
        # legacy `stratified_partial_spearman` consumer bridges.
        z_np: npt.NDArray[np.float64] = np.asarray(zs[0], dtype=np.float64)
        rho, p = stratified_partial_spearman_rho(
            x_np, y_np, z_np, strata,
            min_stratum_size=min_stratum_size,
        )
    else:
        z_matrix: npt.NDArray[np.float64] = np.column_stack(
            [np.asarray(col, dtype=np.float64) for col in zs],
        )
        rho, p = stratified_partial_spearman_rho_multi(
            x_np, y_np, z_matrix, strata,
            min_stratum_size=min_stratum_size,
        )

    counts: dict[object, int] = {}
    for sk in strata:
        counts[sk] = counts.get(sk, 0) + 1
    df_floor = 3 + len(conditioning)
    n_strata = sum(
        1 for c in counts.values()
        if c >= min_stratum_size and c > df_floor
    )

    return PartialSpearmanResult(
        x=x_name, y=y_name, conditioning=cond_names,
        stratify_by=stratify_by, granularity=granularity,
        rho_pooled=float(rho), p_value=float(p),
        n_obs_total=len(xs), n_strata=n_strata,
    )


__all__ = [
    'PartialSpearmanResult',
    'partial_spearman',
]
