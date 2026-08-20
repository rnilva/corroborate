"""`mundlak_paired_g_per_burst` — composite analysis: per-(env,
burst) Hedges' g paired panel + Mundlak within/between
decomposition of a *named* per-burst measurable.

Phase B refactor — the per-burst predictor is identified by
NAME (a registered `@measurable` returning a `(n_bursts,)`
array), not by a callable. Swap moderator candidates by
changing the string; no per-moderator helper functions.

The shape:

  cells × `predictor_name` (registered @measurable)
    → paired_g_per_burst (per-(env, burst) → (g, se))
    → resolve `predictor_name` once per cell (cached array)
    → mundlak_decomposition (env-mean + within-deviation)
    → MundlakResult

Bridge consumers declare `mundlak_paired_g_per_burst:
MundlakResult`; the bridge's `predictor_name` defaulted-kwarg
controls which measurable is decomposed."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

import polars as pl

from corroborate._internals.polars import to_dicts
from corroborate.data.kernel import cells_to_dataframe
from corroborate.analyses.paired.mundlak_decomposition import (
    MundlakResult, mundlak_decomposition,
)
from corroborate.analyses.paired.paired_g_per_burst import (
    DEFAULT_PER_BURST_SOURCE,
    paired_g_per_burst,
)
from corroborate.bridge.analysis import analysis
from corroborate.measurables import Measurable


def _per_env_burst_predictor_mean(
    cells: Sequence[Mapping[str, object]],
    burst_index: int,
    per_cell_array: dict[str, npt.NDArray[np.float64]],
    env_name: str,
    arm_filter: str | None,
    arm_field: str = 'arm_key',
) -> float:
    """Average a precomputed per-burst array across the cells of
    `env_name` at `burst_index`. `per_cell_array` is keyed by
    cell `id`. `arm_filter` restricts the average to a single
    arm (e.g., the baseline arm name)."""
    vals: list[float] = []
    for c in cells:
        if c.get('env_name') != env_name:
            continue
        if arm_filter is not None and c.get(arm_field) != arm_filter:
            continue
        cell_id = c.get('id')
        if not isinstance(cell_id, str):
            continue
        arr = per_cell_array.get(cell_id)
        if arr is None or arr.ndim < 1 or burst_index >= arr.shape[0]:
            continue
        v = float(arr[burst_index])
        if not math.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float('nan')


@analysis
def mundlak_paired_g_per_burst(
    cells: pl.DataFrame,
    *,
    treatment_arm: str,
    baseline_arm: str,
    arm_field: str = 'arm_key',
    pair_by: tuple[str, ...] = ('seed',),
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = DEFAULT_PER_BURST_SOURCE,
    predictor_name: str = 'log_mc_variance_per_burst',
    alpha: float = 0.05,
    cluster_robust: bool = True,
    dedupe_strategy: str = 'mean',
) -> MundlakResult:
    """Per-(env, burst) Hedges' g panel + Mundlak decomposition
    of a per-burst measurable.

    `source` is a typed Measurable returning a per-burst NDArray
    (the Hedges' g panel target). Default: per-burst-mean of
    `mc_return`. See `paired_g_per_burst` for composition shapes.

    `predictor_name` resolves a registered `@measurable` whose
    `__call__(record)` returns a `(n_bursts,)` array (e.g.
    `log_mc_variance_per_burst`, `mc_variance_per_burst`). The
    analysis applies it once per cell, caches the array, then
    averages across cells at each burst index across the
    *baseline arm only* to build the per-(env, burst) predictor
    column. Restricting to the baseline keeps the predictor a
    covariate of the regime (env + HPs as the unmodified
    composition runs them) instead of contaminating it with the
    treatment effect — which is exactly what Mundlak then
    decomposes the g_link panel against. The runner-injected
    `baseline_arm` is used directly; bridge authors do not write
    arm_key strings.

    `cluster_robust` defaults to **True** — bursts within an env
    share the agent's training trajectory and the env's
    structural noise, so per-burst residuals are correlated
    within env. OLS-style SEs would overstate significance.
    Forwarded to `mundlak_decomposition` as the Liang-Zeger CR1
    sandwich (clusters by env).

    `dedupe_strategy` is forwarded to `paired_g_per_burst`:
    defaults to `'mean'` (per-cell aggregation within each
    `(env, arm, pair_by)` bucket); pass `'raise'` to error on
    duplicates."""
    cells_list = [dict(c) for c in to_dicts(cells)]
    per_burst_g = paired_g_per_burst.fn(
        cells, treatment_arm=treatment_arm,
        baseline_arm=baseline_arm, pair_by=pair_by,
        source=source,
        arm_field=arm_field,
        dedupe_strategy=dedupe_strategy,
    )

    # Resolve the named measurable; apply once per cell with
    # transitive dep resolution via `evaluate_with_measurables`,
    # cache the array. Substrate consumers register their
    # measurables via their package's `__init__.py` side effect;
    # this analysis is substrate-neutral and does not register
    # anything itself.
    from corroborate.measurables import (
        get_registered, evaluate_with_measurables,
    )
    predictor_m = get_registered(predictor_name)
    if predictor_m is None:
        raise RuntimeError(
            f'measurable {predictor_name!r} is not registered; '
            f'declare it via `@measurable` and ensure the '
            f'declaring module is imported (substrate consumers '
            f'usually import the substrate package, which '
            f'registers its measurables in its `__init__.py`).',
        )
    per_cell_array: dict[str, npt.NDArray[np.float64]] = {}
    for c in cells_list:
        cell_id = c.get('id')
        if not isinstance(cell_id, str):
            continue
        # Cache-first discipline: if the bridge cache materialised
        # the predictor as a column at build time, the persisted
        # value is authoritative. Recompute only when the column
        # is absent from the schema entirely (legacy corpora,
        # fresh @measurable not yet seen by any prior cache build).
        # Heterogeneous universal-merge corpora often have the
        # column with None for rows from sources without traces;
        # NaN-skip those rather than re-trigger the same KeyError
        # by evaluating the measurable on a leaf-less cell.
        if predictor_name in c:
            cached = c[predictor_name]
            if cached is None:
                continue
            per_cell_array[cell_id] = np.asarray(
                cached, dtype=np.float64,
            )
            continue
        try:
            arr_obj: object = evaluate_with_measurables(
                predictor_m.fn, c,
            )
        except (KeyError, TypeError, ValueError):
            continue
        per_cell_array[cell_id] = np.asarray(arr_obj, dtype=np.float64)

    panel: list[dict[str, object]] = []
    for s in per_burst_g.strata:
        if s.n_pairs < 2 or math.isnan(s.g) or math.isnan(s.se) \
                or s.se <= 0.0:
            continue
        x = _per_env_burst_predictor_mean(
            cells_list, s.burst_index, per_cell_array,
            s.env_name, baseline_arm, arm_field,
        )
        if math.isnan(x):
            continue
        panel.append({
            'stratum_id': s.env_name,
            'x': x, 'y': s.g, 'se': s.se,
        })

    if len(panel) < 4:
        raise ValueError(
            f'mundlak_paired_g_per_burst: panel has only '
            f'{len(panel)} valid (env, burst) strata after '
            f'predictor join + drops; need at least 4.',
        )

    return mundlak_decomposition.fn(
        cells_to_dataframe(panel),
        alpha=alpha, cluster_robust=cluster_robust,
    )


__all__ = ['mundlak_paired_g_per_burst']
