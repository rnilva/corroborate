"""`cross_stratum_property_slope` — cross-stratum Spearman ρ between
a per-stratum scalar covariate and per-stratum effect size.

The substantive question: "does the treatment effect scale with
some stratum-level property?" Examples:
- "DDQN's bias reduction (Δ_jens Cohen's d per env) scales with
  FA-coherence (q_autocorr_vanilla per env)" — env-level property.
- "DDQN's outcome benefit scales inversely with reward density
  per env" — env-level property.
- "DDQN's outcome benefit scales with chain depth (log_horizon
  per γ)" — γ-level property.

Each stratum contributes ONE point (covariate_value, cohen_d).
Spearman ρ across strata returns rank correlation that's robust
at small n and gives a clean trichotomy (HELD / NO_EFFECT-with-
SIGN_FLIP / NO_EFFECT-NULL / POW_INSUF) without paying the
slope-SE inflation tax of small-n meta-regression.

**Sibling of `meta_regression_unpaired_d`** with the same panel
construction (`stratified_arm_diff_pooled.fn` for per-stratum
Cohen's d) but a different cross-stratum test: Spearman ρ vs
OLS slope. At small n_strata (≤ ~15), Spearman is the more
honest form — meta-regression slope CI is bounded by
between-stratum variance / n_strata, giving POWER_INSUFFICIENT
even when the rank order is decisive.

**Sibling of `cross_stratum_arm_diff_slope`** which Spearman-
correlates two per-stratum arm-diff vectors (both are Δs). This
primitive's predictor is a per-stratum SCALAR COVARIATE (env or
γ property), not an arm-diff.

Distinct from:
- `meta_regression_unpaired_d` — same panel, OLS slope on the
  same predictor. Use when n_strata is large enough that slope SE
  resolves.
- `cross_stratum_arm_diff_slope` — both vectors are arm-diff Δs
  (the substrate-level dose-response form).
- `stratified_spearman` — within-stratum Spearman pooled across
  strata via Fisher-z (no cross-stratum slope question).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import scipy.stats as stats

if TYPE_CHECKING:
    # Type-only import (avoids analyses ↔ data runtime cycle).
    from corroborate.data import DerivedSpec as DerivedSpecKernel
else:
    # Runtime-only forward ref so `_derive_per_stratum_covariate`'s
    # signature annotation parses without the import-cycle hit.
    DerivedSpecKernel = object  # placeholder

from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    stratified_arm_diff_pooled,
)
from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class DerivedCovariateSpec:
    """Compute the per-stratum covariate FROM cells in scope rather
    than reading a hardcoded `covariates_per_key` dict.

    `column` — per-cell column to aggregate (e.g., `'lambda_a_late'`).
    `aggregator` — `'mean'`, `'std'`, or `'median'` of the column
    within each stratum's filtered cell pool.
    `arm_filter` — which arm's cells contribute:
      - `'baseline'`: only baseline-arm cells (for properties like
        σ_Λ_a that characterise the vanilla regime);
      - `'treatment'`: only treatment-arm cells;
      - `'both'`: all cells in scope.

    The aggregate is the substantive replacement for hardcoded
    per-env constants — it's recomputed any time the bridge's
    scope changes, so HP-mixing artifacts surface as drift instead
    of hiding in a `MappingProxyType` constant."""
    column: str
    aggregator: Literal['mean', 'std', 'median']
    arm_filter: Literal['baseline', 'treatment', 'both']


@dataclass(frozen=True, slots=True)
class CrossStratumPropertySlopeResult:
    """Spearman ρ across strata of (per-stratum scalar covariate,
    per-stratum Cohen's d on a target measurable).

    `n_strata` is the number of strata that contributed (covariate
    lookup succeeded, Cohen's d finite, ≥ `min_seeds_per_arm` cells
    each arm)."""
    rho: float
    p_value: float
    n_strata: int
    covariate_name: str
    covariate_values: tuple[float, ...]
    cohen_d_per_stratum: tuple[float, ...]


def _derive_per_stratum_covariate(
    cells: list[Mapping[str, object]],
    *,
    spec: 'DerivedCovariateSpec | DerivedSpecKernel',
    treatment_arm: str,
    baseline_arm: str,
    arm_field: str,
    stratify_by: tuple[str, ...],
    key_position: int,
) -> Mapping[object, float]:
    """Derive `{stratum_key: aggregate(column)}` from cells in scope.

    Dispatches on spec type:
    - `DerivedCovariateSpec` (legacy, with `arm_filter:
      Literal[...]`): per-cell loop honouring `arm_filter`.
    - `corroborate.data.DerivedSpec` (framework, with
      `cell_filter: pl.Expr | None`): delegates to the kernel
      (`corroborate.data.kernel.per_stratum_aggregate`) so the
      Panel path and this path share semantics.

    Filters cells by the spec's filter, groups by
    `stratify_by[key_position]`, aggregates `column` via
    `aggregator`. Cells with non-finite column value are dropped
    before aggregation. Returns a frozen mapping suitable as the
    `covariates_per_key` input to the slope analysis."""
    # Lazy import to avoid analyses ↔ data runtime cycle.
    from corroborate.data import DerivedSpec
    from corroborate.data.kernel import (
        cells_to_dataframe, per_stratum_aggregate,
    )
    if isinstance(spec, DerivedSpec):
        # Framework path — delegate to the kernel. The kernel's
        # stratify_by is the full panel-grouping tuple; we
        # project to the single key_position'th key after.
        kernel_out = per_stratum_aggregate(
            cells_to_dataframe(cells),
            column=spec.column,
            aggregator=spec.aggregator,
            stratify_by=stratify_by,
            cell_filter=spec.cell_filter,
            min_n=spec.effective_min_n,
        )
        # Project: the slope analysis's `covariate_key_field`
        # picks ONE stratify key. Aggregate over the other
        # stratify dimensions when present (multiple sub-keys
        # share the same `covariate_key_field` value — rare in
        # implementation use; implementation author typically pass
        # stratify_by=(covariate_key_field,) for cross-env panels.
        # When n_stratify > 1, take the first occurrence per
        # key — deterministic + the implementation-author should
        # collapse upstream.
        out: dict[object, float] = {}
        for stratum_id, v in kernel_out.items():
            if len(stratum_id) <= key_position:
                continue
            key = stratum_id[key_position]
            if key not in out:
                out[key] = v
        return out
    # Legacy DerivedCovariateSpec path — arm_filter Literal +
    # per-cell loop. Same semantics as before the kernel landed.
    grouped: dict[object, list[float]] = {}
    for cell in cells:
        arm = cell.get(arm_field)
        if not isinstance(arm, str):
            continue
        if spec.arm_filter == 'baseline' and arm != baseline_arm:
            continue
        if spec.arm_filter == 'treatment' and arm != treatment_arm:
            continue
        # `both` keeps both
        sid_parts: list[object] = []
        for sb in stratify_by:
            sid_parts.append(cell.get(sb))
        if len(sid_parts) <= key_position:
            continue
        key = sid_parts[key_position]
        v = cell.get(spec.column)
        if not isinstance(v, (int, float)):
            continue
        v_f = float(v)
        if not math.isfinite(v_f):
            continue
        grouped.setdefault(key, []).append(v_f)
    out_legacy: dict[object, float] = {}
    for key, vs in grouped.items():
        if len(vs) < 2:  # SD undefined for n<2
            continue
        arr = np.asarray(vs, dtype=np.float64)
        if spec.aggregator == 'mean':
            out_legacy[key] = float(arr.mean())
        elif spec.aggregator == 'std':
            out_legacy[key] = float(arr.std(ddof=1))
        elif spec.aggregator == 'median':
            out_legacy[key] = float(np.median(arr))
    return out_legacy


@analysis
def cross_stratum_property_slope(
    cells: pl.DataFrame,
    *,
    treatment_arm: str,
    baseline_arm: str,
    arm_field: str = 'arm_key',
    source: str,
    covariate_name: str,
    covariates_per_key: Mapping[object, Mapping[str, float]] | None = None,
    derived_covariate: 'DerivedCovariateSpec | DerivedSpecKernel | None' = None,
    covariate_key_field: str = 'env_name',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
    min_strata: int = 8,
) -> CrossStratumPropertySlopeResult:
    """Compute Spearman ρ across strata of (covariate_value,
    Cohen's d).

    `covariate_key_field` (default `'env_name'`) names which
    `stratify_by` dimension keys the covariates. The analysis
    looks up `covariates_per_key[stratum_id[i]][covariate_name]`
    per stratum (where `i` is the position of `covariate_key_field`
    in `stratify_by`). `covariate_key_field` MUST appear in
    `stratify_by`.

    Covariate source — exactly one of:
    - `covariates_per_key`: hardcoded mapping `{key: {name: value}}`.
      Use when the covariate is exogenous (e.g., env-feature
      `R_max`) and not derived from cells in scope.
    - `derived_covariate`: a `DerivedCovariateSpec` naming a
      per-cell column + aggregator + arm-filter. The analysis
      computes the per-stratum value from cells in scope. Use
      when the covariate IS a within-scope cell aggregate (e.g.,
      σ_Λ_a = SD of `lambda_a_late` over baseline cells per env).
      This path eliminates the HP-mixing risk inherent in
      hardcoded constants: when the bridge's scope changes, the
      derived covariate re-derives instead of staying frozen.

    Per-stratum Cohen's d is produced via
    `stratified_arm_diff_pooled.fn` — same panel construction as
    `meta_regression_unpaired_d`. Strata with NaN Cohen's d
    (saturated outcome → no SD) or failed covariate lookup are
    dropped. Spearman over the surviving (covariate, d) pairs.

    Returns NaN ρ/p when `n_strata < min_strata`."""
    if not stratify_by or covariate_key_field not in stratify_by:
        raise ValueError(
            f'cross_stratum_property_slope: covariate_key_field '
            f'{covariate_key_field!r} must appear in stratify_by; '
            f'got {stratify_by!r}',
        )
    if (covariates_per_key is None) == (derived_covariate is None):
        raise ValueError(
            'cross_stratum_property_slope: pass exactly one of '
            '`covariates_per_key` or `derived_covariate`; '
            f'got covariates_per_key={covariates_per_key is not None}, '
            f'derived_covariate={derived_covariate is not None}',
        )
    key_position = stratify_by.index(covariate_key_field)
    # The covariate-derivation helper streams row dicts;
    # materialise them once. `to_dicts()` returns
    # `list[dict[str, Any]]`; widen to `Mapping[str, object]` via
    # the framework's covariant boundary helper.
    from corroborate._internals.polars import to_dicts as _to_dicts
    cells_list: list[Mapping[str, object]] = list(_to_dicts(cells))
    # Build covariates_per_key from cells if a derived spec is given.
    if derived_covariate is not None:
        derived_map = _derive_per_stratum_covariate(
            cells_list,
            spec=derived_covariate,
            treatment_arm=treatment_arm,
            baseline_arm=baseline_arm,
            arm_field=arm_field,
            stratify_by=stratify_by,
            key_position=key_position,
        )
        effective_covariates: Mapping[object, Mapping[str, float]] = {
            k: {covariate_name: v} for k, v in derived_map.items()
        }
    else:
        assert covariates_per_key is not None  # narrowed by check above
        effective_covariates = covariates_per_key
    pooled = stratified_arm_diff_pooled.fn(
        cells,
        source=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        arm_field=arm_field,
        stratify_by=stratify_by,
        scope_predictor=scope_predictor,
        min_baseline_predictor=min_baseline_predictor,
        min_seeds_per_arm=min_seeds_per_arm,
    )
    cov_values: list[float] = []
    d_values: list[float] = []
    for s in pooled.per_stratum:
        if not s.stratum_id or len(s.stratum_id) <= key_position:
            continue
        key = s.stratum_id[key_position]
        key_covs = effective_covariates.get(key)
        if key_covs is None:
            continue
        cov = key_covs.get(covariate_name)
        if cov is None or math.isnan(float(cov)):
            continue
        if math.isnan(s.cohen_d):
            continue
        cov_values.append(float(cov))
        d_values.append(float(s.cohen_d))

    n = len(cov_values)
    if n < min_strata:
        return CrossStratumPropertySlopeResult(
            rho=float('nan'), p_value=float('nan'),
            n_strata=n,
            covariate_name=covariate_name,
            covariate_values=tuple(cov_values),
            cohen_d_per_stratum=tuple(d_values),
        )

    xs = np.asarray(cov_values, dtype=np.float64)
    ys = np.asarray(d_values, dtype=np.float64)
    rho_raw, p_raw = stats.spearmanr(xs, ys)
    return CrossStratumPropertySlopeResult(
        rho=float(rho_raw),
        p_value=float(p_raw),
        n_strata=n,
        covariate_name=covariate_name,
        covariate_values=tuple(cov_values),
        cohen_d_per_stratum=tuple(d_values),
    )


__all__ = [
    'CrossStratumPropertySlopeResult',
    'DerivedCovariateSpec',
    'cross_stratum_property_slope',
]
