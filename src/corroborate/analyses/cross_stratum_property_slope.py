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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import scipy.stats as stats

from corroborate.analyses.stratified_arm_diff_pooled import (
    stratified_arm_diff_pooled,
)
from corroborate.bridge.analysis import analysis


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


@analysis
def cross_stratum_property_slope(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    source: str,
    covariate_name: str,
    covariates_per_key: Mapping[object, Mapping[str, float]],
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
    `stratify_by` dimension keys `covariates_per_key`. The
    analysis looks up `covariates_per_key[stratum_id[i]][covariate_name]`
    per stratum (where `i` is the position of `covariate_key_field`
    in `stratify_by`). `covariate_key_field` MUST appear in
    `stratify_by`.

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
    key_position = stratify_by.index(covariate_key_field)
    cells_list = list(cells)
    pooled = stratified_arm_diff_pooled.fn(
        cells_list,
        source=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
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
        key_covs = covariates_per_key.get(key)
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
    'cross_stratum_property_slope',
]
