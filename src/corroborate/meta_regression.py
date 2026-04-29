"""Meta-regression — per-stratum effect sizes regressed on
covariates to identify cleavage axes.

The empirical-side counterpart to invariants
(`project_v9_aggregation_reframing.md`). When a corpus-level pool
gives `HELD_WITH_SCOPE_FLAG` (heterogeneous corroboration), the
question becomes "WHICH covariate predicts the per-stratum
effect?" — answered by inverse-variance-weighted OLS on the
per-stratum (g, SE) pairs against a covariate vector.

A significant coefficient is a numeric threshold on a measurable
(env feature, intervention-arm feature, HP grid axis, mediator
value) — exactly the cleavage shape the v9 reframing argues
scope SHOULD take, in opposition to symbolic-condition
implications or env-feature → goal Bridges (the category
mismatch v9 retired).

The function takes typed `StratumObservation` records and returns
a typed `MetaRegressionResult` with per-coefficient CIs at the
configured `alpha`. Cleavage axes are coefficients whose CI
excludes zero — the empirical scope claim's content."""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as ss

if TYPE_CHECKING:
    from corroborate.schema import HypothesisComparisonRow


@dataclass(frozen=True, slots=True)
class StratumObservation:
    """One stratum's observation: an effect size with its standard
    error and a flat covariate vector.

    `g` and `se` typically come from per-stratum aggregation in
    `HypothesisComparisonRow.per_group`; `covariates` is a flat
    `Mapping[str, float]` keyed by author-chosen covariate names.
    Categoricals must be one-hot or label-encoded by the caller
    before reaching this function (the regression is purely
    numeric)."""
    stratum_id: object
    g: float
    se: float
    covariates: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class CovariateCoefficient:
    """One coefficient from meta-regression, with its CI and
    significance flag.

    `is_significant` is `p_value < alpha` (the configured alpha at
    `meta_regression` call time). Authors should rely on the CI +
    p_value for downstream judgment; the flag is a convenience."""
    name: str
    coefficient: float
    ci_lo: float
    ci_hi: float
    p_value: float
    is_significant: bool


@dataclass(frozen=True, slots=True)
class MetaRegressionResult:
    """Output of a meta-regression run on a stratified corpus.

    `intercept` is the fitted constant (β₀); `coefficients` are
    the slopes (β_j) for the covariate columns. `r_squared` is
    the weighted R² (1 minus residual sum of weighted squares
    over total sum of weighted squares around the weighted
    mean). `cleavage_axes` are the names of significant
    covariates — the empirical scope claim's content."""
    n_strata: int
    intercept: float
    coefficients: tuple[CovariateCoefficient, ...]
    r_squared: float

    @property
    def cleavage_axes(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.coefficients if c.is_significant)


def meta_regression(
    observations: Sequence[StratumObservation],
    *,
    alpha: float = 0.05,
) -> MetaRegressionResult:
    """Inverse-variance-weighted least-squares regression of
    per-stratum effect sizes on covariates.

    Each observation contributes one row: outcome `g`, weight
    `1/se²`, and a covariate vector. The fit minimises
    `Σ wᵢ (gᵢ - ŷᵢ)²` where `ŷᵢ = β₀ + Σⱼ βⱼ xᵢⱼ`. CIs use the
    weighted-residual covariance matrix and a t-distribution with
    `n − p` degrees of freedom (p = 1 + n_covariates).

    Covariate names that appear in some `observations[i].covariates`
    but not others default to `0.0` for the missing rows; the
    caller is responsible for ensuring that's the right encoding.

    Raises:
    - `ValueError` when `n_strata == 0`.
    - `ValueError` when `n_strata <= 1 + n_covariates` (not
      enough data for OLS with this many covariates plus the
      intercept).
    - `ValueError` on any `se <= 0` or NaN (inverse-variance
      weighting requires strictly positive SE).
    - `ValueError` when the design matrix is singular (collinear
      covariates)."""
    n = len(observations)
    if n == 0:
        raise ValueError('meta_regression: observations is empty')

    covariate_names = tuple(sorted({
        k for obs in observations for k in obs.covariates
    }))
    p = 1 + len(covariate_names)
    if n <= p:
        raise ValueError(
            f'meta_regression: n={n} <= p={p}; not enough strata '
            f'for {len(covariate_names)} covariates plus intercept',
        )

    for obs in observations:
        if obs.se <= 0.0 or math.isnan(obs.se):
            raise ValueError(
                f'meta_regression: stratum {obs.stratum_id!r} has '
                f'invalid se={obs.se!r}; must be > 0',
            )

    x_mat = np.ones((n, p), dtype=np.float64)
    y_vec = np.zeros(n, dtype=np.float64)
    w_vec = np.zeros(n, dtype=np.float64)
    for i, obs in enumerate(observations):
        y_vec[i] = obs.g
        w_vec[i] = 1.0 / (obs.se ** 2)
        for j, name in enumerate(covariate_names):
            x_mat[i, j + 1] = obs.covariates.get(name, 0.0)

    w_mat = np.diag(w_vec)
    xtw = x_mat.T @ w_mat
    xtwx = xtw @ x_mat
    xtwy = xtw @ y_vec
    try:
        xtwx_inv = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f'meta_regression: design matrix singular ({e}); '
            f'covariates may be collinear',
        ) from e

    # Numpy boundary — `@` and elementwise ops on numpy arrays
    # produce values stub-typed as `Any`. Single-line `float(...)`
    # calls keep the laundering scoped; per-line
    # `pyright: ignore[reportAny]` matches `_json_boundary.py`'s
    # pattern.
    beta = xtwx_inv @ xtwy
    y_hat = x_mat @ beta
    residuals = y_vec - y_hat
    weighted_rss = float((w_vec * residuals ** 2).sum())  # pyright: ignore[reportAny]
    df = n - p
    sigma_sq = weighted_rss / df
    cov_beta = sigma_sq * xtwx_inv

    t_crit = float(ss.t.ppf(1.0 - alpha / 2.0, df=df))

    y_mean = float(np.average(y_vec, weights=w_vec))
    weighted_tss = float((w_vec * (y_vec - y_mean) ** 2).sum())  # pyright: ignore[reportAny]
    r_squared = (
        1.0 - weighted_rss / weighted_tss
        if weighted_tss > 0.0 else float('nan')
    )

    coefficients: list[CovariateCoefficient] = []
    for j, name in enumerate(covariate_names):
        idx = j + 1
        b = float(beta[idx])  # pyright: ignore[reportAny]
        var_b = float(cov_beta[idx, idx])  # pyright: ignore[reportAny]
        se_b = math.sqrt(var_b) if var_b > 0.0 else 0.0
        margin = t_crit * se_b
        t_stat = b / se_b if se_b > 0.0 else float('inf')
        p_val = float(2.0 * (1.0 - ss.t.cdf(abs(t_stat), df=df)))
        coefficients.append(CovariateCoefficient(
            name=name,
            coefficient=b,
            ci_lo=b - margin,
            ci_hi=b + margin,
            p_value=p_val,
            is_significant=(p_val < alpha),
        ))

    return MetaRegressionResult(
        n_strata=n,
        intercept=float(beta[0]),  # pyright: ignore[reportAny]
        coefficients=tuple(coefficients),
        r_squared=r_squared,
    )


def meta_regress_comparison(
    row: 'HypothesisComparisonRow',
    covariate_for: Callable[[object], Mapping[str, float]],
    *,
    alpha: float = 0.05,
) -> MetaRegressionResult:
    """Run meta-regression on a stratified HypothesisComparisonRow.

    Builds one `StratumObservation` per `GroupStats` in
    `row.per_group`, with `g` from `gs.effect_size_g`, `se` from
    `gs.se`, and `covariates` from `covariate_for(gs.group_value)`.

    Strata where `effect_size_g` or `se` is `None`/NaN are
    silently skipped — those are degenerate strata that the
    aggregation already flagged as unsuitable for pooling. The
    skip is silent because the caller already saw the per-group
    verdicts; a separate filter at this boundary would be a
    second voice on the same call.

    Raises:
    - `ValueError` when the row's `per_group` is empty (caller
      passed a non-stratified row).
    - `ValueError` from `meta_regression` when remaining strata
      can't satisfy OLS preconditions."""
    if not row.per_group:
        raise ValueError(
            'meta_regress_comparison: row.per_group is empty — '
            'this row is not stratified. Re-aggregate with '
            'group_by set, or use meta_regression directly.',
        )
    observations: list[StratumObservation] = []
    for gs in row.per_group:
        if gs.effect_size_g is None or gs.se is None:
            continue
        if math.isnan(gs.effect_size_g) or math.isnan(gs.se):
            continue
        if gs.se <= 0.0:
            continue
        observations.append(StratumObservation(
            stratum_id=gs.group_value,
            g=gs.effect_size_g,
            se=gs.se,
            covariates=covariate_for(gs.group_value),
        ))
    return meta_regression(observations, alpha=alpha)


# ============ Cross-validation ============

@dataclass(frozen=True, slots=True)
class FoldResult:
    """One fold's regression fit. Carries the fit's coefficients
    and intercept so cross-fold stability can be assessed
    coefficient-by-coefficient."""
    fold_index: int
    n_train: int
    n_test: int
    intercept: float
    coefficients: tuple[CovariateCoefficient, ...]
    r_squared: float


@dataclass(frozen=True, slots=True)
class CrossValResult:
    """Aggregate cross-validation result.

    `sign_consistency[name]` is the fraction of folds where the
    coefficient's sign matches the modal sign across folds — a
    coefficient that flips sign on different splits is unstable
    and the cleavage claim it would otherwise license is fragile.

    `coefficient_stability[name]` is `(mean, std)` of the
    coefficient across folds; small std relative to |mean| is
    Phase D's robustness signal."""
    n_folds: int
    per_fold: tuple[FoldResult, ...]
    sign_consistency: Mapping[str, float]
    coefficient_stability: Mapping[str, tuple[float, float]]


def _sign(x: float) -> int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def cross_validate_meta_regression(
    observations: Sequence[StratumObservation],
    *,
    k_folds: int = 5,
    alpha: float = 0.05,
    seed: int = 0,
) -> CrossValResult:
    """k-fold cross-validation of the meta-regression coefficients.

    Randomly partitions `observations` into `k_folds` roughly
    equal slices (deterministic given `seed`); on each fold,
    fits `meta_regression` on the remaining `(k-1)/k`. Records
    each fold's coefficient table and aggregates the per-
    coefficient sign-consistency + (mean, std) across folds.

    A coefficient with `sign_consistency == 1.0` voted the same
    sign on every fold — that's the robustness signal Phase D
    needs to defend §7.7 caveat 4 ("rule learned, not validated
    on held-out envs"). Sign consistency below ~0.7 means the
    cleavage axis depends on which strata are in the training
    set — a fragile claim.

    Raises:
    - `ValueError` when `k_folds < 2` or `k_folds > n`.
    - `ValueError` from `meta_regression` when any fold's
      training set fails its OLS preconditions (insufficient n
      vs covariates, collinear covariates, etc.)."""
    n = len(observations)
    if k_folds < 2:
        raise ValueError(
            f'cross_validate_meta_regression: k_folds must be ≥ 2, '
            f'got {k_folds}',
        )
    if k_folds > n:
        raise ValueError(
            f'cross_validate_meta_regression: k_folds ({k_folds}) '
            f'cannot exceed n_observations ({n})',
        )

    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    fold_assignments = [i % k_folds for i in indices]
    by_fold: list[list[int]] = [[] for _ in range(k_folds)]
    for orig_idx, fold in zip(indices, fold_assignments):
        by_fold[fold].append(orig_idx)

    per_fold: list[FoldResult] = []
    for fold_index in range(k_folds):
        test_indices = set(by_fold[fold_index])
        train = [
            obs for i, obs in enumerate(observations)
            if i not in test_indices
        ]
        result = meta_regression(train, alpha=alpha)
        per_fold.append(FoldResult(
            fold_index=fold_index,
            n_train=len(train),
            n_test=len(test_indices),
            intercept=result.intercept,
            coefficients=result.coefficients,
            r_squared=result.r_squared,
        ))

    coef_names: tuple[str, ...] = (
        tuple(c.name for c in per_fold[0].coefficients)
        if per_fold and per_fold[0].coefficients
        else ()
    )

    sign_consistency: dict[str, float] = {}
    coefficient_stability: dict[str, tuple[float, float]] = {}
    for name in coef_names:
        values = [
            next(c for c in fold.coefficients if c.name == name).coefficient
            for fold in per_fold
        ]
        signs = [_sign(v) for v in values]
        if signs:
            modal_sign = max(set(signs), key=signs.count)
            sign_consistency[name] = (
                signs.count(modal_sign) / len(signs)
            )
        else:
            sign_consistency[name] = float('nan')
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        coefficient_stability[name] = (mean, math.sqrt(var))

    return CrossValResult(
        n_folds=k_folds,
        per_fold=tuple(per_fold),
        sign_consistency=sign_consistency,
        coefficient_stability=coefficient_stability,
    )
