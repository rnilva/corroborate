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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
import scipy.stats as ss

type Pool = Literal['fixed', 'random']
"""Pooling regime. `'fixed'` weighs strata by within-stratum
precision only (w_i = 1/v_i). `'random'` adds between-stratum
heterogeneity τ² to the variance (w_i = 1/(v_i + τ²)) — the
honest treatment when strata draw from a population of effects
rather than realizing one shared effect."""


class StratumGProtocol[K](Protocol):
    """Structural protocol for per-stratum (g, se, n_pairs) records.
    `StratumG[K]` (in `corroborate.stratum`) satisfies it; anything
    carrying the four read-only fields works.

    Fields are `@property` (not bare attrs) per CLAUDE.md's
    typing-discipline rule: writable Protocol fields don't match
    immutable concrete fields (frozen-dataclass instance attrs).
    `StratumG` is `frozen=True`; the Protocol must mirror that."""
    @property
    def stratum_id(self) -> K: ...
    @property
    def g(self) -> float: ...
    @property
    def se(self) -> float: ...
    @property
    def n_pairs(self) -> int: ...


@dataclass(frozen=True, slots=True)
class StratumObservation:
    """One stratum's observation: an effect size with its standard
    error and a flat covariate vector.

    `g` and `se` typically come from per-stratum aggregation in
    `PairedComparisonResult.per_group`; `covariates` is a flat
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
    covariates — the empirical scope claim's content.

    Heterogeneity fields (`tau_sq`, `q_statistic`, `i_squared`,
    `pool`) carry the random-effects diagnostics. With an
    intercept-only fit (no covariates), the intercept IS the
    pooled "total mean from population means" — its
    `intercept_se` / `intercept_ci_*` reflect the chosen
    `pool` regime (random-effects widens the CI proportionally
    to τ²). `tau_sq=0` and `i_squared=0` when between-stratum
    heterogeneity is undetectable; `i_squared` near 1.0 means
    almost all variance is between-stratum (the scope-flag
    signal — effects vary with regime more than within)."""
    n_strata: int
    intercept: float
    coefficients: tuple[CovariateCoefficient, ...]
    r_squared: float
    intercept_se: float = 0.0
    intercept_ci_lo: float = 0.0
    intercept_ci_hi: float = 0.0
    intercept_p_value: float = float('nan')
    tau_sq: float = 0.0
    q_statistic: float = 0.0
    i_squared: float = 0.0
    pool: Pool = 'fixed'

    @property
    def cleavage_axes(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.coefficients if c.is_significant)


def _fit_wls(
    x_mat: npt.NDArray[np.float64],
    y_vec: npt.NDArray[np.float64],
    w_vec: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float]:
    """Weighted-least-squares core. Returns
    `(beta, xtwx_inv, weighted_rss)`. Raises `np.linalg.LinAlgError`
    when the weighted normal equations are singular.

    Return uses `np.floating` (not `np.float64`) because numpy stubs
    type `np.linalg.inv` as returning the broader `floating` dtype.
    Downstream consumers narrow back to `float` via `.item(...)`."""
    xtw = x_mat.T * w_vec  # (p, n) — broadcast diag(w) without materialising
    xtwx = xtw @ x_mat
    xtwy = xtw @ y_vec
    xtwx_inv = np.linalg.inv(xtwx)
    beta = xtwx_inv @ xtwy
    residuals = y_vec - x_mat @ beta
    weighted_rss = float(np.sum(w_vec * residuals ** 2))
    return beta, xtwx_inv, weighted_rss


def _dl_tau_sq(
    x_mat: npt.NDArray[np.float64],
    w_fe: npt.NDArray[np.float64],
    q_statistic: float, df: int,
) -> float:
    """DerSimonian-Laird τ² estimator from FE residuals.

    Uses the matrix form τ² = max(0, (Q − df) / c) where
    c = tr(W) − tr((X'WX)⁻¹ X'W²X). Reduces to the textbook
    intercept-only formula c = Σwᵢ − Σwᵢ²/Σwᵢ when X is the
    ones-column. Returns 0 when df ≤ 0 or c ≤ 0 — both cases
    indicate the data carries no information about
    between-stratum heterogeneity beyond the within-stratum
    sampling variance."""
    if df <= 0:
        return 0.0
    xtwx = (x_mat.T * w_fe) @ x_mat
    xtw2x = (x_mat.T * w_fe ** 2) @ x_mat
    try:
        # `np.sum(np.diagonal(...))` rather than `np.trace(...)`: trace
        # returns Any in numpy stubs (overloaded; can't resolve dtype),
        # the diagonal-sum form preserves dtype through both calls.
        c_factor = float(np.sum(w_fe)) - float(
            np.sum(np.diagonal(np.linalg.solve(xtwx, xtw2x))),
        )
    except np.linalg.LinAlgError:
        return 0.0
    if c_factor <= 0.0 or q_statistic <= df:
        return 0.0
    return (q_statistic - df) / c_factor


def meta_regression(
    observations: Sequence[StratumObservation],
    *,
    alpha: float = 0.05,
    pool: Pool = 'random',
) -> MetaRegressionResult:
    """Inverse-variance-weighted least-squares regression of
    per-stratum effect sizes on covariates.

    Each observation contributes one row: outcome `g`, weight
    `1/(vᵢ + τ²)` (`vᵢ = seᵢ²`), and a covariate vector. The fit
    minimises `Σ wᵢ (gᵢ − ŷᵢ)²` where `ŷᵢ = β₀ + Σⱼ βⱼ xᵢⱼ`. CIs
    use the weighted-residual covariance matrix and a
    t-distribution with `n − p` degrees of freedom
    (p = 1 + n_covariates).

    `pool='random'` (default) estimates τ² (DerSimonian-Laird)
    from the FE-fit residuals and refits with RE weights. The
    intercept-only fit is the random-effects pooled mean — the
    "total mean from population means" answer. `pool='fixed'`
    leaves wᵢ = 1/vᵢ; τ² and q_statistic are still computed and
    reported on the result for inspection.

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
    v_vec = np.zeros(n, dtype=np.float64)
    for i, obs in enumerate(observations):
        y_vec[i] = obs.g
        v_vec[i] = obs.se ** 2
        for j, name in enumerate(covariate_names):
            x_mat[i, j + 1] = obs.covariates.get(name, 0.0)

    df = n - p
    w_fe = 1.0 / v_vec
    try:
        _, _, q_statistic = _fit_wls(x_mat, y_vec, w_fe)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f'meta_regression: design matrix singular ({e}); '
            f'covariates may be collinear',
        ) from e

    tau_sq = _dl_tau_sq(x_mat, w_fe, q_statistic, df)
    i_squared = (
        max(0.0, 1.0 - df / q_statistic) if q_statistic > 0.0 else 0.0
    )

    if pool == 'random' and tau_sq > 0.0:
        w_final = 1.0 / (v_vec + tau_sq)
    else:
        w_final = w_fe
    beta, xtwx_inv, weighted_rss = _fit_wls(x_mat, y_vec, w_final)

    sigma_sq = weighted_rss / df
    cov_beta = sigma_sq * xtwx_inv
    t_crit = float(ss.t.ppf(1.0 - alpha / 2.0, df=df))

    y_mean = float(np.average(y_vec, weights=w_final))
    weighted_tss = float(np.sum(w_final * (y_vec - y_mean) ** 2))
    r_squared = (
        1.0 - weighted_rss / weighted_tss
        if weighted_tss > 0.0 else float('nan')
    )

    intercept = beta.item(0)
    intercept_var = cov_beta.item(0, 0)
    intercept_se = math.sqrt(intercept_var) if intercept_var > 0.0 else 0.0
    intercept_margin = t_crit * intercept_se
    if intercept_se > 0.0:
        intercept_t = abs(intercept) / intercept_se
        intercept_p = float(2.0 * (1.0 - ss.t.cdf(intercept_t, df=df)))
    else:
        intercept_p = float('nan')

    coefficients: list[CovariateCoefficient] = []
    for j, name in enumerate(covariate_names):
        idx = j + 1
        b = beta.item(idx)
        var_b = cov_beta.item(idx, idx)
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
        intercept=intercept,
        coefficients=tuple(coefficients),
        r_squared=r_squared,
        intercept_se=intercept_se,
        intercept_ci_lo=intercept - intercept_margin,
        intercept_ci_hi=intercept + intercept_margin,
        intercept_p_value=intercept_p,
        tau_sq=tau_sq,
        q_statistic=q_statistic,
        i_squared=i_squared,
        pool=pool,
    )


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


# ============ Panel → meta-regression bridge ============

def meta_regress_panel[K](
    panel: Sequence[StratumGProtocol[K]],
    *,
    covariates_per_stratum: Mapping[K, Mapping[str, float]],
    alpha: float = 0.05,
    pool: Pool = 'random',
) -> MetaRegressionResult:
    """Project a per-stratum panel of (stratum_id, g, se, n_pairs)
    observations to `StratumObservation`s and run `meta_regression`.

    Both `meta_regression_paired_g` (per-env panel,
    `K = str`) and `meta_regression_per_burst` (per-(env, burst)
    panel, `K = tuple[str, int]`) consume this helper — the
    panel→regression projection is identical apart from the
    stratum-id type.

    Strata with `n_pairs < 2`, NaN g/se, or `se <= 0.0` drop
    silently. Covariate lookup falls back to an empty mapping
    when a stratum-id is absent from `covariates_per_stratum`."""
    observations: list[StratumObservation] = []
    for s in panel:
        if s.n_pairs < 2 or math.isnan(s.g) or math.isnan(s.se):
            continue
        if s.se <= 0.0:
            continue
        observations.append(StratumObservation(
            stratum_id=s.stratum_id,
            g=s.g,
            se=s.se,
            covariates=covariates_per_stratum.get(s.stratum_id, {}),
        ))
    return meta_regression(observations, alpha=alpha, pool=pool)
