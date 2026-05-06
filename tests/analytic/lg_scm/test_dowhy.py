"""Closed-form assertions on `backdoor_ate` / `placebo_refutation`
/ `random_common_cause_refutation` over a multi-env LG-SCM corpus.

The structural truth: per-cell `y_mean ≈ beta_xz * beta_zy * x_mean`
under the X → Z → Y SCM with shared structural coefficients across
all cells. With `beta_xz = 0.5` and `beta_zy = 1.5`, the total
effect of X on Y is `0.5 * 1.5 = 0.75`.

DoWhy with `backdoor.linear_regression` on the `(x_mean, y_mean)`
columns + DAG `X → Y` (no confounders, no adjustment set required)
must recover this slope. The two refuters then check robustness:

- **Placebo**: replaces treatment with a random permutation. The
  closed-form ATE is structurally tied to X — under permutation
  the link should be destroyed → refuted ATE ≈ 0 → drift ≈ 0.75.
- **Random common cause**: adds a synthetic random confounder
  unrelated to either X or Y. The structural ATE is independent of
  random additions → refuted ATE ≈ real ATE → drift ≈ 0.

Test corpus: 5 envs × 30 seeds × single arm. `mu_x` varies across
envs to give x_mean enough variance for OLS to converge tightly.
n_steps is large (200) so per-cell sampling noise on x_mean and
y_mean is small relative to the between-env signal.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.dowhy import (
    backdoor_ate,
    placebo_refutation,
    random_common_cause_refutation,
)
from corroborate.corpus.schema import RunRow

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_arm


_BETA_XZ = 0.5
_BETA_ZY = 1.5
_SIGMA_X = 0.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_PER_ENV = 30

_MU_X_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)

# Closed-form total effect: dY/dX = beta_xz * beta_zy.
_EXPECTED_ATE = _BETA_XZ * _BETA_ZY  # 0.75

# DAG: simplest form. No confounders → no adjustment set needed.
# DoWhy's linear-regression backdoor reduces to OLS slope of y on x.
_DAG: list[tuple[str, str]] = [('x_mean', 'y_mean')]


def _var_x_across_cells() -> float:
    """Closed-form variance of `x_mean` across all cells in the
    multi-env corpus. Per-cell `x_mean ≈ mu_x_env + N(0, σ_x²/n_steps)`
    so the total variance decomposes into between-env (mu_x grid)
    + within-env (sampling). For our parameters the within term
    is ~σ_x²/n_steps = 0.5²/200 = 1.25e-3 — negligible vs the
    grid variance ~0.5; we add it for completeness."""
    grid_mean = sum(_MU_X_GRID) / len(_MU_X_GRID)
    between_var = sum(
        (m - grid_mean) ** 2 for m in _MU_X_GRID
    ) / len(_MU_X_GRID)
    within_var = (_SIGMA_X ** 2) / _N_STEPS
    return between_var + within_var


def _scm(*, mu_x: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x, sigma_x=_SIGMA_X,
        beta_xz=_BETA_XZ, sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_observational_corpus() -> list[Mapping[str, object]]:
    """Single-arm corpus across 5 envs × N seeds. Each cell
    carries scalar `x_mean`, `z_mean`, `y_mean`. Multi-env spread
    in mu_x widens x_mean's range so OLS has signal to regress
    against. Env-specific seed offsets keep the per-env noise
    streams independent (no replicated sampling pattern across
    envs).
    """
    rows: list[RunRow] = []
    for env_index, mu in enumerate(_MU_X_GRID):
        env_seeds = range(
            env_index * 10000, env_index * 10000 + _N_PER_ENV,
        )
        rows.extend(run_arm(
            _scm(mu_x=mu),
            seeds=env_seeds,
            arm_key='single',
            env_name=f'env_mu_{mu:g}',
        ))
    return [r.as_dict() for r in rows]


def _cells() -> Sequence[Mapping[str, object]]:
    """Module-level memoization for the corpus would tie the test
    file to import-order side effects; building the corpus per
    test is fast (a few hundred SCM evaluations) and keeps the
    state isolated."""
    return _build_observational_corpus()


# ============ backdoor_ate ============

def test_backdoor_ate_returns_unidentified_when_no_directed_path() -> None:
    """When the supplied DAG has no directed path from treatment
    to outcome (e.g., common-cause structure z → x, z → y with
    no x → y), DoWhy reports unidentified. The framework returns
    `BackdoorResult(identified=False, ate=NaN, …)`.

    Pin the unidentified-branch construction:
    - `proceed_when_unidentifiable=False` (vs True / None mutants
      that would request DoWhy continue past unidentified)
    - `getattr(identified, 'no_directed_path', False)` (vs the
      string mutations 'NO_DIRECTED_PATH' / 'XXno_directed_pathXX'
      / default value mutations)

    Construct a corpus with z confounding x and y; pass DAG that
    has the confounder structure but no direct x → y edge."""
    cells = [
        {'x_mean': float(i), 'y_mean': float(i * 2), 'z': 0.0}
        for i in range(20)
    ]
    result = backdoor_ate.fn(
        cells, treatment='x_mean', outcome='y_mean',
        dag=[('z', 'x_mean'), ('z', 'y_mean')],    # no x→y path
    )
    assert result.identified is False
    assert math.isnan(result.ate)
    assert result.n_rows == 20


def test_backdoor_ate_recovers_structural_total_effect() -> None:
    """ATE matches `beta_xz * beta_zy = 0.75` within 5%.

    DoWhy with linear-regression backdoor reduces to OLS slope on
    a DAG with no confounders. With ~150 cells spanning a
    deterministic structural relationship `y = 0.75 * x + small
    noise`, the slope is recovered tightly.

    A regression that passed the wrong DAG, swapped treatment and
    outcome, or missed the linear-regression dispatch would fail
    by orders of magnitude."""
    result = backdoor_ate.fn(
        _cells(),
        treatment='x_mean',
        outcome='y_mean',
        dag=_DAG,
    )
    assert result.identified, (
        f'backdoor_ate did not identify the effect on a DAG '
        f'with no confounders; estimand_str={result.estimand_str!r}'
    )
    assert result.n_rows == len(_MU_X_GRID) * _N_PER_ENV, (
        f'n_rows = {result.n_rows}, expected '
        f'{len(_MU_X_GRID) * _N_PER_ENV}'
    )
    rel_err = abs(result.ate - _EXPECTED_ATE) / _EXPECTED_ATE
    assert rel_err < 0.05, (
        f'ATE = {result.ate:.4f}, expected {_EXPECTED_ATE:.4f} '
        f'(rel err {rel_err:.4f}); a >5% drift indicates the '
        f'estimator is not the textbook linear-regression slope, '
        f'or the cell-DataFrame projection mishandled either '
        f'column'
    )


# ============ placebo refutation ============

def test_placebo_refutation_destroys_structural_estimate() -> None:
    """Permuting the treatment must destroy the structural link.
    Real ATE matches the closed form; refuted ATE collapses
    toward zero (within the analytical SE under the permutation
    null); drift is on the order of |real_ATE|.

    Closed-form SE under permutation null:
        Var(refuted_slope) = Var(Y) / (n · Var(X))
    where Var(Y) ≈ (β_xz·β_zy)² · Var(X) (dominant structural
    term, ignoring within-env σ_z, σ_y noise which are ≪ σ_x in
    our parameters). Var(X) is dominated by the mu_x grid spread.

    A refuter that didn't actually permute (e.g., aliased the
    treatment column) would leave refuted_ate ≈ real_ate ≈ 0.75
    and breach the SE bound by orders of magnitude."""
    result = placebo_refutation.fn(
        _cells(),
        treatment='x_mean',
        outcome='y_mean',
        dag=_DAG,
    )
    # real ATE matches closed form (same path as backdoor_ate).
    rel_err = abs(result.real_ate - _EXPECTED_ATE) / _EXPECTED_ATE
    assert rel_err < 0.05, (
        f'real_ate = {result.real_ate:.4f}, expected '
        f'{_EXPECTED_ATE:.4f} — placebo path must compute the '
        f'same baseline ATE as backdoor_ate'
    )
    # Refuted ATE within 4·SE of zero under the permutation null.
    n_cells = len(_MU_X_GRID) * _N_PER_ENV  # 150
    var_x = _var_x_across_cells()  # mu_x grid variance + within-env
    var_y = (_BETA_XZ * _BETA_ZY) ** 2 * var_x  # dominant term
    se_null = math.sqrt(var_y / (n_cells * var_x))
    bound = 4.0 * se_null
    assert abs(result.refuted_ate) < bound, (
        f'refuted_ate = {result.refuted_ate:.4f}, analytical '
        f'4·SE_null = {bound:.4f} (SE_null = {se_null:.4f}). '
        f'Permuted treatment should yield slope ~ 0; if the '
        f'refuter aliased the column instead of permuting, the '
        f'observed value would be near {_EXPECTED_ATE:.4f}'
    )
    # Drift is on the order of |real_ate|.
    assert result.drift > 0.5 * _EXPECTED_ATE, (
        f'drift = {result.drift:.4f}; expected on the order of '
        f'|real_ate| ≈ {_EXPECTED_ATE:.4f} since the placebo '
        f'destroys the structural signal'
    )
    # Pin success-path metadata: a regression replacing any of
    # method_name / refuter_name / treatment / outcome / n_rows
    # with `None` (or the wrong literal) on the success branch
    # would breach. mutmut surfaces these as a 5-mutant cluster
    # on `_run_refuter`.
    assert result.method_name == 'backdoor.linear_regression'
    assert result.refuter_name == 'placebo_treatment_refuter'
    assert result.treatment == 'x_mean'
    assert result.outcome == 'y_mean'
    assert result.n_rows == n_cells


# ============ random common cause refutation ============

def test_random_common_cause_preserves_structural_estimate() -> None:
    """Adding a synthetic random confounder should NOT shift the
    structural estimate (the synthetic node has no real causal
    relationship to either X or Y). Drift must be small relative
    to |real_ate|.

    A refuter that mishandled the synthetic node (e.g., correlated
    it with treatment by accident) would inflate the drift."""
    result = random_common_cause_refutation.fn(
        _cells(),
        treatment='x_mean',
        outcome='y_mean',
        dag=_DAG,
    )
    rel_err = abs(result.real_ate - _EXPECTED_ATE) / _EXPECTED_ATE
    assert rel_err < 0.05, (
        f'real_ate = {result.real_ate:.4f}, expected '
        f'{_EXPECTED_ATE:.4f}'
    )
    # Drift small — random confounder is by construction
    # uncorrelated with both treatment and outcome, so the
    # ATE estimate should be stable. We require drift < 10%
    # of |real_ate|.
    assert result.drift < 0.1 * _EXPECTED_ATE, (
        f'drift = {result.drift:.4f}; a random common cause '
        f'should leave the ATE estimate stable. Drift > 10% of '
        f'|real_ate| ({_EXPECTED_ATE:.4f}) suggests the synthetic '
        f'confounder leaked into the regression'
    )
    # Pin success-path metadata (parallels placebo test above);
    # `refuter_name` differs between RCC and placebo, so this
    # also catches refuter-name swap mutations.
    assert result.method_name == 'backdoor.linear_regression'
    assert result.refuter_name == 'random_common_cause'
    assert result.treatment == 'x_mean'
    assert result.outcome == 'y_mean'
    assert result.n_rows == len(_MU_X_GRID) * _N_PER_ENV


# ============ Unidentified-branch fallback (refuters) ============

def test_placebo_refutation_returns_nan_when_unidentified() -> None:
    """When `_backdoor_estimate` returns `estimate=None` (no
    directed treatment→outcome path under the supplied DAG),
    `_run_refuter` must short-circuit and emit a `RefutationResult`
    with NaN-filled effect fields and the request metadata
    (method_name, refuter_name, treatment, outcome) preserved.

    Pin the unidentified-branch CONSTRUCTOR — this is the
    `_run_refuter` cluster mutmut surfaced (real_ate=NaN→None,
    refuter_name=refuter_method→None, treatment=treatment→None
    field replacements all survive when no test reaches the
    `if estimate is None` branch).

    Construction: same DAG as `test_backdoor_ate_returns_unidentified
    _when_no_directed_path` — z→x, z→y, no x→y edge.
    """
    cells = [
        {'x_mean': float(i), 'y_mean': float(i * 2), 'z': 0.0}
        for i in range(20)
    ]
    result = placebo_refutation.fn(
        cells, treatment='x_mean', outcome='y_mean',
        dag=[('z', 'x_mean'), ('z', 'y_mean')],
    )
    assert math.isnan(result.real_ate), (
        f'real_ate = {result.real_ate!r}; expected NaN on '
        f'unidentified DAG. A regression replacing the NaN '
        f'fallback with `None` would breach.'
    )
    assert math.isnan(result.refuted_ate)
    assert math.isnan(result.drift)
    # Request metadata must round-trip through the unidentified
    # branch — pins the field replacements (method_name=method_name
    # → None, etc.).
    assert result.method_name == 'backdoor.linear_regression'
    assert result.refuter_name == 'placebo_treatment_refuter'
    assert result.treatment == 'x_mean'
    assert result.outcome == 'y_mean'
    assert result.n_rows == 20


def test_random_common_cause_returns_nan_when_unidentified() -> None:
    """Same fallback behavior as placebo when the DAG fails to
    identify. Distinct test pins that the `refuter_method` constant
    differs between placebo and RCC paths — a regression that
    confused the two refuter names would breach the assertion on
    `refuter_name`."""
    cells = [
        {'x_mean': float(i), 'y_mean': float(i * 2), 'z': 0.0}
        for i in range(20)
    ]
    result = random_common_cause_refutation.fn(
        cells, treatment='x_mean', outcome='y_mean',
        dag=[('z', 'x_mean'), ('z', 'y_mean')],
    )
    assert math.isnan(result.real_ate)
    assert math.isnan(result.refuted_ate)
    assert math.isnan(result.drift)
    assert result.refuter_name == 'random_common_cause'
    assert result.method_name == 'backdoor.linear_regression'
    assert result.treatment == 'x_mean'
    assert result.outcome == 'y_mean'
    assert result.n_rows == 20
