"""Closed-form assertions on `classify_variable_scope` and
`assert_stratification_admissible` over LG-SCM cells.

The framework's variable-scope primitive distinguishes:

- `WITHIN_STRATUM`: variance lives within strata. Stratified
  analyses (JCI partial-Spearman, stratified ρ) compatible.
- `ACROSS_STRATUM`: variance lives between strata, constant
  within. Stratified analyses silently return NaN; framework
  must refuse them.
- `BOTH`: variance in both directions. Either analysis applicable.
- `DEGENERATE`: no variance anywhere. Useless for any test.

The auto-memory `framework_variable_scope` calls this out as the
codification of the within-vs-across-stratum admissibility rule —
exercised here on a multi-env LG-SCM corpus where each cell-level
column has a known scope under `stratify_by='env_name'`:

- `mu_x` — env-level structural parameter, constant within env →
  `ACROSS_STRATUM`
- `seed` — uniform seed range per env (0..29 in every env) →
  `WITHIN_STRATUM`
- a constant column (`n_steps`) → `DEGENERATE`
- synthetic `env_index_plus_seed_noise` (env_index + per-cell
  uniform noise) → `BOTH`

The closed-form scopes are determined entirely by the data
construction, not by the implementation's structural arrows. A
regression in the variance decomposition or threshold logic
would surface immediately on these mechanical examples.

`assert_stratification_admissible` should raise on any
`ACROSS_STRATUM` or `DEGENERATE` column — the closed-form scope
classification IS the admissibility check."""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from corroborate.corpus.schema import MeasurementLeaf, RunRow
from corroborate.graph.discovery import (
    VariableScope,
    assert_stratification_admissible,
    classify_variable_scope,
)

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_arm


_SIGMA_X = 0.5
_BETA_XZ = 0.5
_BETA_ZY = 1.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_PER_ENV = 30
_MU_X_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)
_BOTH_COLUMN = 'env_index_plus_seed_noise'


def _scm(*, mu_x: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x, sigma_x=_SIGMA_X,
        beta_xz=_BETA_XZ, sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _augment_with_both_column(
    rows: list[RunRow], env_index: int,
) -> list[RunRow]:
    """Add a synthetic `env_index_plus_seed_noise` column engineered
    to have BOTH within-env (uniform per-cell noise) and between-env
    (env_index offset) variance, both above the 5% relative
    threshold. Variance match is approximate — uniform[0, 5] within
    env (var ≈ 2.08) vs env_index ∈ {0, 1, 2, 3, 4} (var = 2.0)."""
    seeded_rng = random.Random(env_index * 12345)
    out: list[RunRow] = []
    for r in rows:
        m: dict[str, MeasurementLeaf] = dict(r.measurements)
        m[_BOTH_COLUMN] = float(env_index) + seeded_rng.uniform(0.0, 5.0)
        out.append(replace(r, measurements=m))
    return out


def _build_multi_env_corpus() -> list[Mapping[str, object]]:
    """Single-arm multi-env corpus. Uniform seeds per env (0..29
    in every env, no env-specific offset) so `seed` has variance
    within env but constant mean across envs → WITHIN_STRATUM.
    `mu_x` varies between envs only → ACROSS_STRATUM. A synthetic
    `env_index_plus_seed_noise` column carries both axes → BOTH.
    """
    rows: list[RunRow] = []
    for env_index, mu in enumerate(_MU_X_GRID):
        env_rows = run_arm(
            _scm(mu_x=mu),
            seeds=range(_N_PER_ENV),  # uniform 0..29, no env offset
            arm_key='single',
            env_name=f'env_mu_{mu:g}',
        )
        rows.extend(_augment_with_both_column(env_rows, env_index))
    return [r.as_dict() for r in rows]


def _column(
    cells: Sequence[Mapping[str, object]], key: str,
) -> tuple[np.ndarray, list[object]]:
    """Project (key, env_name) columns from cells. Returns
    (values_as_float_array, env_strata_list)."""
    values: list[float] = []
    strata: list[object] = []
    for c in cells:
        v = c.get(key)
        env = c.get('env_name')
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if not isinstance(env, str):
            continue
        values.append(float(v))
        strata.append(env)
    return np.asarray(values, dtype=np.float64), strata


# ============ Per-column scope ============

def test_env_level_parameter_classifies_as_across_stratum() -> None:
    """`mu_x` is set per-env and constant across all 30 seeds in
    that env. Variance lives entirely between envs — the canonical
    `ACROSS_STRATUM` signature. Stratified analyses on `mu_x` would
    get zero within-stratum variance and silently skip."""
    cells = _build_multi_env_corpus()
    arr, strata = _column(cells, 'mu_x')
    scope = classify_variable_scope(arr, strata)
    assert scope is VariableScope.ACROSS_STRATUM, (
        f'mu_x scope = {scope.value!r}; expected ACROSS_STRATUM '
        f'(constant within env, varies between envs)'
    )


def test_uniform_seed_per_env_classifies_as_within_stratum() -> None:
    """Each env uses seeds 0..29; per-env mean is identical. So:
    - within-env variance: full 0..29 range
    - between-env variance: zero (all means equal)
    → WITHIN_STRATUM. The canonical case for stratified primitives."""
    cells = _build_multi_env_corpus()
    arr, strata = _column(cells, 'seed')
    scope = classify_variable_scope(arr, strata)
    assert scope is VariableScope.WITHIN_STRATUM, (
        f'seed scope = {scope.value!r}; expected WITHIN_STRATUM '
        f'(uniform 0..29 per env, identical mean across envs)'
    )


def test_synthetic_both_column_classifies_as_both() -> None:
    """`env_index_plus_seed_noise` mixes per-cell uniform noise
    (within-env variance ≈ 2.08) and env_index offset (between-env
    variance = 2.0). Both above the 0.05 relative threshold → BOTH.

    A regression that mis-decomposed the variance (e.g., conflated
    within-stratum with total variance) would mis-classify this
    column."""
    cells = _build_multi_env_corpus()
    arr, strata = _column(cells, _BOTH_COLUMN)
    scope = classify_variable_scope(arr, strata)
    assert scope is VariableScope.BOTH, (
        f'{_BOTH_COLUMN} scope = {scope.value!r}; expected BOTH '
        f'(within: uniform per-cell noise; between: env_index offset)'
    )


def test_constant_column_classifies_as_degenerate() -> None:
    """`n_steps` is set identically on every cell — no variance
    anywhere. The framework must classify this as DEGENERATE
    (and `assert_stratification_admissible` must refuse to use it)."""
    cells = _build_multi_env_corpus()
    arr, strata = _column(cells, 'n_steps')
    scope = classify_variable_scope(arr, strata)
    assert scope is VariableScope.DEGENERATE, (
        f'n_steps scope = {scope.value!r}; expected DEGENERATE '
        f'(constant across the corpus by construction)'
    )


# ============ assert_stratification_admissible ============

def test_admissibility_passes_on_within_stratum_variables() -> None:
    """`seed` has within-stratum variance (uniform 0..29 per env).
    `assert_stratification_admissible` should pass and report the
    scope classification."""
    cells = _build_multi_env_corpus()
    df = pl.DataFrame([dict(c) for c in cells])
    scopes = assert_stratification_admissible(
        df, variables=['seed'], stratify_by='env_name',
    )
    assert scopes['seed'] is VariableScope.WITHIN_STRATUM


def test_admissibility_raises_on_across_stratum_variable() -> None:
    """`mu_x` has zero within-env variance. A stratified analysis
    on `mu_x` would silently return NaN per stratum.
    `assert_stratification_admissible` must refuse explicitly,
    raising ValueError that names the offending column AND its
    scope.

    This is the canonical "framework refuses the silent failure"
    case. Without it, JCI / partial-Spearman / stratified-ρ on
    env-level features silently lose power."""
    cells = _build_multi_env_corpus()
    df = pl.DataFrame([dict(c) for c in cells])
    with pytest.raises(ValueError) as exc_info:
        _ = assert_stratification_admissible(
            df, variables=['mu_x'], stratify_by='env_name',
        )
    msg = str(exc_info.value)
    assert 'mu_x' in msg, (
        f'error message must name the offending variable; got: {msg}'
    )
    assert (
        'across_stratum' in msg.lower() or 'across' in msg.lower()
    ), (
        f"error message must name the scope class; got: {msg}"
    )


def test_admissibility_message_uses_comma_separator_for_multiple() -> None:
    """When multiple variables are blocked, the message joins
    them with `', '`. Pin against `'XX, XX'.join(...)` mutant
    that would emit a mangled separator."""
    cells = _build_multi_env_corpus()
    df = pl.DataFrame([dict(c) for c in cells])
    with pytest.raises(ValueError) as exc_info:
        _ = assert_stratification_admissible(
            df, variables=['mu_x', 'beta_xz'],
            stratify_by='env_name',
        )
    msg = str(exc_info.value)
    # Both blocked variables present.
    assert 'mu_x' in msg
    assert 'beta_xz' in msg
    # The literal separator ', ' connects them in the details.
    assert 'XX' not in msg, (
        f'separator should be ", " not contain "XX"; got: {msg}'
    )


def test_admissibility_raises_on_degenerate_variable() -> None:
    """A constant column (no variance anywhere) is unusable for
    any analysis. The framework must refuse it loudly rather than
    silently producing NaN downstream."""
    cells = _build_multi_env_corpus()
    df = pl.DataFrame([dict(c) for c in cells])
    with pytest.raises(ValueError) as exc_info:
        _ = assert_stratification_admissible(
            df, variables=['n_steps'], stratify_by='env_name',
        )
    msg = str(exc_info.value)
    assert 'n_steps' in msg
    assert 'degenerate' in msg.lower(), (
        f"error message must name 'degenerate' scope; got: {msg}"
    )


def test_admissibility_collects_all_offenders_in_one_error() -> None:
    """When MULTIPLE variables are inadmissible, the error should
    name all of them in one message rather than raising on the
    first one. This makes upstream debugging tractable on a corpus
    with multiple legacy env-level columns."""
    cells = _build_multi_env_corpus()
    df = pl.DataFrame([dict(c) for c in cells])
    with pytest.raises(ValueError) as exc_info:
        _ = assert_stratification_admissible(
            df,
            variables=['mu_x', 'n_steps', 'sigma_x'],
            stratify_by='env_name',
        )
    msg = str(exc_info.value)
    # All three columns are env-level (or constant) and should
    # appear in the consolidated error.
    assert 'mu_x' in msg
    assert 'n_steps' in msg
    assert 'sigma_x' in msg, (
        f'error message must name all offenders; got: {msg}'
    )
