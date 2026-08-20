"""Closed-form assertions on `cross_env_sign_alignment_binomial`.

Under the LG-SCM, Δ_y = β_zy · (β_xz_t − β_xz_b) · μ_x and
Δ_z = (β_xz_t − β_xz_b) · μ_x. So sign(Δ_y) ≡ sign(Δ_z) (both
controlled by Δβ, β_zy > 0). Constructing envs with mixed sign
of Δβ produces a panel where Δ_y and Δ_z sign-align at every env
(SAME direction).

Tests:
  1. 10 envs, all Δ_y / Δ_z same-direction → SAME alignment HELDs
     at p = 0.5^10 = 0.0010.
  2. 10 envs, all opposite-direction (constructed via flipped
     β_zy at half of envs) — but LG-SCM doesn't natively support
     this; we test 'opposite' via a different measurable
     construction (z and -y proxy via β_zy sign).
  3. null_floor filters out near-zero |d| strata.
  4. Insufficient n → p = NaN.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.analyses.panel.cross_env_sign_alignment_binomial import (
    cross_env_sign_alignment_binomial,
)

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_multi_env_paired_arms
from corroborate.data import cells_to_dataframe


_MU_X = 1.0
_SIGMA_X = 0.5
_SIGMA_Z = 0.1
_BETA_ZY = 1.5
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_SEEDS_PER_ARM = 30


def _scm(beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=_MU_X, sigma_x=_SIGMA_X,
        beta_xz=beta_xz, sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


# 10 envs, all treatment β > baseline β → Δ_z > 0 AND Δ_y > 0
# at every env. Sign-alignment SAME → 10/10.
_SAME_DIRECTION_ENVS: Mapping[str, tuple[float, float]] = {
    f'env_{i:02d}': (0.7 + 0.02 * i, 0.3 + 0.01 * i)
    for i in range(10)
}


def _build_cells(
    env_betas: Mapping[str, tuple[float, float]],
) -> list[Mapping[str, object]]:
    envs = {n: (_scm(t), _scm(b)) for n, (t, b) in env_betas.items()}
    rows = run_multi_env_paired_arms(
        envs=envs, seeds=tuple(range(_N_SEEDS_PER_ARM)),
    )
    return [r.as_dict() for r in rows]


def test_sign_alignment_same_direction_at_every_env() -> None:
    """All 10 envs have β_xz_t > β_xz_b → Δ_z > 0 and Δ_y > 0
    at every env (β_zy > 0). Sign-alignment SAME holds at 10/10
    → one-tailed binomial p = 0.5**10 = 0.000977. SUPPORTED."""
    cells = _build_cells(_SAME_DIRECTION_ENVS)
    result = cross_env_sign_alignment_binomial.fn(
        cells_to_dataframe(cells),
        source_x='z_mean',
        source_y='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
        alignment='same',
        scope_predictor='y_mean',
        min_baseline_predictor=-1e9,
    )
    assert result.n_strata_total == 10, (
        f'expected 10 strata, got {result.n_strata_total}'
    )
    assert result.n_strata_aligned == 10
    expected_p = 0.5 ** 10
    assert abs(result.p_value - expected_p) < 1e-9, (
        f'p={result.p_value}, expected {expected_p}'
    )
    # All d values should be positive
    for d_x, d_y in zip(
        result.cohen_d_x_per_stratum,
        result.cohen_d_y_per_stratum, strict=True,
    ):
        assert d_x > 0, f'd_x={d_x} should be positive'
        assert d_y > 0, f'd_y={d_y} should be positive'


def test_sign_alignment_opposite_direction_rejects() -> None:
    """Same data, but predict OPPOSITE direction → 0/10 aligned
    → binomial p = 1.0 (no evidence of opposite alignment;
    actually evidence of SAME alignment but we asked the wrong
    question)."""
    cells = _build_cells(_SAME_DIRECTION_ENVS)
    result = cross_env_sign_alignment_binomial.fn(
        cells_to_dataframe(cells),
        source_x='z_mean',
        source_y='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
        alignment='opposite',  # predicting opposite — wrong
        scope_predictor='y_mean',
        min_baseline_predictor=-1e9,
    )
    assert result.n_strata_total == 10
    assert result.n_strata_aligned == 0
    # P(X >= 0 | n=10, p=0.5) = 1.0 exactly
    assert abs(result.p_value - 1.0) < 1e-9


# Mixed-direction: half envs flip Δβ sign
_MIXED_SIGN_ENVS: Mapping[str, tuple[float, float]] = {
    **{f'pos_{i:02d}': (0.7, 0.3) for i in range(5)},  # Δβ = +0.4
    **{f'neg_{i:02d}': (0.3, 0.7) for i in range(5)},  # Δβ = -0.4
}


def test_sign_alignment_mixed_direction_same_at_every_env() -> None:
    """Mixed-sign envs: Δ_z and Δ_y flip sign together at every
    env (β_zy > 0 globally → sign(Δ_y) = sign(Δ_z) regardless of
    Δβ direction). SAME alignment HELDs at 10/10."""
    cells = _build_cells(_MIXED_SIGN_ENVS)
    result = cross_env_sign_alignment_binomial.fn(
        cells_to_dataframe(cells),
        source_x='z_mean',
        source_y='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
        alignment='same',
        scope_predictor='y_mean',
        min_baseline_predictor=-1e9,
    )
    assert result.n_strata_total == 10
    assert result.n_strata_aligned == 10
    expected_p = 0.5 ** 10
    assert abs(result.p_value - expected_p) < 1e-9


def test_sign_alignment_null_floor_drops_small_d() -> None:
    """With a very large null_floor on x, every stratum drops out
    → n_total = 0 → p = NaN."""
    cells = _build_cells(_SAME_DIRECTION_ENVS)
    result = cross_env_sign_alignment_binomial.fn(
        cells_to_dataframe(cells),
        source_x='z_mean',
        source_y='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
        alignment='same',
        null_floor_x=1000.0,
        scope_predictor='y_mean',
        min_baseline_predictor=-1e9,
    )
    assert result.n_strata_total == 0
    assert math.isnan(result.p_value)
