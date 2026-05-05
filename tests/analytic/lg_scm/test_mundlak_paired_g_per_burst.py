"""Closed-form assertions on `mundlak_paired_g_per_burst` over a
multi-env phased LG-SCM corpus.

Mundlak decomposition separates a panel predictor `x` into:

    x_e = E[x_burst | env]    (env-mean — between-env effect)
    x_w = x_burst - x_e        (within-env-deviation — burst-level)

The fit `y_burst ~ beta_b * x_e + beta_w * x_w` is well-determined
because `x_e` and `x_w` are orthogonal by construction.

This test wires a synthetic per-burst predictor that has known
between-env AND within-env structure, so closed-form values for
both `beta_b` and `beta_w` are predictable:

    Per-cell predictor value at burst b:
        x_burst(env, b) = mu_x_env + 0.1 * b

    Per-(env, burst) averaged across baseline cells (constant
    per (env, b) since the predictor depends only on env-level
    mu_x and the burst index):
        x_e(env)     = mu_x_env + 0.1 * mean(b)         (between)
        x_w(env, b)  = 0.1 * (b - mean(b))               (within)

    Per-(env, burst) Hedges' g target (under shared noise):
        y(env, b) = mu_x_env * sqrt(n_steps) / sigma_x * c_4
                   (≈ constant within env; varies linearly between)

Closed-form Mundlak:
    beta_b = sqrt(n_steps) / sigma_x * c_4(n_pairs)
            (since x_e differs from mu_x by a constant offset, the
             slope of y_e on x_e equals the slope of y_e on mu_x)
    beta_w ≈ 0
            (within-env y variance is sampling noise from
             paired_g_per_burst at finite n_pairs, independent of
             the deterministic burst-index predictor)
    hausman_p < alpha
            (beta_b clearly nonzero, beta_w near zero — the two
             channels are statistically distinguishable)

A regression that conflated `beta_b` with `beta_w`, or that failed
to demean the within component, would fail these closed-form
assertions on the structural panel."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from corroborate import measurable
from corroborate.analyses.mundlak_paired_g_per_burst import (
    mundlak_paired_g_per_burst,
)
from corroborate.measurables.reductions import from_key, reduce_axis

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import (
    PER_BURST_Y_KEY,
    run_multi_env_paired_phased_arms,
)


_SIGMA_X = 0.5
_BETA_ZY = 1.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_PAIRS = 30
_N_BURSTS = 4
_BURST_SLOPE = 0.1

_BETA_XZ_BASE = 0.3
_BETA_XZ_TREAT = 0.8

_MU_X_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)


_PER_BURST_Y_MEAN = reduce_axis(
    from_key(PER_BURST_Y_KEY), axis=-1, op='mean',
)


# ============ Synthetic per-burst predictor ============

@measurable
def _mundlak_lg_scm_predictor(
    record: Mapping[str, object],
) -> npt.NDArray[np.floating]:
    """Per-burst array `[mu_x + 0.1*0, mu_x + 0.1*1, ...]`.

    Designed so:
    - env-mean (between component) tracks `mu_x` linearly with
      offset = 0.1 * mean(burst_index).
    - within-deviation (within component) is the deterministic
      sequence `0.1 * (b - mean(b))` — orthogonal to mu_x by
      construction.

    Registered globally via @measurable at module import. The
    `_lg_scm` suffix avoids collision with other substrate-defined
    predictors that might share namespace at registration time."""
    mu_x = record.get('mu_x')
    n_bursts_v = record.get('n_bursts')
    if not isinstance(mu_x, (int, float)) or isinstance(mu_x, bool):
        return np.array([], dtype=np.float64)
    if not isinstance(n_bursts_v, int) or isinstance(n_bursts_v, bool):
        return np.array([], dtype=np.float64)
    return np.array(
        [float(mu_x) + _BURST_SLOPE * b for b in range(n_bursts_v)],
        dtype=np.float64,
    )


def _scm(*, mu_x: float, beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x, sigma_x=_SIGMA_X,
        beta_xz=beta_xz, sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_phased_panel() -> list[Mapping[str, object]]:
    """Build a multi-env phased corpus, env-specific seed offsets.

    Each env runs on a disjoint seed range (`env_index * 10000 +
    [0..n_pairs)`) so the noise streams across envs are
    independent. Without this, every env shares the same RNG
    stream → burst-index sampling pattern is replicated across
    envs → spurious within-env Mundlak signal. Within-env pairing
    still works because treatment and baseline share their env's
    seed range."""
    cells: list[Mapping[str, object]] = []
    for env_index, mu in enumerate(_MU_X_GRID):
        treatments = tuple(
            _scm(mu_x=mu, beta_xz=_BETA_XZ_TREAT) for _ in range(_N_BURSTS)
        )
        baselines = tuple(
            _scm(mu_x=mu, beta_xz=_BETA_XZ_BASE) for _ in range(_N_BURSTS)
        )
        env_seeds = range(env_index * 10000, env_index * 10000 + _N_PAIRS)
        cells.extend(run_multi_env_paired_phased_arms(
            envs={f'env_mu_{mu:g}': (treatments, baselines)},
            seeds=env_seeds,
        ))
    return cells


def _expected_beta_b(*, n_pairs: int) -> float:
    """Closed-form between-env slope.

    y_e(env)  = mu_x * sqrt(n_steps) / sigma_x * c_4
    x_e(env)  = mu_x + 0.1 * mean_burst  (linear in mu_x, slope 1)

    So beta_b (slope of y_e on x_e) equals
        sqrt(n_steps) / sigma_x * c_4(n_pairs).
    """
    c4 = 1.0 - 3.0 / (4 * n_pairs - 5)
    return math.sqrt(_N_STEPS) / _SIGMA_X * c4


# ============ Test ============

def test_mundlak_recovers_between_slope_and_zero_within() -> None:
    """beta_b matches the closed-form `sqrt(n)/sigma * c_4`,
    beta_w is statistically indistinguishable from zero, and the
    Hausman test rejects beta_b == beta_w (the two channels are
    structurally distinct).

    The substrate sets up:
    - target g varies BETWEEN env (with mu_x), constant within env
    - predictor x varies BETWEEN env (with mu_x) AND within env
      (with burst index)
    Mundlak should pick up the between covariation, ignore the
    within (which is independent of the constant within-env y)."""
    cells = _build_phased_panel()
    result = mundlak_paired_g_per_burst.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source=_PER_BURST_Y_MEAN,
        predictor_name='_mundlak_lg_scm_predictor',
    )
    expected_b = _expected_beta_b(n_pairs=_N_PAIRS)

    # Sanity on counts: 5 envs * 4 bursts = 20 obs; 5 env clusters
    # for the cluster-robust SE.
    assert result.n_strata == len(_MU_X_GRID), (
        f'n_strata = {result.n_strata}, expected {len(_MU_X_GRID)} '
        f'(one cluster per env)'
    )
    assert result.n_obs == len(_MU_X_GRID) * _N_BURSTS, (
        f'n_obs = {result.n_obs}, expected '
        f'{len(_MU_X_GRID) * _N_BURSTS}'
    )

    # Between coefficient: closed-form recovery within ~5%.
    rel_err_b = abs(result.between.coefficient - expected_b) / expected_b
    assert rel_err_b < 0.05, (
        f'beta_b = {result.between.coefficient:.4f}, expected '
        f'{expected_b:.4f} (rel err {rel_err_b:.4f}). The between-'
        f'env covariation between predictor (x_e ≈ mu_x + offset) '
        f'and target (y_e ≈ slope * mu_x) is closed-form linear; '
        f'a >5% drift indicates the WLS solve or the within/between '
        f'split is broken.'
    )

    # Within coefficient: structurally zero in expectation. The
    # within-env target variation is sampling noise on per-burst g
    # (independent across bursts within env), and the within-env
    # predictor is a deterministic burst-index sequence — the two
    # are independent in expectation, so E[β_w] = 0.
    #
    # Bound: |β_w / SE_w| < 2.5 — direct Z-score test against the
    # framework's own reported SE. Catches a regression in two
    # ways:
    #   (a) point estimate inflates without proportional SE
    #       (β_w grows but SE stays) → Z-score breaches
    #   (b) point estimate stays small but SE collapses to zero
    #       → false-significance signal, breaches differently
    # Replaces the prior "p > 0.05 + CI covers zero" pair which
    # would pass on a framework returning garbage near-zero
    # estimates with overconfident CIs.
    assert result.within.se > 0.0, (
        f'within SE = {result.within.se}; framework reported zero '
        f'or negative SE — degenerate fit'
    )
    z_score = abs(result.within.coefficient) / result.within.se
    assert z_score < 2.5, (
        f'|β_w / SE_w| = {z_score:.4f} (β_w = '
        f'{result.within.coefficient:.4f}, SE = '
        f'{result.within.se:.4f}). The within channel is '
        f'structurally null; a Z-score above 2.5 indicates either '
        f'a spurious within signal or an under-reported SE.'
    )

    # Hausman test: beta_b != beta_w. With beta_b ~ 27.5 and
    # beta_w ~ 0, the difference should be highly significant.
    assert result.hausman_p < 0.05, (
        f'hausman_p = {result.hausman_p:.4f}; expected < 0.05 '
        f'because beta_b is structurally large and beta_w is near '
        f'zero — Mundlak decomposition was justified here'
    )


def test_mundlak_intercept_absorbs_predictor_offset() -> None:
    """The synthetic predictor's env-mean is `mu_x + 0.1 * 1.5
    = mu_x + 0.15` (offset by the burst-index midpoint × slope).

    Closed-form: y_e = beta_b * x_e + intercept. With beta_b ≈
    sqrt(n)/sigma * c_4 and the predictor offset above:

        intercept ≈ -beta_b * 0.15

    A regression that failed to compute the within-deviation
    correctly (e.g., didn't demean by env) would land the
    intercept far from this value.
    """
    cells = _build_phased_panel()
    result = mundlak_paired_g_per_burst.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source=_PER_BURST_Y_MEAN,
        predictor_name='_mundlak_lg_scm_predictor',
    )
    expected_b = _expected_beta_b(n_pairs=_N_PAIRS)
    mean_burst_idx = (_N_BURSTS - 1) / 2.0  # 1.5 for n_bursts=4
    expected_intercept = -expected_b * _BURST_SLOPE * mean_burst_idx
    rel_err = abs(result.intercept - expected_intercept) / abs(
        expected_intercept,
    )
    # Looser bound on intercept than on slope: absolute scale is
    # smaller (~4) and finite-sample noise on g_e bleeds in.
    assert rel_err < 0.1, (
        f'intercept = {result.intercept:.4f}, expected '
        f'{expected_intercept:.4f} (rel err {rel_err:.4f}). The '
        f"closed-form -beta_b * 0.15 reflects the predictor's "
        f'env-mean offset; a >10% drift indicates the env-mean '
        f'projection is mis-computed.'
    )
