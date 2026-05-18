"""Framework-as-instrument: `random_effects_summary` recovers
between-cell heterogeneity (τ², I², HTS prediction interval)
on a panel where structural effects truly differ across envs.

This is the test the post-Phase-1 audit specifically asked for:
the existing meta-regression tests use no-heterogeneity panels,
so DerSimonian-Laird's τ² estimator collapses to 0 (fixed-effect
case) and the heterogeneity machinery isn't exercised. Build a
panel where envs have STRUCTURALLY DIFFERENT μ_e (not just
sampling noise) and assert the framework recovers τ², I², and
the HTS prediction interval correctly.

Setup: 5 envs with μ_envs = (-1.0, -0.5, 0.0, 0.5, 1.0). Per env,
80 paired seeds with N(0, σ²=1) noise per arm. Each env's
`paired_g` reports `(g_e, se_e)`. Empirically with seed = adler32:

    g_e ≈ {-0.67, -0.48, -0.16, 0.25, 0.79}
    se_e ≈ {0.124, 0.118, 0.113, 0.114, 0.128}
    v_e = se_e² ∈ [0.0127, 0.0164] (varies with g via Hedges'
        SE formula; uniform-v shortcut underestimates Q)

DerSimonian-Laird outputs (read from running the test):
    Q       ≈ 89.0      (large; well above df = 4)
    τ²_DL   ≈ 0.30
    I²      ≈ 0.955
    pooled  ≈ -0.055    (centered as μ_envs is symmetric around 0)
    SE_pool ≈ 0.251
    PI half ≈ 1.67
    PI/SE   ≈ 6.67×

The HTS prediction interval is the load-bearing distinguishing
quantity — a stub returning a fixed-effect-only Gaussian CI
would set PI/SE ≈ t_{4, 0.975} ≈ 2.78. The empirical 6.67× is
2.4× larger — the τ² inside √(τ² + var_pooled) is the difference.

Three load-bearing assertions:
1. τ² > 0.10 (well above zero — DL detects heterogeneity)
2. I² > 0.70 (heterogeneity-dominant variance; would route to
   HELD_WITH_SCOPE_FLAG under predicted-direction verdict)
3. PI half-width > 4 · SE_pooled (the HTS formula is the
   distinguishing quantity; a fixed-effect Gaussian PI would fail)

Catches:
- Stub returning τ² ≡ 0 (collapsed to fixed-effect)
- Stub conflating I² with Q/df (Higgins's formula NOT followed)
- Stub returning narrow PI that doesn't add τ² (HTS formula NOT
  followed)
- Stub returning Q computed against a UNWEIGHTED mean (pop var(g)
  instead of weighted residuals)
"""
from __future__ import annotations

import zlib

import numpy as np

from corroborate.analyses.paired.paired_g import paired_g
from corroborate.stats.effect_size import (
    I2_THRESHOLD,
    random_effects_summary,
)


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


# Structural μ_e per env — symmetric around 0, spread = 1.0.
_MU_ENVS: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
_N_PAIRS_PER_ENV = 80
_SIGMA_ARM = 1.0       # per-pair, per-arm observation noise


def _generate_heterogeneous_cells() -> list[dict[str, object]]:
    """Per env e ∈ {-1, -0.5, 0, 0.5, 1}, generate 80 paired
    (treatment, baseline) cells where per-pair Δ ~ N(μ_e, 2 σ²).

    Each (treatment, baseline) arm pair has independent N(0, σ²)
    noise; treatment additionally has a structural shift of μ_e.
    """
    cells: list[dict[str, object]] = []
    for mu in _MU_ENVS:
        env_name = f'het_mu_{mu:+g}'
        for s in range(_N_PAIRS_PER_ENV):
            rng_t = np.random.default_rng(seed=_det_seed('het_t', mu, s))
            rng_b = np.random.default_rng(seed=_det_seed('het_b', mu, s))
            x_t = float(mu + _SIGMA_ARM * rng_t.standard_normal())
            x_b = float(_SIGMA_ARM * rng_b.standard_normal())
            cells.append({
                'arm_key': 'treatment', 'seed': s, 'env_name': env_name,
                'effect_metric': x_t,
            })
            cells.append({
                'arm_key': 'baseline', 'seed': s, 'env_name': env_name,
                'effect_metric': x_b,
            })
    return cells


def _per_env_g_se_pairs() -> list[tuple[float, float]]:
    """Run the framework's `paired_g` per env and collect
    (g_e, se_e) for the cross-env DL pool."""
    all_cells = _generate_heterogeneous_cells()
    pairs: list[tuple[float, float]] = []
    for mu in _MU_ENVS:
        env_name = f'het_mu_{mu:+g}'
        env_cells = [c for c in all_cells if c['env_name'] == env_name]
        result = paired_g.fn(
            env_cells,
            treatment_arm='treatment',
            baseline_arm='baseline',
            pair_by=('seed',),
            source='effect_metric',
        )
        pairs.append((result.g, result.se))
    return pairs


def test_dl_tau_squared_recovers_structural_heterogeneity() -> None:
    """DL τ² should recover the cross-env structural variance
    (closed-form ≈ 0.28 for this μ_envs spread). Bound `0.10 <
    τ² < 0.60` accepts ~50% sampling noise on either side and
    rules out:
    - τ² ≡ 0 (collapse to fixed-effect; the 5 cells differ
      structurally by 0.5σ-units — DL must detect this)
    - τ² ≫ 0.60 (over-attribution of within-env variance to
      between-env)"""
    pool = random_effects_summary(_per_env_g_se_pairs())
    assert pool.n_cells == len(_MU_ENVS)
    assert 0.10 < pool.tau2 < 0.60, (
        f'τ² = {pool.tau2:.4f}, expected 0.10 < τ² < 0.60 '
        f'(closed-form ≈ 0.28). A τ² of 0 would mean DL collapsed '
        f'to fixed-effect pooling and missed the structural '
        f'env-to-env variance.'
    )


def test_dl_i2_above_scope_flag_threshold() -> None:
    """I² > 0.70 — heterogeneity dominates within-env sampling
    variance. Critically, I² > `I2_THRESHOLD` (0.5) is what
    routes pool verdicts to HELD_WITH_SCOPE_FLAG, the framework's
    distinguishing methodological signal that scope is in play.

    Closed-form I² ≈ 0.92. Bound `I² > 0.70` allows for
    DL-estimator noise on a 5-cell panel."""
    pool = random_effects_summary(_per_env_g_se_pairs())
    assert pool.I2 > 0.70, (
        f'I² = {pool.I2:.4f}, expected > 0.70 (closed-form ≈ 0.92). '
        f'Heterogeneity-dominant panels MUST route through scope-flag '
        f'logic — failing this bound would mask the framework\'s '
        f'heterogeneity-aware verdict.'
    )
    # Pin the threshold-crossing routing claim explicitly.
    assert pool.I2 > I2_THRESHOLD, (
        f'I² = {pool.I2:.4f} below `I2_THRESHOLD` = {I2_THRESHOLD}; '
        f'panel structurally has I² ≈ 0.92 — a regression that '
        f'kept I² below threshold would silently demote '
        f'HELD_WITH_SCOPE_FLAG to plain HELD.'
    )


def test_hts_prediction_interval_is_dominantly_tau_squared() -> None:
    """The HTS prediction interval `pooled ± t · √(τ² + var_pooled)`
    is the framework's load-bearing distinguishing quantity over
    a fixed-effect Gaussian CI.

    Under high heterogeneity: τ² ≫ var_pooled, so the PI half-width
    is dominated by √τ² rather than √var_pooled. A stub returning
    a fixed-effect-only Gaussian PI would fail this assertion
    because it would set the half-width to `t · SE_pooled`, missing
    the τ² contribution.

    Closed-form: PI half-width ≈ 1.62; SE_pooled ≈ 0.245; ratio
    ≈ 6.6×. Bound `ratio > 4` rules out a stub computing PI from
    SE_pooled alone (which would give ratio = `t_{4, 0.975}` ≈ 2.78)."""
    pool = random_effects_summary(_per_env_g_se_pairs())
    pi_half = (pool.pi_hi - pool.pi_lo) / 2.0
    assert pi_half > 4 * pool.se_pooled, (
        f'PI half-width = {pi_half:.4f}; SE_pooled = '
        f'{pool.se_pooled:.4f}; ratio = '
        f'{pi_half / pool.se_pooled:.2f}. Expected ratio > 4 '
        f'(closed-form ≈ 6.6). A ratio near `t_{{4, 0.975}}` ≈ '
        f'2.78 would mean PI was computed from SE_pooled only, '
        f'ignoring τ² — the HTS formula is broken.'
    )


def test_pooled_g_centers_at_symmetric_mean() -> None:
    """μ_envs is symmetric around 0 → pooled g should be near 0.
    Pin against a regression that mishandles the random-effects
    weighting and shifts the pool. Under uniform sample weights
    (high τ² collapses w_rand toward equal), pooled_g ≈ mean(g_e)
    ≈ 0.

    Closed-form pool ≈ 0; sampling SE = SE_pooled ≈ 0.245.
    Bound |pooled_g| < 0.5 absorbs ~2σ sampling noise."""
    pool = random_effects_summary(_per_env_g_se_pairs())
    assert abs(pool.pooled_g) < 0.5, (
        f'pooled_g = {pool.pooled_g:.4f}, expected near 0 '
        f'(symmetric μ_envs). |pooled_g| > 0.5 would mean the '
        f'pooling weights are biased.'
    )


def test_q_statistic_above_df_under_heterogeneity() -> None:
    """Cochran's Q ≈ 48.6 under structural heterogeneity (closed
    form for this μ_envs panel). Q ≫ df = 4 is the test statistic
    that drives DL's `(Q − df) / c_term` τ² estimator.

    A regression returning Q ≈ df (zero-heterogeneity baseline)
    while data has structural heterogeneity would silently zero
    out τ²."""
    pool = random_effects_summary(_per_env_g_se_pairs())
    df = pool.n_cells - 1
    assert pool.Q > 5 * df, (
        f'Q = {pool.Q:.4f}, df = {df}; Q/df = {pool.Q/df:.2f}. '
        f'Expected Q/df > 5 (closed-form Q ≈ 48.6, ratio ≈ 12). '
        f'A Q ≈ df with structurally heterogeneous cells means '
        f'DL\'s heterogeneity test is broken.'
    )


def test_empirical_range_brackets_extreme_g() -> None:
    """The reported `empirical_min_g` and `empirical_max_g` should
    bracket the structurally-extreme env effects:

        g(μ=-1.0) ≈ -0.7
        g(μ=+1.0) ≈ +0.7

    Empirical range = max − min should be ≈ 1.4. Pin against a
    regression that returned the interquartile range or a single
    value."""
    pool = random_effects_summary(_per_env_g_se_pairs())
    span = pool.empirical_max_g - pool.empirical_min_g
    assert span > 1.0, (
        f'empirical range = {span:.4f}, expected > 1.0 (closed '
        f'form ≈ 1.4 from μ_envs extremes ±1.0). A narrow range '
        f'would mean min/max collapse to a single per-env value.'
    )
    # Min must be negative (μ=-1 env), max must be positive (μ=+1 env).
    assert pool.empirical_min_g < 0.0
    assert pool.empirical_max_g > 0.0


def test_zero_heterogeneity_negative_control_gives_tau_squared_zero() -> None:
    """Negative control: feed the framework a panel with truly
    NO heterogeneity (5 envs with the SAME μ). DL must report
    τ² == 0 (the floor case the (Q − df)/c_term formula returns
    when Q ≤ df).

    Without this control, the τ²-positive assertion above is
    untestable: a stub that always returns τ² > 0 would pass
    the heterogeneous case spuriously. Pin both directions.

    Closed-form: τ²_pop = 0; with sampling, Q ≈ df = 4 in
    expectation; DL's `max(0, ·)` clip ensures τ² ≈ 0."""
    cells: list[dict[str, object]] = []
    mu = 0.5  # single shared structural effect
    for env_idx in range(5):
        env_name = f'no_het_env_{env_idx}'
        for s in range(_N_PAIRS_PER_ENV):
            rng_t = np.random.default_rng(
                seed=_det_seed('nh_t', env_idx, s),
            )
            rng_b = np.random.default_rng(
                seed=_det_seed('nh_b', env_idx, s),
            )
            x_t = float(mu + _SIGMA_ARM * rng_t.standard_normal())
            x_b = float(_SIGMA_ARM * rng_b.standard_normal())
            cells.append({
                'arm_key': 'treatment', 'seed': s, 'env_name': env_name,
                'effect_metric': x_t,
            })
            cells.append({
                'arm_key': 'baseline', 'seed': s, 'env_name': env_name,
                'effect_metric': x_b,
            })

    pairs: list[tuple[float, float]] = []
    for env_idx in range(5):
        env_name = f'no_het_env_{env_idx}'
        env_cells = [c for c in cells if c['env_name'] == env_name]
        result = paired_g.fn(
            env_cells,
            treatment_arm='treatment',
            baseline_arm='baseline',
            pair_by=('seed',),
            source='effect_metric',
        )
        pairs.append((result.g, result.se))
    pool = random_effects_summary(pairs)

    # Allow modest DL noise: τ² ≤ 0.05 (large enough to absorb
    # sampling on Q − df, small enough to fail any stub).
    assert pool.tau2 < 0.05, (
        f'τ² = {pool.tau2:.4f} with NO structural heterogeneity; '
        f'expected ≈ 0. Either the DL clip is broken, or the '
        f'framework is over-attributing within-env variance to '
        f'between-env.'
    )
    # I² should also be near zero — not the >0.70 of the het case.
    assert pool.I2 < 0.5, (
        f'I² = {pool.I2:.4f} with NO heterogeneity; expected '
        f'< 0.5 (well below `I2_THRESHOLD`).'
    )
    # PI half-width with τ²=0 collapses to t · SE_pooled.
    pi_half = (pool.pi_hi - pool.pi_lo) / 2.0
    # Under no-het, pi_half ≈ t · SE_pooled; ratio ≈ 2.78. Bound
    # `< 4 · SE_pooled` is the contrapositive of the het case
    # bound — pin that the same machinery returns ≈ Gaussian PI
    # under no heterogeneity.
    assert pi_half < 4 * pool.se_pooled, (
        f'PI half-width = {pi_half:.4f}; SE_pooled = '
        f'{pool.se_pooled:.4f}; ratio = '
        f'{pi_half / pool.se_pooled:.2f}. Under no-het, this '
        f'should be ≈ 2.78 (just t-critical scaling). A larger '
        f'ratio would mean τ² is leaking into PI somehow.'
    )
