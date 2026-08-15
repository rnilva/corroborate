r"""Hasselt's max-bias theorem (Hasselt 2010, 2016): the central
theoretical claim that DDQN addresses.

Setup: at a fixed state s, the agent has noisy estimates
`Q̂(s, a) = Q*(s, a) + ε(a)` where ε is iid mean-zero. Vanilla
DQN's bootstrap target uses `max_a Q̂(s, a)` — biased upward by
Jensen's inequality (max is convex). DDQN's double-greedify
target uses `Q_target[argmax_a Q_online]` with independent
Q_online, Q_target — unbiased.

The tabular regime makes the bias **exactly testable**:

| Setting | Closed-form expected V | Closed-form bias |
|---|---|---|
| max_greedify, Q* = 0, ε ~ N(0, σ²), \|A\|=2 | σ/√π | σ/√π |
| max_greedify, Q* = 0, ε ~ N(0, σ²), large \|A\| | σ·√(2 ln \|A\|) approx | same |
| double_greedify, Q_online ⫫ Q_target, Q* = 0 | 0 | 0 |
| double_greedify, Q_online = Q_target | σ/√π (n=2 case) | σ/√π |

This tests the *foundational* theorem at the heart of CLAIM 2 in
the implementation's claim graph (auto-memory `findings_action_dim_sweep`).
"""
from __future__ import annotations

import math

import numpy as np

from corroborate_rl.tabular import (
    double_greedify_tabular,
    hasselt_max_bias_asymptotic,
    hasselt_n2_max_bias,
    max_greedify_tabular,
)


_SIGMA = 1.0
_N_TRIALS = 100_000


def _empirical_bias_max_greedify(
    *, sigma: float, n_actions: int, n_trials: int, seed: int = 0,
) -> tuple[float, float]:
    """Monte Carlo: Q* = 0 vector, draw Q̂ = Q* + iid N(0, σ²) ε
    per trial, compute max_a Q̂. Return (mean, SE_of_mean) over
    n_trials.

    Closed-form expected value of max_a Q̂ when Q* = 0 IS the
    bias (since max_a Q* = 0). This is what the closed-form
    helpers compute analytically."""
    rng = np.random.default_rng(seed)
    samples = np.array([
        max_greedify_tabular(
            (rng.standard_normal(n_actions) * sigma).astype(np.float64),
        )
        for _ in range(n_trials)
    ])
    mean = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / math.sqrt(n_trials))
    return mean, se


# ============ |A|=2 closed form: σ/√π ============

def test_hasselt_n2_max_bias_matches_exact_closed_form() -> None:
    """For |A|=2 with Q*=0 and iid N(0, σ²) noise, E[max_a Q̂] =
    σ/√π exactly. Empirical mean over 100k trials matches within
    4·SE.

    A regression where `max_greedify_tabular` returned the WRONG
    aggregator (sum instead of max, mean, etc.) would breach this
    by orders of magnitude."""
    expected = hasselt_n2_max_bias(_SIGMA)
    assert abs(expected - _SIGMA / math.sqrt(math.pi)) < 1e-12

    mean, se = _empirical_bias_max_greedify(
        sigma=_SIGMA, n_actions=2, n_trials=_N_TRIALS, seed=0,
    )
    assert abs(mean - expected) < 4.0 * se, (
        f'empirical max-bias = {mean:.6f}, closed-form σ/√π = '
        f'{expected:.6f} (4·SE = {4.0 * se:.6f}). max_greedify '
        f'must aggregate via max for the Hasselt theorem to hold.'
    )


def test_hasselt_n2_max_bias_scales_linearly_with_sigma() -> None:
    """The closed-form σ/√π is linear in σ. Catches a regression
    where the bias scales nonlinearly (e.g., σ²) — would fail
    the linearity ratio across two σ values."""
    bias_at_1 = hasselt_n2_max_bias(1.0)
    bias_at_3 = hasselt_n2_max_bias(3.0)
    assert abs(bias_at_3 - 3.0 * bias_at_1) < 1e-12, (
        f'bias not linear in σ: σ=1 → {bias_at_1:.6f}, '
        f'σ=3 → {bias_at_3:.6f}, ratio = {bias_at_3/bias_at_1:.4f} '
        f'≠ 3'
    )

    # And empirically: max_greedify scales linearly too.
    mean_1, se_1 = _empirical_bias_max_greedify(
        sigma=1.0, n_actions=2, n_trials=_N_TRIALS, seed=0,
    )
    mean_3, se_3 = _empirical_bias_max_greedify(
        sigma=3.0, n_actions=2, n_trials=_N_TRIALS, seed=0,
    )
    # Empirical bias at σ=3 should be 3× empirical bias at σ=1.
    assert abs(mean_3 - 3.0 * mean_1) < 4.0 * (se_1 * 3 + se_3), (
        f'empirical bias not linear in σ: σ=1 → {mean_1:.6f}, '
        f'σ=3 → {mean_3:.6f}'
    )


# ============ |A| scaling: σ · √(2 ln |A|) leading term ============

def test_hasselt_max_bias_grows_with_log_n_actions() -> None:
    """Closed-form leading-order: bias ≈ σ · √(2 ln |A|). Empirical
    bias at |A|=2, |A|=10, |A|=100 grows monotonically in |A|.

    The asymptotic σ·√(2 ln |A|) overestimates for small |A|
    (~30% off at |A|=4) but is the correct leading-order term.
    The implementation's `jensen_dormancy_gap` invariant uses this as
    a structural floor.

    A regression that broke the dependence on |A| (e.g., constant
    bias regardless of |A|) would fail the monotonicity check."""
    means: list[float] = []
    for n_actions in (2, 10, 100):
        mean, _ = _empirical_bias_max_greedify(
            sigma=_SIGMA, n_actions=n_actions, n_trials=_N_TRIALS,
            seed=n_actions,
        )
        means.append(mean)
    # Monotone growth.
    assert means[0] < means[1] < means[2], (
        f'bias non-monotone in |A|: 2→{means[0]:.4f}, '
        f'10→{means[1]:.4f}, 100→{means[2]:.4f}'
    )
    # At |A|=100, asymptotic σ·√(2 ln 100) ≈ 3.03 — empirical
    # should be in [2.0, 3.5] (true value ≈ 2.51).
    asymptotic_100 = hasselt_max_bias_asymptotic(_SIGMA, 100)
    assert 2.0 < means[2] < asymptotic_100 + 0.1, (
        f'|A|=100 empirical bias = {means[2]:.4f} outside '
        f'[2.0, {asymptotic_100:.4f}]; asymptotic upper bound is '
        f'σ·√(2 ln 100) = {asymptotic_100:.4f}'
    )


# ============ DDQN with independent estimators: unbiased ============

def test_double_greedify_unbiased_with_independent_estimators() -> None:
    """When Q_online ⫫ Q_target both estimating Q* = 0, DDQN's
    target is exactly unbiased: E[Q_target[argmax_a Q_online]] = 0.

    Reasoning: argmax_a Q_online is a random index from
    Q_target's perspective (the noise streams are independent),
    so Q_target[that index] is just a sample from Q_target's
    distribution (mean 0). No Jensen-inequality bias.

    A regression where double_greedify accidentally used Q_online's
    value (instead of Q_target's) would inherit max_greedify's
    σ/√π bias → fail this test by ~0.56."""
    rng = np.random.default_rng(7)
    samples: list[float] = []
    for _ in range(_N_TRIALS):
        q_online = (rng.standard_normal(2) * _SIGMA).astype(np.float64)
        q_target = (rng.standard_normal(2) * _SIGMA).astype(np.float64)
        samples.append(
            double_greedify_tabular(q_online, q_target),
        )
    mean = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / math.sqrt(_N_TRIALS))
    # Expected = 0 (unbiased). 4·SE bound.
    assert abs(mean) < 4.0 * se, (
        f'double_greedify with independent estimators: empirical '
        f'mean = {mean:.6f}, expected 0, 4·SE = {4.0 * se:.6f}. '
        f'A bias here would mean either (a) the action selection '
        f'is leaking into evaluation, or (b) Q_target is being '
        f'used to select action.'
    )
    # And: the bias is much smaller than max_greedify's.
    max_bias = hasselt_n2_max_bias(_SIGMA)
    assert abs(mean) < max_bias / 5, (
        f'double_greedify bias {abs(mean):.4f} not far below '
        f'max_greedify σ/√π = {max_bias:.4f}; the decoupling '
        f'should reduce the bias by an order of magnitude or more.'
    )


# ============ DDQN with identical estimators: equals max_greedify ============

def test_double_greedify_equals_max_greedify_when_estimators_identical() -> None:
    """When Q_online = Q_target (same vector), DDQN reduces to
    max_greedify exactly: argmax_a Q_online IS argmax_a Q_target,
    so Q_target at that index IS max_a Q_target.

    Negative control for the prior test: the DDQN architecture's
    bias-correction REQUIRES the two estimators to be independent.
    Identical estimators inherit max_greedify's σ/√π bias.

    A regression where double_greedify silently dropped one of
    the estimators (e.g., always used Q_online) would still pass
    this trivially, but would fail the prior unbiased test."""
    rng = np.random.default_rng(11)
    biases_max: list[float] = []
    biases_double: list[float] = []
    for _ in range(_N_TRIALS):
        q = (rng.standard_normal(2) * _SIGMA).astype(np.float64)
        biases_max.append(max_greedify_tabular(q))
        biases_double.append(double_greedify_tabular(q, q))
    # Per-trial equality: with q_online == q_target, the two
    # operators return the SAME number every time.
    for i, (m, d) in enumerate(zip(biases_max, biases_double)):
        if m != d:
            raise AssertionError(
                f'trial {i}: max_greedify={m:.6f}, '
                f'double_greedify(q,q)={d:.6f}; identical-estimator '
                f'DDQN must reduce to max_greedify exactly'
            )
    # And both have the σ/√π bias.
    mean_max = float(np.mean(biases_max))
    mean_double = float(np.mean(biases_double))
    expected = hasselt_n2_max_bias(_SIGMA)
    se = float(np.std(biases_max, ddof=1) / math.sqrt(_N_TRIALS))
    assert abs(mean_max - expected) < 4.0 * se
    assert abs(mean_double - expected) < 4.0 * se
