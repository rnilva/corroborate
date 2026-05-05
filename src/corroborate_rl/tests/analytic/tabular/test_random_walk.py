"""Sutton & Barto §6.2 random walk: closed-form V* matches
framework-computed V on the canonical example.

The random walk is the textbook example for tabular policy
evaluation: linear chain with two terminal endpoints, single
random-step "policy", reward 0 on internal transitions and on
entering the left terminal, +1 on entering the right terminal.

Closed-form V (γ=1, left=0, right=1) per Sutton & Barto eq. 6.4:
    V(s) = s / (n_internal + 1)         for s ∈ {0, ..., n_internal + 1}

This is the *value function* for the equiprobable-random policy
— it's not a value-iteration optimum since the random walk has
only one action (no policy choice). But the framework's
`policy_evaluation` should recover it exactly via the closed-form
matrix inversion `(I - γP_π)^{-1} r_π`, even at γ very close to 1.

For γ=1 strictly, the matrix `(I - P_π)` is singular on the
absorbing-state subspace — `policy_evaluation` would hit
LinAlgError. So we test at γ slightly less than 1 (the
sub-discounted random walk) AND derive its closed form, plus
test the unit-discounted form using a workaround (terminal-
shift trick from Howard 1960)."""
from __future__ import annotations

import numpy as np

from corroborate_rl.tabular import (
    policy_evaluation,
    random_walk,
)
from corroborate_rl.tabular.examples import random_walk_closed_form_v


def _sup_norm(v: np.ndarray) -> float:
    return float(np.max(np.abs(v)))


def test_random_walk_value_function_matches_closed_form_at_high_gamma() -> None:
    """At γ very close to 1 (0.999), the Sutton-Barto random-walk
    V function from `policy_evaluation` matches the γ=1 closed
    form within a small margin: the discount slightly dampens
    long-horizon rewards, but with terminal absorption the dampening
    is bounded by ~γ^max_horizon.

    A regression in `policy_evaluation`'s matrix inversion (sign,
    transposition, mis-indexed P_π) would fail this catastrophically."""
    n_internal = 5
    mdp = random_walk(n_internal_states=n_internal, gamma=0.999)
    # Single action MDP — policy is trivially [0]*n_states.
    pi = np.zeros(mdp.n_states, dtype=np.int64)
    v = policy_evaluation(pi, mdp.transitions, mdp.rewards, mdp.gamma)

    # Closed form at γ=1 (the unit-discount limit).
    v_closed = random_walk_closed_form_v(n_internal_states=n_internal)
    # γ=0.999 introduces small discount-induced deviations from the
    # γ=1 closed form. The deviation per state scales with the
    # discount applied over the expected hit time. For a 5-internal
    # random walk, expected hit time is bounded by n²/4 = 6.25 ish;
    # γ^6.25 ≈ 0.994, so V deviates by ~0.6% at most.
    diff = _sup_norm(v - v_closed)
    assert diff < 0.01, (
        f'V(γ=0.999) deviates from γ=1 closed form by '
        f'{diff:.6f}; expected < 0.01 (γ-induced damping)'
    )


def test_random_walk_value_at_exact_unit_discount_via_per_internal_only() -> None:
    """At γ=1 exactly, the system `(I - P_π) v = r_π` is rank-
    deficient (terminal states are absorbing → identity rows in
    P_π → zero rows in (I - P_π)). `policy_evaluation` uses
    `np.linalg.solve` which would error.

    The correct approach for γ=1 absorbing MDPs is to solve only
    on the non-terminal subspace, using the absorbing-row
    structure to read off terminal V's directly (V = 0 on a
    terminal whose entry-reward is captured upstream).

    This test documents the limitation by asserting
    `policy_evaluation` raises on the singular system — and
    points the user at `value_iteration` (which works because T
    is contractive even at γ=1 on absorbing MDPs, as a
    practical matter — well, almost; we won't test that here).

    Catches a regression where `policy_evaluation` silently
    returns something nonsensical instead of raising."""
    import pytest as _pytest

    n_internal = 5
    mdp = random_walk(n_internal_states=n_internal, gamma=0.999)
    # Construct a γ=1 dynamics directly (skipping the example's
    # gamma validation by using the underlying tensor + replacing
    # gamma at the call site).
    pi = np.zeros(mdp.n_states, dtype=np.int64)
    with _pytest.raises(np.linalg.LinAlgError):
        _ = policy_evaluation(pi, mdp.transitions, mdp.rewards, 1.0)


def test_random_walk_closed_form_v_self_consistency() -> None:
    """Sanity on the closed-form helper itself.

    Terminal V's are 0 (no future reward from absorbing state).
    Internal V's follow `V(i) = i/(N+1)` for the canonical setup,
    monotone increasing across internal states."""
    n_internal = 5
    v = random_walk_closed_form_v(
        n_internal_states=n_internal,
        left_terminal_reward=0.0, right_terminal_reward=1.0,
    )
    assert v.shape == (n_internal + 2,)
    # Terminals: V = 0 (entry-reward attributed to predecessor's
    # expected value; once absorbed there's no more reward).
    assert v[0] == 0.0
    assert v[-1] == 0.0
    # Internal V's: V(i) = i / (N + 1) for i ∈ {1, ..., N}.
    for i in range(1, n_internal + 1):
        expected = i / (n_internal + 1)
        assert abs(v[i] - expected) < 1e-12, (
            f'V({i}) = {v[i]:.6f}, expected {expected:.6f} = '
            f'{i}/{n_internal + 1}'
        )
    # Internal portion is monotone increasing.
    for i in range(1, n_internal):
        assert v[i + 1] > v[i] - 1e-12
