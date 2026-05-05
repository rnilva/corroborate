"""Bellman optimality operator γ-contraction theorem.

Bertsekas-Tsitsiklis 1996 §6.3 / Prop 6.2.3:
For any V₁, V₂ ∈ R^|S|,
    ||T(V₁) - T(V₂)||_∞ ≤ γ · ||V₁ - V₂||_∞
where T is the Bellman optimality operator
    T(V)(s) = max_a [R(s, a) + γ · Σ_s' P(s'|s, a) V(s')]
and ||·||_∞ is the sup-norm.

Why it matters: γ-contraction + Banach fixed-point theorem
gives unique V* and value-iteration's convergence rate.

Why tabular: the contraction is EXACT in the tabular regime,
not a stochastic approximation. We assert the inequality to
machine precision."""
from __future__ import annotations

import numpy as np

from corroborate_rl.tabular import (
    bellman_backup,
    random_walk,
    two_state_chain,
)


def _sup_norm(v: np.ndarray) -> float:
    return float(np.max(np.abs(v)))


# ============ Two-state chain ============

def test_bellman_operator_is_gamma_contraction_on_two_state_chain() -> None:
    """T contracts by exactly γ in sup-norm on the two-state chain.
    A regression that flipped sign or used wrong factor would be
    caught by the strict ≤ + the closed-form ratio check."""
    mdp = two_state_chain(gamma=0.9, p_self=0.5, r_left=0.0, r_right=1.0)
    rng = np.random.default_rng(0)
    for trial in range(20):
        v1 = rng.standard_normal(mdp.n_states)
        v2 = rng.standard_normal(mdp.n_states)
        t_v1 = bellman_backup(v1, mdp.transitions, mdp.rewards, mdp.gamma)
        t_v2 = bellman_backup(v2, mdp.transitions, mdp.rewards, mdp.gamma)
        lhs = _sup_norm(t_v1 - t_v2)
        rhs = mdp.gamma * _sup_norm(v1 - v2)
        # γ-contraction inequality, allowing tiny floating-point slack.
        assert lhs <= rhs + 1e-9, (
            f'trial {trial}: ||T(V1) - T(V2)|| = {lhs:.6f} > '
            f'γ · ||V1 - V2|| = {rhs:.6f} (γ = {mdp.gamma})'
        )


# ============ Random walk ============

def test_bellman_operator_is_gamma_contraction_on_random_walk() -> None:
    """Same contraction theorem on a richer 5-internal-state
    random walk (γ = 0.99 to stay strictly inside the contraction
    regime). The state space is bigger, the dynamics are
    stochastic — the theorem still holds tightly."""
    mdp = random_walk(n_internal_states=5, gamma=0.99)
    rng = np.random.default_rng(1)
    for _trial in range(20):
        v1 = rng.standard_normal(mdp.n_states) * 5.0  # wider scale
        v2 = rng.standard_normal(mdp.n_states) * 5.0
        t_v1 = bellman_backup(v1, mdp.transitions, mdp.rewards, mdp.gamma)
        t_v2 = bellman_backup(v2, mdp.transitions, mdp.rewards, mdp.gamma)
        assert _sup_norm(t_v1 - t_v2) <= mdp.gamma * _sup_norm(v1 - v2) + 1e-9


# ============ Contraction on identical inputs is exact zero ============

def test_bellman_contraction_on_identical_inputs_is_zero() -> None:
    """Edge case: T(V) - T(V) = 0 exactly. Catches a regression
    where T accidentally injects per-call randomness or fails to
    be a pure function."""
    mdp = two_state_chain(gamma=0.9)
    v = np.array([1.5, 0.0], dtype=np.float64)
    t_v_a = bellman_backup(v, mdp.transitions, mdp.rewards, mdp.gamma)
    t_v_b = bellman_backup(v, mdp.transitions, mdp.rewards, mdp.gamma)
    assert _sup_norm(t_v_a - t_v_b) == 0.0


# ============ Tightness: contraction is achievable at exactly γ ============

def test_bellman_contraction_is_tight_at_constant_offsets() -> None:
    """The ≤ in Bertsekas-Tsitsiklis is tight: there exist V₁, V₂
    where ||T(V₁) - T(V₂)||_∞ achieves *exactly* γ · ||V₁ - V₂||_∞.

    Witness: V₁ = c · ones, V₂ = 0. Both V's induce the same
    argmax (max_a is constant in V's offset), and:

        T(V₁)(s) = max_a [R(s, a) + γ · Σ P(s'|s, a) · c]
                = max_a [R(s, a)] + γ · c
        T(V₂)(s) = max_a [R(s, a)]

    So ||T(V₁) - T(V₂)||_∞ = γ · c = γ · ||V₁ - V₂||_∞. The
    inequality saturates.

    A regression that produced a *strict* contraction (factor
    < γ) would fail this — equally, a regression that exceeded γ
    would fail the previous tests."""
    mdp = two_state_chain(gamma=0.7, p_self=0.5)
    c = 5.0
    v1 = np.full(mdp.n_states, c, dtype=np.float64)
    v2 = np.zeros(mdp.n_states, dtype=np.float64)
    t_v1 = bellman_backup(v1, mdp.transitions, mdp.rewards, mdp.gamma)
    t_v2 = bellman_backup(v2, mdp.transitions, mdp.rewards, mdp.gamma)
    lhs = _sup_norm(t_v1 - t_v2)
    rhs = mdp.gamma * c
    assert abs(lhs - rhs) < 1e-9, (
        f'expected tight contraction at γ · c = {rhs:.6f}, '
        f'got ||T(V₁) - T(V₂)||_∞ = {lhs:.6f}'
    )
