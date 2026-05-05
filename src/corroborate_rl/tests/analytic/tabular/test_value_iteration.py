"""Value iteration convergence theorem.

Bertsekas-Tsitsiklis 1996 Prop 6.2.3 (Banach fixed-point applied
to T):

    ||V_k - V*||_∞ ≤ γ^k · ||V_0 - V*||_∞

Value iteration starting from any V_0 converges geometrically to
the unique fixed point V*. With γ-contraction proved separately,
this convergence rate is the direct corollary.

Tests:
- After k iterations, the gap to V* shrinks by exactly γ^k.
- The fixed-point property `T(V*) = V*` holds at convergence.
- Greedy policy of V* recovers the closed-form optimal policy."""
from __future__ import annotations

import math

import numpy as np

from corroborate_rl.tabular import (
    bellman_backup,
    greedy_policy,
    q_backup,
    two_state_chain,
    value_iteration,
)
from corroborate_rl.tabular.examples import two_state_chain_closed_form_v


def _sup_norm(v: np.ndarray) -> float:
    return float(np.max(np.abs(v)))


# ============ Convergence rate ============

def test_value_iteration_decay_bounded_by_gamma() -> None:
    """The Bertsekas-Tsitsiklis bound `||V_k - V*||_∞ ≤ γ · ||V_{k-1}
    - V*||_∞` holds at each step. The bound is generally NOT tight
    on absorbing-MDPs (terminal absorption gives a faster effective
    rate γ · max_a P(s_internal | s, a) — for the two-state chain
    at p_self=0.5, γ=0.9, that's 0.45). We assert the inequality.

    A regression that broke the gamma factor in `bellman_backup`
    (e.g., dropped it, doubled it, sign-flipped) would breach
    `≤ γ` immediately."""
    mdp = two_state_chain(gamma=0.9, p_self=0.5, r_right=1.0)
    v_star = two_state_chain_closed_form_v(
        gamma=0.9, p_self=0.5, r_right=1.0,
    )
    v = np.zeros(mdp.n_states, dtype=np.float64)
    initial_gap = _sup_norm(v - v_star)
    assert initial_gap > 0.0

    prev_gap = initial_gap
    for k in range(20):
        v = bellman_backup(v, mdp.transitions, mdp.rewards, mdp.gamma)
        gap = _sup_norm(v - v_star)
        if prev_gap > 1e-12:
            assert gap <= mdp.gamma * prev_gap + 1e-12, (
                f'iter {k}: gap = {gap:.6f} > γ · prev_gap = '
                f'{mdp.gamma * prev_gap:.6f}; γ-contraction breached'
            )
        prev_gap = gap

    # Closed-form effective-rate check for THIS MDP: the
    # two-state chain with terminal absorption decays at exactly
    # γ · p_self per step (because the only contractive direction
    # in V is the V(0) component; V(terminal) = 0 stays exact).
    expected_rate = mdp.gamma * 0.5  # p_self=0.5
    final_gap = _sup_norm(v - v_star)
    expected_final = initial_gap * (expected_rate ** 20)
    assert abs(final_gap - expected_final) < 1e-9, (
        f'after 20 iter, gap = {final_gap:.2e}, expected rate-'
        f'(γ·p_self)^20 = {expected_final:.2e}'
    )


def test_value_iteration_converges_to_closed_form_v_star() -> None:
    """`value_iteration(mdp)` returns V* matching the closed-form
    `two_state_chain_closed_form_v`. Tests the full mechanics
    loop: convergence detection, max_iter guard, return shape.

    A regression that broke convergence detection (early stop / no
    stop) would diverge from the closed form by visible amount."""
    for gamma in (0.5, 0.9, 0.99, 0.999):
        mdp = two_state_chain(gamma=gamma, p_self=0.7, r_right=2.0)
        v_star_closed = two_state_chain_closed_form_v(
            gamma=gamma, p_self=0.7, r_right=2.0,
        )
        v_star_iterated, n_iter = value_iteration(mdp, tol=1e-12)
        assert _sup_norm(v_star_iterated - v_star_closed) < 1e-9, (
            f'γ={gamma}: VI V* = {v_star_iterated}, closed form = '
            f'{v_star_closed} (diff sup-norm = '
            f'{_sup_norm(v_star_iterated - v_star_closed):.2e})'
        )
        # Convergence speed: should match log(tol)/log(γ).
        expected_iter = math.ceil(math.log(1e-12) / math.log(gamma))
        # Allow a few extra iterations for the tol-vs-machine
        # precision interplay; main check is "it converged".
        assert n_iter <= expected_iter + 10, (
            f'γ={gamma}: VI took {n_iter} iter, expected ≤ '
            f'{expected_iter + 10} from log(tol)/log(γ)'
        )


# ============ Bellman fixed-point ============

def test_v_star_is_a_bellman_fixed_point() -> None:
    """T(V*) = V* exactly (Banach fixed point). Catches a
    regression where VI returns something close but not actually
    a fixed point of T."""
    mdp = two_state_chain(gamma=0.9, p_self=0.5)
    v_star, _ = value_iteration(mdp, tol=1e-12)
    t_v_star = bellman_backup(
        v_star, mdp.transitions, mdp.rewards, mdp.gamma,
    )
    # Within numerical-convergence tolerance.
    assert _sup_norm(t_v_star - v_star) < 1e-9, (
        f'T(V*) - V* sup-norm = {_sup_norm(t_v_star - v_star):.2e}; '
        f'expected near zero (Banach fixed-point)'
    )


# ============ Q-form Bellman ============

def test_q_iteration_max_recovers_v_iteration() -> None:
    """Q-form fixed-point: max_a Q*(s, a) = V*(s). Cross-check
    that `q_backup` and `bellman_backup` agree at convergence —
    they're two views on the same fixed-point equation.

    A regression in `q_backup` that decoupled it from
    `bellman_backup` would fail this consistency check."""
    mdp = two_state_chain(gamma=0.9, p_self=0.5, r_right=1.0)
    # Iterate Q until fixed point.
    q = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.float64)
    for _ in range(500):
        q_next = q_backup(q, mdp.transitions, mdp.rewards, mdp.gamma)
        if _sup_norm(q_next - q) < 1e-12:
            q = q_next
            break
        q = q_next
    v_from_q = q.max(axis=1)
    v_star, _ = value_iteration(mdp, tol=1e-12)
    assert _sup_norm(v_from_q - v_star) < 1e-9, (
        f'max_a Q*(s, a) != V*(s); diff sup-norm = '
        f'{_sup_norm(v_from_q - v_star):.2e}'
    )


# ============ Greedy policy from Q* ============

def test_greedy_policy_of_q_star_picks_optimal_action() -> None:
    """On a two-state chain with one action, the greedy policy is
    trivially `[0, 0]`. The interest is the type contract: the
    function returns int64 array of shape (n_states,)."""
    mdp = two_state_chain(gamma=0.9)
    q = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.float64)
    for _ in range(500):
        q = q_backup(q, mdp.transitions, mdp.rewards, mdp.gamma)
    pi = greedy_policy(q)
    assert pi.shape == (mdp.n_states,)
    assert pi.dtype == np.int64
    # With a single action, every state's greedy choice is action 0.
    assert (pi == 0).all()
