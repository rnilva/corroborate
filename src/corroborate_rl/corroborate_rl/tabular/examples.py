"""Textbook tabular MDP examples.

Each constructor returns a `TabularMDP` matching a published
example whose optimal policy / value function is closed-form
known. Tests assert framework-computed quantities match those
closed forms.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from corroborate_rl.tabular.mdp import TabularMDP


type _FArr = npt.NDArray[np.float64]


def random_walk(
    n_internal_states: int = 5,
    *,
    gamma: float = 1.0,
    left_terminal_reward: float = 0.0,
    right_terminal_reward: float = 1.0,
) -> TabularMDP:
    """Sutton & Barto §6.2 random walk: linear chain of
    `n_internal_states` non-terminal states between two terminals.

    State indexing: 0 is left terminal, 1..n_internal are internal,
    n_internal + 1 is right terminal. Total states: n_internal + 2.

    One action per state (the random walk has no policy choice;
    the agent transitions left/right each with probability 0.5).
    Reward: 0 on every internal transition; `left_terminal_reward`
    on entering the left terminal; `right_terminal_reward` on
    entering the right terminal.

    **Closed-form V*** (Sutton & Barto eq. 6.4): for the canonical
    setup (γ=1, left_term=0, right_term=1), the value of internal
    state `i` (1-indexed) is `i / (n_internal + 1)`. This is the
    canonical "policy evaluation has a closed form" example: a
    random-walk reward problem solves to a linear-in-position
    function.

    Note: γ=1 is OUTSIDE the strict contraction regime (γ < 1),
    so `value_iteration` would not converge from arbitrary V_0
    via Bellman-iteration. Tests that need γ=1 closed forms use
    `policy_evaluation` (closed-form matrix inversion) directly.
    For γ < 1 tests, callers can pass a slightly-discounted gamma
    (e.g. 0.999) — the closed form shifts but stays computable.
    """
    if n_internal_states < 1:
        raise ValueError(
            f'n_internal_states must be ≥ 1; got {n_internal_states}',
        )
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(
            f'gamma must be in [0, 1]; got {gamma}',
        )
    n_states = n_internal_states + 2
    left_term = 0
    right_term = n_states - 1
    n_actions = 1  # random walk: single "step" action samples 50/50

    # Transitions: from internal state i, P(i-1) = 0.5, P(i+1) = 0.5.
    # Terminals absorb.
    transitions = np.zeros(
        (n_states, n_actions, n_states), dtype=np.float64,
    )
    rewards = np.zeros((n_states, n_actions), dtype=np.float64)
    for s in range(1, n_states - 1):  # internal states
        transitions[s, 0, s - 1] = 0.5
        transitions[s, 0, s + 1] = 0.5
    # Terminal absorption.
    transitions[left_term, 0, left_term] = 1.0
    transitions[right_term, 0, right_term] = 1.0
    # Reward credited on the *transition into* a terminal.
    # From state 1 (rightmost left-adjacent), 0.5 chance to enter
    # left terminal → expected reward contribution 0.5·left_term.
    # From state n_internal (leftmost right-adjacent), 0.5 chance
    # to enter right terminal → 0.5·right_term.
    if n_internal_states >= 1:
        rewards[1, 0] = 0.5 * left_terminal_reward
        rewards[n_states - 2, 0] = 0.5 * right_terminal_reward
    # If n_internal_states == 1, both contributions stack on
    # state 1 (since state 1 IS both 1 and n_states - 2).
    if n_internal_states == 1:
        rewards[1, 0] = (
            0.5 * left_terminal_reward
            + 0.5 * right_terminal_reward
        )

    return TabularMDP(
        n_states=n_states,
        n_actions=n_actions,
        transitions=transitions,
        rewards=rewards,
        gamma=gamma,
        terminal_states=frozenset({left_term, right_term}),
    )


def random_walk_closed_form_v(
    n_internal_states: int = 5,
    *,
    left_terminal_reward: float = 0.0,
    right_terminal_reward: float = 1.0,
) -> _FArr:
    """Closed-form V* for the random walk at γ=1.

    Internal state i (1-indexed within internal states, 0-indexed
    in the full state space at position i+1... wait let me redo).

    State indexing per `random_walk`:
        0           = left terminal
        1..N        = internal (where N = n_internal_states)
        N + 1       = right terminal

    Closed form (γ=1):
        V(internal i) = a + (i / (N + 1)) · (b - a)
        V(terminal)   = 0           (no future rewards from
                                     absorbing state — the
                                     entry-reward `a` or `b` is
                                     already baked into the
                                     predecessor's expected value)

    With `a = 0`, `b = 1`, `N = 5`:
        V = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 0]
        ↑ left term  internal 1..5      ↑ right term

    Returns the V vector of length n_internal_states + 2.
    """
    a = left_terminal_reward
    b = right_terminal_reward
    v = np.zeros(n_internal_states + 2, dtype=np.float64)
    for i in range(1, n_internal_states + 1):  # internal states only
        v[i] = a + (i / (n_internal_states + 1)) * (b - a)
    # Terminal V's stay 0 (already initialized).
    return v


def two_state_chain(
    *,
    gamma: float,
    p_self: float = 0.5,
    r_left: float = 0.0,
    r_right: float = 1.0,
) -> TabularMDP:
    """Minimal non-trivial MDP: state 0 (non-terminal) and state 1
    (terminal). One action; from state 0 the agent stays with
    probability `p_self` (collecting `r_left`) or transitions to
    state 1 with probability `1 - p_self` (collecting `r_right`).

    Closed-form V*(0):
        V(0) = E[immediate reward + γ · V(s')]
             = (p_self · r_left + (1 - p_self) · r_right)
                + γ · p_self · V(0) + γ · (1 - p_self) · 0
        V(0) · (1 - γ · p_self) = p_self · r_left + (1 - p_self) · r_right
        V(0) = (p_self · r_left + (1 - p_self) · r_right) / (1 - γ · p_self)

    V(1) = 0 (terminal).
    """
    if not (0.0 <= p_self <= 1.0):
        raise ValueError(f'p_self must be in [0, 1]; got {p_self}')
    transitions = np.zeros((2, 1, 2), dtype=np.float64)
    transitions[0, 0, 0] = p_self
    transitions[0, 0, 1] = 1.0 - p_self
    transitions[1, 0, 1] = 1.0  # terminal absorbs
    rewards = np.zeros((2, 1), dtype=np.float64)
    rewards[0, 0] = p_self * r_left + (1.0 - p_self) * r_right
    return TabularMDP(
        n_states=2, n_actions=1,
        transitions=transitions, rewards=rewards,
        gamma=gamma,
        terminal_states=frozenset({1}),
    )


def two_state_chain_closed_form_v(
    *,
    gamma: float,
    p_self: float = 0.5,
    r_left: float = 0.0,
    r_right: float = 1.0,
) -> _FArr:
    """Closed-form V vector for `two_state_chain`."""
    expected_immediate = p_self * r_left + (1.0 - p_self) * r_right
    v_0 = expected_immediate / (1.0 - gamma * p_self)
    return np.array([v_0, 0.0], dtype=np.float64)


__all__ = [
    'random_walk',
    'random_walk_closed_form_v',
    'two_state_chain',
    'two_state_chain_closed_form_v',
]
