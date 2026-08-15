"""Tabular MDP: frozen-dataclass dynamics + `@claim` Bellman
primitives.

Carries the transition tensor `P[s, a, s']`, expected-reward
tensor `R[s, a]`, discount `gamma`, and the set of terminal
(absorbing) states. The framework's two primitive shapes:

- `TabularMDP` is a config bundle (frozen dataclass, fields are
  the dynamics). Mechanics methods (`from_pairs`, validity check)
  are plain methods.
- `bellman_backup`, `policy_evaluation`, `q_backup`,
  `greedy_policy` are `@claim` Free Claims — each carries a
  theorem in its docstring (Bertsekas-Tsitsiklis §6.3 contraction,
  Howard 1960 PI improvement). `value_iteration` is mechanics
  glue around `bellman_backup`.

Strict typing: numpy arrays with declared dtype, `slots=True`
on the dataclass, no Any / cast / type ignore."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from corroborate import claim


type _FArr = npt.NDArray[np.float64]
type _IArr = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TabularMDP:
    """Finite-state finite-action MDP with explicit dynamics tensors.

    `transitions[s, a, s']` is the probability of transitioning to
    `s'` after taking action `a` in state `s` — must sum to 1
    along the last axis. `rewards[s, a]` is the expected immediate
    reward of taking action `a` in state `s`. `gamma ∈ [0, 1)`.

    `terminal_states` are absorbing: by convention transitions
    from a terminal state with any action go back to that same
    state with zero reward. Callers wiring up terminal dynamics
    should respect this; `_validate` checks it.

    States and actions are indexed by ints; the index → human
    name mapping is the caller's concern (constructed examples
    document their own state semantics)."""
    n_states: int
    n_actions: int
    transitions: _FArr
    rewards: _FArr
    gamma: float
    terminal_states: frozenset[int]

    def __post_init__(self) -> None:
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(
                f'gamma must be in [0, 1); got {self.gamma}',
            )
        expected_t_shape = (self.n_states, self.n_actions, self.n_states)
        if self.transitions.shape != expected_t_shape:
            raise ValueError(
                f'transitions shape {self.transitions.shape} != '
                f'expected {expected_t_shape}',
            )
        expected_r_shape = (self.n_states, self.n_actions)
        if self.rewards.shape != expected_r_shape:
            raise ValueError(
                f'rewards shape {self.rewards.shape} != '
                f'expected {expected_r_shape}',
            )
        # Row-stochastic: transition probs sum to 1 along s'.
        row_sums = np.sum(self.transitions, axis=-1)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            max_dev = float(np.max(np.abs(row_sums - 1.0)))
            raise ValueError(
                f'transition probabilities do not sum to 1 along '
                f'last axis; max deviation = {max_dev}',
            )
        # Terminal-state self-absorption sanity. Each (s_term, a)
        # must put all probability mass on s_term itself with zero
        # reward — otherwise the backup math breaks invariants
        # downstream.
        for s in self.terminal_states:
            if not (0 <= s < self.n_states):
                raise ValueError(
                    f'terminal state {s} out of range '
                    f'[0, {self.n_states})',
                )
            for a in range(self.n_actions):
                if not np.isclose(self.transitions[s, a, s], 1.0):
                    raise ValueError(
                        f'terminal state {s} action {a}: '
                        f'P(s|s,a) = {self.transitions[s, a, s]:.4f} '
                        f'!= 1 (terminal must be absorbing)',
                    )
                if not np.isclose(self.rewards[s, a], 0.0):
                    raise ValueError(
                        f'terminal state {s} action {a}: '
                        f'reward = {self.rewards[s, a]} != 0 '
                        f'(terminal reward must be 0; encode '
                        f'terminal rewards on the *transition into* '
                        f'the terminal state, not after)',
                    )


# ============ Bellman primitives — Free Claims ============

@claim
def bellman_backup(
    v: _FArr, transitions: _FArr, rewards: _FArr, gamma: float,
) -> _FArr:
    """Bellman optimality operator T applied once.

    `T(V)(s) = max_a [R(s, a) + γ · Σ_s' P(s'|s, a) · V(s')]`

    Bertsekas-Tsitsiklis 1996 §6.3: T is a γ-contraction in
    sup-norm on R^|S|. Banach fixed-point gives the unique V*
    satisfying T(V*) = V*, reachable by value iteration at
    geometric rate γ.

    `transitions` and `rewards` are passed explicitly (rather
    than packed into a TabularMDP arg) so this Claim composes
    cleanly with implementation-author hooks that mutate dynamics
    (e.g., a partial reward shaping). Plain functions of arrays
    are easier to canonicalise + record than dataclass-receivers."""
    # Expected next-state value: sum_s' P(s'|s,a) * V(s'). Shape (S, A).
    expected_next_v = transitions @ v  # (S, A, S) @ (S,) → (S, A)
    q_values = rewards + gamma * expected_next_v
    # Optimal action's value per state: max over A.
    return q_values.max(axis=1).astype(np.float64, copy=False)


@claim
def q_backup(
    q: _FArr, transitions: _FArr, rewards: _FArr, gamma: float,
) -> _FArr:
    """Q-form Bellman optimality operator.

    `T_Q(Q)(s, a) = R(s, a) + γ · Σ_s' P(s'|s, a) · max_a' Q(s', a')`

    Same contraction property as `bellman_backup` — fixed point
    is `Q*(s, a)` whose greedy policy is optimal."""
    v_from_q = q.max(axis=1)  # max_a Q(s, a) → V(s) shape (S,)
    expected_next_v = transitions @ v_from_q  # (S, A)
    return (rewards + gamma * expected_next_v).astype(np.float64, copy=False)


@claim
def policy_evaluation(
    policy: _IArr, transitions: _FArr, rewards: _FArr, gamma: float,
) -> _FArr:
    """Closed-form V_π via matrix inversion: `(I - γ P_π)^{-1} r_π`.

    Howard 1960 / Bertsekas §6.2: for a stationary deterministic
    policy π and γ < 1, the policy-induced reward + transition
    operator (P_π[s, s'] = P(s' | s, π(s)), r_π[s] = R(s, π(s)))
    yields V_π = r_π + γ · P_π · V_π → (I - γ P_π) V_π = r_π →
    V_π = (I - γ P_π)^{-1} r_π. The inversion is well-defined
    because (I - γ P_π) is strictly diagonally dominant for γ < 1.

    `policy` is the integer action per state, shape (n_states,)."""
    n_states = transitions.shape[0]
    state_indices = np.arange(n_states)
    p_pi = transitions[state_indices, policy]        # (S, S)
    r_pi = rewards[state_indices, policy]            # (S,)
    identity = np.eye(n_states, dtype=np.float64)
    a_matrix = identity - gamma * p_pi
    # `np.linalg.solve` returns floating[Any] in numpy stubs; the
    # runtime invariant under float64 inputs is float64 output.
    v_pi = np.linalg.solve(a_matrix, r_pi).astype(np.float64, copy=False)
    return v_pi


@claim
def greedy_policy(q: _FArr) -> _IArr:
    """Deterministic greedy policy: π(s) = argmax_a Q(s, a).

    Howard 1960 PI step: improving on V_π yields a policy at
    least as good. The argmax tie-breaking rule (here: numpy's
    leftmost-index) only affects ties' eventual choice, never
    optimality."""
    return q.argmax(axis=1).astype(np.int64, copy=False)


# ============ Mechanics: value iteration ============

def value_iteration(
    mdp: TabularMDP,
    *,
    tol: float = 1e-9,
    max_iter: int = 10_000,
) -> tuple[_FArr, int]:
    """Iterate `bellman_backup` until the sup-norm change drops
    below `tol`. Returns `(V*_estimate, n_iterations)`.

    Convergence is guaranteed at geometric rate γ
    (Bertsekas-Tsitsiklis 1996 Prop 6.2.3): after k iterations,
    `||V_k - V*||_∞ ≤ γ^k · ||V_0 - V*||_∞`. Exact value of V*
    is recoverable to machine precision when γ < 1.

    `max_iter` is a runaway guard for degenerate inputs (γ very
    close to 1 + tight `tol`). Default of 10k iterations covers
    γ ≤ 0.9999 at tol=1e-9 (need ~ log(tol) / log(γ) iterations,
    ~92k worst case for γ=0.9999 — at that scale the caller
    should bump `max_iter`)."""
    v = np.zeros(mdp.n_states, dtype=np.float64)
    for k in range(1, max_iter + 1):
        v_next = bellman_backup(
            v, mdp.transitions, mdp.rewards, mdp.gamma,
        )
        if float(np.max(np.abs(v_next - v))) < tol:
            return v_next, k
        v = v_next
    raise RuntimeError(
        f'value_iteration did not converge in {max_iter} '
        f'iterations at tol={tol} (gamma={mdp.gamma})',
    )


__all__ = [
    'TabularMDP',
    'bellman_backup',
    'greedy_policy',
    'policy_evaluation',
    'q_backup',
    'value_iteration',
]
