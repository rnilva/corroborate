"""Tabular MDP implementation — finite states + actions, exact Bellman
primitives, closed-form theorems testable to numerical precision.

Sibling to `corroborate_rl.dqn` (function-approximation regime,
sampling-based Bellman targets, replay buffer). The tabular regime
strips all approximation and stochasticity: the MDP carries exact
transition + reward tensors, Bellman backup is a tensor product,
and value iteration converges at the textbook γ-rate without
floating-point noise beyond machine precision.

Two roles:

1. **Theorems become testable as exact equalities.** γ-contraction
   in sup-norm, exponential VI convergence, finite-step PI
   termination, Hasselt's max-bias under Gaussian noise — every
   classical RL theorem has a closed form here, not a bound. The
   analytic tests at `tests/analytic/tabular/` assert these
   equalities to within machine precision (1e-9 typical).

2. **Textbook examples for substrate-grounded RL claims.**
   Sutton-and-Barto's random walk, two-state chains, cliff
   walking. The known optimal policies + value functions become
   closed-form ground truth for downstream DQN / DDQN bridges
   that ought to recover them in the limit.

The `@claim` Free Claims (`bellman_backup`, `policy_evaluation`,
`q_backup`, `greedy_policy`) carry their theorem references in
docstrings; the `TabularMDP` config bundle holds the dynamics."""
from corroborate_rl.tabular.bias import (
    double_greedify_tabular,
    hasselt_max_bias_asymptotic,
    hasselt_n2_max_bias,
    max_greedify_tabular,
)
from corroborate_rl.tabular.mdp import (
    TabularMDP,
    bellman_backup,
    greedy_policy,
    policy_evaluation,
    q_backup,
    value_iteration,
)
from corroborate_rl.tabular.examples import (
    random_walk,
    two_state_chain,
)

__all__ = [
    'TabularMDP',
    'bellman_backup',
    'double_greedify_tabular',
    'greedy_policy',
    'hasselt_max_bias_asymptotic',
    'hasselt_n2_max_bias',
    'max_greedify_tabular',
    'policy_evaluation',
    'q_backup',
    'random_walk',
    'two_state_chain',
    'value_iteration',
]
