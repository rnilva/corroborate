"""Analytic tests for the substrate's truncation-aware Bellman
target (Pardo 2018 / Sutton-Barto §6.6 / Gymnasium-API).

Closed-form convergence assertions on a synthetic 1-state cycle env
that flows through the production `dqn_step` JAX training path.
Sibling motivation to `tabular/` (closed-form on the numpy
substrate) and `deadly_triad/` (closed-form on FQI cell panels):
each subpackage carries the analytic shape its substrate exposes.

The cycle env is structurally tabular (single observed state, cycle
back to self every step, +1 reward) so the optimal Q*(s) has a
closed form `1/(1-γ)`. Wrapped with a hard step-cap, the
truncation-aware bootstrap maintains the fixed point; the
treat-as-terminal regime converges instead to the geometric-sum
truncation `(1 − γᴹ)/(1 − γ)`. The two regimes' gap is large
enough (e.g., 5.9 at γ=0.9, M=5) to distinguish across multiple
seeds without ambiguity."""
