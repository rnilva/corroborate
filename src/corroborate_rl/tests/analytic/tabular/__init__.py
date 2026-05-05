"""Analytic tests for the tabular MDP substrate.

Tabular regime → exact computations → exact theorems testable to
within machine precision (1e-9 typical), no sampling SE bounds
needed. Each test asserts a textbook RL theorem holds:

- γ-contraction in sup-norm (Bertsekas-Tsitsiklis 1996 Prop 6.2.3)
- Value-iteration convergence at geometric rate γ
- Policy-iteration finite-step termination + per-step improvement
- Closed-form V_π via matrix inversion (Howard 1960)
- Bellman optimality fixed point: T(V*) = V*
- Closed-form V on Sutton & Barto's random walk

Sibling to `tests/analytic/deadly_triad/` (the FQI bound + Q-
divergence story for the function-approximation regime). Together
these cover the two halves of RL claim-testing: the tabular
half that's exactly testable, and the FA half that's bounded
testable."""
