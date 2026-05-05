"""Analytic substrate tests for the RL substrate's framework
contributions.

Mirrors the framework-side `tests/analytic/lg_scm/` pattern:
substrate-grounded cells with closed-form analytical bounds on
the framework primitives that consume them.

Subpackages:

- `deadly_triad/` — closed-form FQI / Bellman-bound assertions
  on `q_divergence_score`, `fqi_decay_gap`, and the cross-cell
  panel pattern that detects the deadly-triad failure mode."""
