"""corroborate-rl — RL substrate for the corroborate framework.

Provides DQN training (cell_runner, dqn/), env catalogue with
solve thresholds, sweep dispatcher (yaml-driven and programmatic),
and DDQN measurables that compose with the framework's
`@measurable` registry.

This package depends on `corroborate` but is never imported by
the framework. Promoting it from a sub-package of `corroborate`
to a sibling package makes the substrate-neutrality rule structural
rather than cultural — `pip install corroborate` no longer pulls
JAX/optax/gymnax.

Public re-exports are added in a future phase once the migration
of source files is complete."""
