"""Per-sample loss — TD-error → scalar.

`squared_error` is v0's default. Huber / pinball / expectile
losses are alternative @claims for this slot (future work,
matching v9's per-sample-loss alternatives).

**Theorem reference.** Semi-gradient TD (Sutton-Barto §11.2):
the bootstrap target y is treated as constant in the gradient,
so this is *not* a true gradient method. Convergence holds in
tabular Q-learning (Watkins 1992) but linear FA + off-policy
data can diverge (Baird 1995 counterexample). DQN's mitigation
is target-net + experience replay; deadly-triad divergence
remains possible. Principled signature is the projected-
Bellman-operator eigenvalue spectrum; deferred — needs Q-value
probe extension."""
from __future__ import annotations

import jax

from corroborate.core.claim import claim


@claim
def squared_error(predicted: jax.Array, target: jax.Array) -> jax.Array:
    """Squared TD-error per sample. Mnih 2015 used Huber; squared
    is the simpler default and the natural starting point."""
    return (predicted - target) ** 2
