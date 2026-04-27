"""Per-sample loss — TD-error → scalar.

`squared_error` is v0's default. Huber / pinball / expectile
losses are alternative @claims for this slot (future work,
matching v9's per-sample-loss alternatives)."""
from __future__ import annotations

import jax

from corroborate.claim import claim


@claim
def squared_error(predicted: jax.Array, target: jax.Array) -> jax.Array:
    """Squared TD-error per sample. Mnih 2015 used Huber; squared
    is the simpler default and the natural starting point."""
    return (predicted - target) ** 2
