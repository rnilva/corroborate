"""Target sync — how target parameters follow online.

`periodic_copy` is the canonical hard-sync (Mnih 2015). Polyak
averaging (`new_target = τ · online + (1 - τ) · target`) is an
alternative @claim for this slot; future work.

**Theorem reference.** Mnih 2015 §3 informally: freezing the
target net for τ steps makes the regression target stationary,
turning each τ-window into an FQI iteration (Munos 2003 fitted-
Q-iteration). FQI is a γ-contraction in sup-norm under
Lipschitz function-class assumptions. The
`fqi_decay_gap(sync_period, gamma)` measurable in
`invariants.py` reports the empirical decay rate's gap from γ
within each τ-window — the principled FQI signature."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from corroborate import claim
from corroborate.rl.dqn.types import Params


@claim
def periodic_copy(
    *,
    online_params: Params,
    target_params: Params,
    step: jax.Array,
    sync_period: int = 100,
) -> Params:
    """Hard sync every `sync_period` steps. On the sync step,
    copy online parameters into target; otherwise leave target
    unchanged. Mnih 2015 used this with sync_period = 10_000."""
    should_sync = (step % sync_period) == 0

    def select(o: jax.Array, t: jax.Array) -> jax.Array:
        return jnp.where(should_sync, o, t)

    return jax.tree.map(select, online_params, target_params)
