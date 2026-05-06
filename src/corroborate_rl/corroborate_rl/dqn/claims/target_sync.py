"""Target sync — how target parameters follow online.

`periodic_copy` is the canonical hard-sync (Mnih 2015).
`polyak_update` is the soft-sync alternative
(`new_target = τ · online + (1 − τ) · target`, every step).

**Theorem reference.** Mnih 2015 §3 informally: freezing the
target net for τ steps makes the regression target stationary,
turning each τ-window into an FQI iteration (Munos 2003 fitted-
Q-iteration). FQI is a γ-contraction in sup-norm under
Lipschitz function-class assumptions. The
`fqi_decay_gap(sync_period, gamma)` measurable in
`invariants.py` reports the empirical decay rate's gap from γ
within each τ-window — the principled FQI signature.

Polyak's geometric-mixing analog (Lillicrap 2016 §3): with
soft-sync at rate τ, the target's effective "look-back window"
is ~1/τ steps; the equivalent FQI-window grows continuously
with smaller τ rather than stepping at sync_period. Direct
control of target staleness IS the τ knob."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from corroborate import claim
from corroborate_rl.dqn.types import Params


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


@claim
def polyak_update(
    *,
    online_params: Params,
    target_params: Params,
    step: jax.Array,
    sync_period: int,  # protocol-required; ignored by Polyak
    tau: float = 0.005,
) -> Params:
    """Soft sync every step: `target ← (1 − τ) · target + τ ·
    online`. Lillicrap 2016 used τ = 0.001 for DDPG; DQN soft-
    sync variants typically use τ ∈ [10⁻³, 10⁻²].

    `sync_period` is accepted to satisfy the `TargetSync` Protocol
    but unused — Polyak averages every step.

    `do(τ)` directly controls late-training staleness (`mean(|online
    − target| / max(...))` over the last 50%): higher τ → target
    follows online more aggressively → staleness ≈ 0; lower τ →
    target lags → high staleness. With sync_period collinear with
    staleness in periodic_copy sweeps (ρ = +0.95-0.97, cf.
    `findings_target_staleness_collinear.md`), Polyak-τ at FIXED
    sync_period is the canonical Pearl-rung-2 intervention to
    isolate staleness causality."""
    del step, sync_period  # Polyak averages every step.

    def blend(o: jax.Array, t: jax.Array) -> jax.Array:
        return (1.0 - tau) * t + tau * o

    return jax.tree.map(blend, online_params, target_params)
