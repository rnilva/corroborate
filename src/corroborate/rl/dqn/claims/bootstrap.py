"""Bootstrap — Bellman target computation.

**The slot DDQN swaps.** Vanilla and DDQN differ only in whether
the target action is selected by the online or target network:

- vanilla: r + γ · max_a' Q_target(s', a')           — same net for both
- DDQN:    r + γ · Q_target(s', argmax_a' Q_online(s', a'))  — decoupled

Same call signature so the swap is a clean drop-in via
`partial(theory.dqn_step, bootstrap=ddqn_bootstrap)`.

**Theorem reference.** Bellman optimality T* is a γ-contraction
on Q* (Bertsekas-Tsitsiklis 1996 §6.3); vanilla bootstrap
implements T* directly. Hasselt 2010, 2016: vanilla DQN exhibits
a Jensen-inequality bias E[max_a Q̂(s,a)] ≥ max_a E[Q̂(s,a)],
because max is convex in Q. DDQN decouples action selection
(online net) from value evaluation (target net) so the two
estimators are roughly independent → unbiased target. The
`max_q_overestimation_bounded` invariant detects unbounded
Jensen-bias growth empirically."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from corroborate.claim import claim
from corroborate.rl.dqn.types import Params, QNetwork


@claim
def vanilla_bootstrap(
    *,
    online_params: Params,
    target_params: Params,
    q_network: QNetwork,
    next_obs: jax.Array,
    reward: jax.Array,
    done: jax.Array,
    gamma: float,
) -> jax.Array:
    """Vanilla DQN target: r + γ · max_a' Q_target(s', a').

    Both action selection AND evaluation use the target network.
    Mnih 2015 Algorithm 1 line: y = r + γ · max_a' Q̂(s', a'; θ⁻)."""
    del online_params  # unused — vanilla doesn't decouple
    next_q = q_network(target_params, next_obs)
    max_next_q = jnp.max(next_q, axis=-1)
    return reward + gamma * (1.0 - done) * max_next_q


@claim
def ddqn_bootstrap(
    *,
    online_params: Params,
    target_params: Params,
    q_network: QNetwork,
    next_obs: jax.Array,
    reward: jax.Array,
    done: jax.Array,
    gamma: float,
) -> jax.Array:
    """DDQN target: r + γ · Q_target(s', argmax_a' Q_online(s', a')).

    Action selection (online network) is decoupled from value
    evaluation (target network). Hasselt 2016: this asymmetry
    reduces vanilla DQN's Q-overestimation bias."""
    next_q_online = q_network(online_params, next_obs)
    a_star = jnp.argmax(next_q_online, axis=-1)
    next_q_target = q_network(target_params, next_obs)
    chosen = jnp.take_along_axis(
        next_q_target, a_star[..., None], axis=-1,
    ).squeeze(-1)
    return reward + gamma * (1.0 - done) * chosen
