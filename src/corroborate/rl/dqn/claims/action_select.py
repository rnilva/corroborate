"""Action selection — rollout policy claims.

`epsilon_greedy` is the canonical exploratory rollout, a Free
Claim with the ε-schedule as a `schedule` kwarg. Authors swap the
schedule via `partial(epsilon_greedy, schedule=cosine_epsilon)`
or bake schedule HPs via `partial(epsilon_greedy, schedule=
partial(linear_epsilon, anneal_steps=50_000))` — same partial-
composition pattern as bootstrap's greedification slot or the
optimizer factories.

`linear_epsilon` is one schedule shape; alternatives
(exponential, cosine, piecewise) implement the same
`EpsilonSchedule` Protocol.

**Theorem references.**

ε-greedy leans on Watkins 1992 Q-learning convergence:
optimal-policy convergence requires every (s, a) visited
infinitely often. ε-greedy with ε > 0 satisfies this on any
finite MDP under sufficient training.

`linear_epsilon` is a *soft* GLIE schedule (Singh 2000): strict
GLIE requires ε → 0 ∧ Σε = ∞. Linear schedule violates strict
GLIE by construction (floors at `eps_final > 0`), so it
sacrifices asymptotic optimal-policy guarantees for finite-time
exploration. ε ∈ [0, 1] is a Kolmogorov-axiom static check
(asserted in the body)."""
from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp

from corroborate import claim


# ============ Protocols ============

class EpsilonSchedule(Protocol):
    """Schedule mapping global step → ε. Linear / exponential /
    constant implementations all conform to this shape. Lives as
    a kwarg on `epsilon_greedy` (slot Claim), not as a top-level
    slot of `dqn` — substrate-author authors swap via
    `partial(epsilon_greedy, schedule=...)`."""
    def __call__(self, step: jax.Array) -> jax.Array: ...


class ActionSelect(Protocol):
    """Rollout action-selection — Free Claim with signature
    `(q_values, rng_key, step, n_actions) -> action`.

    `step` (not `epsilon` directly) because the action-select
    Claim owns its schedule internally as a kwarg — the slot's
    interface is what the rollout-loop has on hand at call time.
    ε-schedule swaps live as `schedule` kwarg bake-ins
    (`partial(epsilon_greedy, schedule=...)`)."""
    def __call__(
        self,
        q_values: jax.Array,
        rng_key: jax.Array,
        step: jax.Array,
        n_actions: int,
    ) -> jax.Array: ...


# ============ Schedules ============

@claim
def linear_epsilon(
    step: jax.Array,
    *,
    eps_init: float = 1.0,
    eps_final: float = 0.05,
    anneal_steps: int = 10_000,
) -> jax.Array:
    """Linear ε schedule: anneal from `eps_init` at step 0 to
    `eps_final` at `anneal_steps`, constant afterwards."""
    assert 0.0 <= eps_final <= eps_init <= 1.0, (
        f'Kolmogorov axiom: ε must be a probability; got '
        f'eps_init={eps_init}, eps_final={eps_final}'
    )
    assert anneal_steps > 0, f'anneal_steps must be positive; got {anneal_steps}'
    progress = jnp.minimum(step / anneal_steps, 1.0)
    return eps_init + (eps_final - eps_init) * progress


# ============ Action-selection ============

@claim
def epsilon_greedy(
    q_values: jax.Array,
    rng_key: jax.Array,
    step: jax.Array,
    n_actions: int,
    *,
    schedule: EpsilonSchedule = linear_epsilon,
) -> jax.Array:
    """ε-greedy action selection — Free Claim with `schedule`
    sub-slot.

    With probability `schedule(step)`, sample uniformly from the
    action space; else argmax over Q-values. The schedule is a
    keyword Claim slot so authors swap it via `partial(
    epsilon_greedy, schedule=cosine_epsilon)` or pass a different
    `ActionSelect` slot whole.

    Default schedule is `linear_epsilon` (un-baked — its own HP
    defaults `eps_init=1.0, eps_final=0.05, anneal_steps=10_000`
    apply). To bake schedule HPs at composition time, use
    `partial(epsilon_greedy, schedule=partial(linear_epsilon,
    anneal_steps=50_000))`."""
    epsilon = schedule(step)
    explore_key, action_key = jax.random.split(rng_key)
    explore = jax.random.uniform(explore_key) < epsilon
    random_action = jax.random.randint(action_key, (), 0, n_actions)
    greedy_action = jnp.argmax(q_values).astype(jnp.int32)
    return jnp.where(explore, random_action, greedy_action)
