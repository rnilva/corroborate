"""Action selection — rollout policy claims.

`EpsilonGreedy` is the canonical exploratory rollout, a Module
that owns its own ε-schedule as a field. The schedule swaps via
`replace(EpsilonGreedy(), schedule=other_schedule)` —
mechanism_key sees one entry (`action_select`) carrying the
nested configuration.

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

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from corroborate.claim import ClaimBase, claim, record_call


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


# Forward declaration — `EpsilonSchedule` Protocol lives in
# `types.py`. We import lazily inside `EpsilonGreedy` to avoid a
# circular import.
@dataclass(frozen=True, slots=True)
class EpsilonGreedy(ClaimBase):
    """ε-greedy action selection — Module with `schedule` field.

    With probability `schedule(step)`, sample uniformly from the
    action space; else argmax over Q-values. The schedule is a
    Module field so authors swap it via
    `replace(EpsilonGreedy(), schedule=cosine_epsilon)` or
    pass a different `ActionSelect` slot whole.

    Default schedule is `linear_epsilon` (un-baked — its own
    HP defaults `eps_init=1.0, eps_final=0.05, anneal_steps=
    10_000` apply). To bake schedule HPs at composition time,
    use `replace(EpsilonGreedy(), schedule=partial(
    linear_epsilon, anneal_steps=50_000))`."""
    # Schedule's type at runtime is `Claim` (the linear_epsilon
    # FnClaim singleton, or a partial wrapping it). Field type is
    # the EpsilonSchedule Protocol; pyright unions across concrete
    # implementations satisfying it.
    schedule: 'EpsilonSchedule' = field(default=linear_epsilon)  # pyright: ignore[reportAssignmentType]

    def __call__(
        self,
        q_values: jax.Array,
        rng_key: jax.Array,
        step: jax.Array,
        n_actions: int,
    ) -> jax.Array:
        epsilon = self.schedule(step)
        explore_key, action_key = jax.random.split(rng_key)
        explore = jax.random.uniform(explore_key) < epsilon
        random_action = jax.random.randint(action_key, (), 0, n_actions)
        greedy_action = jnp.argmax(q_values).astype(jnp.int32)
        result = jnp.where(explore, random_action, greedy_action)
        record_call(
            self, (q_values, rng_key, step, n_actions), {}, result,
        )
        return result


# Default ε-greedy instance — `epsilon_greedy` re-exported as the
# instance authors use as a default value.
epsilon_greedy = EpsilonGreedy()


# Late-import to satisfy the ForwardRef in `EpsilonGreedy.schedule`'s
# default annotation. Avoids the circular-import boomerang between
# action_select.py and types.py.
from corroborate.rl.dqn.types import EpsilonSchedule  # noqa: E402  pyright: ignore[reportUnusedImport]
