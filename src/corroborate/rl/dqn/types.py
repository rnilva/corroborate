"""Slot Protocols — typed contracts for each `dqn_step` slot.

Each slot in `theory.dqn_step` is filled by a callable conforming
to one of these Protocols. The Protocol IS the slot's interface:
an alternative implementation must structurally match the same
signature. DDQN's intervention is `{'bootstrap': ddqn_bootstrap}`
where `ddqn_bootstrap` satisfies `Bootstrap`.

Discipline-wise, this turns the "default-arg Callable" pattern
into a typed-contract pattern: pyright catches a mismatched
alternative at the call site, not at runtime."""
from __future__ import annotations

from typing import Protocol

import jax


type Params = dict[str, jax.Array]


class QNetwork(Protocol):
    """Q-value forward pass: parameter pytree + observation →
    Q-values. Single-obs and batched-obs both supported."""
    def __call__(self, params: Params, obs: jax.Array) -> jax.Array: ...


class ActionSelect(Protocol):
    """Rollout action selection (e.g. ε-greedy). Takes Q-values
    + RNG + (technique-specific kwargs); returns action index.

    `epsilon` is `jax.Array` (not `float`) because schedules return
    a traced array under `jax.lax.scan`; coercing to float there is
    a ConcretizationTypeError."""
    def __call__(
        self,
        q_values: jax.Array,
        rng_key: jax.Array,
        epsilon: jax.Array,
        n_actions: int,
    ) -> jax.Array: ...


class EpsilonSchedule(Protocol):
    """Schedule mapping global step → ε. Linear / exponential /
    constant implementations all conform to this shape."""
    def __call__(self, step: jax.Array) -> jax.Array: ...


class Bootstrap(Protocol):
    """Bellman target. **The slot DDQN swaps**: vanilla and DDQN
    differ only in whether the online network or target network
    selects the action used in the bootstrap target.

    Keyword-only signature so the swap is a clean call-site drop-
    in (positional args could let an alternative silently re-order
    online_params and target_params)."""
    def __call__(
        self,
        *,
        online_params: Params,
        target_params: Params,
        q_network: QNetwork,
        next_obs: jax.Array,
        reward: jax.Array,
        done: jax.Array,
        gamma: float,
    ) -> jax.Array: ...


class LossFn(Protocol):
    """Per-sample TD-error loss (predicted Q for the taken action
    vs bootstrap target). Squared / Huber / etc."""
    def __call__(self, predicted: jax.Array, target: jax.Array) -> jax.Array: ...


class TargetSync(Protocol):
    """Target-network update rule. Periodic copy / Polyak average."""
    def __call__(
        self,
        *,
        online_params: Params,
        target_params: Params,
        step: jax.Array,
        sync_period: int,
    ) -> Params: ...


# Replay-buffer shape is six arrays — too many to fit a clean
# Protocol per call. Convention: callers pass the buffer's
# component arrays explicitly (as kwargs on `dqn_step`); the
# `buffer_*` claims below take state and a few side-args. A
# tighter Protocol-shaped buffer interface lands when the second
# buffer impl (e.g. PrioritisedReplay) forces the shape.

# Public type alias for `dqn_step`'s record output. The framework
# stores arbitrary scalars per step; this dict keys are
# semantic-role names ('loss', 'epsilon', 'max_q_current', ...).
type StepRecord = dict[str, jax.Array]


# Public alias for the env-handle pytree threaded through the
# step. gymnax envs are opaque structurally; `object` is honest
# (the framework cannot constrain a third-party env's pytree shape).
type EnvState = object


# Optax optimizer state — opaque pytree at runtime; optax exposes
# `optax.OptState` which is a `chex.ArrayTree` alias. We re-export
# their type so the optimizer-boundary signatures match without a
# wrapper.
import optax
OptState = optax.OptState
