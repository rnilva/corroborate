"""Protocols — typed contracts for each component of `dqn`.

Each component in `dqn` is filled by a callable conforming to one
of these Protocols. The Protocol IS the component's interface: an
alternative implementation must structurally match the same
signature. DDQN's intervention is `{'greedification':
double_greedify}` (sub-Protocol of Bootstrap's composition).

The Protocols here are **shape-agnostic** — they say what the
caller needs (a function with this signature) without committing
to which framework primitive provides it. Per
`CLAUDE.md` §"Three-way claim taxonomy", three primitive shapes
satisfy the contracts:

- **Free Claim** (`@claim`-decorated function): one pure
  operation, possibly with `schedule` / `greedification`-style
  sub-Claim kwargs. Conforms to call-signature Protocols
  (`Bootstrap`, `LossFn`, `EpsilonSchedule`, `ActionSelect`,
  `BufferSample`, `Greedification`, `GradientRule`,
  `TargetSync`).
- **Config bundle** (frozen dataclass with mechanics + slot
  Claims): satisfies bundle-shaped Protocols where the caller
  needs both stateful methods and pluggable behaviour. `MLP` /
  `CNN` satisfy `QFunction` (init mechanics + forward
  delegation); `Replay` carries init/add/sample_batch mechanics
  + a `BufferSample` slot.

The walker treats both shapes uniformly: it recurses into a Free
Claim's function signature OR a config bundle's dataclass
fields — surfacing each HP leaf at composition time."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import jax
import optax

# Re-exported for back-compat. Concrete Q-function implementations
# define their own `Params` shapes; dqn threads the opaque pytree.
from corroborate_rl.dqn.claims.q_network import Params

if TYPE_CHECKING:
    # Forward-ref only: `ReplayState` / `Batch` are concrete types
    # defined in `claims/replay.py`, which imports `BufferSample`
    # from this module. The TYPE_CHECKING guard breaks the runtime
    # circular import while still letting pyright resolve the names.
    from corroborate_rl.dqn.claims.q_network import QFunction
    from corroborate_rl.dqn.claims.replay import Batch, ReplayState


# `QFunction` Protocol lives in
# `corroborate_rl.dqn.claims.q_network`. `ActionSelect` and
# `EpsilonSchedule` Protocols live in
# `corroborate_rl.dqn.claims.action_select`. Sibling Protocols
# in this file reference forward types via TYPE_CHECKING-guarded
# strings to avoid runtime cycles (the claims modules import
# nothing from this module).


class Greedification(Protocol):
    """Compute v(s') from Q. The DDQN-vs-vanilla axis lives here.

    - `max_greedify` (vanilla): max_a Q_target(s', a)
    - `double_greedify` (DDQN): Q_target(s', argmax_a Q_online(s', a))

    Pure value computation — no reward, no gamma, no done. The
    `bootstrap` claim composes greedify+gradient_rule into the
    Bellman target."""
    def __call__(
        self,
        *,
        online_params: Params,
        target_params: Params,
        q_network: 'QFunction',
        next_obs: jax.Array,
    ) -> jax.Array: ...


class GradientRule(Protocol):
    """Apply a gradient-flow policy to the bootstrap target. The
    semi-gradient-vs-full-gradient axis lives here.

    - `semi_gradient` (Mnih 2015 default): `stop_gradient(target)`.
    - `full_gradient`: identity — gradient flows through.

    Operates on the assembled target `r + γ·(1−done)·v(s')`,
    not on `v(s')` alone, so the same rule covers both
    semi-gradient TD and full-gradient TD."""
    def __call__(self, target: jax.Array) -> jax.Array: ...


class Bootstrap(Protocol):
    """Bellman target. The default `bootstrap` composition is
    `reward + gamma · (1−done) · gradient_rule(greedification(...))`.

    `gamma` here is the BOOTSTRAP DISCOUNT — for n-step it equals
    γⁿ (computed by dqn_step from the env's γ); for 1-step it's
    just γ. `reward` is the (potentially-aggregated) n-step return
    precomputed during rollout by the `n_step_return` Free Claim.
    bootstrap itself doesn't need to know `n_step` — single leaf
    in the configuration surface.

    Keyword-only signature so the swap is a clean call-site drop-
    in. The DEFAULT swap-axis for DDQN-vs-vanilla is now
    `greedification` (sub-Protocol); swapping the entire
    `Bootstrap` is the wholesale alternative."""
    def __call__(
        self,
        *,
        online_params: Params,
        target_params: Params,
        q_network: 'QFunction',
        next_obs: jax.Array,
        reward: jax.Array,
        done: jax.Array,
        gamma: float,
    ) -> jax.Array: ...


class LossFn(Protocol):
    """Per-sample TD-error loss (predicted Q for the taken action
    vs bootstrap target). Squared / Huber / etc."""
    def __call__(self, predicted: jax.Array, target: jax.Array) -> jax.Array: ...


class BufferSample(Protocol):
    """Replay sampling strategy. Field of `Replay` Module; default
    is `uniform_sample`. PrioritisedReplay's swap is the same
    Protocol with side-car priorities.

    Returns a `Batch` (NamedTuple over the sampled transition
    arrays + indices)."""
    def __call__(
        self,
        *,
        state: 'ReplayState',
        rng_key: jax.Array,
        batch_size: int,
        capacity: int,
    ) -> 'Batch': ...


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


# `OptimizerFactory` Protocol was moved to
# `corroborate_rl.dqn.claims.optimizer` (Module → pure-functional
# refactor — Protocol lives with its implementations).


# Public type alias for `dqn_step`'s record output. The framework
# stores arbitrary scalars per step; this dict keys are
# semantic-role names ('loss', 'epsilon', 'max_q_current', ...).
type StepRecord = dict[str, jax.Array]


# Public alias for the env-handle pytree threaded through the
# step. gymnax envs are opaque structurally; `object` is honest
# (the framework cannot constrain a third-party env's pytree shape).
type EnvState = object


# Optax optimizer state — opaque pytree at runtime; re-export
# `optax.OptState` so the optimizer-boundary signatures match
# without a wrapper.
OptState = optax.OptState
