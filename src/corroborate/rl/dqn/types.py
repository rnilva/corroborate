"""Protocols — typed contracts for each component of `dqn`.

Each component in `dqn` is filled by a callable conforming to one
of these Protocols. The Protocol IS the component's interface: an
alternative implementation must structurally match the same
signature. DDQN's intervention is `{'greedification':
double_greedify}` (sub-Protocol of Bootstrap's composition).

Two flavours of component:

- **Module functor** — frozen-dataclass with paired `init` +
  `__call__` (e.g. `QFunction`). The dataclass fields are the
  construction-time HPs; calling the instance is the forward pass.
- **Stateless callable** — a `@claim` function with no init phase
  (e.g. `Bootstrap`, `LossFn`, `EpsilonSchedule`). The call IS the
  whole component.

The walker treats both uniformly: it recurses into a Claim's
function signature OR a frozen dataclass's fields, surfacing each
HP leaf."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import jax

# Re-exported for back-compat. Concrete Q-function implementations
# define their own `Params` shapes; dqn threads the opaque pytree.
from corroborate.rl.dqn.claims.q_network import Params

if TYPE_CHECKING:
    # Forward-ref only: `ReplayState` / `Batch` are concrete types
    # defined in `claims/replay.py`, which imports `BufferSample`
    # from this module. The TYPE_CHECKING guard breaks the runtime
    # circular import while still letting pyright resolve the names.
    from corroborate.rl.dqn.claims.replay import Batch, ReplayState


class QFunction(Protocol):
    """Q-function Module: paired `init` + `__call__`.

    `init(rng, obs_dim, n_actions) -> params` allocates the
    parameter pytree; `__call__(params, obs) -> q_values` is the
    forward pass. Implementations carry their construction-time
    HPs as frozen-dataclass fields (`MLP.hidden`,
    `SpectralNormMLP.hidden`, etc.) so HPs travel with the
    function — dqn doesn't see them.

    `Params` is opaque PyTree from dqn's perspective. Tabular,
    linear, and MLP Q-functions each define their own internal
    layout; the framework treats it as `dict[str, jax.Array]` for
    convenience but doesn't constrain the pytree shape further."""
    def init(
        self,
        rng_key: jax.Array,
        obs_dim: int,
        n_actions: int,
    ) -> Params: ...
    def __call__(self, params: Params, obs: jax.Array) -> jax.Array: ...


# Historical alias — kept so existing imports don't break while
# call sites migrate. New code should import `QFunction` directly.
QNetwork = QFunction


class ActionSelect(Protocol):
    """Rollout action-selection — e.g. `EpsilonGreedy`. Takes
    Q-values + RNG + the global step + n_actions; returns action
    index.

    `step` (not `epsilon` directly) because Module-style action
    selection owns its schedule internally — the slot's interface
    is what the rollout-loop has on hand at call time. ε-schedule
    swaps live as fields on the action-select Module
    (e.g. `EpsilonGreedy.schedule`)."""
    def __call__(
        self,
        q_values: jax.Array,
        rng_key: jax.Array,
        step: jax.Array,
        n_actions: int,
    ) -> jax.Array: ...


class EpsilonSchedule(Protocol):
    """Schedule mapping global step → ε. Linear / exponential /
    constant implementations all conform to this shape. Lives as
    a field on `ActionSelect` Modules (e.g. `EpsilonGreedy.schedule`),
    not as a top-level slot of `dqn`."""
    def __call__(self, step: jax.Array) -> jax.Array: ...


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
    `r + γ · (1−done) · gradient_rule(greedification(...))`.

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


class OptimizerFactory(Protocol):
    """Constructs an `optax.GradientTransformation` from typed
    construction HPs. Module-shaped (frozen-dataclass with
    `__call__`) so optimizer choice (Adam vs RMSProp vs SGD) +
    its HPs (lr, b1, b2, decay, eps, momentum, ...) canonicalise
    cleanly in mechanism_key.

    dqn calls `optimizer()` once at the top of a run to build
    the optax handle; train_phase consumes the raw handle's
    `.init` / `.update` interface JAX traces over."""
    def __call__(self) -> 'optax.GradientTransformation': ...


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
