"""Minimal gymnax stubs — typed Env / EnvParams / Space surface.

Why this stub exists: gymnax does not ship type stubs upstream. Its
real signatures use a giant `EnvParams` union and Box/Discrete
unions for spaces; both leak `Unknown` / `Any` through the framework
boundary even though the corroborate-side contract is small. Rather
than relax pyright globally for `corroborate_rl`, we narrow at the
boundary with this stub.

Level 3 design: typed end-to-end. `Box` and `Discrete` declare
their attributes; spaces returned from `Env.observation_space` /
`Env.action_space` are typed `Box` / `Discrete` directly (every
registered env in the substrate's catalogue conforms — verified by
walking `env_catalogue` registrations). `EnvParams` declares
`max_steps_in_episode: int` because the gymnax base
`environment.EnvParams` ships it as a default — every concrete
env's params subclass inherits it, so direct attribute access is
type-honest without `isinstance(_, MaxStepsParams)` self-Protocol
narrowing. `StepEnvParams` is kept as an optional refinement
Protocol for consumers that want a tighter bound (rare).
`EnvState` is opaque (an env-specific pytree) but typed-distinct
from `object` so it flows through the substrate's
`DQNState.env_state` slot without collapsing to bare-object.

`Env` is a `Protocol` — gymnax's runtime `Env` class isn't exported
at top-level (it lives at `gymnax.environments.environment.Environment`),
so substrate wrappers can't inherit from `gymnax.Env` at runtime.
A Protocol lets substrate wrappers (`RewardScaledEnv`, …) match
structurally, and gymnax's real `Environment` subclasses match
structurally too because their method signatures align."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax


class EnvState:
    """Opaque environment-state pytree. Concrete shape is per-env;
    the substrate never inspects it (just threads through scan)."""
    ...


class EnvParams:
    """Opaque environment-parameter container. The gymnax runtime
    base (`gymnax.environments.environment.EnvParams`) declares
    `max_steps_in_episode: int = 1` — every concrete env's params
    inherits it. Direct attribute access on a typed `EnvParams`
    therefore needs no `isinstance` narrowing."""
    max_steps_in_episode: int


@runtime_checkable
class StepEnvParams(EnvParams, Protocol):
    """Optional refinement Protocol — same shape as `EnvParams`
    but flagged Protocol for callers that want a tighter bound on
    "env params declaring a per-episode horizon." Most substrate
    code can use `EnvParams` directly; this Protocol exists for
    callers writing generic helpers parametric in "anything with
    `max_steps_in_episode`."""
    max_steps_in_episode: int


class Space:
    """Abstract base for action / observation spaces. `Box` and
    `Discrete` are the concrete classes the substrate actually
    consumes; this base exists only because the gymnax runtime
    declares it."""
    ...


class Box(Space):
    """Continuous-bounded space — exposes `shape`, `low`, `high`.
    Used as the observation space for every registered env in the
    substrate's catalogue (vector obs envs declare per-dim bounds;
    image obs envs declare full-array bounds)."""
    shape: tuple[int, ...]
    low: jax.Array
    high: jax.Array

    def __init__(
        self,
        low: jax.Array | float,
        high: jax.Array | float,
        shape: tuple[int, ...],
        dtype: object = ...,
    ) -> None: ...


class Discrete(Space):
    """Discrete space — exposes `n` (action cardinality) and
    `shape` (always `()` for scalar discrete actions). Used as
    the action space for every registered env in the substrate's
    catalogue (all DQN-eligible envs are discrete-action)."""
    n: int
    shape: tuple[int, ...]

    def __init__(self, num_categories: int) -> None: ...


class Env(Protocol):
    """Gymnax environment Protocol. Methods are typed end-to-end:
    `params: EnvParams` flows in, typed values flow out.

    Method signatures match gymnax's `Environment` base in
    `gymnax/environments/environment.py`. Substrate wrapper envs
    (`RewardScaledEnv`, `RewardClippedEnv`, `ActionDuplicatedEnv`)
    satisfy structurally — no inheritance needed (gymnax's runtime
    `Env` is at `gymnax.environments.environment.Environment`, not
    at `gymnax.Env`).

    `observation_space` returns `Box` and `action_space` returns
    `Discrete` for every env the substrate currently runs (verified
    by walking `env_catalogue`'s registrations). Tighter than the
    runtime upstream signature (which is unannotated `-> Any`) but
    accurate for the substrate's corpus; if a future env breaks
    this — say a continuous-action one — narrow this Protocol to
    `Box | Discrete` at that point."""
    def reset(
        self, rng: jax.Array, params: EnvParams,
    ) -> tuple[jax.Array, EnvState]: ...

    def step(
        self,
        rng: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[
        jax.Array,           # next_obs
        EnvState,            # next_state
        jax.Array,           # reward
        jax.Array,           # done
        dict[str, object],   # info
    ]: ...

    def observation_space(self, params: EnvParams) -> Box: ...
    def action_space(self, params: EnvParams) -> Discrete: ...


def make(env_id: str, **env_kwargs: object) -> tuple[Env, EnvParams]: ...
