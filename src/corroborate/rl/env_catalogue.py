"""Env catalogue — typed metadata for the gymnax envs the
framework runs experiments on.

Mirrors poc_v9's `dqn/env.py` structure: `introspect_env(name)`
auto-extracts `action_dim`, `observation_shape`, `horizon` from
gymnax's spaces; the registration call only declares
non-introspectable metadata (reward bounds, regime,
benchmark family). Two corroborate-specific additions on top of
v9's shape:

- `state_hash` + `state_hash_cardinality`: per-env discretization
  for the Watkins (s, a)-coverage gap measurable. Vector-obs envs
  get a bucket-based factory; image-obs envs (minatar) ship
  `None` because state_hash_cardinality would be astronomical and
  the KL-against-uniform invariant has no useful signal there.

- All envs additionally expose `eval_episode_cap` — read directly
  from gymnax's `max_steps_in_episode`; used by the eval-loop
  infrastructure (Step 4.2) to cap greedy-rollout episode length.

Authors who add a new env: call `_register(name, r_min, r_max,
reward_regime, benchmark_family, state_hash=None|...)` once at
module import."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, TypedDict, runtime_checkable

import gymnax
import jax
import jax.numpy as jnp
import numpy as np


type RewardRegime = Literal[
    'per_step', 'event_triggered', 'shaped', 'terminal_only',
]
type BenchmarkFamily = Literal[
    'classic_control', 'minatar', 'bsuite', 'bandit', 'misc',
]
type ActionType = Literal['discrete', 'continuous']
type ObservationType = Literal['vector', 'image', 'structured']

type StateHash = Callable[[jax.Array], jax.Array]
"""(obs,) → integer bucket id. Single-obs (not batched). Used by
the Watkins (s, a)-coverage gap; image envs ship `None` because
the bucket cardinality is astronomical and KL-against-uniform
has no useful signal there."""


# ============ Structural Protocols for gymnax-side typing ============

@runtime_checkable
class GymnaxEnvLike(Protocol):
    """Structural Protocol for the env surface `corroborate.rl`
    consumes. gymnax `Env` instances satisfy structurally; any
    alternative env library matching this shape works too. Used
    to type `env` parameters in `dqn_step` / `rollout_phase` /
    `eval_episode` without importing `gymnax.Env` everywhere."""
    def reset(
        self, rng: jax.Array, params: object,
    ) -> tuple[jax.Array, object]: ...

    def step(
        self,
        rng: jax.Array,
        state: object,
        action: jax.Array,
        params: object,
    ) -> tuple[
        jax.Array, object, jax.Array, jax.Array, dict[str, object],
    ]: ...

    def observation_space(self, params: object) -> object: ...
    def action_space(self, params: object) -> object: ...


@runtime_checkable
class EnvWrapper(Protocol):
    """Anything that wraps a gymnax-style env in another
    `GymnaxEnvLike`. Frozen-dataclass implementations carry
    their config + a `wrap(inner)` method that returns the
    wrapped env. Composable: `cell_runner` applies a tuple of
    wrappers in order, so `(RewardScale(0.5), RewardClip(0.0,
    None))` first scales then clips.

    Add a new wrapper by:
      1. Define `@dataclass(frozen=True, slots=True) class
         FooWrapper: ... def wrap(self, inner) -> ...`
      2. Register: `_WRAPPER_REGISTRY['foo'] = FooWrapper`
      3. Use in YAML: `wrappers: [{type: foo, ...}]`

    No 7-place plumbing per wrapper — the registry is the only
    surface that grows."""
    def wrap(self, inner: 'GymnaxEnvLike') -> 'GymnaxEnvLike': ...


@dataclass(frozen=True, slots=True)
class RewardScale:
    """Wrapper config: multiply step reward by `scale`."""
    scale: float

    def wrap(self, inner: 'GymnaxEnvLike') -> 'GymnaxEnvLike':
        return RewardScaledEnv(inner=inner, scale=self.scale)


@dataclass(frozen=True, slots=True)
class RewardClip:
    """Wrapper config: clip step reward to `[clip_min, clip_max]`.
    Either bound may be None to disable that side."""
    clip_min: float | None = None
    clip_max: float | None = None

    def wrap(self, inner: 'GymnaxEnvLike') -> 'GymnaxEnvLike':
        return RewardClippedEnv(
            inner=inner, clip_min=self.clip_min, clip_max=self.clip_max,
        )


_WRAPPER_REGISTRY: dict[str, type[EnvWrapper]] = {
    'reward_scale': RewardScale,
    'reward_clip': RewardClip,
}
"""Name → wrapper class. YAML's `wrappers: [{type: <name>, ...}]`
parses each entry by looking up `<name>` here and instantiating
with the remaining kwargs."""


def get_wrapper_class(name: str) -> type[EnvWrapper]:
    """Look up a registered wrapper class by name. Raises
    KeyError with the registry's known names if missing."""
    cls = _WRAPPER_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f'no env wrapper registered as {name!r}; known: '
            f'{sorted(_WRAPPER_REGISTRY)}',
        )
    return cls


def wrappers_canonical_str(wrappers: tuple[EnvWrapper, ...]) -> str:
    """Stable-order canonical string for an env-wrapper tuple.
    Used as the suffix on `env_arm_tag` so `(RewardScale(0.5),
    RewardClip(0.0, None))` and `(RewardClip(0.0, None),
    RewardScale(0.5))` get distinct identities (order matters
    in wrap composition)."""
    if not wrappers:
        return ''
    parts: list[str] = []
    for w in wrappers:
        # Read fields off the dataclass via vars() — frozen + slots
        # means the asdict-style listing is stable.
        from dataclasses import fields
        cls_name = next(
            (k for k, v in _WRAPPER_REGISTRY.items() if v is type(w)),
            type(w).__name__,
        )
        kvs = ','.join(
            f'{f.name}={getattr(w, f.name)}' for f in fields(w)
        )
        parts.append(f'{cls_name}({kvs})')
    return ','.join(parts)


@dataclass(frozen=True, slots=True)
class RewardScaledEnv:
    """Wraps a gymnax-style env, scaling step reward by `scale`.

    Causal-probe lever: scaling reward changes mc_variance by
    `scale²` without altering dynamics, |A|, obs_dim, or the
    optimal policy. Used to test whether observed mc_variance →
    g_link associations are causal (vary the moderator
    interventionally) vs spurious (proxy for something correlated).

    Implements `GymnaxEnvLike` structurally — reset/step/spaces
    delegate to `inner`; only `step` is overridden to multiply
    reward by `scale`. The optimal Q* under the scaled reward is
    `scale * Q*_original`, so DDQN's Jensen gap (in Q-units)
    scales with `scale` while standardized comparisons (Hedges' g)
    are mostly invariant — the deviation from invariance is the
    causal signature."""
    inner: 'GymnaxEnvLike'
    scale: float

    def reset(
        self, rng: jax.Array, params: object,
    ) -> tuple[jax.Array, object]:
        return self.inner.reset(rng, params)

    def step(
        self,
        rng: jax.Array,
        state: object,
        action: jax.Array,
        params: object,
    ) -> tuple[
        jax.Array, object, jax.Array, jax.Array, dict[str, object],
    ]:
        next_obs, next_state, reward, done, info = self.inner.step(
            rng, state, action, params,
        )
        return next_obs, next_state, reward * self.scale, done, info

    def observation_space(self, params: object) -> object:
        return self.inner.observation_space(params)

    def action_space(self, params: object) -> object:
        return self.inner.action_space(params)


@dataclass(frozen=True, slots=True)
class RewardClippedEnv:
    """Wraps a gymnax-style env, clipping step reward to
    `[clip_min, clip_max]`. Either bound may be None to disable
    that side.

    Causal-probe lever for envs with stochastic mixed-sign
    reward (e.g. SpaceInvaders-MinAtar's +1-kill / −1-hit). Tests
    whether DDQN's bias-correction attenuation in such envs is
    driven by the negative-reward stochasticity: with
    `clip_min=0`, the negative tail vanishes; the optimal policy
    changes (no longer needs to weigh hit-cost against kill
    reward), but the test isolates whether DDQN's behaviour
    inherits the same attenuation pattern under positive-only
    reward."""
    inner: 'GymnaxEnvLike'
    clip_min: float | None = None
    clip_max: float | None = None

    def reset(
        self, rng: jax.Array, params: object,
    ) -> tuple[jax.Array, object]:
        return self.inner.reset(rng, params)

    def step(
        self,
        rng: jax.Array,
        state: object,
        action: jax.Array,
        params: object,
    ) -> tuple[
        jax.Array, object, jax.Array, jax.Array, dict[str, object],
    ]:
        next_obs, next_state, reward, done, info = self.inner.step(
            rng, state, action, params,
        )
        if self.clip_min is not None:
            reward = jnp.maximum(reward, self.clip_min)
        if self.clip_max is not None:
            reward = jnp.minimum(reward, self.clip_max)
        return next_obs, next_state, reward, done, info

    def observation_space(self, params: object) -> object:
        return self.inner.observation_space(params)

    def action_space(self, params: object) -> object:
        return self.inner.action_space(params)


@runtime_checkable
class MaxStepsParams(Protocol):
    """Marker Protocol — env params that declare a per-episode
    horizon. `isinstance` narrows to the typed attribute access,
    sidestepping `getattr(..., default)` discipline violation."""
    max_steps_in_episode: int


@runtime_checkable
class HasShape(Protocol):
    """Observation / action space surface — exposes `shape` (and
    optionally `n` for discrete spaces). `isinstance` narrows
    after `env.observation_space(params)` returns `object`."""
    shape: tuple[int, ...]


@runtime_checkable
class HasN(Protocol):
    """Discrete action space — `.n` is the action cardinality."""
    n: int


# ============ TypedDict for introspect_env return ============

class IntrospectedEnv(TypedDict):
    """Auto-derived fields from `gymnax.make(name)`. Replaces
    `dict[str, object]` so consumers (`_register`) get typed
    field access without `# type: ignore`."""
    name: str
    action_type: ActionType
    action_dim: int
    observation_shape: tuple[int, ...]
    observation_type: ObservationType
    horizon: int | None


@dataclass(frozen=True, slots=True)
class EnvSpec:
    """Static metadata for one gymnax env.

    Auto-introspected (read from gymnax at registration time):
    `action_type`, `n_actions`, `observation_shape`, `horizon`.
    For discrete envs `n_actions` is the action-space cardinality
    (gymnax's `act_space.n`); v0 only handles discrete envs, so
    the field name reflects that. Continuous-action envs require
    a separate field added when needed.

    Author-declared (registered via `_register`): `r_min`, `r_max`,
    `reward_regime`, `benchmark_family`, optional `state_hash` +
    `state_hash_cardinality`."""
    name: str
    action_type: ActionType
    n_actions: int
    observation_shape: tuple[int, ...]
    observation_type: ObservationType
    horizon: int | None
    r_min: float
    r_max: float
    reward_regime: RewardRegime
    benchmark_family: BenchmarkFamily
    state_hash: StateHash | None = None
    state_hash_cardinality: int | None = None
    benchmark_params: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({}),
    )

    @property
    def obs_dim(self) -> int:
        """Total observation dimensionality (flattened)."""
        return int(np.prod(self.observation_shape))

    @property
    def eval_episode_cap(self) -> int:
        """Max steps per eval episode. Reads gymnax's
        `max_steps_in_episode`; falls back to 1000 for envs
        without a declared horizon (bandits)."""
        return self.horizon if self.horizon is not None else 1000

    def public_attrs(self) -> dict[str, object]:
        """Whitelist of attributes the YAML sweep's
        `{from_env: <attr>}` placeholder is allowed to bind to.
        Explicit because dataclass-field introspection would let
        a private cache field silently grow the YAML schema —
        this method names the contract."""
        return {
            'name': self.name,
            'n_actions': self.n_actions,
            'observation_shape': self.observation_shape,
            'horizon': self.horizon,
            'r_min': self.r_min,
            'r_max': self.r_max,
        }


# ============ Introspection: read gymnax's spaces ============

def introspect_env(name: str) -> IntrospectedEnv:
    """Extract auto-derivable fields from gymnax's env+params.

    Returns a typed `IntrospectedEnv` so `_register` consumes
    fields without `# type: ignore`. Narrowing for action and
    observation spaces is via runtime-checkable Protocols
    (`HasShape`, `HasN`, `MaxStepsParams`) — no `getattr`."""
    env_obj, env_params = gymnax.make(name)
    act_space = env_obj.action_space(env_params)
    obs_space = env_obj.observation_space(env_params)

    if isinstance(obs_space, HasShape):
        shape = tuple(obs_space.shape)
    else:
        raise TypeError(
            f"env '{name}' observation_space lacks `shape`; "
            f'cannot introspect.',
        )

    if isinstance(act_space, HasN):
        is_discrete = True
        action_dim = int(act_space.n)
    elif isinstance(act_space, HasShape):
        is_discrete = False
        action_dim = int(np.prod(act_space.shape))
    else:
        raise TypeError(
            f"env '{name}' action_space has neither `.n` nor "
            f'`.shape`; cannot introspect.',
        )

    horizon: int | None = (
        env_params.max_steps_in_episode
        if isinstance(env_params, MaxStepsParams)
        else None
    )

    obs_type: ObservationType = (
        'vector' if len(shape) == 1
        else 'image' if len(shape) == 3
        else 'structured'
    )
    return IntrospectedEnv(
        name=name,
        action_type='discrete' if is_discrete else 'continuous',
        action_dim=action_dim,
        observation_shape=shape,
        observation_type=obs_type,
        horizon=horizon,
    )


# ============ State-hash factory for vector envs ============

def bucket_hash(
    lows: jax.Array,
    highs: jax.Array,
    n_buckets_per_dim: int = 10,
) -> tuple[StateHash, int]:
    """Per-dim bucketed hash for vector-obs envs. Returns
    `(state_hash_fn, cardinality)`.

    Discretizes each obs dimension into `n_buckets_per_dim` equal
    buckets between `lows[i]` and `highs[i]`, then encodes as a
    base-`n_buckets_per_dim` integer. For a 4-dim env with 10
    buckets/dim, cardinality = 10^4 = 10000.

    Out-of-bounds obs values clip to the boundary buckets (so the
    hash stays in [0, cardinality) regardless of input). Bounds
    are author-declared per env; gymnax's observation spaces don't
    consistently expose `low`/`high`."""
    obs_dim = int(lows.shape[0])
    cardinality = int(n_buckets_per_dim ** obs_dim)
    spans = highs - lows
    weights = jnp.power(
        n_buckets_per_dim, jnp.arange(obs_dim, dtype=jnp.int32),
    )

    def state_hash(obs: jax.Array) -> jax.Array:
        clipped = jnp.clip(obs, lows, highs)
        scaled = (clipped - lows) / jnp.maximum(spans, 1e-9)
        bucketed = jnp.clip(
            jnp.floor(scaled * n_buckets_per_dim),
            0, n_buckets_per_dim - 1,
        ).astype(jnp.int32)
        return jnp.sum(bucketed * weights)

    return state_hash, cardinality


# ============ Registry ============

ENV_REGISTRY: dict[str, EnvSpec] = {}


def _register(
    name: str,
    *,
    r_min: float,
    r_max: float,
    reward_regime: RewardRegime,
    benchmark_family: BenchmarkFamily,
    state_hash: StateHash | None = None,
    state_hash_cardinality: int | None = None,
    benchmark_params: dict[str, object] | None = None,
) -> None:
    """Register an env with the catalogue. Auto-introspects
    gymnax-derivable fields; the call site only specifies the
    metadata gymnax doesn't expose."""
    introspected = introspect_env(name)
    ENV_REGISTRY[name] = EnvSpec(
        name=introspected['name'],
        action_type=introspected['action_type'],
        n_actions=introspected['action_dim'],
        observation_shape=introspected['observation_shape'],
        observation_type=introspected['observation_type'],
        horizon=introspected['horizon'],
        r_min=r_min,
        r_max=r_max,
        reward_regime=reward_regime,
        benchmark_family=benchmark_family,
        state_hash=state_hash,
        state_hash_cardinality=state_hash_cardinality,
        benchmark_params=MappingProxyType(benchmark_params or {}),
    )


# Vector-obs envs with author-declared bounds for the bucket-
# hash discretization. Bounds are taken from gym's reference
# implementations; conservative envelopes — clipping handles
# excursions outside.

_CARTPOLE_HASH, _CARTPOLE_CARD = bucket_hash(
    lows=jnp.array([-2.4, -3.0, -0.21, -3.0]),
    highs=jnp.array([2.4, 3.0, 0.21, 3.0]),
    n_buckets_per_dim=10,
)
_ACROBOT_HASH, _ACROBOT_CARD = bucket_hash(
    lows=jnp.array([-1.0, -1.0, -1.0, -1.0, -12.566, -28.274]),
    highs=jnp.array([1.0, 1.0, 1.0, 1.0, 12.566, 28.274]),
    n_buckets_per_dim=6,
)
_MOUNTAINCAR_HASH, _MOUNTAINCAR_CARD = bucket_hash(
    lows=jnp.array([-1.2, -0.07]),
    highs=jnp.array([0.6, 0.07]),
    n_buckets_per_dim=20,
)


# Classic-control: vector obs, dense per-step reward.
_register(
    'CartPole-v1',
    r_min=0.0, r_max=1.0,
    reward_regime='per_step',
    benchmark_family='classic_control',
    state_hash=_CARTPOLE_HASH,
    state_hash_cardinality=_CARTPOLE_CARD,
)
_register(
    'Acrobot-v1',
    r_min=-1.0, r_max=0.0,
    reward_regime='per_step',
    benchmark_family='classic_control',
    state_hash=_ACROBOT_HASH,
    state_hash_cardinality=_ACROBOT_CARD,
)
_register(
    'MountainCar-v0',
    r_min=-1.0, r_max=0.0,
    reward_regime='per_step',
    benchmark_family='classic_control',
    state_hash=_MOUNTAINCAR_HASH,
    state_hash_cardinality=_MOUNTAINCAR_CARD,
)

# bsuite — small-scale theoretical benchmarks. Vector obs;
# state_hash deferred (bsuite envs encode their own canonical
# discrete states, but exposing them through gymnax requires
# per-env logic; ship without state_hash for v0).
_register(
    'Catch-bsuite',
    r_min=-1.0, r_max=1.0,
    reward_regime='terminal_only',
    benchmark_family='bsuite',
)
_register(
    'DeepSea-bsuite',
    r_min=-0.01, r_max=1.0,
    reward_regime='event_triggered',
    benchmark_family='bsuite',
)
_register(
    'MemoryChain-bsuite',
    r_min=-1.0, r_max=1.0,
    reward_regime='terminal_only',
    benchmark_family='bsuite',
)
_register(
    'UmbrellaChain-bsuite',
    r_min=-1.0, r_max=1.0,
    reward_regime='shaped',
    benchmark_family='bsuite',
)
_register(
    'DiscountingChain-bsuite',
    r_min=0.0, r_max=1.1,
    reward_regime='event_triggered',
    benchmark_family='bsuite',
)
_register(
    'MNISTBandit-bsuite',
    r_min=-1.0, r_max=1.0,
    reward_regime='terminal_only',
    benchmark_family='bsuite',
)

# Minatar — image obs (10×10×n_channels). state_hash skipped:
# bucket cardinality is astronomical, KL-against-uniform has no
# useful signal; the (s, a)-coverage gap reports `gap=0` (no-data)
# for these envs.
_register(
    'Asterix-MinAtar',
    r_min=0.0, r_max=1.0,
    reward_regime='event_triggered',
    benchmark_family='minatar',
)
_register(
    'Breakout-MinAtar',
    r_min=0.0, r_max=1.0,
    reward_regime='event_triggered',
    benchmark_family='minatar',
)
_register(
    'Freeway-MinAtar',
    r_min=0.0, r_max=1.0,
    reward_regime='event_triggered',
    benchmark_family='minatar',
)
_register(
    'SpaceInvaders-MinAtar',
    r_min=0.0, r_max=1.0,
    reward_regime='event_triggered',
    benchmark_family='minatar',
)

# Misc — small-scale theoretical / pedagogical envs.
_register(
    'FourRooms-misc',
    r_min=0.0, r_max=1.0,
    reward_regime='terminal_only',
    benchmark_family='misc',
)
_register(
    'MetaMaze-misc',
    r_min=0.0, r_max=10.0,
    reward_regime='event_triggered',
    benchmark_family='misc',
)
_register(
    'Pong-misc',
    r_min=0.0, r_max=1.0,
    reward_regime='per_step',
    benchmark_family='misc',
)

# Bandits — single-state, action-only.
_register(
    'BernoulliBandit-misc',
    r_min=0.0, r_max=1.0,
    reward_regime='per_step',
    benchmark_family='bandit',
)
_register(
    'GaussianBandit-misc',
    r_min=-5.0, r_max=5.0,
    reward_regime='per_step',
    benchmark_family='bandit',
)


# ============ Public lookups ============

def get(name: str) -> EnvSpec:
    """Lookup by gymnax name. Raises `KeyError` with a helpful
    message listing registered envs if `name` isn't registered."""
    if name not in ENV_REGISTRY:
        raise KeyError(
            f"env '{name}' not registered. Registered envs: "
            f'{sorted(ENV_REGISTRY)}',
        )
    return ENV_REGISTRY[name]


def envs_in_family(family: BenchmarkFamily) -> tuple[EnvSpec, ...]:
    """All registered envs in the given benchmark family.
    Iteration order is registration order."""
    return tuple(
        e for e in ENV_REGISTRY.values()
        if e.benchmark_family == family
    )


def all_envs() -> tuple[EnvSpec, ...]:
    """All registered envs, in registration order."""
    return tuple(ENV_REGISTRY.values())
