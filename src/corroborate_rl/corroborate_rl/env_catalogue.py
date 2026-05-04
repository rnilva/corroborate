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
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, runtime_checkable

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from gymnax import EnvParams, EnvState
from gymnax.environments import spaces

if TYPE_CHECKING:
    # `Env` / `Box` / `Discrete` are stub-only typed surfaces —
    # gymnax's runtime exposes `Environment` (not `Env`) via
    # `gymnax.environments.environment`, and `Box` / `Discrete` via
    # `gymnax.environments.spaces` (the runtime `spaces` module
    # imported above). With `from __future__ import annotations`
    # enabled, the substrate's annotations referencing these names
    # are stringified — never resolved at runtime — so the
    # TYPE_CHECKING import is the right tool: pyright sees the
    # typed Protocol surface from the stub, the runtime never
    # tries to import a name that doesn't exist on the live
    # gymnax module.
    from gymnax import Box, Discrete, Env


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


# Substrate-side env / space / params types come straight from the
# gymnax stub — `Env` (Protocol) / `Box` / `Discrete` / `EnvParams`
# / `EnvState`. Wrapper envs below match `Env` structurally; spaces
# returned from `env.action_space(params)` are already typed
# `Discrete`, so the prior `HasN` / `HasShape` / `MaxStepsParams`
# self-Protocols are redundant and have been dropped.


@runtime_checkable
class EnvWrapper(Protocol):
    """Anything that wraps a gymnax-style `Env` in another `Env`.
    Frozen-dataclass implementations carry their config + a
    `wrap(inner)` method that returns the wrapped env. Composable:
    `cell_runner` applies a tuple of wrappers in order, so
    `(RewardScale(0.5), RewardClip(0.0, None))` first scales then
    clips.

    Each wrapper also declares its own `measurement_keys()` —
    the per-cell scalar columns it contributes to the persisted
    `RunRow.measurements`. This keeps cell_runner generic: it
    iterates `for w in wrappers: cols.update(w.measurement_keys())`
    rather than hardcoding `if isinstance(w, RewardScale): …`.

    Add a new wrapper by:
      1. Define `@dataclass(frozen=True, slots=True) class
         FooWrapper: ... def wrap(self, inner) -> ...
                      def measurement_keys(self) -> Mapping[...]: ...`
      2. Register: `_WRAPPER_REGISTRY['foo'] = FooWrapper`
      3. Use in YAML: `wrappers: [{type: foo, ...}]`

    No 7-place plumbing per wrapper — the wrapper class is the
    only surface that grows."""
    def wrap(self, inner: Env) -> Env: ...
    def measurement_keys(self) -> Mapping[str, float]: ...


@dataclass(frozen=True, slots=True)
class RewardScale:
    """Wrapper config: multiply step reward by `scale`."""
    scale: float

    def wrap(self, inner: Env) -> Env:
        return RewardScaledEnv(inner=inner, scale=self.scale)

    def measurement_keys(self) -> Mapping[str, float]:
        return {'reward_scale': float(self.scale)}


@dataclass(frozen=True, slots=True)
class RewardClip:
    """Wrapper config: clip step reward to `[clip_min, clip_max]`.
    Either bound may be None to disable that side."""
    clip_min: float | None = None
    clip_max: float | None = None

    def wrap(self, inner: Env) -> Env:
        return RewardClippedEnv(
            inner=inner, clip_min=self.clip_min, clip_max=self.clip_max,
        )

    def measurement_keys(self) -> Mapping[str, float]:
        out: dict[str, float] = {}
        if self.clip_min is not None:
            out['reward_clip_min'] = float(self.clip_min)
        if self.clip_max is not None:
            out['reward_clip_max'] = float(self.clip_max)
        return out


@dataclass(frozen=True, slots=True)
class ActionDuplicate:
    """Wrapper config: inflate action space by integer factor `k`.
    Action `i` for `i in [0, k * inner_n)` maps to inner action
    `i % inner_n`. Same dynamics, same optimal Q*, only declared
    `|A|` changes — Hasselt floor scales as √(2 log(k * inner_n))."""
    k: int

    def wrap(self, inner: Env) -> Env:
        return ActionDuplicatedEnv(inner=inner, k=self.k)

    def measurement_keys(self) -> Mapping[str, float]:
        return {'action_duplicate_k': float(self.k)}


_WRAPPER_REGISTRY: dict[str, type[EnvWrapper]] = {
    'reward_scale': RewardScale,
    'reward_clip': RewardClip,
    'action_duplicate': ActionDuplicate,
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
        # Read fields off the dataclass via fields() — frozen +
        # slots means the asdict-style listing is stable. The
        # `is_dataclass(w) and not isinstance(w, type)` narrow
        # satisfies stdlib's `DataclassInstance` Protocol; the
        # EnvWrapper Protocol itself doesn't declare
        # `__dataclass_fields__` because not every conforming
        # impl needs to be a dataclass (a hand-rolled wrapper
        # could expose `wrap`/`measurement_keys` without
        # @dataclass).
        from dataclasses import fields, is_dataclass
        cls_name = next(
            (k for k, v in _WRAPPER_REGISTRY.items() if v is type(w)),
            type(w).__name__,
        )
        if is_dataclass(w) and not isinstance(w, type):
            kvs = ','.join(
                f'{f.name}={getattr(w, f.name)}' for f in fields(w)
            )
            parts.append(f'{cls_name}({kvs})')
        else:
            parts.append(f'{cls_name}()')
    return ','.join(parts)


@dataclass(frozen=True, slots=True)
class RewardScaledEnv:
    """Wraps a gymnax-style env, scaling step reward by `scale`.

    Causal-probe lever: scaling reward changes mc_variance by
    `scale²` without altering dynamics, |A|, obs_dim, or the
    optimal policy. Used to test whether observed mc_variance →
    g_link associations are causal (vary the moderator
    interventionally) vs spurious (proxy for something correlated).

    Implements `gymnax.Env` structurally — reset/step/spaces
    delegate to `inner`; only `step` is overridden to multiply
    reward by `scale`. The optimal Q* under the scaled reward is
    `scale * Q*_original`, so DDQN's Jensen gap (in Q-units)
    scales with `scale` while standardized comparisons (Hedges' g)
    are mostly invariant — the deviation from invariance is the
    causal signature."""
    inner: Env
    scale: float

    def reset(
        self, rng: jax.Array, params: EnvParams,
    ) -> tuple[jax.Array, EnvState]:
        return self.inner.reset(rng, params)

    def step(
        self,
        rng: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[
        jax.Array, EnvState, jax.Array, jax.Array, dict[str, object],
    ]:
        next_obs, next_state, reward, done, info = self.inner.step(
            rng, state, action, params,
        )
        return next_obs, next_state, reward * self.scale, done, info

    def observation_space(self, params: EnvParams) -> Box:
        return self.inner.observation_space(params)

    def action_space(self, params: EnvParams) -> Discrete:
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
    inner: Env
    clip_min: float | None = None
    clip_max: float | None = None

    def reset(
        self, rng: jax.Array, params: EnvParams,
    ) -> tuple[jax.Array, EnvState]:
        return self.inner.reset(rng, params)

    def step(
        self,
        rng: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[
        jax.Array, EnvState, jax.Array, jax.Array, dict[str, object],
    ]:
        next_obs, next_state, reward, done, info = self.inner.step(
            rng, state, action, params,
        )
        if self.clip_min is not None:
            reward = jnp.maximum(reward, self.clip_min)
        if self.clip_max is not None:
            reward = jnp.minimum(reward, self.clip_max)
        return next_obs, next_state, reward, done, info

    def observation_space(self, params: EnvParams) -> Box:
        return self.inner.observation_space(params)

    def action_space(self, params: EnvParams) -> Discrete:
        return self.inner.action_space(params)


@dataclass(frozen=True, slots=True)
class ActionDuplicatedEnv:
    """Wraps a gymnax-style env, inflating its discrete action
    space by factor `k`. Step delegates to inner with action
    folded back via `action % inner_n_actions` — every duplicate
    is dynamically identical to its inner counterpart.

    Causal-probe lever: declared |A| varies (k * inner_n) while
    dynamics, reward, optimal Q*, and observation space are
    unchanged. Hasselt 2010's max-bias floor is ε ≤
    σ_Q · √(2 log|A|), so DDQN's bias-correction headroom should
    grow with k. If DDQN's outcome benefit is bottlenecked by
    Hasselt floor, varying k cleanly resolves whether |A|∈{3,4}
    is a true sweet spot or a corpus-specific artifact."""
    inner: Env
    k: int

    def reset(
        self, rng: jax.Array, params: EnvParams,
    ) -> tuple[jax.Array, EnvState]:
        return self.inner.reset(rng, params)

    def step(
        self,
        rng: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[
        jax.Array, EnvState, jax.Array, jax.Array, dict[str, object],
    ]:
        inner_space = self.inner.action_space(params)
        inner_action = action % inner_space.n
        return self.inner.step(rng, state, inner_action, params)

    def observation_space(self, params: EnvParams) -> Box:
        return self.inner.observation_space(params)

    def action_space(self, params: EnvParams) -> Discrete:
        inner_space = self.inner.action_space(params)
        return spaces.Discrete(inner_space.n * self.k)


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
    fields without `# type: ignore`. Spaces and params are typed
    end-to-end via the gymnax stub — `obs_space` is `Box`,
    `act_space` is `Discrete`, and `env_params.max_steps_in_episode`
    reads directly off the typed `EnvParams` (the gymnax runtime
    declares the field with a default of `1`, so every concrete
    env's params inherits it; the stub's `EnvParams` exposes it
    as `int` accordingly)."""
    env_obj, env_params = gymnax.make(name)
    act_space = env_obj.action_space(env_params)
    obs_space = env_obj.observation_space(env_params)

    shape = tuple(int(d) for d in obs_space.shape)
    action_dim = int(act_space.n)
    is_discrete = True

    horizon: int | None = int(env_params.max_steps_in_episode)

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
