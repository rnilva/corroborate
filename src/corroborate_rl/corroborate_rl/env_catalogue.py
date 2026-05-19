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
    'jumanji',
]
type ActionType = Literal['discrete', 'continuous']
type ObservationType = Literal['vector', 'image', 'structured']
type EnvBackend = Literal[
    'gymnax', 'jumanji', 'lunar_lander', 'synthetic',
]

type ThresholdConfidence = Literal[
    'literature', 'derived', 'sample_relative', 'absent',
]
"""Provenance tier for a per-env solve threshold:
- `literature` — converted from gymnasium / bsuite / paper-canonical
  raw thresholds via the discount-factor formula.
- `derived` — chosen as a fraction of a literature DQN baseline
  (MinAtar envs use 50% of Young & Tian 2019), then converted.
- `sample_relative` — defined relative to the corpus.
- `absent` — no defensible threshold; the env exists but
  `is_solved` returns `None` (caller decides what to do)."""

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


@dataclass(frozen=True, slots=True)
class ActionDiscretize:
    """Wrapper config: discretize a 1-D continuous action space
    into `n_bins` uniformly-spaced buckets. Inner action space
    must be a `Box` of shape `(1,)`. Discrete action `i ∈
    [0, n_bins)` maps to `low + (high − low) · i / (n_bins − 1)`
    (endpoints included).

    Use to bring continuous-action gymnax envs (Pendulum,
    MountainCarContinuous) into the discrete-action DQN substrate
    without re-engineering the agent. Provides REACH-polarity
    cohort additions where the discrete env catalogue is thin —
    DQN learning behavior is well-studied at modest n_bins (5-9)
    on Pendulum.

    Multi-D continuous action spaces are not handled (n_bins^d
    inflates combinatorially; outside the substrate's scope)."""
    n_bins: int

    def wrap(self, inner: Env) -> Env:
        return ActionDiscretizedEnv(inner=inner, n_bins=self.n_bins)

    def measurement_keys(self) -> Mapping[str, float]:
        return {'action_discretize_n_bins': float(self.n_bins)}


@dataclass(frozen=True, slots=True)
class RewardSparsify:
    """Wrapper config: zero per-step reward; emit `terminal_bonus`
    only at SUCCESS-terminal steps (distinguished from timeout-
    terminal by `success_threshold`).

    Causal-probe lever for the action-selection mechanism:
    `findings_action_selection_fourrooms_specific` showed DDQN
    concentrates argmax distribution only on FourRooms (sparse-
    positive-terminal reward). Sparsifying a dense-reward env
    (e.g., Acrobot) tests whether converting that env to FR-shape
    activates the same entropy-concentration mechanism — i.e.
    whether reward shape is sufficient.

    `success_threshold` distinguishes success-terminal from
    timeout-terminal by inner reward at the terminal step:

      - Acrobot: success has inner reward = 0 (not swung up = -1).
        Set `success_threshold=0.0` to treat r≥0 at terminal as
        success.
      - FourRooms / MetaMaze: success has inner reward = +1
        (timeout = 0). Set `success_threshold=0.5` (or 1.0)
        to require positive reward.

    Required parameter (no default) to force authors to commit
    to an env-specific success criterion. The earlier "no
    threshold, just inner_reward + bonus" formulation was
    fragile: it relied on Acrobot's −1/step exactly cancelling
    the bonus on timeout, which doesn't hold for FR/MetaMaze
    where timeout inner reward is 0 (so they'd also receive the
    bonus, losing the success-only property).

    Implementation:
      per-step (not done): reward → 0
      done & inner_reward ≥ success_threshold (success): bonus
      done & inner_reward < success_threshold (timeout): 0
    """
    terminal_bonus: float
    success_threshold: float

    def wrap(self, inner: Env) -> Env:
        return RewardSparsifiedEnv(
            inner=inner,
            terminal_bonus=self.terminal_bonus,
            success_threshold=self.success_threshold,
        )

    def measurement_keys(self) -> Mapping[str, float]:
        return {
            'reward_sparsify_terminal_bonus': float(self.terminal_bonus),
            'reward_sparsify_success_threshold': float(self.success_threshold),
        }


@dataclass(frozen=True, slots=True)
class ActionNoise:
    """Wrapper config: with probability `prob`, replace agent's
    action with a uniformly-random action before passing to the
    inner env's step.

    Causal-probe lever for the action-selection mechanism.
    `findings_action_selection_fourrooms_specific`: DDQN's
    argmax-concentration mechanism is FR-specific. Reward-shape
    interventions (CLAIM 7g/7h) didn't transfer it. Last-
    standing candidate: FR's native `fail_prob=0.333` (action
    randomized ~44% of the time) makes Q values across actions
    converge toward similar values, so DDQN's denoising matters
    as the only signal in a noise-dominated env.

    Tests sufficiency: stochastify Acrobot/MetaMaze (deterministic-
    action envs that don't show the mechanism) and check whether
    DDQN now concentrates argmax. If yes → action stochasticity
    is sufficient, FR-specificity explained. If no → some other
    structural property of FR remains."""
    prob: float

    def wrap(self, inner: Env) -> Env:
        return ActionNoisedEnv(inner=inner, prob=self.prob)

    def measurement_keys(self) -> Mapping[str, float]:
        return {'action_noise_prob': float(self.prob)}


@dataclass(frozen=True, slots=True)
class RewardDensify:
    """Wrapper config: add `per_step` constant to every step
    reward (typically negative for penalty-shaping).

    Symmetric counterpart to `RewardSparsify`. Densifying a
    sparse-reward env (FourRooms with `per_step=-0.01`) tests
    whether removing FR-shape attenuates the action-selection
    mechanism — i.e. whether reward shape is necessary."""
    per_step: float = 0.0

    def wrap(self, inner: Env) -> Env:
        return RewardDensifiedEnv(inner=inner, per_step=self.per_step)

    def measurement_keys(self) -> Mapping[str, float]:
        return {'reward_densify_per_step': float(self.per_step)}


@dataclass(frozen=True, slots=True)
class PotentialReward:
    """Wrapper config: potential-based reward shaping per
    Ng-Harada-Russell 1999.

      r'(s, a, s') = r(s, a, s') + γ · Φ(s') − Φ(s)

    Φ is a STATE-DEPENDENT potential function. Theorem 1 of
    Ng 1999 proves this shaping preserves the optimal policy
    (under any γ-discounted reward MDP) while transforming
    the per-step reward into a state-varying, informative
    signal.

    Causal-probe lever for the FA-degeneracy theory
    (`findings_fa_depth_within_env`): under (high α, sparse r,
    high γ), the TD bootstrap `Q(s,a) ← r + γ max_a' Q(s',a')`
    degenerates to a self-referential map `Q(s,a) ≈ γ Q(s,a*)`
    when r=0 most steps. Potential shaping replaces the
    zero-reward chain with an INFORMATIVE per-step signal
    `γΦ(s') − Φ(s)` (proportional to progress toward goal),
    breaking the self-reference WITHOUT changing the optimal
    policy. Test prediction: shaped FR at high γ + deep FA
    should NOT show vanilla bias collapse (the degeneracy
    requires the uninformative-reward condition).

    Distinct from `RewardDensify(per_step=C)`: that adds a
    CONSTANT per-step term, which doesn't break the self-
    reference (Q just converges to a different constant
    fixed point). Potential shaping is what's needed.

    `gamma` must match the agent's training γ for policy-
    invariance to hold. `potential_kind` selects the Φ
    function — currently FR-specific (`fr_manhattan_to_goal`:
    Φ(s) = −|agent_pos − goal_pos|_1)."""
    gamma: float
    potential_kind: str = 'fr_manhattan_to_goal'

    def wrap(self, inner: Env) -> Env:
        return PotentialShapedEnv(
            inner=inner, gamma=self.gamma, kind=self.potential_kind,
        )

    def measurement_keys(self) -> Mapping[str, float]:
        # Stringly-typed kind serialised as a stable hash so
        # the measurement column is queryable. The float value
        # carries the bridge-relevant scalar (gamma).
        return {
            'potential_reward_gamma': float(self.gamma),
            f'potential_kind_{self.potential_kind}': 1.0,
        }


_WRAPPER_REGISTRY: dict[str, type[EnvWrapper]] = {
    'reward_scale': RewardScale,
    'reward_clip': RewardClip,
    'action_duplicate': ActionDuplicate,
    'action_discretize': ActionDiscretize,
    'reward_sparsify': RewardSparsify,
    'reward_densify': RewardDensify,
    'action_noise': ActionNoise,
    'potential_reward': PotentialReward,
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

    Causal-probe lever for envs with genuinely mixed-sign
    reward — e.g. `Catch-bsuite` (+1 catch / −1 miss), classic-
    control envs that emit `−1` per step (MountainCar, Acrobot).
    Tests whether DDQN's bias-correction attenuation depends on
    the negative-tail stochasticity: `clip_min=0` removes the
    negative tail, the optimal policy under the clipped reward
    differs from the unclipped one, but the test isolates whether
    DDQN's behaviour inherits the same attenuation pattern under
    positive-only reward.

    **Caveat: gymnax's MinAtar suite (Asterix, Breakout, Freeway,
    SpaceInvaders) emits kill-only `+1` reward** — collisions
    terminate the episode but never emit a negative reward — so
    `clip_min=0` is a no-op on every MinAtar env. (Classic Atari
    SpaceInvaders is a different env with different reward
    semantics; do not confuse the two.) Pick a genuinely mixed-
    sign env when authoring sweeps that intervene on this
    wrapper."""
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


@dataclass(frozen=True, slots=True)
class ActionDiscretizedEnv:
    """Wraps a gymnax-style env with a 1-D continuous (Box(1,))
    action space, exposing a Discrete(n_bins) action space.

    Discrete action `i ∈ [0, n_bins)` maps to continuous action
    `low + (high − low) · i / (n_bins − 1)`. Both endpoints
    included; at `n_bins=1` the single action picks `low`.

    Observation space and dynamics are inherited unchanged."""
    inner: Env
    n_bins: int

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
        low = jnp.asarray(inner_space.low, dtype=jnp.float32)
        high = jnp.asarray(inner_space.high, dtype=jnp.float32)
        denom = jnp.float32(max(self.n_bins - 1, 1))
        t = action.astype(jnp.float32) / denom
        continuous = low + (high - low) * t
        return self.inner.step(rng, state, continuous, params)

    def observation_space(self, params: EnvParams) -> Box:
        return self.inner.observation_space(params)

    def action_space(self, params: EnvParams) -> Discrete:
        return spaces.Discrete(self.n_bins)


@dataclass(frozen=True, slots=True)
class RewardSparsifiedEnv:
    """Wraps a gymnax-style env, zeroing per-step reward; emits
    `terminal_bonus` only at SUCCESS-terminal steps, distinguished
    from timeout-terminal by `success_threshold` on inner reward.

    Use for testing whether sparsifying a dense-reward env to FR-
    shape activates DDQN's argmax-concentration mechanism."""
    inner: Env
    terminal_bonus: float
    success_threshold: float

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
        # Per-step (not done) → 0.
        # done & reward ≥ success_threshold (success) → terminal_bonus.
        # done & reward < success_threshold (timeout) → 0.
        success = done & (reward >= self.success_threshold)
        new_reward = jnp.where(
            success,
            jnp.full_like(reward, self.terminal_bonus),
            jnp.zeros_like(reward),
        )
        return next_obs, next_state, new_reward, done, info

    def observation_space(self, params: EnvParams) -> Box:
        return self.inner.observation_space(params)

    def action_space(self, params: EnvParams) -> Discrete:
        return self.inner.action_space(params)


@dataclass(frozen=True, slots=True)
class ActionNoisedEnv:
    """Wraps a gymnax-style env, replacing agent's action with
    a uniformly-random action with probability `prob`. Otherwise
    passes the agent's action through unchanged.

    Mirrors FourRooms's native `fail_prob` mechanism (which
    randomizes action with prob `fail_prob * 4/3`)."""
    inner: Env
    prob: float

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
        key_random, key_action, key_step = jax.random.split(rng, 3)
        choose_random = jax.random.uniform(key_random, ()) < self.prob
        random_action = self.inner.action_space(params).sample(key_action)
        effective_action = jax.lax.select(choose_random, random_action, action)
        return self.inner.step(key_step, state, effective_action, params)

    def observation_space(self, params: EnvParams) -> Box:
        return self.inner.observation_space(params)

    def action_space(self, params: EnvParams) -> Discrete:
        return self.inner.action_space(params)


@dataclass(frozen=True, slots=True)
class RewardDensifiedEnv:
    """Wraps a gymnax-style env, adding `per_step` constant to
    every step reward.

    Symmetric to `RewardSparsifiedEnv`. Use for testing whether
    densifying FR with per-step penalty attenuates DDQN's
    argmax-concentration mechanism."""
    inner: Env
    per_step: float

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
        return next_obs, next_state, reward + self.per_step, done, info

    def observation_space(self, params: EnvParams) -> Box:
        return self.inner.observation_space(params)

    def action_space(self, params: EnvParams) -> Discrete:
        return self.inner.action_space(params)


@dataclass(frozen=True, slots=True)
class PotentialShapedEnv:
    """Wraps a gymnax-style env with potential-based shaping
    `r'(s,a,s') = r + γ Φ(s') − Φ(s)`.

    `kind` selects the Φ function. Currently:
      - 'fr_manhattan_to_goal': Φ(s) = −|agent_pos − goal_pos|_1
        for `gymnax.environments.misc.FourRooms` (uses
        `state.pos` and `state.goal` directly).

    Terminal-step handling: at done, Φ(s_terminal) is treated
    as 0 (absorbing-state convention per Ng 1999), so the
    shaped reward at the terminal step is `r − Φ(s_pre)` —
    the agent's pre-terminal distance is credited as the
    final shaping term."""
    inner: Env
    gamma: float
    kind: str

    def reset(
        self, rng: jax.Array, params: EnvParams,
    ) -> tuple[jax.Array, EnvState]:
        return self.inner.reset(rng, params)

    def _phi(self, state: EnvState) -> jax.Array:
        if self.kind == 'fr_manhattan_to_goal':
            # FourRooms state carries `.pos` and `.goal`, each (2,) int.
            # Negative manhattan distance — higher Φ = closer to goal.
            pos = state.pos  # pyright: ignore[reportAttributeAccessIssue]
            goal = state.goal  # pyright: ignore[reportAttributeAccessIssue]
            return -jnp.sum(jnp.abs(pos - goal)).astype(jnp.float32)
        raise ValueError(
            f'PotentialShapedEnv: unknown potential_kind={self.kind!r}; '
            f"known: ['fr_manhattan_to_goal']",
        )

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
        phi_curr = self._phi(state)
        phi_next = self._phi(next_state)
        # Absorbing-state convention: Φ(terminal) = 0 in the
        # shaping. At done, shaped contribution is just −Φ(s).
        # Done-aware: jnp.where to keep this jit-compatible.
        gamma_phi_next = jnp.where(
            done, jnp.float32(0.0), self.gamma * phi_next,
        )
        shaped = reward + gamma_phi_next - phi_curr
        return next_obs, next_state, shaped, done, info

    def observation_space(self, params: EnvParams) -> Box:
        return self.inner.observation_space(params)

    def action_space(self, params: EnvParams) -> Discrete:
        return self.inner.action_space(params)


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
    `state_hash_cardinality`, optional `solve_threshold` +
    `solve_threshold_source` + `solve_threshold_confidence` +
    `solve_threshold_outcome_path`.

    **Solve thresholds.** A canonical-literature outcome value at
    or above which a cell counts as solved. The framework's eval
    records *discounted* MC return (`mc_return = Σ_t γ^t r_t`),
    so threshold values are stored in discounted units at γ=0.99
    — see env_catalogue source for per-env conversion notes.
    `solve_threshold=None` + `solve_threshold_confidence='absent'`
    means the env was considered and judged unthresholdable;
    `is_solved()` returns `None` for those, distinct from
    KeyError on unregistered envs."""
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
    solve_threshold: float | None = None
    solve_threshold_source: str | None = None
    solve_threshold_confidence: ThresholdConfidence = 'absent'
    solve_threshold_outcome_path: str = 'eval_final_mean'
    backend: EnvBackend = 'gymnax'

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


# ============ Jumanji per-env factories ============
#
# Each registered jumanji env defines a factory closure that builds
# a typed `JumanjiEnv` adapter at call time. Factories live in
# their own module (`jumanji_envs.py`) where per-env Observation
# imports are local — keeps env_catalogue.py free of jumanji-
# backend code paths and preserves typing on the obs-extract closure.
#
# Factory signature returns `tuple[Env, EnvParams]` matching the
# gymnax surface that `cell_runner` consumes; the registry maps
# our internal env name (e.g., "Snake-jumanji") to that factory.

type JumanjiFactory = Callable[[], 'tuple[Env, EnvParams]']

_JUMANJI_FACTORIES: dict[str, JumanjiFactory] = {}


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
    # Box (continuous) → action_dim is the product of shape dims;
    # is_discrete=False. Caller must apply `ActionDiscretize` (or
    # similar) before the substrate's discrete-action consumers see
    # the env. Discrete envs follow the original path.
    # `Discrete` is TYPE_CHECKING-only; duck-type at runtime via the
    # `n` attribute (Discrete has `n`, Box has `shape`).
    if hasattr(act_space, 'n'):
        action_dim = int(act_space.n)
        is_discrete = True
    else:
        action_dim = int(np.prod(act_space.shape)) if act_space.shape else 1
        is_discrete = False

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


def image_bucket_hash(
    obs_shape: tuple[int, ...],
    *,
    n_proj_dims: int = 4,
    n_buckets_per_dim: int = 4,
    proj_seed: int = 0,
    proj_low: float = -3.0,
    proj_high: float = +3.0,
) -> tuple[StateHash, int]:
    """Random-projection state hash for image-obs envs (MinAtar
    10×10×C, jumanji PacMan 31×28×5, etc.).

    Constructs a fixed random projection `R: (obs_flat_dim,
    n_proj_dims)` seeded reproducibly. Each obs flattened, then
    projected to `n_proj_dims`-vector; each dim bucketed in
    `[proj_low, proj_high]` via `n_buckets_per_dim` equal slots.

    Johnson-Lindenstrauss preserves rank-ordering of obs distance
    approximately, so the resulting bucket-id captures coarse
    state similarity. The (proj_low, proj_high) bounds default
    to a reasonable [-3σ, +3σ] window for normalised-projection
    output (assumes obs values are roughly bounded; pre-norm
    isn't done here so authors should pick bounds matching the
    env's obs scale).

    Cardinality = `n_buckets_per_dim ** n_proj_dims`. Default
    4^4 = 256 buckets — coarse enough to retain signal across
    1M-step trajectories without sparsity.

    Returns `(state_hash_fn, cardinality)`. The function is
    `jit`-compatible (`obs: jax.Array` → `jax.Array` int32).

    Use for envs where direct per-dim bucketing isn't feasible
    (image obs, high-dim). Sibling of `bucket_hash`."""
    import numpy as np
    obs_flat_dim = int(np.prod(obs_shape))
    rng = jax.random.PRNGKey(proj_seed)
    # Normalised projection: divide by sqrt(obs_flat_dim) so that
    # projection of bounded inputs stays in a comparable range.
    R = jax.random.normal(rng, (obs_flat_dim, n_proj_dims)) / jnp.sqrt(
        jnp.float32(obs_flat_dim)
    )
    cardinality = int(n_buckets_per_dim ** n_proj_dims)
    lows = jnp.full((n_proj_dims,), float(proj_low), dtype=jnp.float32)
    highs = jnp.full((n_proj_dims,), float(proj_high), dtype=jnp.float32)
    span = highs - lows
    weights = jnp.power(
        n_buckets_per_dim, jnp.arange(n_proj_dims, dtype=jnp.int32),
    )

    def state_hash(obs: jax.Array) -> jax.Array:
        flat = obs.astype(jnp.float32).flatten()
        projected = flat @ R
        clipped = jnp.clip(projected, lows, highs)
        scaled = (clipped - lows) / jnp.maximum(span, 1e-9)
        bucketed = jnp.clip(
            jnp.floor(scaled * n_buckets_per_dim),
            0, n_buckets_per_dim - 1,
        ).astype(jnp.int32)
        return jnp.sum(bucketed * weights)

    return state_hash, cardinality


def image_downsample_hash(
    obs_shape: tuple[int, ...],
    *,
    pool_size: int = 3,
    n_buckets_per_dim: int = 2,
    channel_agg: Literal['sum', 'max', 'none'] = 'sum',
    feature_low: float = 0.0,
    feature_high: float | None = None,
) -> tuple[StateHash, int]:
    """Go-Explore-style spatial-downsample state hash for image-
    obs envs. Pools the obs to `pool_size × pool_size` cells via
    `avg`-pool, optionally aggregates channels (`sum` / `max`),
    then buckets each resulting feature.

    Preserves spatial structure (different ball-positions in
    Breakout → different pool cells active → different buckets)
    that `image_bucket_hash`'s random projection collapses.

    For MinAtar 10×10×C with `pool_size=3, channel_agg='sum',
    n_buckets_per_dim=2`: 3×3 = 9 features × 2 buckets each =
    `2^9 = 512` buckets. Compact + spatial.

    For higher-resolution envs (jumanji PacMan 31×28×5) use the
    random-projection `image_bucket_hash` fallback — pooling to
    2×2 over 31×28 is too coarse to preserve policy-relevant
    state.

    Args:
      obs_shape: full obs (H, W, C) or (H, W) shape.
      pool_size: target downsampled resolution per spatial dim.
      n_buckets_per_dim: bucket count per pooled feature.
      channel_agg: 'sum' (typical for MinAtar binary channels —
        feature = total active across types), 'max' (highest
        activation), or 'none' (per-channel).
      feature_low / feature_high: bucket range for pooled
        features. If `feature_high` is None, derived from
        obs_shape (sum-agg over a pool cell of binary input
        max ≈ pool_area; max-agg max = 1.0).

    Cardinality = `n_buckets_per_dim ** n_features` where
    `n_features = pool_size² × (1 if channel_agg != 'none' else C)`.

    Returns `(state_hash_fn, cardinality)`. `state_hash_fn` is
    `jit`-compatible.
    """
    if len(obs_shape) < 2:
        raise ValueError(f'image_downsample_hash needs ≥2 spatial dims, got {obs_shape}')
    H, W = obs_shape[0], obs_shape[1]
    C = obs_shape[2] if len(obs_shape) >= 3 else 1
    n_spatial = pool_size * pool_size
    n_features = n_spatial if channel_agg != 'none' else n_spatial * C
    cardinality = int(n_buckets_per_dim ** n_features)
    # Derive feature_high default from pool-cell area + channel agg
    pool_cell_h = max(H // pool_size, 1)
    pool_cell_w = max(W // pool_size, 1)
    pool_area = pool_cell_h * pool_cell_w
    if feature_high is None:
        if channel_agg == 'sum':
            feature_high = float(pool_area * C * 0.5)
        else:
            feature_high = float(pool_area * 0.5)
    feature_high_f = float(feature_high)
    weights = jnp.power(
        n_buckets_per_dim, jnp.arange(n_features, dtype=jnp.int32),
    )
    span = float(feature_high_f - feature_low)
    # h_idx/w_idx depend only on (H, W, pool_size) — all static. Precompute
    # a (pool_size, pool_size, H, W) one-hot pooling mask in numpy at factory
    # time so the jit'd hash is a single einsum, not 100 dynamic-index
    # `.at[].add()` scatters.
    h_idx_np = np.minimum(np.arange(H) * pool_size // H, pool_size - 1)
    w_idx_np = np.minimum(np.arange(W) * pool_size // W, pool_size - 1)
    pool_mask_np = np.zeros((pool_size, pool_size, H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            pool_mask_np[h_idx_np[i], w_idx_np[j], i, j] = 1.0
    pool_mask = jnp.asarray(pool_mask_np)

    def state_hash(obs: jax.Array) -> jax.Array:
        x = obs.astype(jnp.float32)
        # Ensure (H, W, C) — add channel dim if missing
        if x.ndim == 2:
            x = x[..., None]
        # Spatial pool via the precomputed static mask:
        # (pool_size, pool_size, H, W) × (H, W, C) → (pool_size, pool_size, C)
        pooled = jnp.einsum('pqHW,HWc->pqc', pool_mask, x)
        # Channel aggregation
        if channel_agg == 'sum':
            features = pooled.sum(axis=-1).flatten()
        elif channel_agg == 'max':
            features = pooled.max(axis=-1).flatten()
        else:
            features = pooled.flatten()
        # Bucket each feature
        scaled = (features - feature_low) / max(span, 1e-9)
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
    solve_threshold: float | None = None,
    solve_threshold_source: str | None = None,
    solve_threshold_confidence: ThresholdConfidence = 'absent',
    solve_threshold_outcome_path: str = 'eval_final_mean',
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
        solve_threshold=solve_threshold,
        solve_threshold_source=solve_threshold_source,
        solve_threshold_confidence=solve_threshold_confidence,
        solve_threshold_outcome_path=solve_threshold_outcome_path,
        backend='gymnax',
    )


def _register_jumanji(
    name: str,
    *,
    factory: JumanjiFactory,
    n_actions: int,
    observation_shape: tuple[int, ...],
    horizon: int | None,
    r_min: float,
    r_max: float,
    reward_regime: RewardRegime,
    state_hash: StateHash | None = None,
    state_hash_cardinality: int | None = None,
    solve_threshold: float | None = None,
    solve_threshold_source: str | None = None,
    solve_threshold_confidence: ThresholdConfidence = 'absent',
    solve_threshold_outcome_path: str = 'eval_final_mean',
) -> None:
    """Register a jumanji-backed env with the catalogue.

    Metadata (`n_actions`, `observation_shape`, `horizon`) is
    supplied explicitly rather than introspected from the factory,
    so registration doesn't construct the underlying jumanji env
    at module-import time. This matters for envs whose constructor
    triggers a network call (e.g. Sokoban-v0 downloads its level
    dataset from HuggingFace Hub on first instantiation) — without
    explicit metadata the import-time HF call fires for every
    sub-process loading the substrate, even when no jumanji cells
    will run.

    The factory is stashed in `_JUMANJI_FACTORIES` for later
    `make_env` calls — the substrate runs N seeds against the same
    EnvSpec and we want one freshly-constructed jumanji env per
    cell, not a shared singleton."""
    obs_type: ObservationType = (
        'vector' if len(observation_shape) == 1
        else 'image' if len(observation_shape) == 3
        else 'structured'
    )

    _JUMANJI_FACTORIES[name] = factory
    ENV_REGISTRY[name] = EnvSpec(
        name=name,
        action_type='discrete',
        n_actions=n_actions,
        observation_shape=observation_shape,
        observation_type=obs_type,
        horizon=horizon,
        r_min=r_min,
        r_max=r_max,
        reward_regime=reward_regime,
        benchmark_family='jumanji',
        state_hash=state_hash,
        state_hash_cardinality=state_hash_cardinality,
        benchmark_params=MappingProxyType({}),
        solve_threshold=solve_threshold,
        solve_threshold_source=solve_threshold_source,
        solve_threshold_confidence=solve_threshold_confidence,
        solve_threshold_outcome_path=solve_threshold_outcome_path,
        backend='jumanji',
    )


def make_env(env_spec: EnvSpec) -> 'tuple[Env, EnvParams]':
    """Construct an `(env, env_params)` pair routed by backend.

    The runtime entry point that `cell_runner` calls. Replaces a
    direct `gymnax.make(name)` so jumanji-backed envs flow through
    the same code path."""
    if env_spec.backend == 'jumanji':
        factory = _JUMANJI_FACTORIES.get(env_spec.name)
        if factory is None:
            raise KeyError(
                f"Jumanji env '{env_spec.name}' has no registered "
                f"factory in _JUMANJI_FACTORIES",
            )
        return factory()
    if env_spec.backend == 'lunar_lander':
        # Lazy import — the lunar_lander module pulls in flax/jax;
        # we already import jax above, but the import is kept lazy
        # to mirror the jumanji factory pattern.
        from corroborate_rl.lunar_lander_jax import make_lunar_lander
        env, params = make_lunar_lander()
        # Structural conformance to the gymnax Env / EnvParams
        # Protocol surface — the LunarLander types declare the
        # same method shape (reset/step/spaces) and field
        # (`max_steps_in_episode`).
        return env, params  # type: ignore[return-value]
    if env_spec.backend == 'synthetic':
        factory = _SYNTHETIC_FACTORIES.get(env_spec.name)
        if factory is None:
            raise KeyError(
                f"Synthetic env '{env_spec.name}' has no "
                f"registered factory in _SYNTHETIC_FACTORIES",
            )
        env, params = factory()
        return env, params  # type: ignore[return-value]
    return gymnax.make(env_spec.name)


# ============ Synthetic env factories ============
#
# Synthetic envs (controlled-substrate causal-test envs) match
# the gymnax `Env` Protocol structurally without going through
# `gymnax.make`. Each registered name maps to a factory closure
# that builds the env + its params with the per-name config
# baked in (parameters that change the action/state space —
# `n_states`, `n_actions` — must be fixed-per-name; sweepable
# scalar knobs go into `BiasTypeBParams` and are set at factory
# time).

type SyntheticFactory = Callable[[], 'tuple[Env, EnvParams]']

_SYNTHETIC_FACTORIES: dict[str, SyntheticFactory] = {}


def _register_synthetic(
    name: str,
    *,
    factory: SyntheticFactory,
    n_actions: int,
    observation_shape: tuple[int, ...],
    horizon: int | None,
    r_min: float,
    r_max: float,
    reward_regime: RewardRegime,
    solve_threshold: float | None = None,
    solve_threshold_source: str | None = None,
    solve_threshold_confidence: ThresholdConfidence = 'absent',
    solve_threshold_outcome_path: str = 'eval_final_mean',
) -> None:
    """Register a synthetic-backed env with the catalogue.

    Mirrors `_register_jumanji` — metadata is supplied explicitly
    rather than introspected so registration is import-time cheap.
    The factory closure constructs the env + params at
    `make_env` call time."""
    obs_type: ObservationType = (
        'vector' if len(observation_shape) == 1
        else 'image' if len(observation_shape) == 3
        else 'structured'
    )
    _SYNTHETIC_FACTORIES[name] = factory
    ENV_REGISTRY[name] = EnvSpec(
        name=name,
        action_type='discrete',
        n_actions=n_actions,
        observation_shape=observation_shape,
        observation_type=obs_type,
        horizon=horizon,
        r_min=r_min,
        r_max=r_max,
        reward_regime=reward_regime,
        benchmark_family='misc',
        state_hash=None,
        state_hash_cardinality=None,
        benchmark_params=MappingProxyType({}),
        solve_threshold=solve_threshold,
        solve_threshold_source=solve_threshold_source,
        solve_threshold_confidence=solve_threshold_confidence,
        solve_threshold_outcome_path=solve_threshold_outcome_path,
        backend='synthetic',
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


# Solve threshold provenance — see ThresholdConfidence docstring.
# All values are in **discounted units at γ=0.99** so they align
# with the framework's eval (`mc_return = Σ_t γ^t r_t`).
#
# Classic control — exact conversion from gymnasium docs raw
# thresholds via `R * (1 - γ^T) / (1 - γ)`:
#   CartPole-v1: raw 475 → 100 * (1 - 0.99^475) ≈ 99.16
#   Acrobot-v1: raw -100 → -100 * (1 - 0.99^100) ≈ -63.40
#   MountainCar-v0: raw -110 → -100 * (1 - 0.99^110) ≈ -67.33
#
# bsuite — sparse terminal reward, γ^L attenuation small for
# short episodes; conservative downward adjustment from raw 0.5
# (Osband 2019 `score=0.5` convention). DiscountingChain has its
# own internal discount that compounds with the framework's, so
# threshold left at the env's near-max raw value.
#
# MinAtar — derived from Young & Tian 2019's DQN baselines at
# 50% (the "decent" threshold), then approximate-converted via
#   discounted ≈ raw × (1 - γ^L_avg) / (L_avg * (1 - γ))
# at typical episode length L_avg ≈ 500 (factor ≈ 0.199 at γ=0.99):
#   Asterix raw 6.8 → discounted ≈ 1.35
#   Breakout raw 6.2 → discounted ≈ 1.23
#   Freeway raw 12.9 → discounted ≈ 2.57
#   SpaceInvaders raw 3.7 → discounted ≈ 0.74
# Variable-per-step-reward envs don't have an exact raw→discounted
# formula without knowing the within-episode reward timing, so
# these are documented as approximate.

# Classic-control: vector obs, dense per-step reward.
_register(
    'CartPole-v1',
    r_min=0.0, r_max=1.0,
    reward_regime='per_step',
    benchmark_family='classic_control',
    state_hash=_CARTPOLE_HASH,
    state_hash_cardinality=_CARTPOLE_CARD,
    solve_threshold=99.0,
    solve_threshold_source='gymnasium-docs-475-discounted-gamma-0.99',
    solve_threshold_confidence='literature',
)
_register(
    'Acrobot-v1',
    r_min=-1.0, r_max=0.0,
    reward_regime='per_step',
    benchmark_family='classic_control',
    state_hash=_ACROBOT_HASH,
    state_hash_cardinality=_ACROBOT_CARD,
    solve_threshold=-63.4,
    solve_threshold_source='gymnasium-docs-(-100)-discounted-gamma-0.99',
    solve_threshold_confidence='literature',
)
_register(
    'MountainCar-v0',
    r_min=-1.0, r_max=0.0,
    reward_regime='per_step',
    benchmark_family='classic_control',
    state_hash=_MOUNTAINCAR_HASH,
    state_hash_cardinality=_MOUNTAINCAR_CARD,
    solve_threshold=-67.3,
    solve_threshold_source='gymnasium-docs-(-110)-discounted-gamma-0.99',
    solve_threshold_confidence='literature',
)

# Continuous-action classic-control envs, used via the
# `ActionDiscretize` wrapper to bring them into the discrete-action
# DQN substrate. Solve thresholds are approximate (literature
# values vary by author + discretisation choice). Reward bounds
# below are the inner (continuous-action) values.
_register(
    'Pendulum-v1',
    r_min=-16.27, r_max=0.0,
    reward_regime='per_step',
    benchmark_family='classic_control',
    solve_threshold=-200.0,
    solve_threshold_source='gymnasium-docs-(-200)-undiscounted-near-optimal',
    solve_threshold_confidence='derived',
)
_register(
    'MountainCarContinuous-v0',
    r_min=-0.1, r_max=100.0,
    reward_regime='event_triggered',
    benchmark_family='classic_control',
    solve_threshold=90.0,
    solve_threshold_source='gymnasium-docs-(90)-undiscounted-summit-reached',
    solve_threshold_confidence='derived',
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
    solve_threshold=0.45,
    solve_threshold_source='osband-2019-score-0.5-discounted-approx',
    solve_threshold_confidence='literature',
)
_register(
    'DeepSea-bsuite',
    r_min=-0.01, r_max=1.0,
    reward_regime='event_triggered',
    benchmark_family='bsuite',
    solve_threshold=0.45,
    solve_threshold_source='osband-2019-score-0.5-discounted-approx',
    solve_threshold_confidence='literature',
)
_register(
    'MemoryChain-bsuite',
    r_min=-1.0, r_max=1.0,
    reward_regime='terminal_only',
    benchmark_family='bsuite',
    solve_threshold=0.45,
    solve_threshold_source='osband-2019-score-0.5-discounted-approx',
    solve_threshold_confidence='literature',
)
_register(
    'UmbrellaChain-bsuite',
    r_min=-1.0, r_max=1.0,
    reward_regime='shaped',
    benchmark_family='bsuite',
    solve_threshold=0.45,
    solve_threshold_source='osband-2019-score-0.5-discounted-approx',
    solve_threshold_confidence='literature',
)
_register(
    'DiscountingChain-bsuite',
    r_min=0.0, r_max=1.1,
    reward_regime='event_triggered',
    benchmark_family='bsuite',
    solve_threshold=1.0,
    solve_threshold_source='osband-2019-near-max-1.1-env-internal-discount',
    solve_threshold_confidence='literature',
)
_register(
    'MNISTBandit-bsuite',
    r_min=-1.0, r_max=1.0,
    reward_regime='terminal_only',
    benchmark_family='bsuite',
    solve_threshold=0.5,
    solve_threshold_source='osband-2019-score-0.5-bandit-no-discount',
    solve_threshold_confidence='literature',
)

# Minatar — image obs (10×10×n_channels). state_hash via
# image_downsample_hash (Go-Explore-style spatial-pool to 3×3 +
# channel-sum + 2-buckets per cell = 2^9 = 512 buckets).
# Preserves spatial structure that random projection collapses;
# unlocks state-conditional argmax measurables for the policy-
# channel verification per memory
# `project_image_state_hash_for_substrate`.
_ASTERIX_HASH, _ASTERIX_CARD = image_downsample_hash(
    (10, 10, 4), pool_size=3, n_buckets_per_dim=2, channel_agg='sum',
    feature_low=0.0, feature_high=2.0,
)
_register(
    'Asterix-MinAtar',
    r_min=0.0, r_max=1.0,
    reward_regime='event_triggered',
    benchmark_family='minatar',
    state_hash=_ASTERIX_HASH,
    state_hash_cardinality=_ASTERIX_CARD,
    solve_threshold=1.35,
    solve_threshold_source='young-tian-2019-50pct-discounted-approx',
    solve_threshold_confidence='derived',
)
_BREAKOUT_HASH, _BREAKOUT_CARD = image_downsample_hash(
    (10, 10, 4), pool_size=3, n_buckets_per_dim=2, channel_agg='sum',
    feature_low=0.0, feature_high=2.0,
)
_register(
    'Breakout-MinAtar',
    r_min=0.0, r_max=1.0,
    reward_regime='event_triggered',
    benchmark_family='minatar',
    state_hash=_BREAKOUT_HASH,
    state_hash_cardinality=_BREAKOUT_CARD,
    solve_threshold=1.23,
    solve_threshold_source='young-tian-2019-50pct-discounted-approx',
    solve_threshold_confidence='derived',
)
_FREEWAY_HASH, _FREEWAY_CARD = image_downsample_hash(
    (10, 10, 7), pool_size=3, n_buckets_per_dim=2, channel_agg='sum',
    feature_low=0.0, feature_high=2.0,
)
_register(
    'Freeway-MinAtar',
    r_min=0.0, r_max=1.0,
    reward_regime='event_triggered',
    benchmark_family='minatar',
    state_hash=_FREEWAY_HASH,
    state_hash_cardinality=_FREEWAY_CARD,
    solve_threshold=2.57,
    solve_threshold_source='young-tian-2019-50pct-discounted-approx',
    solve_threshold_confidence='derived',
)
_SI_HASH, _SI_CARD = image_downsample_hash(
    (10, 10, 6), pool_size=3, n_buckets_per_dim=2, channel_agg='sum',
    feature_low=0.0, feature_high=2.0,
)
_register(
    'SpaceInvaders-MinAtar',
    r_min=0.0, r_max=1.0,
    reward_regime='event_triggered',
    benchmark_family='minatar',
    state_hash=_SI_HASH,
    state_hash_cardinality=_SI_CARD,
    solve_threshold=0.74,
    solve_threshold_source='young-tian-2019-50pct-discounted-approx',
    solve_threshold_confidence='derived',
)

# Misc — small-scale theoretical / pedagogical envs. No canonical
# literature thresholds; ship as 'absent' so analyses skip rather
# than miss them by accident.
_register(
    'FourRooms-misc',
    r_min=0.0, r_max=1.0,
    reward_regime='terminal_only',
    benchmark_family='misc',
    solve_threshold_source='no-canonical-criterion',
)
_register(
    'MetaMaze-misc',
    r_min=0.0, r_max=10.0,
    reward_regime='event_triggered',
    benchmark_family='misc',
    solve_threshold_source='no-canonical-criterion',
)
_register(
    'Pong-misc',
    r_min=0.0, r_max=1.0,
    reward_regime='per_step',
    benchmark_family='misc',
    solve_threshold_source='no-canonical-criterion',
)

# Bandits — single-state, action-only. No canonical solve criterion.
_register(
    'BernoulliBandit-misc',
    r_min=0.0, r_max=1.0,
    reward_regime='per_step',
    benchmark_family='bandit',
    solve_threshold_source='no-canonical-criterion',
)
_register(
    'GaussianBandit-misc',
    r_min=-5.0, r_max=5.0,
    reward_regime='per_step',
    benchmark_family='bandit',
    solve_threshold_source='no-canonical-criterion',
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


# ============ Solve-threshold record + accessors ============


@dataclass(frozen=True, slots=True)
class SolveThreshold:
    """Per-env solve threshold + provenance — a flat view of the
    EnvSpec's threshold fields. Materialised for the
    `SOLVE_THRESHOLDS` mapping below; convergence-audit consumers
    consume this shape directly. New code should prefer reading
    `EnvSpec.solve_threshold*` off the registry; this view exists
    for callers that hold a per-env record without the rest of
    the EnvSpec metadata."""
    env_name: str
    threshold: float | None
    source: str
    confidence: ThresholdConfidence
    outcome_path_assumed: str = 'eval_final_mean'


# Trigger jumanji registrations BEFORE SOLVE_THRESHOLDS is built,
# so the snapshot mapping captures jumanji envs alongside gymnax.
# Side-effect import — `jumanji_envs` calls `_register_jumanji`
# at module-load time. Placed here (not at the top of this file)
# because `_register_jumanji` must be defined first.
from corroborate_rl import jumanji_envs as _jumanji_envs  # noqa: F401, E402


def _register_synthetic_bias_typeb_panel() -> None:
    """Register the synthetic bias Type-A/B controlled-substrate
    envs.

    The naming convention encodes the structural knobs in the env
    name (K, n_states) plus the swept knobs (rvs = reward variance
    scale, sp = reward sparsity, ns = noise scale, h = horizon).
    Each named env is fixed-config; sweep yamls list one cell per
    named env. This mirrors the per-env naming pattern MinAtar
    uses (Asterix-MinAtar, Breakout-MinAtar) rather than the
    parameterised-single-env approach.

    Phase 1 panel (n=12): 3 reward_variance_scale × 2 sparsity ×
    2 reward_noise_scale levels at K=4, n_states=4, h=64. Designed
    to span the Asterix-like (high rvs, low sparsity, low noise)
    → Breakout-like (low rvs, high sparsity, high noise) axis.
    The TRUE Var_a[Q*] is (rvs × sp / (1 - γ))² up to a state-
    independent constant; γ varies via the substrate's standard γ
    sweep so we get the (env-feature × γ) cross-axis.

    Reward bounds: per-step reward ∈ [-rvs - 3·ns, +rvs + 3·ns]
    (3σ Gaussian tail); registered as ±(rvs + 3*ns)."""
    from corroborate_rl.synthetic_bias_typeb import (
        make_synthetic_bias_typeb,
    )

    # Phase 1 panel — 12 cells spanning rvs ∈ {0.2, 1.0, 3.0} ×
    # sparsity ∈ {0.2, 1.0} × noise ∈ {0.1, 0.5}.
    rvs_values = (0.2, 1.0, 3.0)
    sparsity_values = (0.2, 1.0)
    noise_values = (0.1, 0.5)
    n_states = 4
    n_actions = 4
    horizon = 64

    for rvs in rvs_values:
        for sp in sparsity_values:
            for ns in noise_values:
                # Naming: "TypeBChain-K4-rvs1.0-sp1.0-ns0.1-synthetic"
                name = (
                    f"TypeBChain-K{n_actions}-rvs{rvs}-sp{sp}-ns{ns}"
                    f"-synthetic"
                )

                def make(
                    rvs: float = rvs, sp: float = sp,
                    ns: float = ns,
                ) -> 'tuple[Env, EnvParams]':
                    env, params = make_synthetic_bias_typeb(
                        n_states=n_states,
                        n_actions=n_actions,
                        reward_variance_scale=rvs,
                        reward_sparsity=sp,
                        reward_noise_scale=ns,
                        max_steps_in_episode=horizon,
                    )
                    return env, params  # type: ignore[return-value]

                r_bound = float(rvs + 3.0 * ns)
                _register_synthetic(
                    name=name,
                    factory=make,
                    n_actions=n_actions,
                    observation_shape=(n_states,),
                    horizon=horizon,
                    r_min=-r_bound,
                    r_max=+r_bound,
                    reward_regime=(
                        'event_triggered' if sp < 1.0 else 'per_step'
                    ),
                )


_register_synthetic_bias_typeb_panel()


def _register_lunar_lander() -> None:
    """Register the pure-JAX LunarLander port. Single-env backend
    (no factory dict needed) — `make_env` routes via the
    `lunar_lander` backend tag.

    Reward bounds are loose: per-step shaped contribution ≈ ±100 +
    fuel cost, terminal ±100. Author-declared range covers the
    typical range observed in random rollouts. Solve threshold
    follows gymnasium's `score >= 200` convention; left as
    'derived' confidence because the JAX port's simplifications
    (no articulated legs, flat ground) make the threshold a less
    exact match for Box2D-trained baselines."""
    lunar_hash, lunar_card = bucket_hash(
        # x, y ∈ ±2.5 (normalised viewport); vx, vy ∈ ±10; angle
        # ∈ ±π; ang_vel ∈ ±10; legs ∈ {0, 1}. 6 buckets per dim ×
        # 6 dims of continuous obs + 2 binary = 6^6 × 4 = 186_624
        # buckets — large but tractable for the (s, a)-coverage gap.
        lows=jnp.array(
            [-2.5, -2.5, -10.0, -10.0, -3.1416, -10.0, 0.0, 0.0],
        ),
        highs=jnp.array(
            [2.5, 2.5, 10.0, 10.0, 3.1416, 10.0, 1.0, 1.0],
        ),
        n_buckets_per_dim=4,
    )
    ENV_REGISTRY['LunarLander-v2-jax'] = EnvSpec(
        name='LunarLander-v2-jax',
        action_type='discrete',
        n_actions=4,
        observation_shape=(8,),
        observation_type='vector',
        horizon=1000,
        r_min=-300.0,
        r_max=300.0,
        reward_regime='shaped',
        benchmark_family='classic_control',
        state_hash=lunar_hash,
        state_hash_cardinality=lunar_card,
        benchmark_params=MappingProxyType({}),
        solve_threshold=200.0,
        solve_threshold_source='gymnasium-docs-(200)-undiscounted',
        solve_threshold_confidence='derived',
        solve_threshold_outcome_path='eval_final_mean',
        backend='lunar_lander',
    )


_register_lunar_lander()


SOLVE_THRESHOLDS: Mapping[str, SolveThreshold] = MappingProxyType({
    name: SolveThreshold(
        env_name=name,
        threshold=spec.solve_threshold,
        source=(
            spec.solve_threshold_source
            if spec.solve_threshold_source is not None
            else 'no-canonical-criterion'
        ),
        confidence=spec.solve_threshold_confidence,
        outcome_path_assumed=spec.solve_threshold_outcome_path,
    )
    for name, spec in ENV_REGISTRY.items()
})
"""Per-env solve thresholds, derived from the registry. Read-only;
authoring is via `_register(...)` calls. 18 envs total in v0."""


def is_solved(
    env_name: str, outcome_value: float,
) -> bool | None:
    """Did this cell solve the env, given its registered solve
    threshold?

    Returns:
    - `True` if `outcome_value >= spec.solve_threshold`.
    - `False` if `outcome_value < spec.solve_threshold`.
    - `None` when `spec.solve_threshold is None` (the env was
      considered and judged unthresholdable — caller decides what
      to do, e.g. exclude or treat as unknown).
    - Raises `KeyError` when `env_name` isn't registered, so
      consumers can't silently mis-classify."""
    spec = get(env_name)
    if spec.solve_threshold is None:
        return None
    return outcome_value >= spec.solve_threshold


def envs_with_threshold() -> tuple[str, ...]:
    """Env names where a defensible solve threshold exists
    (`solve_threshold_confidence` is `'literature'` or
    `'derived'`). Excludes `'absent'` and `'sample_relative'`
    envs so analyses can default to a sound subset."""
    return tuple(sorted(
        name for name, spec in ENV_REGISTRY.items()
        if spec.solve_threshold_confidence in ('literature', 'derived')
    ))
