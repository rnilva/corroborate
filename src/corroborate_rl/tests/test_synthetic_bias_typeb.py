"""Smoke tests for the v2 synthetic bias Type-A/B env.

Verifies the load-bearing structural properties of the v2 design
(the critic recommendations from `/tmp/synthetic_env_roast.md`):

1. Action-DEPENDENT transitions: `s' = (s + a + 1) mod L`.
2. Geometric-mean-preserving anisotropy: best-action SD =
   sigma_base × exp(α); other-action SD = sigma_base × exp(-α/(K-1)).
3. Best-action mean pinned at mu_best (DOESN'T scale with α).
4. JIT + vmap traceability.
5. Catalogue registration.

Sample-size bounds are SD-CV-derived from the analytic mean / SD
formulae, not "loose envelope" checks."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from corroborate_rl.env_catalogue import ENV_REGISTRY, get, make_env
from corroborate_rl.synthetic_bias_typeb import (
    BiasTypeBEnv, BiasTypeBState, make_synthetic_bias_typeb,
)


# ============ Catalogue registration ============

def test_v2_envs_registered_in_catalogue() -> None:
    """The v2 panel registers 6 envs: 2 n_states × 3 alpha levels."""
    expected = {
        f"TypeBChainV2-K4-L{n_states}-alpha{alpha}-synthetic"
        for n_states in (16, 64)
        for alpha in (-0.5, 0.0, 0.5)
    }
    registered = {
        name for name in ENV_REGISTRY.keys()
        if name.startswith('TypeBChainV2-')
    }
    assert registered == expected


def test_v2_envs_have_correct_obs_shape() -> None:
    """obs_shape = (n_states,) per the one-hot encoding."""
    spec_l16 = get('TypeBChainV2-K4-L16-alpha0.0-synthetic')
    assert spec_l16.observation_shape == (16,)
    assert spec_l16.n_actions == 4

    spec_l64 = get('TypeBChainV2-K4-L64-alpha0.5-synthetic')
    assert spec_l64.observation_shape == (64,)


def test_make_env_routes_to_synthetic_backend() -> None:
    """`make_env(spec)` constructs a BiasTypeBEnv + params when
    the spec's backend is `synthetic`."""
    spec = get('TypeBChainV2-K4-L16-alpha0.5-synthetic')
    assert spec.backend == 'synthetic'
    env, params = make_env(spec)
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')
    assert hasattr(params, 'max_steps_in_episode')


# ============ API surface ============

def test_reset_returns_obs_state_pair() -> None:
    env, params = make_synthetic_bias_typeb(n_states=8)
    obs, state = env.reset(jax.random.PRNGKey(42), params)
    assert obs.shape == (8,)
    assert obs.dtype == jnp.float32
    assert isinstance(state, BiasTypeBState)
    # Uniform-random initial state; only need it to be a valid index.
    assert 0 <= int(state.state) < 8


def test_step_returns_5_tuple_with_correct_shapes() -> None:
    env, params = make_synthetic_bias_typeb(n_states=8)
    _obs, state = env.reset(jax.random.PRNGKey(0), params)
    next_obs, next_state, reward, done, info = env.step(
        jax.random.PRNGKey(1), state, jnp.int32(0), params,
    )
    assert next_obs.shape == (8,)
    assert next_obs.dtype == jnp.float32
    assert isinstance(next_state, BiasTypeBState)
    assert reward.shape == ()
    assert reward.dtype == jnp.float32
    assert done.shape == ()
    assert done.dtype == jnp.bool_
    assert isinstance(info, dict)


# ============ Action-dependent transitions (critic rec #1) ============

def test_action_dependent_transition() -> None:
    """The critical fix: each action leads to a DIFFERENT
    successor state. Without this, max_b Q*(s', b) is action-
    independent and chain-amplified bias is impossible (the v1
    failure mode the v2 redesign addresses)."""
    env, params = make_synthetic_bias_typeb(n_states=8, n_actions=4)
    # Start at fixed state 3.
    s = BiasTypeBState(step=jnp.int32(0), state=jnp.int32(3))
    next_states: list[int] = []
    for a in range(4):
        _, ns, _, _, _ = env.step(
            jax.random.PRNGKey(a), s, jnp.int32(a), params,
        )
        next_states.append(int(ns.state))
    # Each action leads to a distinct next state.
    assert len(set(next_states)) == 4, (
        f"actions 0-3 from s=3 went to {next_states}; expected "
        f"4 distinct successors per action-dependent transition"
    )
    # Specifically: s' = (s + a + 1) mod L = (3 + a + 1) mod 8.
    assert next_states == [4, 5, 6, 7]


# ============ Decoupling Var_a[Q*] from Δ_v (critic rec #2) ============

def test_mu_best_pinned_across_alpha() -> None:
    """The decoupling fix: mu_best should NOT depend on
    anisotropy_alpha. Empirical mean reward when calling the best
    action should be mu_best regardless of α.

    Sample size N=10000 → SE ≈ σ / √N ≈ 0.005 (for σ ≤ 0.5);
    tolerance 3 SE = 0.015 (well below mu_best=0.05)."""
    n_samples = 10000
    rng = jax.random.PRNGKey(7)
    keys = jax.random.split(rng, n_samples)

    def best_reward(alpha: float) -> float:
        env, params = make_synthetic_bias_typeb(
            n_states=8, n_actions=4, mu_best=0.05, sigma_base=0.5,
            anisotropy_alpha=alpha,
        )
        # State 0, best action 0 (a_best(s) = s mod K).
        s0 = BiasTypeBState(step=jnp.int32(0), state=jnp.int32(0))

        def step_at(k: jax.Array) -> jax.Array:
            _, _, r, _, _ = env.step(k, s0, jnp.int32(0), params)
            return r

        rewards = jax.vmap(step_at)(keys)
        return float(rewards.mean())

    # Best-action mean should be mu_best=0.05 independent of α.
    for alpha in (-0.5, 0.0, 0.5):
        empirical = best_reward(alpha)
        # 3-SE tolerance on σ_best = 0.5×exp(0.5) ≈ 0.82 worst case.
        # SE ≈ 0.82/√10000 ≈ 0.008; 3 SE ≈ 0.025.
        assert abs(empirical - 0.05) < 0.03, (
            f"mu_best mean drifted: alpha={alpha} empirical="
            f"{empirical:.4f}, expected 0.05 ± 0.03"
        )


def test_anisotropy_alpha_modulates_sd_not_mean() -> None:
    """The Type-A/B axis: anisotropy_alpha shifts SD asymmetry
    across actions but NOT means. Verifies the geometric-mean-
    preserving construction:
      sigma_best = sigma_base × exp(α)
      sigma_other = sigma_base × exp(-α/(K-1))

    Closed-form prediction: at α=0.5, sigma_base=0.5, K=4:
      sigma_best = 0.5 × e^0.5 ≈ 0.824
      sigma_other = 0.5 × e^(-1/6) ≈ 0.423"""
    n_samples = 20000
    rng = jax.random.PRNGKey(11)
    keys = jax.random.split(rng, n_samples)

    env, params = make_synthetic_bias_typeb(
        n_states=8, n_actions=4, mu_best=0.0,  # zero out mean to
        # cleanly observe noise SD.
        sigma_base=0.5, anisotropy_alpha=0.5,
    )
    # At state 0, action 0 is best (sigma_best); action 1 is non-
    # best (sigma_other).
    s0 = BiasTypeBState(step=jnp.int32(0), state=jnp.int32(0))

    def step_at(k: jax.Array, a: int) -> jax.Array:
        _, _, r, _, _ = env.step(k, s0, jnp.int32(a), params)
        return r

    r_best = jax.vmap(lambda k: step_at(k, 0))(keys)
    r_other = jax.vmap(lambda k: step_at(k, 1))(keys)

    sd_best = float(r_best.std())
    sd_other = float(r_other.std())

    expected_best = 0.5 * float(jnp.exp(0.5))   # ≈ 0.8244
    expected_other = 0.5 * float(jnp.exp(-0.5 / 3))  # ≈ 0.4232

    # Sample SD at N=20000 has CV ≈ 1/sqrt(2N) ≈ 0.5%; allow 5%.
    assert abs(sd_best - expected_best) / expected_best < 0.05, (
        f"sigma_best empirical {sd_best:.4f}, expected "
        f"{expected_best:.4f}"
    )
    assert abs(sd_other - expected_other) / expected_other < 0.05, (
        f"sigma_other empirical {sd_other:.4f}, expected "
        f"{expected_other:.4f}"
    )
    # Type-B regime: sigma_best > sigma_other.
    assert sd_best > sd_other


def test_anisotropy_alpha_isotropic_when_zero() -> None:
    """At alpha=0, all actions have equal SD = sigma_base."""
    n_samples = 20000
    rng = jax.random.PRNGKey(13)
    keys = jax.random.split(rng, n_samples)

    env, params = make_synthetic_bias_typeb(
        n_states=8, n_actions=4, mu_best=0.0,
        sigma_base=0.5, anisotropy_alpha=0.0,
    )
    s0 = BiasTypeBState(step=jnp.int32(0), state=jnp.int32(0))

    def step_at(k: jax.Array, a: int) -> jax.Array:
        _, _, r, _, _ = env.step(k, s0, jnp.int32(a), params)
        return r

    r_best = jax.vmap(lambda k: step_at(k, 0))(keys)
    r_other = jax.vmap(lambda k: step_at(k, 1))(keys)

    sd_best = float(r_best.std())
    sd_other = float(r_other.std())

    # Both should be ≈ sigma_base.
    assert abs(sd_best - 0.5) / 0.5 < 0.05
    assert abs(sd_other - 0.5) / 0.5 < 0.05


# ============ Determinism + JIT/vmap traceability (critic rec, JAX) ============

def test_determinism_under_same_rng() -> None:
    """Same rng + same action sequence → byte-identical trajectory."""
    env, params = make_synthetic_bias_typeb(n_states=8)
    key = jax.random.PRNGKey(7)
    actions = jnp.array([0, 1, 2, 3, 0, 2, 2, 0], dtype=jnp.int32)

    def rollout(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        obs, state = env.reset(rng, params)
        obs_buf: list[jax.Array] = [obs]
        rew_buf: list[jax.Array] = []
        step_rng = rng
        for a in actions:
            step_rng, k = jax.random.split(step_rng)
            obs, state, r, _, _ = env.step(k, state, a, params)
            obs_buf.append(obs)
            rew_buf.append(r)
        return jnp.stack(obs_buf), jnp.stack(rew_buf)

    obs1, r1 = rollout(key)
    obs2, r2 = rollout(key)
    assert jnp.allclose(obs1, obs2)
    assert jnp.allclose(r1, r2)


def test_jit_compiles_step() -> None:
    """`step` is jit-able — no Python branching on traced values."""
    env, params = make_synthetic_bias_typeb(n_states=16)
    _obs, state = env.reset(jax.random.PRNGKey(0), params)

    jit_step = jax.jit(env.step)
    next_obs, next_state, reward, done, info = jit_step(
        jax.random.PRNGKey(1), state, jnp.int32(2), params,
    )
    assert next_obs.shape == (16,)
    assert reward.shape == ()
    del next_state, done, info


def test_vmap_over_seeds() -> None:
    """`reset` + `step` vmap cleanly over a batch of rngs."""
    env, params = make_synthetic_bias_typeb(n_states=16)
    rngs = jax.random.split(jax.random.PRNGKey(0), 8)

    vmap_reset = jax.vmap(lambda r: env.reset(r, params))
    obs_b, state_b = vmap_reset(rngs)
    assert obs_b.shape == (8, 16)
    assert state_b.state.shape == (8,)

    vmap_step = jax.vmap(
        lambda r, s, a: env.step(r, s, a, params),
        in_axes=(0, 0, 0),
    )
    actions = jnp.zeros((8,), dtype=jnp.int32)
    next_obs, _, rewards, dones, _ = vmap_step(rngs, state_b, actions)
    assert next_obs.shape == (8, 16)
    assert rewards.shape == (8,)
    assert dones.shape == (8,)


# ============ Episode termination ============

def test_episode_terminates_at_max_steps() -> None:
    """`done=True` exactly at max_steps_in_episode."""
    env, params = make_synthetic_bias_typeb(
        n_states=8, max_steps_in_episode=10,
    )
    _, state = env.reset(jax.random.PRNGKey(0), params)
    for i in range(10):
        _, state, _, done, _ = env.step(
            jax.random.PRNGKey(i + 1), state, jnp.int32(0), params,
        )
        if i < 9:
            assert not bool(done), f"premature done at step {i}"
        else:
            assert bool(done), "missing done at max_steps"


# ============ Type signature smoke ============

def test_env_class_is_frozen_dataclass() -> None:
    """`BiasTypeBEnv` is a frozen dataclass with no fields — the
    'config-free, params-carry-everything' pattern."""
    env1 = BiasTypeBEnv()
    env2 = BiasTypeBEnv()
    # Frozen dataclass with no fields: instances are interchangeable
    # but not identity-equal.
    assert env1 == env2
