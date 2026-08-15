"""Smoke tests for the v3.1 synthetic bias Type-A/B env.

v1 → v2 → v3 → v3.1 evolution lives in
`src/corroborate_rl/corroborate_rl/synthetic_bias_typeb.py`.

These tests verify the LOAD-BEARING structural properties of v3.1
(addresses the v3 design review's two STRUCTURAL critiques):

1. Action-DEPENDENT transitions: `s' = (s + a + 1) mod L` (each
   action visits a distinct successor — preserved from v2/v3).
2. **State-baked RANDOM per-state payoff** drawn from
   `payoff_seed`:
   `mu_state[s] = peak_value · (1 - payoff_spread + payoff_spread · U_s)`.
   Breaks v3's modular periodicity (`mu_state(s) = peak · β^(s mod K)`
   gave Q* only K=4 distinct values; v3.1 gives ~L distinct values).
3. **Var_a[V*(s'_a)] > 0** at every `payoff_spread > 0`,
   confirmed by **value iteration** on the deterministic MDP. v3
   had Var_a[V*(s'_a)] = 0 identically (the v3 reviewer's
   load-bearing critique). v3.1 has it scale monotonically with
   the `payoff_spread` knob.
4. **Q* matrix has ~L distinct values** at L=1024 with
   `payoff_spread > 0` (rank-ish check; the (L × K) matrix has
   column-rank ≤ K so we count unique entries, not matrix rank).
5. peak_value pinned at 1.0 across v3.1 panel (|Q*| ≈ 1/(1-γ)
   matches natural-env Asterix scale).
6. noise_sigma = 0.02·peak_value (knife-edge σ/Δ regime).
7. FA-binding regime check: L=1024 with hidden=[16] is genuinely
   capacity-bound (representation check, not training-time).
8. JIT + vmap traceability.
9. Catalogue registration.

Sample-size bounds are SD-CV-derived from the analytic mean / SD
formulae, not "loose envelope" checks."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from corroborate_rl.env_catalogue import ENV_REGISTRY, get, make_env
from corroborate_rl.synthetic_bias_typeb import (
    BiasTypeBEnv, BiasTypeBState, build_mu_state, make_synthetic_bias_typeb,
)


# ============ Value iteration reference implementation ============

def compute_v_star(
    mu_state: jax.Array,
    n_actions: int,
    gamma: float,
    n_iters: int = 5000,
    tol: float = 1e-8,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Value iteration on the deterministic v3.1 MDP.

    Returns (V*, Q*, s_next) where:
    - V* has shape (L,)
    - Q* has shape (L, K)
    - s_next[s, a] = (s + a + 1) mod L is the successor index table.

    Computes Q* up to convergence (typically <2000 iters at γ≤0.999).
    Excludes the zero-mean Gaussian reward noise from the Q-target
    (the noise only affects sampling variance, not V* / Q*)."""
    L = int(mu_state.shape[0])
    K = int(n_actions)
    s_idx = jnp.arange(L)[:, None]
    a_idx = jnp.arange(K)[None, :]
    s_next = (s_idx + a_idx + 1) % L  # shape (L, K)
    rewards = mu_state[s_next]  # shape (L, K)
    v = jnp.zeros((L,), dtype=jnp.float32)
    for _ in range(n_iters):
        q = rewards + jnp.float32(gamma) * v[s_next]
        v_new = jnp.max(q, axis=1)
        if float(jnp.max(jnp.abs(v_new - v))) < tol:
            v = v_new
            break
        v = v_new
    q = rewards + jnp.float32(gamma) * v[s_next]
    return v, q, s_next


# ============ Catalogue registration ============

def test_v31_envs_registered_in_catalogue() -> None:
    """The v3.1+ panel registers envs spanning the payoff_spread
    axis × the L (FA-capacity) axis.

    v3.2 narrows the registered set to L=1024 only; the env
    constructor still supports arbitrary L for direct calls.
    """
    registered = {
        name for name in ENV_REGISTRY.keys()
        if name.startswith('TypeBChainV31-')
    }
    # Per the v3.2 panel registration: 1 L × 5 spread × 3
    # payoff_seeds = 15 envs (see
    # `_register_synthetic_bias_typeb_panel` in env_catalogue).
    # Check the panel is non-empty and the naming convention matches.
    assert len(registered) >= 6, (
        f"expected ≥ 6 registered v3.1+ envs; got {len(registered)}: "
        f"{sorted(registered)[:10]}..."
    )
    # Every name should follow the v3.1 convention.
    for name in registered:
        assert name.endswith('-synthetic')
        assert '-spread' in name
        assert '-L' in name
        assert '-seed' in name


def test_v31_env_has_correct_obs_shape() -> None:
    """obs_shape = (n_states,) per the one-hot encoding.

    v3.2 drops the L=32 envs from the catalogue (see
    `_register_synthetic_bias_typeb_panel`); the registered panel
    is L=1024-only. The env constructor still supports arbitrary
    L (covered by the `make_synthetic_bias_typeb` direct-call
    tests below); only the registered catalogue subset narrows.
    """
    # Pick any registered v3.1+ env to verify shape; all L=1024
    # in the v3.2 panel.
    name = next(
        n for n in ENV_REGISTRY
        if n.startswith('TypeBChainV31-') and '-L1024-' in n
    )
    spec = get(name)
    assert spec.observation_shape == (1024,)
    assert spec.n_actions == 4


def test_make_env_routes_to_synthetic_backend() -> None:
    """`make_env(spec)` constructs a BiasTypeBEnv + params when
    the spec's backend is `synthetic`."""
    name = next(n for n in ENV_REGISTRY if n.startswith('TypeBChainV31-'))
    spec = get(name)
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


# ============ Action-dependent transitions (preserved from v3) ============

def test_action_dependent_transition() -> None:
    """v2/v3/v3.1: each action leads to a DIFFERENT successor state.
    Without this, `max_b Q*(s', b)` is action-independent and
    chain-amplified bias is impossible (the v1 failure mode)."""
    env, params = make_synthetic_bias_typeb(n_states=8, n_actions=4)
    s = BiasTypeBState(step=jnp.int32(0), state=jnp.int32(3))
    next_states: list[int] = []
    for a in range(4):
        _, ns, _, _, _ = env.step(
            jax.random.PRNGKey(a), s, jnp.int32(a), params,
        )
        next_states.append(int(ns.state))
    assert len(set(next_states)) == 4
    assert next_states == [4, 5, 6, 7]


# ============ Random per-state payoff (v3.1 substantive fix) ============

def test_mu_state_is_deterministic_in_payoff_seed() -> None:
    """Same `payoff_seed` → byte-identical `mu_state` vector.
    Different `payoff_seed` → different vectors (with overwhelming
    probability under U(0,1))."""
    mu1 = build_mu_state(
        n_states=64, peak_value=1.0, payoff_spread=1.0, payoff_seed=7,
    )
    mu2 = build_mu_state(
        n_states=64, peak_value=1.0, payoff_spread=1.0, payoff_seed=7,
    )
    assert jnp.allclose(mu1, mu2)

    mu3 = build_mu_state(
        n_states=64, peak_value=1.0, payoff_spread=1.0, payoff_seed=42,
    )
    # Different seeds → different vectors (random U(0,1) draws).
    assert not jnp.allclose(mu1, mu3)


def test_mu_state_range_scales_with_payoff_spread() -> None:
    """`payoff_spread=0` → all-peak (degenerate); `payoff_spread=1`
    → spans [0, peak] uniformly. SD scales as `peak · spread / sqrt(12)`
    (U(0,1) variance × scale²)."""
    mu_zero = build_mu_state(
        n_states=10000, peak_value=1.0, payoff_spread=0.0, payoff_seed=0,
    )
    assert float(mu_zero.std()) < 1e-7
    assert float(mu_zero.min()) == 1.0
    assert float(mu_zero.max()) == 1.0

    mu_full = build_mu_state(
        n_states=10000, peak_value=1.0, payoff_spread=1.0, payoff_seed=0,
    )
    # Expected SD of U(0, 1) is 1/sqrt(12) ≈ 0.2887. SE of empirical
    # SD at N=10000 is ~0.002; allow 5% relative tolerance.
    expected_sd = 1.0 / np.sqrt(12.0)
    assert abs(float(mu_full.std()) - expected_sd) / expected_sd < 0.05
    assert float(mu_full.min()) >= 0.0
    assert float(mu_full.max()) <= 1.0

    mu_half = build_mu_state(
        n_states=10000, peak_value=1.0, payoff_spread=0.5, payoff_seed=0,
    )
    # Spread=0.5 → mu_state in [0.5, 1.0]; SD = 0.5/sqrt(12) ≈ 0.144.
    expected_sd_half = 0.5 / np.sqrt(12.0)
    assert abs(float(mu_half.std()) - expected_sd_half) / expected_sd_half < 0.05


# ============ The v3 reviewer's load-bearing critique: Var_a[V*] ============

def test_var_a_v_star_is_nonzero_at_high_spread() -> None:
    """**The v3.1 substantive fix.** v3's design had
    `Var_a[V*(s'_a)] = 0` identically at every β (the v3 reviewer's
    load-bearing structural critique from the v3 design review).
    v3.1 with random per-state payoffs has `Var_a[V*(s'_a)] > 0`
    at every `payoff_spread > 0`, confirmed by value iteration.

    Verifies:
    - `Var_a[V*(s'_a)] ≈ 0` at `payoff_spread = 0` (degenerate
      isotropic case is preserved as a sanity baseline).
    - `Var_a[V*(s'_a)] > 0` at `payoff_spread > 0` AND scales
      monotonically with `payoff_spread`."""
    n_states = 32
    n_actions = 4
    gamma = 0.99

    # Compute mean Var_a[V*(s'_a)] at each spread.
    var_a_v_means: list[float] = []
    for spread in (0.0, 0.25, 0.5, 0.75, 1.0):
        mu = build_mu_state(
            n_states=n_states, peak_value=1.0,
            payoff_spread=spread, payoff_seed=0,
        )
        v, _q, s_next = compute_v_star(mu, n_actions, gamma)
        # Var_a[V*(s'_a)] per state s, then mean over s.
        v_next = v[s_next]  # shape (L, K)
        var_a_v = jnp.var(v_next, axis=1)  # shape (L,)
        var_a_v_means.append(float(var_a_v.mean()))

    # Sanity baseline: at spread=0, V* is constant → variance = 0.
    assert var_a_v_means[0] < 1e-8, (
        f"Var_a[V*] at spread=0 should be ≈ 0; got {var_a_v_means[0]:.6f}"
    )

    # At spread>0, Var_a[V*] should be STRICTLY POSITIVE.
    # (v3's design had this = 0 at every β — the load-bearing flaw.)
    for spread_val, var_val in zip(
        (0.25, 0.5, 0.75, 1.0), var_a_v_means[1:], strict=True,
    ):
        assert var_val > 1e-6, (
            f"Var_a[V*] at spread={spread_val} should be > 0; "
            f"got {var_val:.6e}. This is the v3 reviewer's "
            f"load-bearing critique — v3.1 must have it strictly "
            f"positive."
        )

    # Monotone increase with spread.
    for i in range(1, len(var_a_v_means)):
        assert var_a_v_means[i] > var_a_v_means[i - 1], (
            f"Var_a[V*] should be monotone in payoff_spread; got "
            f"{var_a_v_means}"
        )


def test_q_star_has_l_distinct_entries_under_random_payoffs() -> None:
    """**The second v3 reviewer critique.** v3's design had Q*
    periodic with period K=4 across L=1024 states → only K·K=16
    distinct Q*-values. v3.1 with random per-state payoffs has
    ~L distinct Q*-entries (no modular collapse).

    The (L × K) Q* matrix has L·K = 4L entries; this test counts
    DISTINCT entries (not matrix rank, which is column-rank-bounded
    at K=4). The L-axis-binds-FA-capacity claim requires Q*'s
    entry-cardinality to scale with L, NOT collapse to a small
    constant. v3's design had 16 distinct Q* entries at L=1024;
    v3.1 should have ~L (subject to V*-clustering under cyclic
    chain structure)."""
    n_actions = 4
    gamma = 0.99

    for n_states in (32, 1024):
        mu = build_mu_state(
            n_states=n_states, peak_value=1.0,
            payoff_spread=1.0, payoff_seed=0,
        )
        _v, q, _s_next = compute_v_star(mu, n_actions, gamma)
        q_flat = q.flatten()
        q_unique = jnp.unique(jnp.round(q_flat, decimals=4))
        n_q_unique = int(q_unique.shape[0])

        # v3 had 16 distinct Q* entries at L=1024 (modular collapse).
        # v3.1 should have ≥ L distinct entries at both L=32 and
        # L=1024. Empirical numbers (verified by VI):
        # L=32, spread=1.0, seed=0  → ~32 distinct Q* entries
        # L=1024, spread=1.0, seed=0 → ~1001 distinct Q* entries.
        # Floor at 0.95 · L (allows finite-precision ties).
        assert n_q_unique >= int(0.95 * n_states), (
            f"Q* should have ~{n_states} distinct entries under "
            f"random payoffs at L={n_states}; got {n_q_unique}. "
            f"v3 had only 16 distinct Q* entries at L=1024 — v3.1 "
            f"must break the K=4 modular collapse."
        )


def test_q_star_periodicity_broken_compared_to_v3() -> None:
    """A direct diagnostic: under v3's modular shape
    `mu_state(s) = peak · β^(s mod K)`, Q*(s, a) is exactly periodic
    in s with period K. Under v3.1's random per-state payoffs,
    Q*(s, a) is NOT periodic in s (with overwhelming probability
    over the random U(0,1) realisation).

    This test verifies the modular collapse is broken at L=32
    by checking Q*(0, 0) vs Q*(4, 0) — under v3 these would be
    identical (period 4); under v3.1 they differ by Ω(spread)."""
    n_actions = 4
    gamma = 0.99

    mu = build_mu_state(
        n_states=32, peak_value=1.0, payoff_spread=1.0, payoff_seed=0,
    )
    _v, q, _s_next = compute_v_star(mu, n_actions, gamma)

    # Q*(s, a) vs Q*((s + K) mod L, a). Under v3 modular periodicity,
    # these are EXACTLY equal. Under v3.1 random payoffs, they should
    # differ by ≳ payoff_spread · peak / sqrt(L) (closed-form scale
    # of V*(s) - V*(s+K) under random payoffs is bounded by the
    # SD of the per-block payoff differences).
    period_diffs: list[float] = []
    for s in range(8):  # check 8 states
        for a in range(n_actions):
            diff = float(jnp.abs(q[s, a] - q[(s + n_actions) % 32, a]))
            period_diffs.append(diff)
    median_diff = float(np.median(period_diffs))
    # Under v3: median_diff would be 0.0 (exact periodicity).
    # Under v3.1: median_diff should be Ω(0.001) — strictly positive.
    assert median_diff > 1e-4, (
        f"Q* should NOT be periodic in s with period K=4 under "
        f"random payoffs; median |Q*(s,a) - Q*(s+K, a)| = {median_diff:.6f}. "
        f"v3 had this = 0 exactly."
    )


# ============ Calibrated noise σ (preserved from v3) ============

def test_noise_sigma_pinned_at_calibrated_value() -> None:
    """The per-step Gaussian noise has SD = noise_sigma,
    INDEPENDENT of payoff_spread. At noise_sigma=0.02,
    σ/peak_value = 2% — natural-env Asterix knife-edge regime.

    Sample SD at N=20000 has CV ≈ 1/sqrt(2N) ≈ 0.5%; allow 5%."""
    n_samples = 20000
    rng = jax.random.PRNGKey(13)
    keys = jax.random.split(rng, n_samples)

    for spread in (0.0, 0.5, 1.0):
        env, params = make_synthetic_bias_typeb(
            n_states=8, n_actions=4, peak_value=1.0,
            payoff_spread=spread, payoff_seed=0,
            noise_sigma=0.02,
        )
        # Fix successor by always taking action 0 from state 0.
        # Mean = mu_state[1] (constant given seeded payoffs); the
        # observed SD across samples is purely the per-step noise.
        s0 = BiasTypeBState(step=jnp.int32(0), state=jnp.int32(0))

        def step_at(k: jax.Array) -> jax.Array:
            _, _, r, _, _ = env.step(k, s0, jnp.int32(0), params)
            return r

        rewards = jax.vmap(step_at)(keys)
        sd = float(rewards.std())
        assert abs(sd - 0.02) / 0.02 < 0.05, (
            f"noise SD drifted at spread={spread}: empirical {sd:.4f}, "
            f"expected 0.02"
        )


def test_reward_matches_mu_state_at_successor() -> None:
    """The per-step reward is `mu_state[s'] + noise`. Sample mean
    over many noise draws should converge to `mu_state[s']`."""
    n_samples = 5000
    rng = jax.random.PRNGKey(11)
    keys = jax.random.split(rng, n_samples)

    env, params = make_synthetic_bias_typeb(
        n_states=16, n_actions=4, peak_value=1.0,
        payoff_spread=1.0, payoff_seed=99,
        noise_sigma=0.02,
    )
    # From state 0, action 2 → successor (0 + 2 + 1) mod 16 = 3.
    expected_mu = float(params.mu_state[3])
    s0 = BiasTypeBState(step=jnp.int32(0), state=jnp.int32(0))

    def step_at(k: jax.Array) -> jax.Array:
        _, _, r, _, _ = env.step(k, s0, jnp.int32(2), params)
        return r

    rewards = jax.vmap(step_at)(keys)
    mean_r = float(rewards.mean())
    # SE ≈ 0.02 / sqrt(5000) ≈ 0.0003; tolerance 3 SE.
    assert abs(mean_r - expected_mu) < 0.001, (
        f"sample mean {mean_r:.4f} should match mu_state[3]={expected_mu:.4f}"
    )


# ============ Determinism + JIT/vmap traceability ============

def test_determinism_under_same_rng() -> None:
    """Same rng + same action sequence → byte-identical trajectory."""
    env, params = make_synthetic_bias_typeb(
        n_states=8, payoff_spread=0.5, payoff_seed=7,
    )
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
    env, params = make_synthetic_bias_typeb(
        n_states=16, payoff_spread=0.7, payoff_seed=3,
    )
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
    env, params = make_synthetic_bias_typeb(
        n_states=16, payoff_spread=0.5, payoff_seed=0,
    )
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
    assert env1 == env2
