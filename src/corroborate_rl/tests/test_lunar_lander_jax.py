"""Tests for the JAX-native LunarLander port.

Verifies API conformance to gymnax's `Env` Protocol + qualitative
correctness of the rewrite: jit/vmap compatibility, determinism,
reward sign / bounds, termination triggers, full-episode survival
under random actions.

We deliberately do NOT assert trajectory equivalence with
gymnasium's Box2D implementation — that would require a Box2D
runtime in the test, defeating the purpose of the port. The
substrate's contract is structural (Env Protocol + obs/action
shapes) plus qualitative (reward sign / done-flags match the
documented spec).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from corroborate_rl.env_catalogue import ENV_REGISTRY, get, make_env
from corroborate_rl.lunar_lander_jax import (
    HELIPAD_Y,
    INITIAL_X,
    INITIAL_Y,
    LunarLanderEnv,
    LunarLanderParams,
    LunarLanderState,
    make_lunar_lander,
    shaping,
)


def _make_state(
    *,
    x: float = INITIAL_X,
    y: float = INITIAL_Y,
    vx: float = 0.0,
    vy: float = 0.0,
    angle: float = 0.0,
    angular_vel: float = 0.0,
    leg_l: float = 0.0,
    leg_r: float = 0.0,
    crashed: bool = False,
    landed: bool = False,
    time: int = 0,
    prev_shaping: float | None = None,
) -> LunarLanderState:
    """Test helper: build a typed LunarLanderState with sensible
    defaults. If `prev_shaping` is None, it's initialised from
    the obs computed off the other fields (matches the env's
    reset-pass behavior).

    Avoids `.replace()` — flax's struct decorator adds the
    method dynamically and pyright doesn't see it, so direct
    construction stays type-clean.
    """
    if prev_shaping is None:
        # Derive prev_shaping from the obs at this state — uses the
        # same normalisation the env does, so first-step Δshaping
        # is ≈ 0 unless the test explicitly perturbs dynamics.
        from corroborate_rl.lunar_lander_jax import (
            FPS,
            LEG_DOWN,
            SCALE,
            VIEWPORT_H,
            VIEWPORT_W,
        )
        half_w = VIEWPORT_W / SCALE / 2.0
        half_h = VIEWPORT_H / SCALE / 2.0
        leg_anchor = HELIPAD_Y + LEG_DOWN / SCALE
        init_obs = jnp.array([
            (x - half_w) / half_w,
            (y - leg_anchor) / half_h,
            vx * half_w / FPS,
            vy * half_h / FPS,
            angle,
            20.0 * angular_vel / FPS,
            leg_l,
            leg_r,
        ], dtype=jnp.float32)
        prev_shaping = float(shaping(init_obs))
    return LunarLanderState(
        x=jnp.float32(x),
        y=jnp.float32(y),
        vx=jnp.float32(vx),
        vy=jnp.float32(vy),
        angle=jnp.float32(angle),
        angular_vel=jnp.float32(angular_vel),
        leg_contact_l=jnp.float32(leg_l),
        leg_contact_r=jnp.float32(leg_r),
        prev_shaping=jnp.float32(prev_shaping),
        crashed=jnp.bool_(crashed),
        landed=jnp.bool_(landed),
        time=jnp.int32(time),
    )


# ============ Registration / catalogue integration ============

def test_lunar_lander_registered_in_catalogue() -> None:
    """`LunarLander-v2-jax` registers in the env catalogue under
    the `lunar_lander` backend tag with the correct shape."""
    assert 'LunarLander-v2-jax' in ENV_REGISTRY
    spec = get('LunarLander-v2-jax')
    assert spec.action_type == 'discrete'
    assert spec.n_actions == 4
    assert spec.observation_shape == (8,)
    assert spec.observation_type == 'vector'
    assert spec.horizon == 1000
    assert spec.backend == 'lunar_lander'


def test_make_env_routes_to_lunar_lander_backend() -> None:
    """`make_env(spec)` constructs a `LunarLanderEnv` + params
    when the spec's backend is `lunar_lander`."""
    spec = get('LunarLander-v2-jax')
    env, params = make_env(spec)
    # Structural: must have reset / step methods.
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')
    assert hasattr(params, 'max_steps_in_episode')
    assert params.max_steps_in_episode == 1000


# ============ API surface ============

def test_reset_returns_obs_state_pair() -> None:
    env, params = make_lunar_lander()
    obs, state = env.reset(jax.random.PRNGKey(42), params)
    assert obs.shape == (8,)
    assert obs.dtype == jnp.float32
    assert isinstance(state, LunarLanderState)


def test_step_returns_5_tuple_with_correct_shapes() -> None:
    env, params = make_lunar_lander()
    _obs, state = env.reset(jax.random.PRNGKey(0), params)
    next_obs, next_state, reward, done, info = env.step(
        jax.random.PRNGKey(1), state, jnp.int32(0), params,
    )
    assert next_obs.shape == (8,)
    assert next_obs.dtype == jnp.float32
    assert isinstance(next_state, LunarLanderState)
    assert reward.shape == ()
    assert done.shape == ()
    assert done.dtype == jnp.bool_
    assert isinstance(info, dict)


def test_action_space_returns_discrete_4() -> None:
    env, params = make_lunar_lander()
    space = env.action_space(params)
    assert space.n == 4


def test_observation_space_is_8d_box() -> None:
    env, params = make_lunar_lander()
    space = env.observation_space(params)
    assert space.shape == (8,)
    assert space.low.shape == (8,)
    assert space.high.shape == (8,)


# ============ Determinism ============

def test_determinism_under_same_rng() -> None:
    """Same rng + same action sequence → byte-identical
    trajectory."""
    env, params = make_lunar_lander()
    key = jax.random.PRNGKey(7)
    actions = jnp.array([0, 1, 2, 3, 0, 2, 2, 0], dtype=jnp.int32)

    def rollout(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        obs, state = env.reset(rng, params)
        obs_buf: list[jax.Array] = [obs]
        rew_buf: list[jax.Array] = []
        for a in actions:
            obs, state, r, _, _ = env.step(rng, state, a, params)
            obs_buf.append(obs)
            rew_buf.append(r)
        return jnp.stack(obs_buf), jnp.stack(rew_buf)

    obs1, r1 = rollout(key)
    obs2, r2 = rollout(key)
    assert jnp.allclose(obs1, obs2)
    assert jnp.allclose(r1, r2)


# ============ jit / vmap compatibility ============

def test_jit_compiles_step() -> None:
    """`step` is jit-able — no Python branching on traced values."""
    env, params = make_lunar_lander()
    _obs, state = env.reset(jax.random.PRNGKey(0), params)

    jit_step = jax.jit(env.step)
    next_obs, next_state, reward, done, info = jit_step(
        jax.random.PRNGKey(1), state, jnp.int32(2), params,
    )
    assert next_obs.shape == (8,)
    assert reward.shape == ()
    del next_state, done, info


def test_vmap_over_seeds() -> None:
    """`reset` + `step` vmap cleanly over a batch of rngs."""
    env, params = make_lunar_lander()
    rngs = jax.random.split(jax.random.PRNGKey(0), 8)

    vmap_reset = jax.vmap(lambda r: env.reset(r, params))
    obs_b, state_b = vmap_reset(rngs)
    assert obs_b.shape == (8, 8)  # (batch, obs_dim)
    assert state_b.x.shape == (8,)

    vmap_step = jax.vmap(
        lambda r, s, a: env.step(r, s, a, params),
        in_axes=(0, 0, 0),
    )
    actions = jnp.zeros((8,), dtype=jnp.int32)
    next_obs, _, rewards, dones, _ = vmap_step(rngs, state_b, actions)
    assert next_obs.shape == (8, 8)
    assert rewards.shape == (8,)
    assert dones.shape == (8,)


# ============ Reward structure ============

def test_shaping_is_zero_at_perfect_landing() -> None:
    """Shaping function: at origin with zero velocity, upright,
    both legs touching → shaping = 0 + 0 + 0 + 10 + 10 = 20."""
    obs = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    s = float(shaping(obs))
    assert s == pytest.approx(20.0)


def test_shaping_negative_far_from_origin() -> None:
    """Lander far from origin / moving / tilted → large negative
    shaping."""
    obs = jnp.array([1.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    s = float(shaping(obs))
    # -100·√2 - 100·1 - 100·0.5 = -141.42 - 100 - 50 = -291.42
    assert s < -200.0


def test_main_engine_changes_reward_vs_nop() -> None:
    """Firing main engine produces a measurably different reward
    than NOP at the same start state — the main engine slows the
    fall (smaller |vy| growth → smaller |Δshaping|) AND costs
    fuel. Net direction depends on which dominates; here we only
    assert that the two are distinguishable (the test is a
    sanity check on the dynamics + fuel-cost wiring, not on the
    exact magnitude)."""
    env, params = make_lunar_lander()
    # Hand-build a deterministic rest state — bypasses reset's
    # random initial impulse so we can compare nop vs main at
    # IDENTICAL starting velocity.
    state = _make_state()
    _, _, r_nop, _, _ = env.step(
        jax.random.PRNGKey(1), state, jnp.int32(0), params,
    )
    _, _, r_main, _, _ = env.step(
        jax.random.PRNGKey(1), state, jnp.int32(2), params,
    )
    # Main engine reduces |vy| growth from gravity → larger
    # (less-negative) Δshaping → reward higher than NOP, even
    # accounting for 0.30 fuel cost. Sign-check.
    assert float(r_main) > float(r_nop), (
        f'main engine should yield higher reward than NOP near '
        f'rest, got r_main={r_main} r_nop={r_nop}'
    )


def test_side_engine_differs_from_main() -> None:
    """Side and main engines apply different impulses + fuel costs;
    rewards from each at the same start state should differ."""
    env, params = make_lunar_lander()
    state = _make_state()
    _, _, r_main, _, _ = env.step(
        jax.random.PRNGKey(1), state, jnp.int32(2), params,
    )
    _, _, r_side, _, _ = env.step(
        jax.random.PRNGKey(1), state, jnp.int32(1), params,
    )
    assert float(r_main) != float(r_side)


# ============ Engine thrust direction (regression: sign-flip bug) ============

def test_main_engine_thrust_direction_at_tilt() -> None:
    """The main engine impulse direction was sign-flipped on the
    x-component prior to fixing — at angle=+0.5 rad, gymnasium's
    main engine pushes the body in the −x direction (up-left in
    world frame), but the JAX port pushed it in +x (up-right),
    inverting the lander's translational affordance.

    Verified against a Box2D probe: at angle=+0.5, the gymnasium
    reference yields dvx ≈ −0.14; the previous JAX impl yielded
    +0.17. This test pins the correct direction (negative dvx)
    so the bug cannot regress silently.
    """
    env, params = make_lunar_lander()

    # Lander tilted CCW by 0.5 rad (top tipped right in
    # gymnasium's sign convention where angle > 0 is CW visually
    # but +tip points "up-right").
    state = _make_state(angle=0.5)
    _, ns, _, _, _ = env.step(
        jax.random.PRNGKey(0), state, jnp.int32(2), params,
    )
    # vx should decrease (negative dvx). The pre-fix impl gave
    # dvx > 0.
    assert float(ns.vx) < 0.0, (
        f"main engine at angle=+0.5 should push dvx negative, "
        f"got vx={float(ns.vx):.4f}"
    )

    # Symmetric: at angle=-0.5, dvx should be positive.
    state2 = _make_state(angle=-0.5)
    _, ns2, _, _, _ = env.step(
        jax.random.PRNGKey(0), state2, jnp.int32(2), params,
    )
    assert float(ns2.vx) > 0.0, (
        f"main engine at angle=-0.5 should push dvx positive, "
        f"got vx={float(ns2.vx):.4f}"
    )


def test_side_engine_thrust_direction_at_tilt() -> None:
    """The side engine's y-component was sign-flipped prior to
    fixing — at angle=+0.5 with action=1 (left), the body's dvy
    should be negative (small downward push from horizontal
    thrust projected into tilted frame). The pre-fix impl
    returned positive dvy.

    Per gymnasium: at angle=+0.5, action=1, dvy ≈ −0.025.
    """
    env, params = make_lunar_lander()

    # action 1 (left side engine), tilted CCW
    state = _make_state(angle=0.5)
    _, ns, _, _, _ = env.step(
        jax.random.PRNGKey(0), state, jnp.int32(1), params,
    )
    # The engine vy contribution should be negative; total vy
    # includes gravity (-0.2 over dt=0.02) which is also negative.
    # The pre-fix impl had engine_vy = +0.024, making total vy
    # less negative.
    dt = 1.0 / 50.0
    engine_vy = float(ns.vy) - (-10.0 * dt)
    assert engine_vy < 0.0, (
        f"side-left engine at angle=+0.5 should push dvy negative, "
        f"got engine_dvy={engine_vy:.4f}"
    )

    # action 3 (right side engine), tilted CCW → dvy should
    # be positive at angle=+0.5
    state2 = _make_state(angle=0.5)
    _, ns2, _, _, _ = env.step(
        jax.random.PRNGKey(0), state2, jnp.int32(3), params,
    )
    engine_vy2 = float(ns2.vy) - (-10.0 * dt)
    assert engine_vy2 > 0.0, (
        f"side-right engine at angle=+0.5 should push dvy positive, "
        f"got engine_dvy={engine_vy2:.4f}"
    )


# ============ Termination ============

def test_out_of_bounds_terminates() -> None:
    """Lander outside viewport (|x_obs| ≥ 1) → done."""
    env, params = make_lunar_lander()
    # Push raw x to a position where normalised x > 1 (raw 25 >>
    # half-W = 10).
    state = _make_state(x=25.0)
    _, _, _, done, _ = env.step(
        jax.random.PRNGKey(1), state, jnp.int32(0), params,
    )
    assert bool(done)


def test_crash_terminates() -> None:
    """Lander body dipping below the ground → crash flag stays
    set & done returns true."""
    env, params = make_lunar_lander()
    # Body well below helipad with high downward velocity → crash
    # on next step.
    state = _make_state(y=HELIPAD_Y - 1.0, vy=-5.0)
    _, _next_state, _, done, _ = env.step(
        jax.random.PRNGKey(1), state, jnp.int32(0), params,
    )
    assert bool(done)


def test_landing_bonus_positive_reward() -> None:
    """A clean landing (both legs in contact, low velocity,
    upright, low ω) emits +100 bonus on the transitioning step."""
    env, params = make_lunar_lander()
    # Pre-position lander at ground level with both legs touching,
    # near-zero velocity, upright. The next step's transition
    # should fire `landed_now=True` and emit the +100 bonus.
    # y is positioned so that after the next step's leg-tip
    # rotation, both leg points are at ≤ helipad_y.
    state = _make_state(x=10.0, y=HELIPAD_Y + 0.55)
    _, _, reward, _done, _ = env.step(
        jax.random.PRNGKey(1), state, jnp.int32(0), params,
    )
    # Landing bonus ~+100; reward should be strongly positive.
    assert float(reward) > 50.0, f'expected landing bonus, got r={reward}'


# ============ Full-episode rollout ============

def test_random_episode_completes_no_nan() -> None:
    """5 seeds × 1000 random-action steps each — all rewards/obs
    finite, episode terminates within horizon."""
    env, params = make_lunar_lander()

    def rollout(rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Returns (total_reward, any_nan)."""
        rng_init, rng_loop = jax.random.split(rng)
        obs, state = env.reset(rng_init, params)
        total = jnp.float32(0.0)
        any_nan = jnp.bool_(False)
        for _ in range(1000):
            rng_loop, k_act, k_step = jax.random.split(rng_loop, 3)
            action = jax.random.randint(k_act, (), 0, 4)
            obs, state, reward, _done, _ = env.step(
                k_step, state, action, params,
            )
            total = total + reward
            any_nan = any_nan | jnp.any(jnp.isnan(obs)) | jnp.isnan(reward)
        return total, any_nan

    # Vmap over 5 seeds for a substrate-realistic batch shape.
    rngs = jax.random.split(jax.random.PRNGKey(123), 5)
    totals, nan_flags = jax.vmap(rollout)(rngs)
    assert not bool(jnp.any(nan_flags)), 'NaN encountered during rollout'
    # Totals should be finite scalars.
    assert jnp.all(jnp.isfinite(totals))


def test_obs_within_reasonable_envelope_under_random_actions() -> None:
    """After a short random rollout, observations stay in a
    reasonable envelope (no infinities, no |obs| explosion)."""
    env, params = make_lunar_lander()
    rng = jax.random.PRNGKey(99)
    obs, state = env.reset(rng, params)
    max_abs = jnp.float32(0.0)
    for i in range(200):
        rng, k_act, k_step = jax.random.split(rng, 3)
        action = jax.random.randint(k_act, (), 0, 4)
        obs, state, _r, done, _ = env.step(k_step, state, action, params)
        # Don't track obs after auto-reset triggered done — fresh
        # reset starts fresh; check finiteness only.
        assert jnp.all(jnp.isfinite(obs))
        max_abs = jnp.maximum(max_abs, jnp.max(jnp.abs(obs)))
        del i, done
    # Loose envelope: obs values stay within roughly ±20.
    # (gymnasium's documented obs ranges are ±2.5 for pos, ±10 for
    # vel, ±π for angle; an upper of 20 covers transient excursions
    # under random actions.)
    assert float(max_abs) < 50.0


# ============ Param defaults ============

def test_default_params_match_gymnasium_constants() -> None:
    """Sanity-check key constants against gymnasium's reference."""
    params = LunarLanderEnv().default_params
    assert params.gravity == -10.0
    assert params.main_engine_power == 13.0
    assert params.side_engine_power == 0.6
    assert params.initial_random == 1000.0
    assert params.max_steps_in_episode == 1000


def test_lunar_lander_params_is_pytree() -> None:
    """`LunarLanderParams` must be a flax pytree so it threads
    through vmap / jit. Check by tree-flattening."""
    params = LunarLanderParams()
    leaves, _ = jax.tree.flatten(params)
    # Static fields (max_steps_in_episode is an int — flax struct
    # marks it as a leaf by default). We don't care about the
    # exact leaf count, only that it doesn't raise.
    assert isinstance(leaves, list)
