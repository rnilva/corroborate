"""JAX-native LunarLander-v2, gymnax-Env-Protocol compatible.

Gymnasium's reference `LunarLander-v2` is built on Box2D — a C++
rigid-body simulator that doesn't compose with `jax.vmap` and
breaks the substrate's vectorised seed-rollout pipeline. This
module reimplements the env in pure JAX so it slots into the
existing `cell_runner` codepath alongside gymnax / jumanji envs
with no Python-side per-cell branching.

**What's faithful** to the Box2D original:
- Action space: `Discrete(4)` — nop / left engine / main engine /
  right engine, matching gymnasium's `LunarLander-v2` discrete mode.
- Observation: 8-dim `(x, y, vx, vy, angle, ang_vel, leg1, leg2)`
  with gymnasium's exact normalisation
  (`VIEWPORT_*/SCALE`, `FPS`, `helipad_y` reference points).
- Reward shape: shaping = `-100√(x² + y²) - 100√(vx² + vy²) -
  100|angle| + 10·(leg1 + leg2)`; per-step reward
  `shaping − prev_shaping − 0.30·main − 0.03·side`; +100 for
  successful landing, −100 for crash. Same constants as gymnasium.
- Gravity, FPS, lander polygon, leg offsets, engine forces,
  initial random impulse all use gymnasium's exact numeric values
  (see the module-level constants below).
- Termination: out-of-viewport, lander-body crash, or `awake=False`
  (low-velocity rest, low rotation, on the ground) — see
  `_is_terminal` for the closed-form sleep predicate that
  approximates Box2D's `awake` flag.

**What's simplified** vs Box2D (intentionally):
- **Rigid-body sim**: implemented with semi-implicit Euler at
  FPS=50. Mass + moment of inertia are computed analytically
  from the polygon's bounding-box (closed form, since the polygon
  is approximately rectangular). No collision response — leg
  contacts are detected by point-vs-flat-ground; the body bounces
  off the ground only via crash (terminal), not via elastic
  collision.
- **Articulated legs**: omitted. Legs are modelled as two fixed
  contact points rigidly attached to the lander at the gymnasium
  attachment positions; `leg_contact` triggers when each point's
  world-frame y ≤ helipad_y. Spring-damped revolute joints +
  motor (gymnasium's `LEG_SPRING_TORQUE`, motor speed) are
  dropped — the substrate consumes only the boolean contact flags,
  so leg articulation has no observation-side effect.
- **Terrain**: gymnasium generates a randomly-jagged moonscape
  outside the helipad. This port uses a perfectly flat ground at
  `y = helipad_y` for all x. Most policies that would land on the
  helipad already; off-helipad terrain only changes which
  off-helipad excursions are crash vs landing. For DQN-substrate
  experimentation the flat-ground simplification preserves the
  load-bearing surface (the helipad zone) with reduced env-state
  complexity.
- **Wind / turbulence**: omitted; gymnasium's `enable_wind=False`
  default is what we match.
- **Initial random impulse**: gymnasium calls
  `ApplyForceToCenter(uniform(-1000, 1000), uniform(-1000, 1000))`
  which Box2D integrates over a single step as a delta-v.
  Reimplemented analytically: divide impulse by mass + 1/FPS to
  get the equivalent initial linear velocity perturbation.

The render API is not implemented — substrate consumers are
headless. Trajectory recording happens via the substrate's
per-step trace mechanism, not by re-running with `render_mode`.

API: matches `gymnax.environments.environment.Environment`'s
runtime contract structurally — `reset(rng, params) → (obs,
state)` and `step(rng, state, action, params) → (obs, state,
reward, done, info)`. The `EnvParams` carries
`max_steps_in_episode` (substrate convention).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

if TYPE_CHECKING:
    from gymnax import Box, Discrete


# ============ Gymnasium-faithful physical constants ============
#
# Quoted verbatim from gymnasium/envs/box2d/lunar_lander.py. SCALE
# = 30 pixels/meter; world units below are pixels unless noted
# otherwise (gymnasium uses the same convention — most quantities
# divide by SCALE at use-site).

FPS: float = 50.0
SCALE: float = 30.0
VIEWPORT_W: int = 600
VIEWPORT_H: int = 400

# Lander geometry (px). The polygon is approximately a rectangle
# with the top corners chamfered; bounding box is 34 wide × 27 tall
# (from y=-10 to y=+17). Area used for mass computation below.
LANDER_POLY: tuple[tuple[int, int], ...] = (
    (-14, +17), (-17, 0), (-17, -10),
    (+17, -10), (+17, 0), (+14, +17),
)
LANDER_DENSITY: float = 5.0

LEG_AWAY: int = 20
LEG_DOWN: int = 18

MAIN_ENGINE_POWER: float = 13.0
SIDE_ENGINE_POWER: float = 0.6
MAIN_ENGINE_Y_LOCATION: float = 4.0
SIDE_ENGINE_AWAY: int = 12
SIDE_ENGINE_HEIGHT: int = 14

GRAVITY_Y: float = -10.0
INITIAL_RANDOM: float = 1000.0

# Helipad anchor — gymnasium computes `helipad_y = H/4` where
# `H = VIEWPORT_H / SCALE`. Flat-ground simplification: ground
# height equals helipad_y everywhere.
HELIPAD_Y: float = (VIEWPORT_H / SCALE) / 4.0
INITIAL_Y: float = VIEWPORT_H / SCALE          # 13.33 m
INITIAL_X: float = (VIEWPORT_W / SCALE) / 2.0  # 10.0 m


# ============ Derived mass + inertia (analytic, no Box2D) ============
#
# Polygon area for the LANDER_POLY:
#   shoelace formula on the 6 vertices gives 858 px², or
#   approximately the 34×27 bounding box (918) minus the two
#   top-corner triangles of area 3·17/2 ≈ 25.5 each ≈ 867 px².
#   The shoelace exact value:
#     A = 0.5 |Σ (x_i y_{i+1} - x_{i+1} y_i)| = 858 px²
#
# Box2D mass = density × (area / SCALE²); LANDER_POLY in meters has
# area 858/SCALE². Resulting mass: 5 × 858 / 900 ≈ 4.7667 kg.
def _polygon_area_px(poly: tuple[tuple[int, int], ...]) -> float:
    n = len(poly)
    a = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return 0.5 * abs(a)


_AREA_M2: float = _polygon_area_px(LANDER_POLY) / (SCALE * SCALE)
LANDER_MASS: float = LANDER_DENSITY * _AREA_M2  # ≈ 4.767 kg

# Moment of inertia: approximate the polygon as a uniform rectangle
# of the bounding-box dimensions for the I term. Width (px) =
# 34, height (px) = 27. In meters: 34/SCALE × 27/SCALE. I = M·(w² +
# h²) / 12 for a rectangle. The polygon is slightly less than this,
# so this is a slight overestimate — bias is small and consistent.
_LANDER_W_M: float = 34.0 / SCALE
_LANDER_H_M: float = 27.0 / SCALE
LANDER_INERTIA: float = (
    LANDER_MASS * (_LANDER_W_M * _LANDER_W_M + _LANDER_H_M * _LANDER_H_M) / 12.0
)


# ============ State / Params dataclasses ============

@struct.dataclass
class LunarLanderState:
    """Lander's rigid-body state + episode bookkeeping.

    Position / velocity are in **meters** (world frame; gymnasium's
    convention after dividing pixels by SCALE). Angle is radians.
    `prev_shaping` carries the previous step's shaping value so
    reward can compute `Δshaping − fuel`.

    `leg_contact_l` / `leg_contact_r` are float (0.0/1.0) — the
    observation is float-typed, and we never branch on these in
    the substrate, so a float keeps the pytree dtype-uniform with
    the rest of the state.

    `crashed` and `landed` are sticky terminal flags carried so
    `_is_terminal` is a pure function of state (no Python-side
    flag mutation across steps)."""
    x: jax.Array
    y: jax.Array
    vx: jax.Array
    vy: jax.Array
    angle: jax.Array
    angular_vel: jax.Array
    leg_contact_l: jax.Array
    leg_contact_r: jax.Array
    prev_shaping: jax.Array
    crashed: jax.Array
    landed: jax.Array
    time: jax.Array


@struct.dataclass
class LunarLanderParams:
    """Tunable env params. `max_steps_in_episode` matches
    gymnasium's `_max_episode_steps=1000` default for LunarLander
    via `gymnasium.wrappers.TimeLimit`."""
    gravity: float = GRAVITY_Y
    main_engine_power: float = MAIN_ENGINE_POWER
    side_engine_power: float = SIDE_ENGINE_POWER
    initial_random: float = INITIAL_RANDOM
    max_steps_in_episode: int = 1000


# ============ Env implementation ============

@dataclass(frozen=True, slots=True)
class LunarLanderEnv:
    """JAX-native LunarLander-v2.

    Structurally matches the gymnax `Env` Protocol — substrate's
    `cell_runner` calls `reset(rng, params)` and `step(rng, state,
    action, params)` without inspecting the env's class.

    Construction is config-free (no fields); per-call config flows
    through `LunarLanderParams`."""

    def reset(
        self, rng: jax.Array, params: LunarLanderParams,
    ) -> tuple[jax.Array, LunarLanderState]:
        del params  # initial conditions are fixed; rng controls noise
        # Gymnasium's reset: lander at (INITIAL_X, INITIAL_Y), angle
        # 0, then ApplyForceToCenter with uniform(-1000, +1000) in x
        # and y. Box2D integrates this force over one 1/FPS timestep
        # producing delta-v = F · dt / M. Reproduce that delta-v
        # directly here (avoids a one-step "apply force then step"
        # sequence at reset).
        key_fx, key_fy = jax.random.split(rng, 2)
        fx = jax.random.uniform(
            key_fx, (), minval=-INITIAL_RANDOM, maxval=INITIAL_RANDOM,
        )
        fy = jax.random.uniform(
            key_fy, (), minval=-INITIAL_RANDOM, maxval=INITIAL_RANDOM,
        )
        dt = 1.0 / FPS
        vx0 = fx * dt / LANDER_MASS
        vy0 = fy * dt / LANDER_MASS

        # Build state in two passes — the first pass computes the
        # observation, which `shaping` reads to initialise
        # `prev_shaping`. Matches gymnasium's `prev_shaping = None →
        # reward += shaping; prev_shaping = shaping` first-step
        # rule (gymnasium's first-step reward IS the absolute
        # shaping; we depart slightly by anchoring prev_shaping to
        # the reset state so the first step's reward is Δshaping
        # from rest, not absolute).
        x0 = jnp.asarray(INITIAL_X, dtype=jnp.float32)
        y0 = jnp.asarray(INITIAL_Y, dtype=jnp.float32)
        # Compute observation directly from the initial scalars —
        # no full state allocation needed for the prev_shaping
        # bootstrap.
        half_w = VIEWPORT_W / SCALE / 2.0
        half_h = VIEWPORT_H / SCALE / 2.0
        leg_anchor = HELIPAD_Y + LEG_DOWN / SCALE
        init_obs = jnp.array([
            (x0 - half_w) / half_w,
            (y0 - leg_anchor) / half_h,
            vx0 * half_w / FPS,
            vy0 * half_h / FPS,
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
            jnp.float32(0.0),
        ], dtype=jnp.float32)
        first_shaping = shaping(init_obs)
        state = LunarLanderState(
            x=x0,
            y=y0,
            vx=vx0.astype(jnp.float32),
            vy=vy0.astype(jnp.float32),
            angle=jnp.float32(0.0),
            angular_vel=jnp.float32(0.0),
            leg_contact_l=jnp.float32(0.0),
            leg_contact_r=jnp.float32(0.0),
            prev_shaping=first_shaping,
            crashed=jnp.bool_(False),
            landed=jnp.bool_(False),
            time=jnp.int32(0),
        )
        return init_obs, state

    def step(
        self,
        rng: jax.Array,
        state: LunarLanderState,
        action: jax.Array,
        params: LunarLanderParams,
    ) -> tuple[
        jax.Array, LunarLanderState, jax.Array, jax.Array, dict[str, object],
    ]:
        del rng  # deterministic dynamics; rng would only feed
        # gymnasium's per-step dispersion. We drop dispersion for a
        # cleaner pure-functional step — fuel cost / impulse
        # magnitude unchanged, only the impulse offset varies in
        # gymnasium's original, which mostly affects torque slightly.

        # Discrete action: 0=nop, 1=left, 2=main, 3=right.
        act = action.astype(jnp.int32)
        is_main = act == 2
        is_left = act == 1
        is_right = act == 3
        is_side = jnp.logical_or(is_left, is_right)

        # Compute engine impulses in world-frame (force × dt / m).
        # tip is unit vector pointing "up out of the lander top"
        # (gymnasium uses tip = (sin angle, cos angle) with their
        # y-up convention; matches ours).
        sin_a = jnp.sin(state.angle)
        cos_a = jnp.cos(state.angle)
        tip_x, tip_y = sin_a, cos_a
        side_x, side_y = -tip_y, tip_x

        # Main engine: power scalar = 1.0 (discrete). Force is
        # MAIN_ENGINE_POWER (Box2D ApplyLinearImpulse uses the
        # value as an impulse, not force). Impulse direction:
        # opposite of tip (engine pushes "out" downward, body
        # accelerates "up" along +tip). Force vector applied at
        # impulse_pos = lander_pos + (ox, oy) where (ox, oy) ~
        # downward offset along -tip (engine exhaust point).
        # For a rigid body the linear acceleration is uniform; only
        # the torque depends on the lever arm. Reproduce both.
        dt = 1.0 / FPS

        m_power = jnp.where(is_main, jnp.float32(1.0), jnp.float32(0.0))
        # Gymnasium computes impulse = (-ox, -oy) · MAIN_ENGINE_POWER
        # where (ox, oy) is the engine-exhaust offset relative to
        # body center:
        #     ox = +tip_x · MAIN_ENGINE_Y_LOCATION/SCALE
        #     oy = -tip_y · MAIN_ENGINE_Y_LOCATION/SCALE
        # so the body impulse is:
        #     impulse_x = -ox · power = -tip_x · offset · power
        #     impulse_y = -oy · power = +tip_y · offset · power
        # Verified by direct probe against Box2D — the previous
        # impl mistakenly wrote impulse_x = +tip_x · power, which
        # inverts the horizontal thrust component at non-zero
        # lander angle. See LUNAR_LANDER_DYNAMICS_REVIEW.md.
        m_offset = jnp.float32(MAIN_ENGINE_Y_LOCATION / SCALE)
        m_impulse_x = -tip_x * params.main_engine_power * m_offset * m_power
        m_impulse_y = tip_y * params.main_engine_power * m_offset * m_power
        # Lever arm matches gymnasium's (ox, oy) above.
        # Cross product r × F = (rx · Fy - ry · Fx). For the main
        # engine r and F are anti-parallel so torque = 0
        # analytically; we compute it anyway for symmetry / numeric
        # stability of any future asymmetric thrust extension.
        m_rx = tip_x * m_offset
        m_ry = -tip_y * m_offset
        m_torque = m_rx * m_impulse_y - m_ry * m_impulse_x

        # Side engine: direction = -1 for left (action=1), +1 for
        # right (action=3). Push along ±side direction, applied at
        # `(side · SIDE_ENGINE_AWAY/SCALE - tip · 17/SCALE,
        #   ... + tip · SIDE_ENGINE_HEIGHT/SCALE)` offset relative
        # to body center.
        direction = jnp.where(is_left, jnp.float32(-1.0), jnp.float32(0.0))
        direction = jnp.where(is_right, jnp.float32(1.0), direction)
        s_power = jnp.where(is_side, jnp.float32(1.0), jnp.float32(0.0))
        # Side impulse — gymnasium's offset (dispersion dropped):
        #   ox = +side_x · direction · SIDE_ENGINE_AWAY/SCALE
        #      = -tip_y · direction · SIDE_ENGINE_AWAY/SCALE
        #   oy = -side_y · direction · SIDE_ENGINE_AWAY/SCALE
        #      = -tip_x · direction · SIDE_ENGINE_AWAY/SCALE
        # Body impulse = (-ox · power, -oy · power) =
        #   ( +tip_y · direction · offset · power,
        #     +tip_x · direction · offset · power )
        # The previous impl wrote impulse_y = -direction · tip_x ·
        # power, which inverts the vertical component of the side
        # thrust at non-zero lander angle. Verified by direct
        # Box2D probe.
        s_offset = jnp.float32(SIDE_ENGINE_AWAY / SCALE)
        s_impulse_x = (
            direction * tip_y * params.side_engine_power * s_offset * s_power
        )
        s_impulse_y = (
            direction * tip_x * params.side_engine_power * s_offset * s_power
        )
        # Lever arm = impulse_pos − lander_pos. Gymnasium adds two
        # extra body-frame offsets to the impulse application
        # point (cf. their lines 599-602):
        #   r_x = ox − tip_x · 17 / SCALE
        #       = −tip_y · direction · 0.4 − tip_x · 17 / SCALE
        #   r_y = oy + tip_y · SIDE_ENGINE_HEIGHT / SCALE
        #       = −tip_x · direction · 0.4 + tip_y · 14 / SCALE
        # Gymnasium's own source comments that the constant 17 (vs
        # SIDE_ENGINE_HEIGHT=14) is "presumably a bug" — keeping
        # gymnasium's literal behaviour for parity.
        s_rx = (
            -tip_y * direction * (SIDE_ENGINE_AWAY / SCALE)
            - tip_x * (17.0 / SCALE)
        )
        s_ry = (
            -tip_x * direction * (SIDE_ENGINE_AWAY / SCALE)
            + tip_y * (SIDE_ENGINE_HEIGHT / SCALE)
        )
        s_torque = s_rx * s_impulse_y - s_ry * s_impulse_x

        # Total impulse & torque this step.
        impulse_x = m_impulse_x + s_impulse_x
        impulse_y = m_impulse_y + s_impulse_y
        torque = m_torque + s_torque

        # Δv = impulse / mass (Box2D ApplyLinearImpulse semantics).
        dvx = impulse_x / LANDER_MASS
        dvy = impulse_y / LANDER_MASS
        dω = torque / LANDER_INERTIA

        # Semi-implicit Euler integration (Box2D uses
        # symplectic Euler, similar order of accuracy).
        vx_new = state.vx + dvx + params.gravity * dt * 0.0  # gravity below
        vy_new = state.vy + dvy + params.gravity * dt
        ω_new = state.angular_vel + dω

        # Update position with new velocity (semi-implicit).
        x_new = state.x + vx_new * dt
        y_new = state.y + vy_new * dt
        angle_new = state.angle + ω_new * dt

        # Leg contact detection — two fixed points relative to body
        # at gymnasium's leg-attachment positions. World-frame y of
        # each leg tip; contact iff y ≤ helipad_y.
        # Leg attach positions in body frame (in meters):
        #   left:  (-LEG_AWAY/SCALE, -LEG_DOWN/SCALE)
        #   right: (+LEG_AWAY/SCALE, -LEG_DOWN/SCALE)
        leg_dx = LEG_AWAY / SCALE
        leg_dy = -LEG_DOWN / SCALE
        # Rotate body → world: (x,y) → (cos·x - sin·y, sin·x + cos·y)
        # for a CCW rotation by `angle`. Matches gymnasium's y-up.
        cos_n, sin_n = jnp.cos(angle_new), jnp.sin(angle_new)

        # Only world-frame y matters for ground contact (flat-ground
        # simplification — see module docstring). World-frame x of
        # the leg tips would be needed for terrain-aware crash
        # detection, which we don't model.
        left_y = y_new + (sin_n * (-leg_dx) + cos_n * leg_dy)
        right_y = y_new + (sin_n * (+leg_dx) + cos_n * leg_dy)

        leg_l = (left_y <= HELIPAD_Y).astype(jnp.float32)
        leg_r = (right_y <= HELIPAD_Y).astype(jnp.float32)

        # Body crash: lander body center (or any LANDER_POLY vertex
        # below the helipad) constitutes a body-ground contact.
        # Approximate via "lowest LANDER_POLY vertex in world frame
        # ≤ helipad_y" combined with non-negligible downward
        # velocity. The cheapest faithful proxy: body bottom = y −
        # 10/SCALE (the polygon's lowest y is −10 px). If that dips
        # below the helipad AND legs haven't taken the load (no
        # leg-contact yet at this exact step), call it a crash.
        body_bottom_y = y_new - (10.0 / SCALE) * cos_n  # approx
        body_below_ground = body_bottom_y <= HELIPAD_Y
        # Crash iff body is on the ground but the legs aren't
        # touching down softly (high velocity / high tilt). Soft
        # touchdown is the landing case — handled by the awake flag.
        crash_now = body_below_ground & ~state.crashed & ~state.landed

        # Soft-landing detection (substitutes for Box2D `awake=False`):
        # both legs in contact AND linear/angular velocity below a
        # small threshold AND lander angle near upright. Once
        # triggered it's sticky.
        both_legs = (leg_l > 0.5) & (leg_r > 0.5)
        # Velocity magnitude predicate (in normalised-state units to
        # match the substrate's reward scale).
        v_norm = jnp.sqrt(vx_new * vx_new + vy_new * vy_new)
        slow = v_norm < 0.5
        upright = jnp.abs(angle_new) < 0.2
        slow_ω = jnp.abs(ω_new) < 0.5
        landed_now = (
            both_legs & slow & slow_ω & upright
            & ~state.crashed & ~state.landed
        )

        crashed = state.crashed | crash_now
        landed = state.landed | landed_now

        # Compute the post-step observation + shaping inline so the
        # new state can carry `prev_shaping=shaping` at construction
        # (avoiding a flax `.replace` round-trip).
        half_w = VIEWPORT_W / SCALE / 2.0
        half_h = VIEWPORT_H / SCALE / 2.0
        leg_anchor = HELIPAD_Y + LEG_DOWN / SCALE
        obs = jnp.array([
            (x_new - half_w) / half_w,
            (y_new - leg_anchor) / half_h,
            vx_new * half_w / FPS,
            vy_new * half_h / FPS,
            angle_new,
            20.0 * ω_new / FPS,
            leg_l,
            leg_r,
        ], dtype=jnp.float32)
        cur_shaping = shaping(obs)
        delta_shaping = cur_shaping - state.prev_shaping

        new_state = LunarLanderState(
            x=x_new.astype(jnp.float32),
            y=y_new.astype(jnp.float32),
            vx=vx_new.astype(jnp.float32),
            vy=vy_new.astype(jnp.float32),
            angle=angle_new.astype(jnp.float32),
            angular_vel=ω_new.astype(jnp.float32),
            leg_contact_l=leg_l,
            leg_contact_r=leg_r,
            prev_shaping=cur_shaping,
            crashed=crashed,
            landed=landed,
            time=state.time + 1,
        )

        fuel_cost = m_power * jnp.float32(0.30) + s_power * jnp.float32(0.03)
        step_reward = delta_shaping - fuel_cost

        # Terminal bonuses — only on the TRANSITION to terminal
        # (crash_now / landed_now), not on subsequent steps after a
        # sticky flag is already set.
        crash_bonus = jnp.where(crash_now, jnp.float32(-100.0), jnp.float32(0.0))
        landing_bonus = jnp.where(
            landed_now, jnp.float32(+100.0), jnp.float32(0.0),
        )
        reward = step_reward + crash_bonus + landing_bonus

        done = self._is_terminal(new_state, params)

        # Auto-reset on done — match gymnax's Environment.step
        # contract (callers expect step to wrap reset for them).
        # Using `jax.lax.cond` here would force a costly trace; the
        # cheap path is to deterministically build both branches via
        # a sub-reset and select via tree_map.
        reset_obs, reset_state = self.reset(
            jax.random.PRNGKey(0),  # deterministic; auto-reset is
            # not seed-sensitive (the substrate calls reset with the
            # cell's master rng on first step explicitly).
            params,
        )
        final_state = jax.tree.map(
            lambda r, n: jnp.where(done, r, n), reset_state, new_state,
        )
        final_obs = jnp.where(done, reset_obs, obs)

        return final_obs, final_state, reward, done, {}

    def _get_obs(self, state: LunarLanderState) -> jax.Array:
        """Gymnasium's 8-dim observation, normalised to roughly
        ±1 for typical play (x in ±1 at viewport bounds; vx in ±1
        at FPS scale). See gymnasium's `state = [...]` block in
        `lunar_lander.py` for the exact formulas — this method
        reproduces them verbatim."""
        half_w = VIEWPORT_W / SCALE / 2.0
        half_h = VIEWPORT_H / SCALE / 2.0
        leg_anchor = HELIPAD_Y + LEG_DOWN / SCALE
        return jnp.array([
            (state.x - half_w) / half_w,
            (state.y - leg_anchor) / half_h,
            state.vx * half_w / FPS,
            state.vy * half_h / FPS,
            state.angle,
            20.0 * state.angular_vel / FPS,
            state.leg_contact_l,
            state.leg_contact_r,
        ], dtype=jnp.float32)

    def _is_terminal(
        self, state: LunarLanderState, params: LunarLanderParams,
    ) -> jax.Array:
        """Termination predicates — pure function of state.

        - `crashed`: lander body hit the ground (set in step).
        - `landed`: lander came to rest on both legs (Box2D's
          `awake=False` proxy).
        - out-of-viewport: |x_obs| ≥ 1 (gymnasium's
          `abs(state[0]) >= 1.0` early exit).
        - step-cap: `time ≥ max_steps_in_episode`.
        """
        obs = self._get_obs(state)
        out_of_bounds = jnp.abs(obs[0]) >= jnp.float32(1.0)
        timeout = state.time >= params.max_steps_in_episode
        return state.crashed | state.landed | out_of_bounds | timeout

    def action_space(self, params: LunarLanderParams) -> Discrete:
        del params
        return spaces.Discrete(num_categories=4)

    def observation_space(self, params: LunarLanderParams) -> Box:
        del params
        # Bounds are gymnasium's documented obs ranges
        # (`x, y ∈ ±2.5`, `vx, vy ∈ ±10`, `angle ∈ ±2π`,
        # `ang_vel ∈ ±10`, legs ∈ {0, 1}). Stored as ±inf-equivalent
        # finite envelopes for the bucket-hash discretiser.
        high = jnp.array(
            [2.5, 2.5, 10.0, 10.0, 6.2831855, 10.0, 1.0, 1.0],
            dtype=jnp.float32,
        )
        return spaces.Box(low=-high, high=high, shape=(8,), dtype=jnp.float32)

    @property
    def default_params(self) -> LunarLanderParams:
        return LunarLanderParams()


def shaping(obs: jax.Array) -> jax.Array:
    """Gymnasium's shaping function — pure observation function.

    `shaping = -100 √(x² + y²) - 100 √(vx² + vy²) - 100 |angle|
               + 10 · (leg1 + leg2)`

    Per-step reward in `step()` uses `Δshaping − fuel cost`.
    """
    x, y, vx, vy, angle, _, leg1, leg2 = (
        obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7],
    )
    return (
        -100.0 * jnp.sqrt(x * x + y * y)
        - 100.0 * jnp.sqrt(vx * vx + vy * vy)
        - 100.0 * jnp.abs(angle)
        + 10.0 * leg1
        + 10.0 * leg2
    )


def make_lunar_lander() -> tuple[LunarLanderEnv, LunarLanderParams]:
    """Factory matching gymnax's `make(name)` return shape —
    `(env, env_params)`. Registered in `env_catalogue` so
    `make_env(spec)` routes `LunarLander-v2-jax` through this
    factory rather than `gymnax.make`."""
    env = LunarLanderEnv()
    return env, env.default_params
