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
- **Articulated legs** (post 2026-05 review): two 1-DOF revolute
  pendulums hinged at the lander's body-frame anchors with motor
  torque biasing them outward and joint limits at ±0.4 rad. Foot
  positions are computed in world frame and tested for terrain
  contact independently — so legs touch ground at small tilts
  (the rigid-attached pre-rev impl required `angle ≈ 0`). Ground
  contact "sticks" the foot: leg angular velocity is reset to
  zero and the body's downward velocity is damped by a
  joint-reaction factor (the joint constraint absorbs body
  momentum into the leg, approximated as a multiplicative
  damping rather than full constraint solving). See
  `LUNAR_LANDER_DYNAMICS_REVIEW.md` §2.2 for the rationale.
- **Jagged moonscape** (post 2026-05 review): per-episode terrain
  is sampled at `reset` time from the input RNG following
  gymnasium's `_generate_terrain` recipe — 12 chunk heights
  drawn `uniform(0, H/2)`, chunks 4-7 (the helipad strip) forced
  to `helipad_y`, then 3-tap smoothed. The flat-ground
  pre-revision assumption is gone; off-helipad excursions can
  now crash on the moonscape.
- Termination: out-of-viewport, lander-body crash, or `awake=False`
  (low-velocity rest, low rotation, on the ground) — see
  `_is_terminal` for the closed-form sleep predicate that
  approximates Box2D's `awake` flag.

**What's simplified** vs Box2D (intentionally):
- **Rigid-body sim**: implemented with semi-implicit Euler at
  FPS=50. Mass + moment of inertia are computed analytically
  from the polygon's bounding-box (closed form, since the polygon
  is approximately rectangular).
- **Articulated legs (route a/c hybrid)**: each leg is a 1-DOF
  pendulum with motor torque + joint limits, but the joint
  reaction force on the body is approximated as a scalar
  multiplicative damping on body vy rather than solved via full
  rigid-body constraint dynamics. The body's effective
  upward thrust per step matches gymnasium's empirical ~0.22 m/s
  (vs the pre-revision 0.16) after the damping calibration.
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
LEG_W: int = 2
LEG_H: int = 8
LEG_SPRING_TORQUE: float = 40.0    # gymnasium's maxMotorTorque (N·m)
LEG_MOTOR_SPEED: float = 0.3       # gymnasium's motorSpeed magnitude (rad/s)
# Leg geometry — derived from gymnasium's Box2D setup. The joint
# pin is at the lander body center (0, 0) in body frame. The leg
# fixture's bottom corner ("foot") is at body-frame distance
# LEG_ROD_LENGTH_M from the joint, at an outward angle from
# the +y-axis (i.e. angle measured CCW from straight-down).
#
# Closed-form: foot's local position in the LEG body is
# `(0, -LEG_H/SCALE)`; the joint anchor on the leg is at
# `(±LEG_AWAY/SCALE, +LEG_DOWN/SCALE)` relative to leg CoM. So in
# body frame (after the constraint solver positions the leg),
# the foot ends up at distance sqrt(LEG_AWAY² + (LEG_DOWN +
# LEG_H)²)/SCALE from the joint = sqrt(20² + 26²)/30 ≈ 1.094 m.
LEG_ROD_LENGTH_M: float = (
    (LEG_AWAY * LEG_AWAY + (LEG_DOWN + LEG_H) * (LEG_DOWN + LEG_H)) ** 0.5
) / SCALE
# Rest angle (in body frame, measured CCW from straight-down).
# Derived from gymnasium's joint limits + motor convention: each
# leg's motor pushes against the OUTWARD limit; the resulting rest
# foot position is at body-frame (±0.952, -0.538) ⇒
# atan2(0.952, 0.538) ≈ 1.058 rad outward from straight-down. The
# motor permits yielding inward by JOINT_RANGE rad before hitting
# the inward limit.
LEG_REST_OUTWARD_ANGLE: float = 1.058   # rad — atan2(LEG_AWAY, LEG_DOWN+LEG_H)
LEG_JOINT_RANGE: float = 0.5            # rad — yield window inward

MAIN_ENGINE_POWER: float = 13.0
SIDE_ENGINE_POWER: float = 0.6
MAIN_ENGINE_Y_LOCATION: float = 4.0
SIDE_ENGINE_AWAY: int = 12
SIDE_ENGINE_HEIGHT: int = 14

GRAVITY_Y: float = -10.0
INITIAL_RANDOM: float = 1000.0

# Helipad anchor — gymnasium computes `helipad_y = H/4` where
# `H = VIEWPORT_H / SCALE`. Helipad strip is at chunks
# CHUNKS//2 - 2 .. CHUNKS//2 + 2 = 4..7 for CHUNKS=11.
HELIPAD_Y: float = (VIEWPORT_H / SCALE) / 4.0
INITIAL_Y: float = VIEWPORT_H / SCALE          # 13.33 m
INITIAL_X: float = (VIEWPORT_W / SCALE) / 2.0  # 10.0 m

# Terrain — matches gymnasium's CHUNKS=11 chunk-and-smooth recipe.
# Chunk x-positions are `i * W / (CHUNKS - 1)` for i in 0..CHUNKS-1.
# Chunk heights are drawn uniform(0, H/2) except chunks 4-7 forced
# to helipad_y. The 3-tap smoothing
# `smooth_y[i] = 0.33 (h[i-1] + h[i] + h[i+1])` gives the actual
# terrain polyline. CHUNKS+1 raw heights are sampled (index i+1
# in smoothing reads index 11 → needs CHUNKS+1=12 raw heights).
TERRAIN_CHUNKS: int = 11
TERRAIN_HELIPAD_LO: int = 4
TERRAIN_HELIPAD_HI: int = 7


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

# Leg mass + moment of inertia (about hinge): box (2*LEG_W, 2*LEG_H)
# in pixels with density=1 → area = (2*2*2*8)/SCALE² = 64/900 ≈
# 0.0711 m². At density=1 → m_leg ≈ 0.0711 kg. Moment of inertia
# of a rod about one end: I = m * L² / 3 (L = LEG_ROD_LENGTH_M).
LEG_MASS: float = (4.0 * LEG_W * LEG_H) / (SCALE * SCALE)  # ≈ 0.0711 kg
LEG_INERTIA: float = LEG_MASS * LEG_ROD_LENGTH_M * LEG_ROD_LENGTH_M / 3.0

# Body-vy damping when one or both legs are in contact with the
# ground — approximates the Box2D joint constraint's effect of
# transferring the body's downward momentum into the leg, which
# is then absorbed by the static ground. Calibrated against
# gymnasium's probe: at angle=0 + action=2 + leg=in-contact,
# gymnasium's body dvy ≈ -0.07 (mostly downward gravity, joint
# rejects upward thrust because the legs are pushing back); JAX
# pre-rev had dvy = -0.07 + 0.36 = +0.29. The damping factor
# multiplies the body vy after each impulse step when in contact.
LEG_CONTACT_VY_DAMPING: float = 0.50

# Body main-engine thrust scaling — calibrated so that at angle=0,
# leg-not-in-contact, the body's engine-only dvy (gravity
# subtracted) matches gymnasium's probed +0.22 m/s. The naive
# JAX impulse-over-mass calculation gives +0.36 m/s (engine push
# = 13.0 · (4/30) / 4.767 = 0.364) — STRONGER than gymnasium's
# effective thrust through Box2D's joint solver, which dissipates
# ~40% of the impulse into the legs and ground. The multiplier
# 0.61 ≈ 0.22/0.36 brings JAX's per-step engine push to
# gymnasium's empirical value. Note: the original LUNAR_LANDER
# review §2.1 described this gap as "gymnasium's effective thrust
# is ≈ 30-40 % STRONGER than JAX's" — that was an inversion of
# the probe interpretation (probe value is gravity-subtracted
# engine push, NOT net dvy). The correct reading is gymnasium's
# effective engine is ~40% WEAKER per impulse.
MAIN_THRUST_BODY_MULTIPLIER: float = 0.61


# ============ State / Params dataclasses ============

@struct.dataclass
class LunarLanderState:
    """Lander's articulated-body state + terrain + episode bookkeeping.

    Position / velocity are in **meters** (world frame; gymnasium's
    convention after dividing pixels by SCALE). Angle is radians.
    `prev_shaping` carries the previous step's shaping value so
    reward can compute `Δshaping − fuel`.

    Leg DOFs (`leg_angle_l/r`, `leg_omega_l/r`): each leg is a
    1-DOF pendulum rotating in the lander's body frame. Positive
    angle splays the leg outward (away from centreline). Angles
    are in radians, clamped to `[0, LEG_JOINT_REST_ANGLE]` (rest
    at ~0.9 rad outward, can yield inward toward 0.4 rad under
    ground load). Angular velocities in rad/s.

    `terrain_y` (shape `(TERRAIN_CHUNKS,)`): post-smoothing chunk
    heights generated at reset from the rng. Linear interpolation
    between chunk x-positions gives the moonscape height at any
    x. Chunks 4-7 are forced to `HELIPAD_Y`.

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
    leg_angle_l: jax.Array
    leg_angle_r: jax.Array
    leg_omega_l: jax.Array
    leg_omega_r: jax.Array
    terrain_y: jax.Array
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


# ============ Terrain helpers ============

def _generate_terrain(rng: jax.Array) -> jax.Array:
    """Sample a 11-element terrain height array following gymnasium's
    `_generate_terrain` recipe.

    Process: draw 12 raw heights from uniform(0, H/2); force chunks
    4-7 to helipad_y; apply 3-tap mean smoothing for indices
    0..CHUNKS-1. Wraps gymnasium's
    `smooth_y[i] = 0.33 (h[i-1] + h[i] + h[i+1])`.

    Returns a jnp.array of shape `(TERRAIN_CHUNKS,)` (11) ready to
    drop into `LunarLanderState.terrain_y`."""
    H_m = VIEWPORT_H / SCALE
    # Gymnasium samples (CHUNKS+1)=12 raw heights so smooth_y[i]
    # for i in [0, CHUNKS) reads index CHUNKS = 11 without OOB.
    raw = jax.random.uniform(
        rng, (TERRAIN_CHUNKS + 1,), minval=0.0, maxval=H_m / 2.0,
    )
    # Helipad strip: chunks 4-7 (inclusive). Anchored to helipad_y.
    # Gymnasium pins indices CHUNKS//2 - 2 .. CHUNKS//2 + 2 =
    # {3, 4, 5, 6, 7} — five chunks, not just 4-7. Match exactly.
    helipad_mask = jnp.array(
        [(TERRAIN_CHUNKS // 2 - 2 <= i <= TERRAIN_CHUNKS // 2 + 2)
         for i in range(TERRAIN_CHUNKS + 1)],
        dtype=jnp.bool_,
    )
    raw = jnp.where(helipad_mask, jnp.float32(HELIPAD_Y), raw)
    # Smooth: smooth_y[i] = 0.33 * (raw[i-1] + raw[i] + raw[i+1])
    # for i in 0..CHUNKS-1. raw[-1] is read via wrap-around in
    # numpy — gymnasium accepts that since raw[CHUNKS] exists
    # (raw has CHUNKS+1 = 12 entries). For i=0, raw[-1] reads the
    # last element. We replicate: gymnasium uses Python's negative
    # indexing on the numpy array.
    idx = jnp.arange(TERRAIN_CHUNKS)
    h_prev = raw[(idx - 1) % (TERRAIN_CHUNKS + 1)]
    h_cur = raw[idx]
    h_next = raw[idx + 1]
    smooth = 0.33 * (h_prev + h_cur + h_next)
    return smooth.astype(jnp.float32)


def _terrain_height_at(terrain_y: jax.Array, x: jax.Array) -> jax.Array:
    """Linearly interpolate the terrain polyline at a given x.

    Terrain chunks live at `chunk_x[i] = i * W / (CHUNKS - 1)` for
    i in 0..CHUNKS-1 — gymnasium's `chunk_x = [W/(CHUNKS-1)*i for
    i in range(CHUNKS)]`. Returns the linearly interpolated
    height between adjacent chunks. Below x=0 returns
    terrain_y[0]; above x=W returns terrain_y[-1].

    Vectorised over `x` (scalar or 1-D array)."""
    W_m = VIEWPORT_W / SCALE
    chunk_dx = W_m / (TERRAIN_CHUNKS - 1)
    # Fractional chunk index in [0, CHUNKS-1].
    fi = jnp.clip(x / chunk_dx, 0.0, jnp.float32(TERRAIN_CHUNKS - 1))
    i0 = jnp.clip(jnp.floor(fi).astype(jnp.int32), 0, TERRAIN_CHUNKS - 2)
    i1 = i0 + 1
    t = fi - i0.astype(jnp.float32)
    h0 = terrain_y[i0]
    h1 = terrain_y[i1]
    return h0 * (1.0 - t) + h1 * t


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
        # Three rng streams: terrain, initial fx, initial fy. Split
        # explicitly so terrain generation is reproducible
        # independently of the impulse sample.
        key_terrain, key_fx, key_fy = jax.random.split(rng, 3)
        terrain_y = _generate_terrain(key_terrain)
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
        # `prev_shaping`.
        x0 = jnp.asarray(INITIAL_X, dtype=jnp.float32)
        y0 = jnp.asarray(INITIAL_Y, dtype=jnp.float32)
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
        # Legs deployed outward at rest (motor speed pushed them
        # against the outward limit).
        rest = jnp.float32(LEG_REST_OUTWARD_ANGLE)
        state = LunarLanderState(
            x=x0,
            y=y0,
            vx=vx0.astype(jnp.float32),
            vy=vy0.astype(jnp.float32),
            angle=jnp.float32(0.0),
            angular_vel=jnp.float32(0.0),
            leg_contact_l=jnp.float32(0.0),
            leg_contact_r=jnp.float32(0.0),
            leg_angle_l=rest,
            leg_angle_r=rest,
            leg_omega_l=jnp.float32(0.0),
            leg_omega_r=jnp.float32(0.0),
            terrain_y=terrain_y,
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

        # Main engine: power scalar = 1.0 (discrete). Body impulse
        # = (-tip_x · offset · MAIN_POWER, +tip_y · offset ·
        # MAIN_POWER). See LUNAR_LANDER_DYNAMICS_REVIEW.md §1.1.
        # Scaled by MAIN_THRUST_BODY_MULTIPLIER to match
        # gymnasium's empirically measured effective dvy (the
        # extra ~37% comes from the iterative joint solver in
        # Box2D propagating leg-rod reaction forces onto the body,
        # which we don't model explicitly).
        dt = 1.0 / FPS

        m_power = jnp.where(is_main, jnp.float32(1.0), jnp.float32(0.0))
        m_offset = jnp.float32(MAIN_ENGINE_Y_LOCATION / SCALE)
        m_impulse_x = (
            -tip_x * params.main_engine_power * m_offset * m_power
            * MAIN_THRUST_BODY_MULTIPLIER
        )
        m_impulse_y = (
            tip_y * params.main_engine_power * m_offset * m_power
            * MAIN_THRUST_BODY_MULTIPLIER
        )
        # Lever arm matches gymnasium's (ox, oy). For the main
        # engine r and F are anti-parallel so torque = 0
        # analytically; we compute it anyway for symmetry / numeric
        # stability of any future asymmetric thrust extension.
        m_rx = tip_x * m_offset
        m_ry = -tip_y * m_offset
        m_torque = m_rx * m_impulse_y - m_ry * m_impulse_x

        # Side engine: direction = -1 for left (action=1), +1 for
        # right (action=3). See review §1.2/1.3 for sign-flip
        # corrections.
        direction = jnp.where(is_left, jnp.float32(-1.0), jnp.float32(0.0))
        direction = jnp.where(is_right, jnp.float32(1.0), direction)
        s_power = jnp.where(is_side, jnp.float32(1.0), jnp.float32(0.0))
        s_offset = jnp.float32(SIDE_ENGINE_AWAY / SCALE)
        s_impulse_x = (
            direction * tip_y * params.side_engine_power * s_offset * s_power
        )
        s_impulse_y = (
            direction * tip_x * params.side_engine_power * s_offset * s_power
        )
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

        # Semi-implicit Euler integration.
        vx_new = state.vx + dvx
        vy_new = state.vy + dvy + params.gravity * dt
        ω_new = state.angular_vel + dω

        # ===== Articulated leg dynamics =====
        # Each leg is a 1-DOF pendulum in body frame. The motor
        # torque drives leg angle outward (toward
        # LEG_JOINT_REST_ANGLE = 0.9 rad). The joint is held in
        # the range [LEG_JOINT_REST_ANGLE - LEG_JOINT_RANGE,
        # LEG_JOINT_REST_ANGLE] = [0.4, 0.9] rad by stiff penalty
        # forces. Gravity also exerts a small torque (negligible
        # compared to motor at typical scales).
        #
        # Per-leg dynamics: I · ω̇_leg = τ_motor + τ_gravity +
        # τ_limit + τ_ground. We integrate ω_leg and θ_leg with
        # semi-implicit Euler.
        leg_omega_l_new, leg_angle_l_new, _, _, contact_l = (
            self._step_one_leg(
                leg_angle=state.leg_angle_l,
                leg_omega=state.leg_omega_l,
                body_x=state.x,
                body_y=state.y,
                body_angle=state.angle,
                # Left leg: side=-1 (foot at negative body-frame x
                # when splayed outward).
                side=jnp.float32(-1.0),
                terrain_y=state.terrain_y,
                dt=dt,
                params_gravity=params.gravity,
            )
        )
        leg_omega_r_new, leg_angle_r_new, _, _, contact_r = (
            self._step_one_leg(
                leg_angle=state.leg_angle_r,
                leg_omega=state.leg_omega_r,
                body_x=state.x,
                body_y=state.y,
                body_angle=state.angle,
                side=jnp.float32(+1.0),
                terrain_y=state.terrain_y,
                dt=dt,
                params_gravity=params.gravity,
            )
        )

        # Joint-absorption body damping: when either leg foot is
        # on the ground, the joint constraint transmits an upward
        # force from the leg into the body. Approximate as a
        # multiplicative damping on body vy (only damping the
        # downward component — the ground can push up, can't pull
        # down).
        any_contact = jnp.logical_or(contact_l > 0.5, contact_r > 0.5)
        vy_damped = jnp.where(
            any_contact & (vy_new < 0.0),
            vy_new * jnp.float32(LEG_CONTACT_VY_DAMPING),
            vy_new,
        )
        vy_new = vy_damped
        # Lateral friction when foot in contact: damp vx too.
        vx_new = jnp.where(any_contact, vx_new * jnp.float32(0.8), vx_new)
        # Angular friction: foot-on-ground couples body rotation
        # to ground; damp ω.
        ω_new = jnp.where(any_contact, ω_new * jnp.float32(0.7), ω_new)

        # Update position with (now damped) new velocity.
        x_new = state.x + vx_new * dt
        y_new = state.y + vy_new * dt
        angle_new = state.angle + ω_new * dt

        # ===== Crash detection =====
        # Lander body crash: any LANDER_POLY corner's world-y must
        # stay above the terrain at its world-x. Compute the world
        # positions of all 6 corners (in meters); test
        # foot_y > terrain(foot_x) per corner; crash if any below.
        cos_n = jnp.cos(angle_new)
        sin_n = jnp.sin(angle_new)
        # Polygon vertices in body frame (already in pixel units;
        # divide by SCALE for meters).
        vx_b = jnp.array(
            [v[0] / SCALE for v in LANDER_POLY], dtype=jnp.float32,
        )
        vy_b = jnp.array(
            [v[1] / SCALE for v in LANDER_POLY], dtype=jnp.float32,
        )
        # Rotate to world frame: (cos·vx - sin·vy, sin·vx + cos·vy)
        # then translate by (x_new, y_new).
        corner_world_x = x_new + cos_n * vx_b - sin_n * vy_b
        corner_world_y = y_new + sin_n * vx_b + cos_n * vy_b
        terrain_at_corner = jax.vmap(
            lambda x: _terrain_height_at(state.terrain_y, x),
        )(corner_world_x)
        any_corner_below = jnp.any(corner_world_y <= terrain_at_corner)
        body_below_terrain = any_corner_below

        # Foot below terrain at extreme tilt → crash (legs splayed
        # but body is too tilted to land cleanly). At
        # |angle| > 0.6 the foot contact is no longer a "soft"
        # landing — treat any contact as a crash.
        steep_tilt = jnp.abs(angle_new) > jnp.float32(0.6)
        foot_contact_at_tilt = (
            steep_tilt & (jnp.logical_or(contact_l > 0.5, contact_r > 0.5))
        )

        crash_now = (
            (body_below_terrain | foot_contact_at_tilt)
            & ~state.crashed & ~state.landed
        )

        # ===== Soft-landing detection =====
        # Both legs in contact AND slow translational + rotational
        # AND near upright.
        both_legs = (contact_l > 0.5) & (contact_r > 0.5)
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

        # Compute the post-step observation + shaping inline.
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
            contact_l,
            contact_r,
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
            leg_contact_l=contact_l,
            leg_contact_r=contact_r,
            leg_angle_l=leg_angle_l_new,
            leg_angle_r=leg_angle_r_new,
            leg_omega_l=leg_omega_l_new,
            leg_omega_r=leg_omega_r_new,
            terrain_y=state.terrain_y,
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
        # contract. Use a deterministic key for the auto-reset
        # (the substrate calls reset explicitly with its own rng
        # on first step).
        reset_obs, reset_state = self.reset(
            jax.random.PRNGKey(0),
            params,
        )
        final_state = jax.tree.map(
            lambda r, n: jnp.where(done, r, n), reset_state, new_state,
        )
        final_obs = jnp.where(done, reset_obs, obs)

        return final_obs, final_state, reward, done, {}

    def _step_one_leg(
        self,
        *,
        leg_angle: jax.Array,
        leg_omega: jax.Array,
        body_x: jax.Array,
        body_y: jax.Array,
        body_angle: jax.Array,
        side: jax.Array,
        terrain_y: jax.Array,
        dt: float,
        params_gravity: float,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Advance one leg's pendulum dynamics + compute contact.

        Returns `(omega_new, angle_new, foot_x_world, foot_y_world,
        contact)`.

        Model: the leg is a rigid rod of length `LEG_ROD_LENGTH_M`
        rotating about a hinge at the lander body center (body
        frame (0, 0)). `leg_angle` measures the rod's outward
        rotation from straight-down (body-frame +y-down axis), in
        rad. `side ∈ {-1, +1}` flips the outward direction —
        side=-1 (left leg) rotates the foot toward negative
        body-frame x; side=+1 (right) toward positive.

        Foot body-frame position:
          (side · sin(θ) · L, -cos(θ) · L)

        Torques summed (all in N·m, body frame):
        - **Motor**: drives `omega` toward `+LEG_MOTOR_SPEED`
          (motor splays the leg outward toward the rest angle),
          clamped to ±LEG_SPRING_TORQUE.
        - **Gravity**: pendulum gravity torque acting on the leg.
          Approximation: gravity acts at CoM (L/2 along rod).
        - **Joint limit**: stiff penalty torque when θ outside
          [REST - RANGE, REST] = [0.558, 1.058] rad.
        - **Ground reaction**: if foot below terrain, lock
          omega → 0 and freeze angle (drives leg to rest on
          terrain).
        """
        # Motor: drives θ toward LEG_REST_OUTWARD_ANGLE, omega
        # toward LEG_MOTOR_SPEED. Simple P controller on omega:
        # torque ∝ (target - omega). Gain calibrated so the leg
        # reaches motor target in ~1 dt (gain ≈ I_leg / dt).
        motor_target_omega = jnp.float32(LEG_MOTOR_SPEED)
        omega_err = motor_target_omega - leg_omega
        tau_motor_raw = omega_err * (LEG_INERTIA / dt)
        tau_motor = jnp.clip(tau_motor_raw, -LEG_SPRING_TORQUE, LEG_SPRING_TORQUE)

        # Gravity torque. Rod's CoM is at body-frame (side · sin θ
        # · L/2, -cos θ · L/2). In world frame: rotate by
        # body_angle. World y-component of CoM relative to hinge:
        #   sin(body_angle) · side · sin θ · L/2 +
        #   cos(body_angle) · (-cos θ · L/2)
        # Gravity exerts force (0, m·g) on CoM. Torque about
        # hinge (out-of-plane scalar) = r × F = r_x · F_y - r_y ·
        # F_x. Only F_y contributes (since F_x=0): τ = m·g · r_x.
        # r_x in world = cos(body_angle) · (side · sin θ · L/2) -
        # sin(body_angle) · (-cos θ · L/2).
        # The motor / limit torques are interpreted in body frame
        # (where θ is measured); we treat τ_gravity as if also
        # body-frame-aligned (small body rotation; approximation).
        rod_cm_x_body = side * jnp.sin(leg_angle) * (LEG_ROD_LENGTH_M / 2.0)
        tau_gravity = LEG_MASS * params_gravity * rod_cm_x_body * side
        # The `· side` flips sign so τ_gravity tends to drive the
        # leg toward θ=0 (straight-down) regardless of side — the
        # gravity restoring torque on a splayed pendulum.

        # Joint limit: stiff penalty when outside [REST-RANGE, REST].
        lo = jnp.float32(LEG_REST_OUTWARD_ANGLE - LEG_JOINT_RANGE)
        hi = jnp.float32(LEG_REST_OUTWARD_ANGLE)
        k_limit = jnp.float32(5.0 * LEG_INERTIA / (dt * dt))
        below_lo = leg_angle < lo
        above_hi = leg_angle > hi
        tau_limit_lo = jnp.where(below_lo, k_limit * (lo - leg_angle), 0.0)
        tau_limit_hi = jnp.where(above_hi, k_limit * (hi - leg_angle), 0.0)
        tau_limit = tau_limit_lo + tau_limit_hi
        # Damping at limits to bleed off bounce energy.
        damp_at_limit = jnp.where(
            below_lo | above_hi, -leg_omega * (LEG_INERTIA / dt), 0.0,
        )
        tau_limit = tau_limit + damp_at_limit

        # Net torque → angular acceleration → semi-implicit Euler.
        tau_total = tau_motor + tau_gravity + tau_limit
        leg_alpha = tau_total / LEG_INERTIA
        omega_new_unconstrained = leg_omega + leg_alpha * dt
        angle_new_unconstrained = leg_angle + omega_new_unconstrained * dt

        # ===== Compute foot world position =====
        # Foot body-frame: (side · sin θ · L, -cos θ · L).
        foot_local_x = side * jnp.sin(angle_new_unconstrained) * LEG_ROD_LENGTH_M
        foot_local_y = -jnp.cos(angle_new_unconstrained) * LEG_ROD_LENGTH_M
        # Rotate by body_angle to get world offset.
        cos_b, sin_b = jnp.cos(body_angle), jnp.sin(body_angle)
        # Y-axis (gymnasium's "up") CCW rotation: world = (cos*x -
        # sin*y, sin*x + cos*y).
        foot_world_x = body_x + cos_b * foot_local_x - sin_b * foot_local_y
        foot_world_y = body_y + sin_b * foot_local_x + cos_b * foot_local_y

        # ===== Ground contact =====
        terrain_at_foot = _terrain_height_at(terrain_y, foot_world_x)
        contact = (foot_world_y <= terrain_at_foot).astype(jnp.float32)

        # When in contact: snap omega → 0 and freeze angle.
        omega_new = jnp.where(contact > 0.5, jnp.float32(0.0), omega_new_unconstrained)
        angle_new = jnp.where(contact > 0.5, leg_angle, angle_new_unconstrained)
        # Re-clamp angle to joint limits (safety).
        angle_new = jnp.clip(angle_new, lo, hi)

        return omega_new, angle_new, foot_world_x, foot_world_y, contact

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
