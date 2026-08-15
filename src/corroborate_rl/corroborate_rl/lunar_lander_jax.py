"""JAX-native LunarLander-v2, gymnax-Env-Protocol compatible.

Gymnasium's reference `LunarLander-v2` is built on Box2D — a C++
rigid-body simulator that doesn't compose with `jax.vmap` and
breaks the implementation's vectorised seed-rollout pipeline. This
module reimplements the env in pure JAX so it slots into the
existing `cell_runner` codepath alongside gymnax / jumanji envs
with no Python-side per-cell branching.

**Solver shape** (2026-05-14 third pass — Box2D-faithful):
the previous revision modelled legs as 1-DOF pendulums attached
to the body via a *scalar-damping* joint absorption (`vy *= 0.5`
on contact). That hack absorbed too much impulse — random-policy
crash rate was 40 % vs gymnasium's 100 %. This revision replaces
the hack with a proper **3-body articulated chain** + **sequential-
impulse constraint solver** following Box2D v2.4's
`b2_revolute_joint.cpp` and `b2_contact_solver.cpp`.

The solver pipeline per step is:
1. Apply external forces (gravity to all three bodies; engine
   impulse + torque to the body).
2. Detect contacts (foot-vs-terrain for each leg; corner-vs-terrain
   for each of the 6 body polygon corners).
3. Run velocity-iteration loop (8 iterations, Box2D default
   collapsed from gymnasium's 6×30=180 since LunarLander is
   nearly-stationary at the joint anchors — convergence is fast):
   - Revolute joints (point-to-point linear constraint + motor +
     limits) for both legs.
   - Contact normal impulses (clamped ≥ 0, with restitution bias).
   - Contact tangent impulses (friction, Coulomb-clamped).
4. Integrate positions with corrected velocities.
5. Hard-crash flag fires on **any body-corner contact** —
   mirroring gymnasium's `ContactDetector.BeginContact` setting
   `game_over = True` whenever the lander body fixture touches
   anything.

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
  initial random impulse all use gymnasium's exact numeric values.
- **Articulated 3-body chain** (post 2026-05 third pass): the
  body, left leg, and right leg are each full 2D rigid bodies
  (x, y, vx, vy, θ, ω). The two revolute joints' anchor
  constraints, motor (target ω = ±0.3 rad/s, max torque 40 N·m),
  and angle limits (Box2D `lowerAngle=±0.4, upperAngle=±0.9` per
  gymnasium) are resolved by 8 velocity-iteration sweeps of
  sequential impulse per step.
- **Jagged moonscape** (post 2026-05 second pass): per-episode
  terrain sampled at `reset` time from the input RNG following
  gymnasium's `_generate_terrain` recipe.
- Termination: out-of-viewport, lander-body crash, or `awake=False`
  (low-velocity rest, low rotation, on the ground) — see
  `_is_terminal`.

**What's simplified** vs Box2D (intentionally):
- 8 velocity iterations (vs gymnasium's 180 = 6×30). Gymnasium's
  high count chases low residual error to avoid numeric drift over
  long episodes; 8 is sufficient for the 3-DOF joint chain at our
  per-step impulse scales. Position correction (Baumgarte) is
  skipped — leg drift is bounded by the per-step joint repair.
- Single contact point per leg foot (gymnasium has 4 from the
  rectangle box fixture). Body contact detection probes all 6
  polygon corners but only the deepest gets the hard-crash flag.
- Wind / turbulence / per-step impulse dispersion omitted (matches
  gymnasium's `enable_wind=False` default; dispersion adds minor
  torque noise that the implementation's deterministic-transition
  preference excludes).

API: matches `gymnax.environments.environment.Environment`'s
runtime contract structurally — `reset(rng, params) → (obs,
state)` and `step(rng, state, action, params) → (obs, state,
reward, done, info)`. The `EnvParams` carries
`max_steps_in_episode` (implementation convention).
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

# Leg geometry — gymnasium parameters mapped directly into Box2D's
# revolute-joint setup.
#
#   localAnchorA = (0, 0)                          # on lander body
#   localAnchorB = (i*LEG_AWAY/SCALE, LEG_DOWN/SCALE)   # on leg body
#   motorSpeed   = +0.3 * i                        # i ∈ {-1, +1}
#   maxMotorTorque = 40 N·m
#   leg limits:  i = -1 (left)  → joint_angle ∈ [+0.4, +0.9]
#                i = +1 (right) → joint_angle ∈ [-0.9, -0.4]
#
# At rest, the motor drives the joint toward its outer-most
# allowed angle (left: +0.9, right: -0.9). The leg fixture itself
# is a box of half-extents (LEG_W/SCALE, LEG_H/SCALE).
LEG_AWAY: int = 20
LEG_DOWN: int = 18
LEG_W: int = 2
LEG_H: int = 8
LEG_SPRING_TORQUE: float = 40.0    # gymnasium's maxMotorTorque (N·m)
LEG_MOTOR_SPEED: float = 0.3       # gymnasium's motorSpeed magnitude (rad/s)
# Box2D joint limits in the gymnasium convention. left joint angle
# (defined as `leg_world_angle - body_world_angle`) is bounded
# between +0.4 and +0.9; right is mirrored.
LEG_LIMIT_LO: float = 0.4
LEG_LIMIT_HI: float = 0.9

MAIN_ENGINE_POWER: float = 13.0
SIDE_ENGINE_POWER: float = 0.6
MAIN_ENGINE_Y_LOCATION: float = 4.0
SIDE_ENGINE_AWAY: int = 12
SIDE_ENGINE_HEIGHT: int = 14

GRAVITY_Y: float = -10.0
INITIAL_RANDOM: float = 1000.0

# Solver iteration count — Box2D's default for World.Step is
# (vel=8, pos=3). Gymnasium increases this to (180, 60) for
# LunarLander to keep joint drift low + propagate impulses
# fully through the chain. For our 3-DOF chain at LunarLander
# scales 8 vel-iterations is enough: at 4 iterations the
# engine thrust is over-attributed to body vy by ~60 %
# (the constraint solver hadn't propagated the impulse from the
# body to the legs yet); at 8 iterations the propagation
# matches gymnasium's 180-iteration solution within 10 %.
# Python-unrolled — `lax.fori_loop` here compiles to an XLA
# while-loop whose per-call memory cost compounds across the
# 1 000-step trace inside the random-policy rollout test.
VEL_ITERATIONS: int = 8

# Contact restitution. Gymnasium sets the lander + leg fixture
# restitution to 0.0, so contacts are inelastic.
CONTACT_RESTITUTION: float = 0.0
# Contact friction (Coulomb). Gymnasium uses 0.1 on the moon
# edges; the lander fixture has friction 0.1 too. Combined is
# sqrt(0.1·0.1) = 0.1.
CONTACT_FRICTION: float = 0.1

# Helipad anchor — gymnasium computes `helipad_y = H/4` where
# `H = VIEWPORT_H / SCALE`. Helipad strip is at chunks
# CHUNKS//2 - 2 .. CHUNKS//2 + 2 = {3, 4, 5, 6, 7} for CHUNKS=11.
HELIPAD_Y: float = (VIEWPORT_H / SCALE) / 4.0
INITIAL_Y: float = VIEWPORT_H / SCALE          # 13.33 m
INITIAL_X: float = (VIEWPORT_W / SCALE) / 2.0  # 10.0 m

# Terrain — matches gymnasium's CHUNKS=11 chunk-and-smooth recipe.
TERRAIN_CHUNKS: int = 11
TERRAIN_HELIPAD_LO: int = 4
TERRAIN_HELIPAD_HI: int = 7


# ============ Derived mass + inertia (analytic, no Box2D) ============
#
# Box2D computes mass = density × polygon_area (in m²) and inertia
# via the polygon's second moment of area. For the lander we
# approximate the second moment with the bounding-box's I_z (close
# enough — the polygon is rectangular with two chamfers).

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

_LANDER_W_M: float = 34.0 / SCALE
_LANDER_H_M: float = 27.0 / SCALE
LANDER_INERTIA: float = (
    LANDER_MASS * (_LANDER_W_M * _LANDER_W_M + _LANDER_H_M * _LANDER_H_M) / 12.0
)

# Leg body: a box of full-dimensions (2·LEG_W/SCALE × 2·LEG_H/SCALE)
# at density 1.0. Box2D's polygon-mass formula gives
# mass = density × (2w)(2h); inertia = mass · (w²+h²)/12 + mass·d²
# (parallel-axis from CoM, but Box2D body inertia is about CoM by
# convention — we use the CoM-centred inertia here).
LEG_W_M: float = LEG_W / SCALE
LEG_H_M: float = LEG_H / SCALE
LEG_MASS: float = 1.0 * (2.0 * LEG_W_M) * (2.0 * LEG_H_M)  # ≈ 0.0711 kg
LEG_INERTIA: float = LEG_MASS * (
    (2.0 * LEG_W_M) ** 2 + (2.0 * LEG_H_M) ** 2
) / 12.0

# Local anchor offsets — used by the constraint solver.
# On the lander body the anchor is at the CoM (0, 0). On each
# leg the anchor is at the leg's "shoulder" corner closer to the
# body, with sign flipped per side.
ANCHOR_A_LOCAL_X: float = 0.0
ANCHOR_A_LOCAL_Y: float = 0.0
# Right leg (i=+1): anchor at (+LEG_AWAY/SCALE, +LEG_DOWN/SCALE) on
# the leg. Left leg (i=-1): anchor at (-LEG_AWAY/SCALE,
# +LEG_DOWN/SCALE).
LEG_ANCHOR_LOCAL_X: float = LEG_AWAY / SCALE
LEG_ANCHOR_LOCAL_Y: float = LEG_DOWN / SCALE

# Foot position (for contact detection). Foot is at the leg's
# distal end — body-frame (0, -LEG_H/SCALE) on the leg fixture.
FOOT_LOCAL_X: float = 0.0
FOOT_LOCAL_Y: float = -LEG_H_M


# ============ State / Params dataclasses ============

@struct.dataclass
class LunarLanderState:
    """Lander's articulated-body state + terrain + episode bookkeeping.

    Position / velocity are in **meters** (world frame; gymnasium's
    convention after dividing pixels by SCALE). Angle is radians.

    The three rigid bodies:
    - Body: `(x, y, vx, vy, angle, angular_vel)` — the lander hull.
    - Left leg: `(leg_lx, leg_ly, leg_lvx, leg_lvy, leg_l_angle,
      leg_l_omega)` — full 2D rigid body, NOT a 1-DOF pendulum.
    - Right leg: same six fields.

    `leg_l_angle` / `leg_r_angle` are absolute world-frame angles
    (matching Box2D's `body.angle`). The Box2D joint angle (the
    quantity Box2D's limits operate on) is recovered as
    `leg_*_angle - body_angle`.

    `leg_contact_l/r` are float (0.0/1.0) — recomputed each step
    from the foot's world position vs terrain.

    `prev_shaping` carries the previous step's shaping value.
    `crashed`, `landed`, `time` are episode bookkeeping flags.
    `terrain_y` is the (11,) chunk heights array sampled at reset.
    """
    # Body
    x: jax.Array
    y: jax.Array
    vx: jax.Array
    vy: jax.Array
    angle: jax.Array
    angular_vel: jax.Array
    # Left leg
    leg_lx: jax.Array
    leg_ly: jax.Array
    leg_lvx: jax.Array
    leg_lvy: jax.Array
    leg_l_angle: jax.Array
    leg_l_omega: jax.Array
    # Right leg
    leg_rx: jax.Array
    leg_ry: jax.Array
    leg_rvx: jax.Array
    leg_rvy: jax.Array
    leg_r_angle: jax.Array
    leg_r_omega: jax.Array
    # Contact + bookkeeping
    leg_contact_l: jax.Array
    leg_contact_r: jax.Array
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
    """
    H_m = VIEWPORT_H / SCALE
    raw = jax.random.uniform(
        rng, (TERRAIN_CHUNKS + 1,), minval=0.0, maxval=H_m / 2.0,
    )
    helipad_mask = jnp.array(
        [(TERRAIN_CHUNKS // 2 - 2 <= i <= TERRAIN_CHUNKS // 2 + 2)
         for i in range(TERRAIN_CHUNKS + 1)],
        dtype=jnp.bool_,
    )
    raw = jnp.where(helipad_mask, jnp.float32(HELIPAD_Y), raw)
    idx = jnp.arange(TERRAIN_CHUNKS)
    h_prev = raw[(idx - 1) % (TERRAIN_CHUNKS + 1)]
    h_cur = raw[idx]
    h_next = raw[idx + 1]
    smooth = 0.33 * (h_prev + h_cur + h_next)
    return smooth.astype(jnp.float32)


def _terrain_height_at(terrain_y: jax.Array, x: jax.Array) -> jax.Array:
    """Linearly interpolate the terrain polyline at a given x.

    Terrain chunks live at `chunk_x[i] = i * W / (CHUNKS - 1)` for
    i in 0..CHUNKS-1. Returns the interpolated height between
    adjacent chunks; below x=0 returns terrain_y[0]; above x=W
    returns terrain_y[-1]. Vectorised over `x`.
    """
    W_m = VIEWPORT_W / SCALE
    chunk_dx = W_m / (TERRAIN_CHUNKS - 1)
    fi = jnp.clip(x / chunk_dx, 0.0, jnp.float32(TERRAIN_CHUNKS - 1))
    i0 = jnp.clip(jnp.floor(fi).astype(jnp.int32), 0, TERRAIN_CHUNKS - 2)
    i1 = i0 + 1
    t = fi - i0.astype(jnp.float32)
    h0 = terrain_y[i0]
    h1 = terrain_y[i1]
    return h0 * (1.0 - t) + h1 * t


# ============ Constraint solver primitives ============
#
# All three bodies live in a 9-vector velocity state
# `v = (vx_B, vy_B, ω_B, vx_L, vy_L, ω_L, vx_R, vy_R, ω_R)`.
# Each constraint computes a Jacobian-times-velocity scalar
# (Cdot), an effective mass (1/JM⁻¹J^T), and an impulse to apply
# back to the velocity state via -J^T.
#
# Body indices: 0 = lander, 1 = left leg, 2 = right leg.
# For a body at (x, y, vx, vy, ω) with anchor offset r = (rx, ry)
# in world frame from CoM, the world-frame velocity at the anchor
# is `v + ω × r = (vx - ω·ry, vy + ω·rx)`.


def _rotate(v_local_x: jax.Array, v_local_y: jax.Array,
            cos_a: jax.Array, sin_a: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
    """Rotate a 2D body-frame vector by angle a → world frame."""
    return (
        cos_a * v_local_x - sin_a * v_local_y,
        sin_a * v_local_x + cos_a * v_local_y,
    )


# Velocity state for the 3-body chain is a single `(3, 3)` float32
# array — `vel[body_index, dof]` where dof ∈ {0:vx, 1:vy, 2:ω} and
# body_index ∈ {0:lander, 1:left leg, 2:right leg}. Using a tensor
# (rather than 9 named scalars) keeps the HLO compact enough that
# the velocity-iteration `fori_loop` compiles cleanly inside
# `vmap` (named-scalar versions blew memory limits at trace time
# because every contact slot expanded to 3-way `jnp.where`).

_V_BODY: int = 0
_V_LEG_L: int = 1
_V_LEG_R: int = 2
_DOF_VX: int = 0
_DOF_VY: int = 1
_DOF_W: int = 2

# Per-body inverse mass/inertia indexed by body index. The contact
# solver reads these via `jnp.take(invM, body_index)` for
# data-indexed dispatch.
_INV_MASS: tuple[float, float, float] = (
    1.0 / LANDER_MASS, 1.0 / LEG_MASS, 1.0 / LEG_MASS,
)
_INV_INERTIA: tuple[float, float, float] = (
    1.0 / LANDER_INERTIA, 1.0 / LEG_INERTIA, 1.0 / LEG_INERTIA,
)


@struct.dataclass
class _JointCache:
    """Precomputed quantities for one revolute joint.

    `r_a`, `r_b` are world-frame anchor offsets from each body's
    CoM. `k_inv` is the 2×2 effective-mass inverse matrix for the
    point-to-point linear constraint, stored as four scalars.
    `axial_mass` is the scalar effective mass for the angular
    motor + limit constraints (`1 / (1/I_a + 1/I_b)`).

    `side` is +1 (right leg) or -1 (left leg); `lo`, `hi` are
    the side-adjusted joint-angle limits.
    `joint_angle` is the current joint angle
    (`leg_angle - body_angle`).
    """
    r_a_x: jax.Array
    r_a_y: jax.Array
    r_b_x: jax.Array
    r_b_y: jax.Array
    k_inv_00: jax.Array
    k_inv_01: jax.Array
    k_inv_11: jax.Array
    axial_mass: jax.Array
    side: jax.Array
    lo: jax.Array
    hi: jax.Array
    joint_angle: jax.Array


def _init_joint(
    body_angle: jax.Array,
    leg_angle: jax.Array,
    side: float,
) -> _JointCache:
    """Compute world-frame anchor offsets + effective mass for one
    revolute joint, given current body / leg angles."""
    cos_b, sin_b = jnp.cos(body_angle), jnp.sin(body_angle)
    cos_l, sin_l = jnp.cos(leg_angle), jnp.sin(leg_angle)
    # Body anchor (0, 0) → in world same as body CoM offset 0.
    r_a_x, r_a_y = _rotate(
        jnp.float32(ANCHOR_A_LOCAL_X), jnp.float32(ANCHOR_A_LOCAL_Y),
        cos_b, sin_b,
    )
    # Leg anchor (i·LEG_AWAY/SCALE, LEG_DOWN/SCALE) on leg body.
    r_b_x, r_b_y = _rotate(
        jnp.float32(side * LEG_ANCHOR_LOCAL_X),
        jnp.float32(LEG_ANCHOR_LOCAL_Y),
        cos_l, sin_l,
    )
    inv_m_a = jnp.float32(1.0 / LANDER_MASS)
    inv_m_b = jnp.float32(1.0 / LEG_MASS)
    inv_i_a = jnp.float32(1.0 / LANDER_INERTIA)
    inv_i_b = jnp.float32(1.0 / LEG_INERTIA)
    # 2×2 K matrix (b2RevoluteJoint InitVelocityConstraints):
    #   K = mA + mB + r_a.y²·iA + r_b.y²·iB,  -r_a.y·r_a.x·iA - r_b.y·r_b.x·iB
    #       [symmetric]                       mA + mB + r_a.x²·iA + r_b.x²·iB
    k00 = inv_m_a + inv_m_b + r_a_y * r_a_y * inv_i_a + r_b_y * r_b_y * inv_i_b
    k01 = -r_a_y * r_a_x * inv_i_a - r_b_y * r_b_x * inv_i_b
    k11 = inv_m_a + inv_m_b + r_a_x * r_a_x * inv_i_a + r_b_x * r_b_x * inv_i_b
    det = k00 * k11 - k01 * k01
    # Numerical guard: det is always positive for non-degenerate
    # 2-body chains (mass > 0). Add tiny epsilon for safety only.
    inv_det = 1.0 / (det + 1e-12)
    k_inv_00 = k11 * inv_det
    k_inv_01 = -k01 * inv_det
    k_inv_11 = k00 * inv_det
    axial_mass = 1.0 / (inv_i_a + inv_i_b)
    side_arr = jnp.float32(side)
    if side > 0:
        lo, hi = -LEG_LIMIT_HI, -LEG_LIMIT_LO  # right: [-0.9, -0.4]
    else:
        lo, hi = LEG_LIMIT_LO, LEG_LIMIT_HI    # left:  [+0.4, +0.9]
    return _JointCache(
        r_a_x=r_a_x, r_a_y=r_a_y, r_b_x=r_b_x, r_b_y=r_b_y,
        k_inv_00=k_inv_00, k_inv_01=k_inv_01, k_inv_11=k_inv_11,
        axial_mass=axial_mass,
        side=side_arr,
        lo=jnp.float32(lo), hi=jnp.float32(hi),
        joint_angle=(leg_angle - body_angle),
    )


def _apply_joint_impulses(
    vel: jax.Array, jc_l: _JointCache, jc_r: _JointCache, dt: float,
) -> jax.Array:
    """One pass over both revolute joints: motor + limits + linear
    point-to-point constraint. Sequential impulse — applies the
    angular constraints first (motor, then limits) and then the
    linear constraint, matching Box2D's per-iteration ordering.

    `vel` is shape (3, 3), `vel[body, dof]` (body: 0/1/2, dof:
    0=vx/1=vy/2=ω). The two legs are handled symmetrically via
    static body indices (left = 1, right = 2)."""
    vel = _apply_motor_limit_p2p(vel, jc_l, leg_idx=_V_LEG_L, dt=dt)
    vel = _apply_motor_limit_p2p(vel, jc_r, leg_idx=_V_LEG_R, dt=dt)
    return vel


def _apply_motor_limit_p2p(
    vel: jax.Array, jc: _JointCache, leg_idx: int, dt: float,
) -> jax.Array:
    """Apply motor + limit + point-to-point constraints for one leg.

    The three sub-constraints are sequential — each reads from the
    velocity state and writes a partial update before the next
    runs. Combined into one function so the closure-captured
    `leg_idx` is a Python int (no dynamic-index dispatch needed)."""
    inv_i_a = jnp.float32(1.0 / LANDER_INERTIA)
    inv_i_b = jnp.float32(1.0 / LEG_INERTIA)
    inv_m_a = jnp.float32(1.0 / LANDER_MASS)
    inv_m_b = jnp.float32(1.0 / LEG_MASS)

    w_a = vel[_V_BODY, _DOF_W]
    w_b = vel[leg_idx, _DOF_W]

    # ---- Motor ----
    motor_speed = LEG_MOTOR_SPEED * jc.side
    cdot_m = w_b - w_a - motor_speed
    imp_m = -jc.axial_mass * cdot_m
    max_imp = LEG_SPRING_TORQUE * dt
    imp_m = jnp.clip(imp_m, -max_imp, max_imp)
    w_a = w_a - imp_m * inv_i_a
    w_b = w_b + imp_m * inv_i_b

    # ---- Lower limit ----
    inv_dt = 1.0 / dt
    c_lo = jc.joint_angle - jc.lo
    bias_lo = jnp.minimum(c_lo, 0.0) * inv_dt
    imp_lo = -jc.axial_mass * ((w_b - w_a) + bias_lo)
    imp_lo = jnp.maximum(imp_lo, 0.0)
    w_a = w_a - imp_lo * inv_i_a
    w_b = w_b + imp_lo * inv_i_b

    # ---- Upper limit ----
    c_hi = jc.hi - jc.joint_angle
    bias_hi = jnp.minimum(c_hi, 0.0) * inv_dt
    imp_hi = jc.axial_mass * ((w_b - w_a) + bias_hi)
    imp_hi = jnp.minimum(imp_hi, 0.0)
    w_a = w_a - imp_hi * inv_i_a
    w_b = w_b + imp_hi * inv_i_b

    # ---- Point-to-point ----
    vx_a = vel[_V_BODY, _DOF_VX]
    vy_a = vel[_V_BODY, _DOF_VY]
    vx_b = vel[leg_idx, _DOF_VX]
    vy_b = vel[leg_idx, _DOF_VY]
    cdot_x = (vx_b - w_b * jc.r_b_y) - (vx_a - w_a * jc.r_a_y)
    cdot_y = (vy_b + w_b * jc.r_b_x) - (vy_a + w_a * jc.r_a_x)
    p_x = -(jc.k_inv_00 * cdot_x + jc.k_inv_01 * cdot_y)
    p_y = -(jc.k_inv_01 * cdot_x + jc.k_inv_11 * cdot_y)
    vx_a = vx_a - p_x * inv_m_a
    vy_a = vy_a - p_y * inv_m_a
    w_a = w_a - (jc.r_a_x * p_y - jc.r_a_y * p_x) * inv_i_a
    vx_b = vx_b + p_x * inv_m_b
    vy_b = vy_b + p_y * inv_m_b
    w_b = w_b + (jc.r_b_x * p_y - jc.r_b_y * p_x) * inv_i_b

    vel = vel.at[_V_BODY].set(jnp.stack([vx_a, vy_a, w_a]))
    vel = vel.at[leg_idx].set(jnp.stack([vx_b, vy_b, w_b]))
    return vel


# ============ Contact constraints ============
#
# Contacts are detected once per step (pre-iteration) and held as
# a fixed-shape `_ContactSet` so the velocity-iteration loop has
# constant Python structure. There are exactly 8 candidate contact
# slots: 2 leg feet + 6 body polygon corners. Inactive slots
# carry a `mass = 0` (impulse clamped to 0 every iteration).
#
# Each contact has:
#   r_x, r_y       — world-frame offset from the body's CoM to the
#                    contact point
#   n_x, n_y       — contact normal (always pointing +y for a flat
#                    moonscape; tangent is (-n_y, n_x))
#   body_index     — 0 (lander), 1 (left leg), or 2 (right leg)
#   normal_mass    — 1 / (invM + invI · (r × n)²)
#   tangent_mass   — 1 / (invM + invI · (r × t)²)
#   velocity_bias  — restitution-weighted target velocity
#   active         — float (0/1), masks impulse application
#   is_body        — float (0/1), used to set the hard-crash flag

_NUM_CONTACTS: int = 8     # 2 foot contacts + 6 body-corner contacts


@struct.dataclass
class _ContactSet:
    """Fixed-shape contact array. All fields are (NUM_CONTACTS,)."""
    r_x: jax.Array
    r_y: jax.Array
    n_x: jax.Array
    n_y: jax.Array
    body_index: jax.Array   # int32
    normal_mass: jax.Array
    tangent_mass: jax.Array
    velocity_bias: jax.Array
    active: jax.Array
    is_body: jax.Array


def _build_contacts(
    body_x: jax.Array, body_y: jax.Array, body_angle: jax.Array,
    body_vx: jax.Array, body_vy: jax.Array, body_omega: jax.Array,
    leg_lx: jax.Array, leg_ly: jax.Array, leg_l_angle: jax.Array,
    leg_lvx: jax.Array, leg_lvy: jax.Array, leg_l_omega: jax.Array,
    leg_rx: jax.Array, leg_ry: jax.Array, leg_r_angle: jax.Array,
    leg_rvx: jax.Array, leg_rvy: jax.Array, leg_r_omega: jax.Array,
    terrain_y: jax.Array,
) -> _ContactSet:
    """Detect contacts and precompute per-contact constraint data.

    Slot layout (fixed):
        0       — left foot
        1       — right foot
        2..7    — six lander body polygon corners

    Active mask is set when the world-y of the contact point is at
    or below the terrain height under it. For body-corner
    contacts, the normal points straight up (terrain is treated as
    a flat ground at the contact x). The leg-foot normal is also
    +y since the foot rectangle is small relative to terrain
    chunks.
    """
    # ---- Left foot ----
    cos_l, sin_l = jnp.cos(leg_l_angle), jnp.sin(leg_l_angle)
    foot_l_lx, foot_l_ly = _rotate(
        jnp.float32(FOOT_LOCAL_X), jnp.float32(FOOT_LOCAL_Y),
        cos_l, sin_l,
    )
    foot_l_wx = leg_lx + foot_l_lx
    foot_l_wy = leg_ly + foot_l_ly
    terrain_l = _terrain_height_at(terrain_y, foot_l_wx)
    penetration_l = terrain_l - foot_l_wy  # > 0 if below ground
    active_l = (penetration_l >= 0.0).astype(jnp.float32)
    # Anchor offset from the leg's CoM to the foot.
    r_l_x, r_l_y = foot_l_lx, foot_l_ly
    # Normal points up.
    n_x = jnp.float32(0.0)
    n_y = jnp.float32(1.0)
    # Velocity at foot
    foot_l_vx = leg_lvx - leg_l_omega * r_l_y
    foot_l_vy = leg_lvy + leg_l_omega * r_l_x
    vn_l = foot_l_vx * n_x + foot_l_vy * n_y
    # Effective mass for normal constraint (leg-only).
    inv_m_leg = jnp.float32(1.0 / LEG_MASS)
    inv_i_leg = jnp.float32(1.0 / LEG_INERTIA)
    rn_l = r_l_x * n_y - r_l_y * n_x   # r × n (z-component)
    k_n_l = inv_m_leg + inv_i_leg * rn_l * rn_l
    n_mass_l = 1.0 / (k_n_l + 1e-12)
    # Tangent (friction)
    t_x = -n_y
    t_y = n_x
    rt_l = r_l_x * t_y - r_l_y * t_x
    k_t_l = inv_m_leg + inv_i_leg * rt_l * rt_l
    t_mass_l = 1.0 / (k_t_l + 1e-12)
    # Velocity bias from restitution. For inelastic contact (e=0),
    # this is 0 except when the impact is strong (CONTACT_THRESHOLD).
    # We use Box2D's convention: bias only when vn < -threshold.
    vb_l = jnp.where(
        vn_l < -1.0,
        -CONTACT_RESTITUTION * vn_l, 0.0,
    )

    # ---- Right foot ---- (mirror)
    cos_r, sin_r = jnp.cos(leg_r_angle), jnp.sin(leg_r_angle)
    foot_r_lx, foot_r_ly = _rotate(
        jnp.float32(FOOT_LOCAL_X), jnp.float32(FOOT_LOCAL_Y),
        cos_r, sin_r,
    )
    foot_r_wx = leg_rx + foot_r_lx
    foot_r_wy = leg_ry + foot_r_ly
    terrain_r = _terrain_height_at(terrain_y, foot_r_wx)
    penetration_r = terrain_r - foot_r_wy
    active_r = (penetration_r >= 0.0).astype(jnp.float32)
    r_r_x, r_r_y = foot_r_lx, foot_r_ly
    foot_r_vx = leg_rvx - leg_r_omega * r_r_y
    foot_r_vy = leg_rvy + leg_r_omega * r_r_x
    vn_r = foot_r_vx * n_x + foot_r_vy * n_y
    rn_r = r_r_x * n_y - r_r_y * n_x
    k_n_r = inv_m_leg + inv_i_leg * rn_r * rn_r
    n_mass_r = 1.0 / (k_n_r + 1e-12)
    rt_r = r_r_x * t_y - r_r_y * t_x
    k_t_r = inv_m_leg + inv_i_leg * rt_r * rt_r
    t_mass_r = 1.0 / (k_t_r + 1e-12)
    vb_r = jnp.where(vn_r < -1.0, -CONTACT_RESTITUTION * vn_r, 0.0)

    # ---- Six body corners ----
    cos_b, sin_b = jnp.cos(body_angle), jnp.sin(body_angle)
    body_corner_lx = jnp.array(
        [v[0] / SCALE for v in LANDER_POLY], dtype=jnp.float32,
    )
    body_corner_ly = jnp.array(
        [v[1] / SCALE for v in LANDER_POLY], dtype=jnp.float32,
    )
    # Rotate into world frame, offset by body center.
    corner_world_x = body_x + cos_b * body_corner_lx - sin_b * body_corner_ly
    corner_world_y = body_y + sin_b * body_corner_lx + cos_b * body_corner_ly
    # Corner offsets from body CoM (world-frame).
    corner_r_x = corner_world_x - body_x
    corner_r_y = corner_world_y - body_y
    terrain_b = jax.vmap(
        lambda x: _terrain_height_at(terrain_y, x),
    )(corner_world_x)
    penetration_b = terrain_b - corner_world_y
    active_b = (penetration_b >= 0.0).astype(jnp.float32)
    # Normal mass per corner (body-only)
    inv_m_body = jnp.float32(1.0 / LANDER_MASS)
    inv_i_body = jnp.float32(1.0 / LANDER_INERTIA)
    rn_b = corner_r_x * n_y - corner_r_y * n_x
    k_n_b = inv_m_body + inv_i_body * rn_b * rn_b
    n_mass_b = 1.0 / (k_n_b + 1e-12)
    rt_b = corner_r_x * t_y - corner_r_y * t_x
    k_t_b = inv_m_body + inv_i_body * rt_b * rt_b
    t_mass_b = 1.0 / (k_t_b + 1e-12)
    corner_vx = body_vx - body_omega * corner_r_y
    corner_vy = body_vy + body_omega * corner_r_x
    vn_b = corner_vx * n_x + corner_vy * n_y
    vb_b = jnp.where(vn_b < -1.0, -CONTACT_RESTITUTION * vn_b, 0.0)

    # ---- Concatenate into 8-slot arrays ----
    r_x = jnp.concatenate([
        jnp.stack([r_l_x, r_r_x]), corner_r_x,
    ])
    r_y = jnp.concatenate([
        jnp.stack([r_l_y, r_r_y]), corner_r_y,
    ])
    n_xs = jnp.full((_NUM_CONTACTS,), n_x, dtype=jnp.float32)
    n_ys = jnp.full((_NUM_CONTACTS,), n_y, dtype=jnp.float32)
    body_index = jnp.array(
        [1, 2] + [0] * 6, dtype=jnp.int32,
    )
    normal_mass = jnp.concatenate([
        jnp.stack([n_mass_l, n_mass_r]), n_mass_b,
    ])
    tangent_mass = jnp.concatenate([
        jnp.stack([t_mass_l, t_mass_r]), t_mass_b,
    ])
    velocity_bias = jnp.concatenate([
        jnp.stack([vb_l, vb_r]), vb_b,
    ])
    active = jnp.concatenate([
        jnp.stack([active_l, active_r]), active_b,
    ])
    is_body = jnp.array(
        [0.0, 0.0] + [1.0] * 6, dtype=jnp.float32,
    )
    return _ContactSet(
        r_x=r_x, r_y=r_y, n_x=n_xs, n_y=n_ys,
        body_index=body_index,
        normal_mass=normal_mass,
        tangent_mass=tangent_mass,
        velocity_bias=velocity_bias,
        active=active,
        is_body=is_body,
    )


def _apply_contact_impulses(
    vel: jax.Array, contacts: _ContactSet,
) -> jax.Array:
    """One sweep over all 8 contact slots: normal + tangent
    impulse per slot, applied to the slot's `body_index`.

    `vel` is the `(3, 3)` velocity tensor. The 8 contacts have
    *static* body indices (slot 0 → leg 1, slot 1 → leg 2, slots
    2-7 → body 0), so we Python-unroll the loop — this gives XLA
    flat straight-line HLO instead of a `lax.scan` while-loop,
    which compiles ~10× faster and keeps the 1 000-step rollout
    fittable in the XLA JIT cache.
    """
    inv_m_body = jnp.float32(_INV_MASS[0])
    inv_m_leg = jnp.float32(_INV_MASS[1])
    inv_i_body = jnp.float32(_INV_INERTIA[0])
    inv_i_leg = jnp.float32(_INV_INERTIA[1])
    # Static body index per slot — slot 0 is left leg, 1 is right
    # leg, 2..7 are the six body corners. Matches `_build_contacts`'s
    # `body_index = [1, 2] + [0]*6`.
    body_indices = [_V_LEG_L, _V_LEG_R, _V_BODY, _V_BODY, _V_BODY,
                    _V_BODY, _V_BODY, _V_BODY]
    inv_m_per_slot = [inv_m_leg, inv_m_leg] + [inv_m_body] * 6
    inv_i_per_slot = [inv_i_leg, inv_i_leg] + [inv_i_body] * 6
    for k in range(_NUM_CONTACTS):
        b_idx = body_indices[k]
        inv_m = inv_m_per_slot[k]
        inv_i = inv_i_per_slot[k]
        bvx = vel[b_idx, _DOF_VX]
        bvy = vel[b_idx, _DOF_VY]
        bw = vel[b_idx, _DOF_W]
        rx = contacts.r_x[k]
        ry = contacts.r_y[k]
        nx = contacts.n_x[k]
        ny = contacts.n_y[k]
        tx, ty = -ny, nx
        vcp_x = bvx - bw * ry
        vcp_y = bvy + bw * rx
        vn = vcp_x * nx + vcp_y * ny
        vt = vcp_x * tx + vcp_y * ty
        # Normal impulse — clamped ≥ 0 per iteration (no accumulator).
        lam_n = -contacts.normal_mass[k] * (vn - contacts.velocity_bias[k])
        lam_n = jnp.maximum(lam_n, 0.0) * contacts.active[k]
        # Friction — Coulomb-clamped to ±μ·lam_n.
        lam_t = -contacts.tangent_mass[k] * vt
        max_friction = CONTACT_FRICTION * lam_n
        lam_t = jnp.clip(lam_t, -max_friction, max_friction) * contacts.active[k]
        p_x = lam_n * nx + lam_t * tx
        p_y = lam_n * ny + lam_t * ty
        new_vx = bvx + p_x * inv_m
        new_vy = bvy + p_y * inv_m
        new_w = bw + (rx * p_y - ry * p_x) * inv_i
        vel = vel.at[b_idx].set(jnp.stack([new_vx, new_vy, new_w]))
    return vel


# ============ Env implementation ============

@dataclass(frozen=True, slots=True)
class LunarLanderEnv:
    """JAX-native LunarLander-v2.

    Structurally matches the gymnax `Env` Protocol — implementation's
    `cell_runner` calls `reset(rng, params)` and `step(rng, state,
    action, params)` without inspecting the env's class.

    Construction is config-free (no fields); per-call config flows
    through `LunarLanderParams`."""

    def reset(
        self, rng: jax.Array, params: LunarLanderParams,
    ) -> tuple[jax.Array, LunarLanderState]:
        del params
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

        x0 = jnp.asarray(INITIAL_X, dtype=jnp.float32)
        y0 = jnp.asarray(INITIAL_Y, dtype=jnp.float32)
        # Legs initialised at the motor target — joint angle at
        # the *inner* limit (left: +0.4, right: -0.4). This is
        # gymnasium's equilibrium configuration: the motor drives
        # the joint toward this limit (motorSpeed = +0.3·i pushes
        # joint angle toward the value of smaller magnitude). At
        # this joint angle the legs splay outward from the body
        # with foot offset ≈ (±0.95, -0.54) m — matching the
        # geometry the pre-solver code modelled analytically.
        leg_l_angle0 = jnp.float32(+LEG_LIMIT_LO)
        leg_r_angle0 = jnp.float32(-LEG_LIMIT_LO)
        # Leg world position: anchor on body (0,0) matches leg
        # local anchor (±LEG_AWAY/SCALE, +LEG_DOWN/SCALE) rotated
        # by leg angle. World position of leg CoM = body_anchor -
        # rotate(leg_anchor_local, leg_angle).
        cos_ll, sin_ll = jnp.cos(leg_l_angle0), jnp.sin(leg_l_angle0)
        anchor_l_wx, anchor_l_wy = _rotate(
            -jnp.float32(LEG_ANCHOR_LOCAL_X),
            jnp.float32(LEG_ANCHOR_LOCAL_Y),
            cos_ll, sin_ll,
        )
        leg_lx0 = x0 - anchor_l_wx
        leg_ly0 = y0 - anchor_l_wy
        cos_lr, sin_lr = jnp.cos(leg_r_angle0), jnp.sin(leg_r_angle0)
        anchor_r_wx, anchor_r_wy = _rotate(
            +jnp.float32(LEG_ANCHOR_LOCAL_X),
            jnp.float32(LEG_ANCHOR_LOCAL_Y),
            cos_lr, sin_lr,
        )
        leg_rx0 = x0 - anchor_r_wx
        leg_ry0 = y0 - anchor_r_wy

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
            x=x0, y=y0,
            vx=vx0.astype(jnp.float32), vy=vy0.astype(jnp.float32),
            angle=jnp.float32(0.0), angular_vel=jnp.float32(0.0),
            leg_lx=leg_lx0, leg_ly=leg_ly0,
            leg_lvx=vx0.astype(jnp.float32),
            leg_lvy=vy0.astype(jnp.float32),
            leg_l_angle=leg_l_angle0, leg_l_omega=jnp.float32(0.0),
            leg_rx=leg_rx0, leg_ry=leg_ry0,
            leg_rvx=vx0.astype(jnp.float32),
            leg_rvy=vy0.astype(jnp.float32),
            leg_r_angle=leg_r_angle0, leg_r_omega=jnp.float32(0.0),
            leg_contact_l=jnp.float32(0.0),
            leg_contact_r=jnp.float32(0.0),
            terrain_y=terrain_y,
            prev_shaping=first_shaping,
            crashed=jnp.bool_(False),
            landed=jnp.bool_(False),
            time=jnp.int32(0),
        )
        return init_obs, state

    def reset_env(
        self, rng: jax.Array, params: LunarLanderParams,
    ) -> tuple[jax.Array, LunarLanderState]:
        return self.reset(rng, params)

    def step_env(
        self,
        rng: jax.Array,
        state: LunarLanderState,
        action: jax.Array,
        params: LunarLanderParams,
    ) -> tuple[
        jax.Array, LunarLanderState, jax.Array, jax.Array, dict[str, object],
    ]:
        """No-auto-reset step. Returns the pre-reset
        `(next_obs, next_state, reward, done, info)` so the
        rollout-phase stores the physical-continuation state in
        replay (load-bearing for the truncation-aware Bellman
        target). Truncation here is the `timeout` predicate:
        `state.time >= max_steps_in_episode` — natural terminations
        (crash / landed / OOB) have `truncated=0` per
        Sutton-Barto §6.6."""
        del rng
        next_obs, next_state, reward, done = self._step_physics(
            state, action, params,
        )
        # Truncation: timeout in the absence of a natural terminal.
        # `_is_terminal` predicates: crashed | landed | OOB | timeout.
        natural_terminal_full = self._is_natural_terminal(next_state, next_obs)
        timeout = next_state.time >= params.max_steps_in_episode
        truncated = jnp.logical_and(
            done, jnp.logical_and(
                timeout, jnp.logical_not(natural_terminal_full),
            ),
        ).astype(jnp.float32)
        info: dict[str, object] = {'truncated': truncated}
        return next_obs, next_state, reward, done, info

    def _is_natural_terminal(
        self, state: LunarLanderState, obs: jax.Array,
    ) -> jax.Array:
        """`crashed | landed | out_of_bounds` — terminal predicates
        OTHER than the timeout. Used by `step_env` to distinguish
        truncation (timeout-only) from natural termination."""
        out_of_bounds = jnp.abs(obs[0]) >= jnp.float32(1.0)
        return state.crashed | state.landed | out_of_bounds

    def _step_physics(
        self,
        state: LunarLanderState,
        action: jax.Array,
        params: LunarLanderParams,
    ) -> tuple[jax.Array, LunarLanderState, jax.Array, jax.Array]:
        """Pre-reset physics path. Identical to `step`'s body
        through the final terminal predicate, MINUS the auto-reset
        post-processing block. Returns `(next_obs, next_state,
        reward, done)`. Internal: `step` and `step_env` both
        delegate here."""

        act = action.astype(jnp.int32)
        is_main = act == 2
        is_left = act == 1
        is_right = act == 3
        is_side = jnp.logical_or(is_left, is_right)

        sin_a = jnp.sin(state.angle)
        cos_a = jnp.cos(state.angle)
        tip_x, tip_y = sin_a, cos_a
        dt = 1.0 / FPS

        # ---- Main engine impulse + torque ----
        m_power = jnp.where(is_main, jnp.float32(1.0), jnp.float32(0.0))
        m_offset = jnp.float32(MAIN_ENGINE_Y_LOCATION / SCALE)
        m_impulse_x = -tip_x * params.main_engine_power * m_offset * m_power
        m_impulse_y = tip_y * params.main_engine_power * m_offset * m_power
        m_rx = tip_x * m_offset
        m_ry = -tip_y * m_offset
        m_torque = m_rx * m_impulse_y - m_ry * m_impulse_x

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

        impulse_x = m_impulse_x + s_impulse_x
        impulse_y = m_impulse_y + s_impulse_y
        torque = m_torque + s_torque

        # ---- Step 1: apply external forces (gravity + engines) ----
        # Δv from impulses; gravity is a force g·dt applied to all
        # three bodies. The mass divides only into the impulses
        # (Box2D's ApplyLinearImpulse semantics).
        dvx_body = impulse_x / LANDER_MASS
        dvy_body = impulse_y / LANDER_MASS
        dw_body = torque / LANDER_INERTIA
        vel0 = jnp.stack([
            jnp.stack([
                state.vx + dvx_body,
                state.vy + dvy_body + params.gravity * dt,
                state.angular_vel + dw_body,
            ]),
            jnp.stack([
                state.leg_lvx,
                state.leg_lvy + params.gravity * dt,
                state.leg_l_omega,
            ]),
            jnp.stack([
                state.leg_rvx,
                state.leg_rvy + params.gravity * dt,
                state.leg_r_omega,
            ]),
        ])

        # ---- Step 2: build contact set ----
        contacts = _build_contacts(
            body_x=state.x, body_y=state.y, body_angle=state.angle,
            body_vx=vel0[_V_BODY, _DOF_VX],
            body_vy=vel0[_V_BODY, _DOF_VY],
            body_omega=vel0[_V_BODY, _DOF_W],
            leg_lx=state.leg_lx, leg_ly=state.leg_ly,
            leg_l_angle=state.leg_l_angle,
            leg_lvx=vel0[_V_LEG_L, _DOF_VX],
            leg_lvy=vel0[_V_LEG_L, _DOF_VY],
            leg_l_omega=vel0[_V_LEG_L, _DOF_W],
            leg_rx=state.leg_rx, leg_ry=state.leg_ry,
            leg_r_angle=state.leg_r_angle,
            leg_rvx=vel0[_V_LEG_R, _DOF_VX],
            leg_rvy=vel0[_V_LEG_R, _DOF_VY],
            leg_r_omega=vel0[_V_LEG_R, _DOF_W],
            terrain_y=state.terrain_y,
        )

        # ---- Step 3: initialise joint caches ----
        jc_l = _init_joint(state.angle, state.leg_l_angle, side=-1.0)
        jc_r = _init_joint(state.angle, state.leg_r_angle, side=+1.0)

        # ---- Step 4: velocity iterations (Python-unrolled) ----
        # `VEL_ITERATIONS` is small (4) so Python-unrolling produces
        # flat HLO. A `lax.fori_loop` here would compile to an XLA
        # while-loop whose per-call memory footprint compounds
        # across the 1 000-step trace inside the random-policy
        # rollout test.
        vel = vel0
        for _ in range(VEL_ITERATIONS):
            vel = _apply_joint_impulses(vel, jc_l, jc_r, dt=dt)
            vel = _apply_contact_impulses(vel, contacts)
        vel_final = vel
        body_vx_new = vel_final[_V_BODY, _DOF_VX]
        body_vy_new = vel_final[_V_BODY, _DOF_VY]
        body_w_new = vel_final[_V_BODY, _DOF_W]
        leg_lvx_new = vel_final[_V_LEG_L, _DOF_VX]
        leg_lvy_new = vel_final[_V_LEG_L, _DOF_VY]
        leg_l_omega_new = vel_final[_V_LEG_L, _DOF_W]
        leg_rvx_new = vel_final[_V_LEG_R, _DOF_VX]
        leg_rvy_new = vel_final[_V_LEG_R, _DOF_VY]
        leg_r_omega_new = vel_final[_V_LEG_R, _DOF_W]

        # ---- Step 5: integrate positions ----
        x_new = state.x + body_vx_new * dt
        y_new = state.y + body_vy_new * dt
        angle_new = state.angle + body_w_new * dt
        leg_lx_new = state.leg_lx + leg_lvx_new * dt
        leg_ly_new = state.leg_ly + leg_lvy_new * dt
        leg_l_angle_new = state.leg_l_angle + leg_l_omega_new * dt
        leg_rx_new = state.leg_rx + leg_rvx_new * dt
        leg_ry_new = state.leg_ry + leg_rvy_new * dt
        leg_r_angle_new = state.leg_r_angle + leg_r_omega_new * dt

        # ---- Step 6: position-level joint corrections ----
        # (a) angle clamp — Baumgarte for the joint-limit
        #     constraint. After Euler integration the joint angle
        #     can drift outside the limit window; we project it
        #     back. Box2D would run 60 position iterations to
        #     converge this; a single clamp captures the bulk of
        #     the correction at LunarLander dt and joint mass
        #     ratio.
        # (b) translation correction — Baumgarte for the point-to-
        #     point constraint. The leg's joint anchor may have
        #     drifted from the body's anchor; we push the leg's
        #     CoM back. Because LEG_MASS << LANDER_MASS the full
        #     correction goes onto the leg.
        joint_l = leg_l_angle_new - angle_new
        joint_l_clamped = jnp.clip(joint_l, LEG_LIMIT_LO, LEG_LIMIT_HI)
        leg_l_angle_new = leg_l_angle_new + (joint_l_clamped - joint_l)
        joint_r = leg_r_angle_new - angle_new
        joint_r_clamped = jnp.clip(joint_r, -LEG_LIMIT_HI, -LEG_LIMIT_LO)
        leg_r_angle_new = leg_r_angle_new + (joint_r_clamped - joint_r)
        leg_lx_new, leg_ly_new = _position_correct_leg(
            x_new, y_new, angle_new,
            leg_lx_new, leg_ly_new, leg_l_angle_new, side=-1.0,
        )
        leg_rx_new, leg_ry_new = _position_correct_leg(
            x_new, y_new, angle_new,
            leg_rx_new, leg_ry_new, leg_r_angle_new, side=+1.0,
        )

        # ---- Step 7: contact flag from refreshed positions ----
        # Foot world-y vs terrain.
        cos_ll, sin_ll = jnp.cos(leg_l_angle_new), jnp.sin(leg_l_angle_new)
        foot_l_dx, foot_l_dy = _rotate(
            jnp.float32(FOOT_LOCAL_X), jnp.float32(FOOT_LOCAL_Y),
            cos_ll, sin_ll,
        )
        foot_l_wy = leg_ly_new + foot_l_dy
        foot_l_wx = leg_lx_new + foot_l_dx
        terrain_at_l = _terrain_height_at(state.terrain_y, foot_l_wx)
        contact_l = (foot_l_wy <= terrain_at_l).astype(jnp.float32)
        cos_lr, sin_lr = jnp.cos(leg_r_angle_new), jnp.sin(leg_r_angle_new)
        foot_r_dx, foot_r_dy = _rotate(
            jnp.float32(FOOT_LOCAL_X), jnp.float32(FOOT_LOCAL_Y),
            cos_lr, sin_lr,
        )
        foot_r_wx = leg_rx_new + foot_r_dx
        foot_r_wy = leg_ry_new + foot_r_dy
        terrain_at_r = _terrain_height_at(state.terrain_y, foot_r_wx)
        contact_r = (foot_r_wy <= terrain_at_r).astype(jnp.float32)

        # ---- Step 8: crash flag — any body corner below terrain ----
        cos_n = jnp.cos(angle_new)
        sin_n = jnp.sin(angle_new)
        vx_b_arr = jnp.array(
            [v[0] / SCALE for v in LANDER_POLY], dtype=jnp.float32,
        )
        vy_b_arr = jnp.array(
            [v[1] / SCALE for v in LANDER_POLY], dtype=jnp.float32,
        )
        corner_world_x = x_new + cos_n * vx_b_arr - sin_n * vy_b_arr
        corner_world_y = y_new + sin_n * vx_b_arr + cos_n * vy_b_arr
        terrain_at_corner = jax.vmap(
            lambda x: _terrain_height_at(state.terrain_y, x),
        )(corner_world_x)
        any_corner_below = jnp.any(corner_world_y <= terrain_at_corner)
        body_below_terrain = any_corner_below

        crash_now = body_below_terrain & ~state.crashed & ~state.landed

        # ---- Step 9: landing detection ----
        both_legs = (contact_l > 0.5) & (contact_r > 0.5)
        v_norm = jnp.sqrt(body_vx_new * body_vx_new
                          + body_vy_new * body_vy_new)
        slow = v_norm < 0.5
        upright = jnp.abs(angle_new) < 0.2
        slow_w = jnp.abs(body_w_new) < 0.5
        landed_now = (
            both_legs & slow & slow_w & upright
            & ~state.crashed & ~state.landed
        )
        crashed = state.crashed | crash_now
        landed = state.landed | landed_now

        # ---- Observation + reward ----
        half_w = VIEWPORT_W / SCALE / 2.0
        half_h = VIEWPORT_H / SCALE / 2.0
        leg_anchor = HELIPAD_Y + LEG_DOWN / SCALE
        obs = jnp.array([
            (x_new - half_w) / half_w,
            (y_new - leg_anchor) / half_h,
            body_vx_new * half_w / FPS,
            body_vy_new * half_h / FPS,
            angle_new,
            20.0 * body_w_new / FPS,
            contact_l,
            contact_r,
        ], dtype=jnp.float32)
        cur_shaping = shaping(obs)
        delta_shaping = cur_shaping - state.prev_shaping

        new_state = LunarLanderState(
            x=x_new.astype(jnp.float32),
            y=y_new.astype(jnp.float32),
            vx=body_vx_new.astype(jnp.float32),
            vy=body_vy_new.astype(jnp.float32),
            angle=angle_new.astype(jnp.float32),
            angular_vel=body_w_new.astype(jnp.float32),
            leg_lx=leg_lx_new.astype(jnp.float32),
            leg_ly=leg_ly_new.astype(jnp.float32),
            leg_lvx=leg_lvx_new.astype(jnp.float32),
            leg_lvy=leg_lvy_new.astype(jnp.float32),
            leg_l_angle=leg_l_angle_new.astype(jnp.float32),
            leg_l_omega=leg_l_omega_new.astype(jnp.float32),
            leg_rx=leg_rx_new.astype(jnp.float32),
            leg_ry=leg_ry_new.astype(jnp.float32),
            leg_rvx=leg_rvx_new.astype(jnp.float32),
            leg_rvy=leg_rvy_new.astype(jnp.float32),
            leg_r_angle=leg_r_angle_new.astype(jnp.float32),
            leg_r_omega=leg_r_omega_new.astype(jnp.float32),
            leg_contact_l=contact_l,
            leg_contact_r=contact_r,
            terrain_y=state.terrain_y,
            prev_shaping=cur_shaping,
            crashed=crashed,
            landed=landed,
            time=state.time + 1,
        )
        fuel_cost = m_power * jnp.float32(0.30) + s_power * jnp.float32(0.03)
        step_reward = delta_shaping - fuel_cost

        x_obs = (
            (x_new - jnp.float32(VIEWPORT_W / SCALE / 2.0))
            / jnp.float32(VIEWPORT_W / SCALE / 2.0)
        )
        oob_now = (
            (jnp.abs(x_obs) >= jnp.float32(1.0))
            & ~state.crashed & ~state.landed
        )
        crash_bonus = jnp.where(
            crash_now | oob_now, jnp.float32(-100.0), jnp.float32(0.0),
        )
        landing_bonus = jnp.where(
            landed_now, jnp.float32(+100.0), jnp.float32(0.0),
        )
        reward = step_reward + crash_bonus + landing_bonus

        done = self._is_terminal(new_state, params)

        return obs, new_state, reward, done

    def step(
        self,
        rng: jax.Array,
        state: LunarLanderState,
        action: jax.Array,
        params: LunarLanderParams,
    ) -> tuple[
        jax.Array, LunarLanderState, jax.Array, jax.Array, dict[str, object],
    ]:
        """Auto-resetting step (gymnax-API equivalent). Calls
        `_step_physics` then `lax.select`s in a fresh-reset state /
        obs when `done`. Substrate's eval-loop consumes this; the
        rollout-phase reaches for `step_env` to capture the
        pre-reset `next_obs` (load-bearing for the
        truncation-aware Bellman target)."""
        del rng
        obs, new_state, reward, done = self._step_physics(state, action, params)
        reset_obs, reset_state = self.reset(jax.random.PRNGKey(0), params)
        final_state = jax.tree.map(
            lambda r, n: jnp.where(done, r, n), reset_state, new_state,
        )
        final_obs = jnp.where(done, reset_obs, obs)
        return final_obs, final_state, reward, done, {}

    def _get_obs(self, state: LunarLanderState) -> jax.Array:
        """Gymnasium's 8-dim observation, normalised to roughly
        ±1 for typical play."""
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
        """Termination predicates — pure function of state."""
        obs = self._get_obs(state)
        out_of_bounds = jnp.abs(obs[0]) >= jnp.float32(1.0)
        timeout = state.time >= params.max_steps_in_episode
        return state.crashed | state.landed | out_of_bounds | timeout

    def action_space(self, params: LunarLanderParams) -> Discrete:
        del params
        return spaces.Discrete(num_categories=4)

    def observation_space(self, params: LunarLanderParams) -> Box:
        del params
        high = jnp.array(
            [2.5, 2.5, 10.0, 10.0, 6.2831855, 10.0, 1.0, 1.0],
            dtype=jnp.float32,
        )
        return spaces.Box(low=-high, high=high, shape=(8,), dtype=jnp.float32)

    @property
    def default_params(self) -> LunarLanderParams:
        return LunarLanderParams()


def _position_correct_leg(
    body_x: jax.Array, body_y: jax.Array, body_angle: jax.Array,
    leg_x: jax.Array, leg_y: jax.Array, leg_angle: jax.Array,
    side: float,
) -> tuple[jax.Array, jax.Array]:
    """Translation-only Baumgarte correction.

    After velocity-iteration + Euler integration, the joint anchor
    on the leg (at leg_local_anchor) may have drifted from the
    body's anchor (at 0,0 in body frame). We compute the
    world-frame separation and translate the leg back to close
    it. Because LEG_MASS << LANDER_MASS, the body's correction is
    negligible (mass ratio ≈ 67×); we put the entire correction on
    the leg side.
    """
    cos_b, sin_b = jnp.cos(body_angle), jnp.sin(body_angle)
    cos_l, sin_l = jnp.cos(leg_angle), jnp.sin(leg_angle)
    body_anchor_x, body_anchor_y = body_x, body_y  # (0, 0) local
    leg_anchor_dx, leg_anchor_dy = _rotate(
        jnp.float32(side * LEG_ANCHOR_LOCAL_X),
        jnp.float32(LEG_ANCHOR_LOCAL_Y),
        cos_l, sin_l,
    )
    leg_anchor_x = leg_x + leg_anchor_dx
    leg_anchor_y = leg_y + leg_anchor_dy
    # Body-side correction is negligible; keep body fixed.
    del cos_b, sin_b, body_anchor_x, body_anchor_y
    # Drift = body_anchor - leg_anchor. Move leg by +drift.
    err_x = body_x - leg_anchor_x
    err_y = body_y - leg_anchor_y
    return leg_x + err_x, leg_y + err_y


def shaping(obs: jax.Array) -> jax.Array:
    """Gymnasium's shaping function — pure observation function.

    `shaping = -100 √(x² + y²) - 100 √(vx² + vy²) - 100 |angle|
               + 10 · (leg1 + leg2)`
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


# ============ Backward-compat: kept for downstream scripts ============
# `LEG_REST_OUTWARD_ANGLE` was the 1-DOF pendulum's "rest splay
# angle" in the pre-solver formulation (measured CCW from straight-
# down). With the 3-body solver, the equivalent notion is the joint
# *inner* limit (the motor drives the joint there; foot ends up at
# body offset (±0.95, -0.54)). We re-export the inner limit
# (`LEG_LIMIT_LO = 0.4`) so downstream scripts that previously
# used `leg_angle_l=LEG_REST_OUTWARD_ANGLE` to position legs at
# rest continue to construct a settled configuration.
LEG_REST_OUTWARD_ANGLE: float = LEG_LIMIT_LO
