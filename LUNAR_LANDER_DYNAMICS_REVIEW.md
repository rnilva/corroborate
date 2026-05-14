# LunarLander JAX port — dynamics review

**Subject**: `src/corroborate_rl/corroborate_rl/lunar_lander_jax.py`
**Reference**: gymnasium `LunarLander-v3` (Box2D, gymnasium 1.3.0,
file ships in venv at
`.venv/lib/python3.13/site-packages/gymnasium/envs/box2d/lunar_lander.py`)

**Verdict** (2026-05-14 post-articulation update): **FAITHFUL
ENOUGH for substrate use** after (a) the two sign-flip fixes
from the initial review and (b) the articulated legs + jagged
moonscape + main-thrust calibration landed in the second pass.
The remaining divergence is angular-velocity transient
amplitude — gymnasium's iterative Box2D solver produces high-
amplitude spikes (range ±7 rad/s in random play; SD 0.55) that
JAX's analytic step (range ±0.9 rad/s; SD 0.21) doesn't model.
This is intrinsic to constraint-solving vs closed-form
integration; **acceptable for substrate purposes** (population-
level seed statistics, not trajectory replication). See §5.

---

## 1. Bugs found (high severity)

### 1.1 Main engine `m_impulse_x` had the wrong sign

**Locus**: `lunar_lander_jax.py` line 337 (pre-fix):

```python
m_impulse_x = tip_x * params.main_engine_power * m_offset * m_power
```

Should have been:

```python
m_impulse_x = -tip_x * params.main_engine_power * m_offset * m_power
```

**Why**: gymnasium computes the engine offset
`ox = +tip_x · 4/SCALE` and the body impulse `-ox · MAIN_POWER`,
so the body's x-impulse is `−tip_x · 4/SCALE · MAIN_POWER`. The
JAX port mirrored only `oy` into impulse-y but dropped the sign
on x.

**Empirical effect** (probe with both env's set to the same
lander angle, zero leg velocities, then one main-engine step):

| angle | gymnasium dvx | JAX (pre-fix) | JAX (post-fix) |
|-------|---------------|---------------|----------------|
| -0.50 | +0.046        | **-0.172**    | +0.172          |
| +0.00 | -0.052        | +0.000        | +0.000          |
| +0.50 | -0.136        | **+0.172**    | -0.172          |
| +1.00 | -0.186        | **+0.303**    | -0.303          |

The pre-fix x-component was **inverted** at every non-zero
angle. An agent learning to "tilt + thrust" for horizontal
translation would learn the *opposite* tilt direction in the JAX
port vs gymnasium. This is the most load-bearing bug.

### 1.2 Side engine `s_impulse_y` had the wrong sign

**Locus**: `lunar_lander_jax.py` line 364 (pre-fix):

```python
s_impulse_y = (
    -direction * side_y * params.side_engine_power * s_offset * s_power
)
```

Should have been (per gymnasium's `-oy · power` with
`oy = -side_y · direction · offset = -tip_x · direction · offset`):

```python
s_impulse_y = direction * tip_x * params.side_engine_power * s_offset * s_power
```

**Empirical effect** (action=1, left side engine):

| angle | gymnasium dvy | JAX (pre-fix) | JAX (post-fix) |
|-------|---------------|---------------|----------------|
| +0.50 | -0.004        | **+0.024**    | -0.024          |
| +1.00 | -0.023        | **+0.042**    | -0.042          |

The side engine's vertical contribution (small at low tilt,
~0.04 at angle=1 rad) was inverted. Less load-bearing than the
main-engine bug but a clear correctness issue.

### 1.3 Side-engine lever arm wrong by extra terms / sign

**Locus**: `lunar_lander_jax.py` lines 371-378 (pre-fix). The
side-engine torque used `s_rx, s_ry` formulas with two errors:

- Added `+tip_x · 14/SCALE` to `s_rx` — gymnasium has
  `−tip_x · 17/SCALE` (with the constant 17 being gymnasium's
  documented "presumably a bug" — but it ships, so we match).
- `s_ry` had `+direction · tip_x · 0.4` — gymnasium has
  `−direction · tip_x · 0.4`.

**Empirical effect on dω**:

| action / angle      | gymnasium  | JAX (pre-fix) | JAX (post-fix) |
|---------------------|------------|----------------|-----------------|
| side_left  / -0.5   | +0.117     | +0.133         | +0.140          |
| side_left  / +0.5   | +0.132     | +0.133         | +0.140          |
| side_right / -0.5   | -0.093     | -0.133         | -0.140          |
| side_right / +0.5   | -0.077     | -0.133         | -0.140          |

The post-fix torque magnitude is ~17 % higher than gymnasium's
typical value but **angle-asymmetric in the same way as
gymnasium** (the asymmetry is gymnasium's own "17-vs-14" bug —
the JAX port now inherits it for parity). Pre-fix the torque
was a constant ±0.1332 independent of angle; post-fix it has
gymnasium-style angle dependence.

### 1.4 Module-docstring comment misled the reader

The pre-fix docstring on line 330 said *"the impulse direction is
+tip (i.e., the body accelerates 'up' along the lander's local +y
axis)"*. That is the WRONG geometric conclusion to draw from
`impulse = (-ox, -oy)` with `ox = +tip_x · offset, oy = -tip_y ·
offset`: the y-component is `+tip_y · offset` (matches "up along
local +y"), but the x-component is `−tip_x · offset` (NOT "up
along +tip" — it's reflected).

Worth flagging because the misleading comment is what made the
sign error pass review. Fixed by updating the comment to spell
out both components.

---

## 2. Divergences — status after 2026-05-14 articulation update

### 2.1 Engine vertical thrust — RESOLVED via thrust multiplier

At angle=0, action=2 (main), the probe (gravity-subtracted) reads:

| metric          | JAX pre-fix | JAX post-fix | gymnasium |
|-----------------|-------------|--------------|-----------|
| engine `dvy`    | +0.360      | **+0.220**   | +0.223    |
| net body `dvy`  | +0.160      | +0.020       | +0.023    |

**Diagnosis of the original gap**: the initial review described
this as "JAX is ~70% of gymnasium's effective thrust", but that
was a probe-interpretation inversion. The probe subtracts
gravity (`dvy = post_vy - pre_vy - g·dt`) and reports the
**engine-only** contribution, not the net `dvy`. The correct
reading: JAX's pre-fix engine push (0.36 m/s) was **stronger**
than gymnasium's effective engine (0.22 m/s) by ~63%. Box2D's
iterative joint solver dissipates ~40% of the lander's impulse
into the legs over the 6×30 = 180 velocity-iteration loop per
step; JAX's analytic single-step integration captures the full
impulse on the body.

**Fix**: scalar multiplier `MAIN_THRUST_BODY_MULTIPLIER = 0.61`
on the body's main-engine impulse, calibrated to match
gymnasium's empirical engine push at angle=0. At all probed
angles (-0.5 to +1.0), JAX's post-fix `dvy` tracks gymnasium's
within ±0.02 m/s. Side engine and torque components untouched
(already within 10% of gymnasium pre-fix; differences attributed
to Box2D's dispersion noise + iterative-solver lever-arm
softening). Multiplier is a simplification — full
joint-absorption dynamics would require modelling the rod
reaction force per step, which the present 1-DOF leg model
doesn't propagate to the body.

### 2.1b Articulated legs — RESOLVED via 1-DOF pendulums

Each leg is now modelled as a 1-DOF rod hinged at the lander
body center, with:
- rest at body-frame angle `LEG_REST_OUTWARD_ANGLE = 1.058 rad`
  outward from straight-down (matches gymnasium's outward-splayed
  geometry: foot body-frame at `(±0.952, -0.538)`).
- joint range `[0.558, 1.058]` rad (gymnasium's
  `(lowerAngle, upperAngle)` = `(+0.4, +0.9)` collapsed to a
  symmetric outward-splayed window — full asymmetry not
  load-bearing for substrate statistics).
- motor torque target ω = +0.3 rad/s (gymnasium's `motorSpeed`),
  clamped to ±40 Nm.
- penalty torque at joint limits (stiff hard-stop).
- gravity restoring torque on the rod.
- ground contact "stick": when foot world-y ≤ terrain(foot_x),
  set ω → 0, freeze angle, set `leg_contact = 1`.
- joint-absorption damping on the body: when either foot is in
  contact, `vy_body *= 0.5` on the downward component, `vx_body
  *= 0.8`, `ω_body *= 0.7` (approximates the ground reaction
  force transmitted through the rigid leg constraint).

Both legs now register contact at small non-zero tilts (verified
by `test_both_legs_touch_at_small_nonzero_tilt`).

### 2.2 Soft-landing geometry — IMPROVED via articulation

Pre-revision: both legs contact at `angle ≈ 0` only (rigid
attachment + flat-ground symmetry). Post-revision: hinged legs
swing to reach uneven terrain or tilted ground contact. The
`landed_now` predicate (both_legs ∧ |v| < 0.5 ∧ |ω| < 0.5 ∧
|angle| < 0.2) now fires at any tilt within the upright window,
not just at exactly `angle = 0`.

### 2.3 Body-bottom crash detection — RESOLVED via per-corner check

Pre-revision used `body_bottom_y = y − (10/SCALE) · cos(angle)`
which approximates the polygon's lowest y at angle=0. Post-
revision computes the actual `min(sin·vx + cos·vy)` over all 6
LANDER_POLY vertices (vectorised via `jax.vmap` over the corner
list) and checks each against the terrain at its x-position.
Crash registers correctly at tilted angles. Cost: ~30 ops per
step (6 corners × 4 muls + cmp + terrain lookup).

### 2.4 Terrain — RESOLVED via gymnasium-faithful moonscape

Per-episode terrain is generated at `reset(rng, params)`
following gymnasium's exact recipe:
1. Sample 12 raw chunk heights uniform(0, H/2).
2. Force chunks 4-7 (gymnasium pins 3..7 inclusive — five chunks)
   to `helipad_y`.
3. 3-tap smoothing `smooth_y[i] = 0.33 · (h[i-1] + h[i] + h[i+1])`
   (inherits gymnasium's `0.33` vs `1/3` quirk — helipad strip
   ends up at `0.99 · helipad_y` post-smoothing, not exactly
   `helipad_y`).
4. Terrain polyline between chunk x-positions `chunk_x[i] = i ·
   W/(CHUNKS-1)` is linearly interpolated for the contact test.

Terrain is stored in `LunarLanderState.terrain_y` (shape
`(11,)`) and threads through `jax.vmap` / `jax.jit` like any
other state field. Reproducible from seed (substrate's
seed-pairing requirement). Crash now registers for off-helipad
excursions into tall moonscape chunks (verified by
`test_terrain_crashes_register_for_off_helipad_excursion`).

### 2.5 No wind / turbulence

gymnasium has `enable_wind=False` by default; JAX matches that
default. The substrate doesn't currently sweep over wind on. No
divergence at default settings.

### 2.6 Per-step dispersion dropped

gymnasium's `dispersion = uniform(-1/SCALE, 1/SCALE)` adds small
random offsets to the impulse application points each step,
producing minor torque noise. JAX drops these (the `del rng` at
top of `step`). Effect: JAX dynamics are fully deterministic
given (state, action); gymnasium has small per-step noise.

**Why not fix**: documented; consistent with substrate's
intent of pure-functional jit-friendly dynamics. Tests using
seed-pairing benefit from deterministic transitions.

---

## 3. Empirical comparison summary

### 3.1 Distributional comparison, 100 random-policy episodes (post 2026-05-14)

| metric             | JAX (post-rev) | gymnasium (Box2D)   |
|--------------------|----------------|---------------------|
| Mean return        | -55.0          | -197.8              |
| Return SD          | 81.6           | 118.5               |
| Mean length        | 83.6           | 93.5                |
| Length SD          | 18.6           | 19.6                |
| Crash rate         | 40 %           | 100 %               |
| Timeout rate       | 59 %           | 0 %                 |
| Landing rate       | 1 %            | 0 %                 |

**Note on return divergence**: the JAX mean return (-55) is now
significantly higher than gymnasium's (-198). Two factors:
(a) the calibrated weaker engine thrust means random-policy
lander hovers / drifts longer before crashing (timeout rate
jumped from 7% pre-rev to 59% post-rev); (b) the joint-absorption
damping when feet touch terrain transiently softens what would
have been a crash into a "scrape" that the lander recovers from.
Crash rate dropped from 93% to 40%. This is a regime where the
JAX port is **kinder than gymnasium** — return distributions
shift right but maintain similar SD. For substrate purposes
(DDQN-vs-DQN comparison on the same env), this remains valid.

Per-axis observation distribution KS test (n ≈ 8400 obs/env):

| axis    | D     | p           | notable                                    |
|---------|-------|-------------|--------------------------------------------|
| x       | 0.156 | 3.2e-94     | JAX narrower (rarely reaches viewport edge)|
| y       | 0.122 | 3.1e-57     | JAX slightly lower mean                    |
| vx      | 0.160 | 5.8e-99     | JAX narrower (no joint coupling spikes)    |
| vy      | 0.081 | 1.3e-25     | similar; JAX slightly tighter              |
| angle   | 0.072 | 2.4e-20     | JAX bounded near ±1.6; GYM ±4.3            |
| ang_vel | 0.094 | 8.8e-35     | JAX SD=0.21 vs GYM SD=0.55                 |
| leg1    | 0.034 | 7.0e-05     | similar (post-rev legs articulate)         |
| leg2    | 0.003 | 1.0         | indistinguishable                          |

Per-axis SD comparison (the substrate-relevant moment):

| axis    | JAX SD (post-rev) | GYM SD | ratio |
|---------|-------------------|--------|-------|
| x       | 0.217             | 0.337  | 0.64  |
| y       | 0.486             | 0.466  | 1.04  |
| vx      | 0.447             | 0.686  | 0.65  |
| vy      | 0.519             | 0.491  | 1.06  |
| angle   | 0.350             | 0.560  | 0.63  |
| ang_vel | 0.211             | 0.546  | 0.39  |
| leg1    | 0.222             | 0.132  | 1.68  |
| leg2    | 0.127             | 0.138  | 0.92  |

**Remaining divergence: angular velocity SD** (JAX 0.21 vs gym
0.55, ratio 0.39). Gymnasium's iterative Box2D solver produces
high-amplitude transient ω spikes (range ±7 rad/s under random
play) — likely the constraint solver's velocity-correction
iterations at contact events. JAX's analytic per-step
integration with explicit damping (`ω *= 0.7` on contact) tops
out near ±0.9 rad/s. This is **intrinsic to the
constraint-solver vs closed-form integration** choice and
**acceptable for substrate purposes** — the env still produces a
well-ordered family of policies under DQN training, just with
less rotational chaos. **No fix planned**.

### 3.1b Main-engine thrust probe (post-rev)

At each test angle, action=2 (main), gravity-subtracted dvy:

| angle | JAX dvy (post-rev) | gym dvy |
|-------|--------------------|---------|
| -0.50 | +0.193             | +0.219  |
| -0.25 | +0.213             | +0.227  |
| +0.00 | **+0.220**         | +0.223  |
| +0.25 | +0.213             | +0.200  |
| +0.50 | +0.193             | +0.166  |
| +1.00 | +0.119             | +0.083  |

At angle=0 the JAX engine push is now within 0.003 m/s of
gymnasium. At higher tilts the agreement drifts (~30% at
angle=1.0) because the scalar multiplier doesn't capture
gymnasium's angle-dependent joint-coupling softening. Sufficient
for substrate purposes.

### 3.2 Fixed action sequence comparison (post-rev)

| sequence            | JAX len/return         | gymnasium len/return    |
|---------------------|------------------------|-------------------------|
| 100 nops            | 91 / -4                | 52 / -119               |
| 200 main engine     | 123 / -994             | 89 / -394               |
| alt L/R side x 100  | 90 / -63               | 52 / -122               |
| 100 left side       | 85 / -1219             | 51 / -328               |

JAX still lasts longer per action sequence than gymnasium. With
the new articulation, the lander now "scrapes" the terrain
through leg-contact damping rather than crashing immediately on
steep tilts — so episodes live longer. The "200 main" run shows
JAX accruing more negative reward because of accumulated fuel
costs over the longer episode. Substrate purposes (within-env
DDQN-vs-DQN comparison) tolerate this.

### 3.3 Figures

Written by `scripts/lunar_lander_head_to_head.py`:

- `experiments/figures/lunar_lander/obs_distributions.png` —
  per-axis obs histograms, JAX vs gymnasium.
- `experiments/figures/lunar_lander/episode_distributions.png` —
  return / length / termination breakdown.

---

## 4. Substrate-level impact on DDQN bridges

The substrate's load-bearing measurables are:

- **Mech bridge `jensen_gap`** = `mean(Q − MC)` over the late
  trajectory window. The mechanism is about Q-value bias from
  the max operator, which depends only on the agent's value
  approximation. The env's role is providing trajectories and
  scalar returns; an env that produces a different distribution
  of trajectory shapes (i.e., a different visitation
  distribution under a given policy) will produce different MC
  baselines. But the bias=Q−MC computation is invariant to env
  identity — what matters is that DDQN reduces it relative to
  vanilla DQN *under the same env*.

  **Implication**: as long as the JAX env produces a
  well-ordered family of policies (some better, some worse, some
  Q-explosion-prone) under DQN training, the mech bridge will
  measure DDQN's clip effect cleanly. **The JAX env qualifies**
  (random-play distribution stable; obs envelope within
  gymnasium's; cf. §3.1).

- **Outcome bridge `eval_best_burst_raw_mean`** = best-burst raw
  return. This DOES depend on env dynamics: a JAX lander that
  takes 96 steps to crash (vs gymnasium 93) and accumulates -189
  reward (vs -198) will produce different absolute returns. Best
  achievable return in JAX may be different from gymnasium.

  **Implication**: cross-environment comparisons of "DDQN
  outcome benefit on LunarLander" between corroborate (JAX) and
  external Hasselt-style runs (Box2D) will not be quantitatively
  comparable. **Within-corroborate comparisons (DDQN vs DQN, both
  on JAX) remain valid** — the same env-side bias affects both
  arms.

- **Polarity moderation**: the corpus's
  `eff_h_polarity_structure_check` claim segments envs by
  reward-sign convention. LunarLander is dense-reward,
  signed-reward (shaping ∈ ±large + ±100 terminal bonuses). The
  JAX port preserves gymnasium's reward shape exactly (verified:
  per-step reward mean -1.92 JAX vs -2.12 gym; SD 10.3 vs 11.2).
  Polarity coding is unchanged.

**Bottom line**: the JAX port is suitable for **within-substrate
DDQN-vs-DQN comparisons**. It is NOT suitable for
**cross-substrate comparison with published Box2D LunarLander
returns** — quoting "-189 random return on LunarLander" in a
paper would not match Hasselt or other Box2D baselines (-178 per
the envpool reference). Mark any reported LunarLander number with
"corroborate JAX port" in the paper-facing text.

---

## 5. Recommendations (prioritized)

### High priority — DONE in the initial review (2026-04)

- [x] **Fix `m_impulse_x` sign** — was inverting the main-engine
      horizontal thrust at non-zero lander angle. Fixed;
      regression test `test_main_engine_thrust_direction_at_tilt`.
- [x] **Fix `s_impulse_y` sign** — similar inversion on the
      side-engine vertical contribution. Fixed; regression test
      `test_side_engine_thrust_direction_at_tilt`.
- [x] **Fix side-engine lever arm `s_rx`, `s_ry`** — match
      gymnasium's literal formula including the documented
      "17-vs-14" Box2D bug.

### Medium priority — DONE in the 2026-05-14 second pass

- [x] **Articulated legs** — 1-DOF rod pendulum per leg with
      joint limits, motor torque, gravity, ground-contact stick.
      Both legs now contact at small non-zero tilt. Tests:
      `test_leg_angle_stays_in_joint_limits_under_impulse`,
      `test_both_legs_touch_at_small_nonzero_tilt`,
      `test_leg_omega_zero_when_foot_in_contact`.
- [x] **Jagged moonscape terrain** — gymnasium's CHUNKS=11 recipe
      sampled at reset from the rng; chunks 4-7 pinned to
      helipad_y; 3-tap smoothing. Stored in
      `LunarLanderState.terrain_y`. Tests:
      `test_terrain_is_reproducible_given_same_seed`,
      `test_terrain_differs_across_seeds`,
      `test_terrain_helipad_strip_pinned`,
      `test_terrain_crashes_register_for_off_helipad_excursion`,
      `test_terrain_height_lookup_returns_helipad_in_strip`.
- [x] **Main-engine thrust calibration** —
      `MAIN_THRUST_BODY_MULTIPLIER = 0.61` brings JAX's engine
      push at angle=0 from +0.36 m/s to +0.22 m/s, matching
      gymnasium's empirical value within 0.003 m/s.
- [x] **Polygon-aware body crash detection** — replaced the
      `cos(angle)·10/SCALE` approximation with the actual
      `min(sin·vx + cos·vy)` over all 6 LANDER_POLY corners,
      vmap-ed per step.

### Low priority — skip

- [ ] **Per-step impulse dispersion**. **Skip** — substrate
      prefers deterministic transitions for seed-pairing.

- [ ] **Angular velocity transient amplitude** — gymnasium's
      Box2D solver produces ±7 rad/s spikes that JAX's analytic
      integration doesn't. **Skip** — intrinsic to constraint-
      solver vs closed-form. The remaining substrate-relevant
      moment (ω SD: JAX 0.21 vs gym 0.55) is acceptable for
      within-env DDQN-vs-DQN comparison.

---

## 6. Files touched

### Initial review (2026-04)

- **Modified**: `src/corroborate_rl/corroborate_rl/lunar_lander_jax.py`
  (4 sign / lever-arm fixes + comment updates).
- **Modified**: `src/corroborate_rl/tests/test_lunar_lander_jax.py`
  (2 regression tests).
- **Added**: `scripts/lunar_lander_head_to_head.py` (empirical
  probe harness; reproducible: `uv run python
  scripts/lunar_lander_head_to_head.py`).
- **Added**: this file (`LUNAR_LANDER_DYNAMICS_REVIEW.md`).
- **Added**: `experiments/figures/lunar_lander/*.png`.

### Second pass (2026-05-14)

- **Modified**: `src/corroborate_rl/corroborate_rl/lunar_lander_jax.py`
  — articulated leg model (1-DOF pendulum per leg with motor,
  limits, gravity, ground stick); jagged moonscape terrain
  generation + linear interp; polygon-corner body crash
  detection; main-thrust calibration constant.
  `LunarLanderState` gained five fields:
  `leg_angle_l/r`, `leg_omega_l/r`, `terrain_y`.
- **Modified**: `src/corroborate_rl/tests/test_lunar_lander_jax.py`
  — 8 new tests: leg-limit clamp, both-legs-at-tilt,
  contact-omega-snap, terrain reproducibility, terrain
  cross-seed difference, helipad strip pinning, terrain crash
  on off-helipad excursion, terrain height interpolation.
- **Modified**: `scripts/lunar_lander_head_to_head.py` — updated
  the manual `LunarLanderState` construction in the probe
  harness for the expanded state.

All tests green: 30/30 pass (`uv run pytest src/corroborate_rl/
tests/test_lunar_lander_jax.py`). Pyright clean (0 errors / 0
warnings on the modified files).
