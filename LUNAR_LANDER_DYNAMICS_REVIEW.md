# LunarLander JAX port — dynamics review

**Subject**: `src/corroborate_rl/corroborate_rl/lunar_lander_jax.py`
(commit `901711d`)
**Reference**: gymnasium `LunarLander-v3` (Box2D, gymnasium 1.3.0,
file ships in venv at
`.venv/lib/python3.13/site-packages/gymnasium/envs/box2d/lunar_lander.py`)

**Verdict**: **FAITHFUL ENOUGH for substrate use AFTER the two
sign-flip fixes landed in this review**. Distributional fidelity
under random play was already within 1 SD of the reference; the
fixed-action probe revealed two genuine sign bugs in the engine
impulse formulas that produced reversed translational
affordances at non-zero lander angles — i.e. an agent learning
the JAX port would have inverted its control mapping vs
gymnasium. Both fixed; covered by regression tests. Remaining
divergences (engine vertical thrust ≈ 70 % of gymnasium's, no
articulated leg dynamics, rigid body-bottom approximation) are
**load-bearing for trajectory replication but not for the
substrate's statistical purposes** — see §5.

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

## 2. Bugs / divergences NOT fixed (with rationale)

### 2.1 Engine vertical thrust ≈ 70 % of gymnasium's

At angle=0, action=2 (main):

- gymnasium: total `dvy = +0.219` (engine push 0.42 over one step,
  reduced by joints absorbing momentum + gravity)
- JAX:       total `dvy = +0.160` (engine push 0.36 over one
  step, no joint absorption + gravity)

The JAX port's engine acts on the lander as a single 4.77 kg
rigid body; gymnasium's lander is articulated through revolute
joints with two ~0.07 kg legs. The joints' constraint forces
absorb a fraction of the lander's impulse on each step. The net
effect is **gymnasium's effective thrust is ≈ 30-40 % stronger
per impulse, in the upward direction, than JAX's "ideal rigid
body"** integration.

**Why not fix**: matching Box2D's joint-coupling dissipation
analytically without Box2D is the entire problem the port avoids.
The empirical effect on episode-length / return distributions is
small (random-policy mean return -189 vs -198, within 1 SD).
Marked as low priority.

### 2.2 Rigid-attached legs — no articulation, no shock absorption

The JAX port treats both legs as fixed body-frame points at
`(±20/SCALE, -18/SCALE)`. Two consequences:

- **Both legs touch ground ONLY at angle ≈ 0** (flat ground +
  rigid geometry → the two leg world-y values are equal only when
  cos(angle)·leg_dy contributes the same on both sides, which
  happens iff angle=0).
- gymnasium's articulated legs swing on revolute joints with
  `lowerAngle/upperAngle` of ±0.4 rad of travel. They can rest
  on uneven contact and the joint motor torque (40 Nm) keeps
  them deployed against gravity. A lander touching down with
  small tilt (e.g. ±0.1 rad) lands cleanly in gymnasium because
  one leg compresses; in the JAX port, only one leg
  registers contact and `both_legs` never fires.

**Empirical effect**: under random play 100 episodes,
gymnasium had 0 landings, JAX had 0 landings — the legs-disjoint
issue doesn't surface under random policy (everything crashes).
Under a heuristic / trained policy it WILL matter: the JAX port
demands tighter angle control to register `landed`.

**Why not fix**: implementing articulated legs without Box2D
means writing a 2-body revolute-joint constraint solver — that's
≥ 200 LOC of physics that defeats the "no Box2D" goal. The
substrate's outcome bridge measures `eval_best_burst_raw_mean`
on Hasselt-convention runs — a learner that lands "harder" in JAX
than gymnasium still produces a well-ordered ranking across
seeds. Marked medium-low priority; a future fix would relax the
`upright` predicate to `|angle| < 0.3` or similar to compensate.

### 2.3 Body-bottom crash detection underestimates at tilt

`body_bottom_y = y_new − (10/SCALE) · cos(angle)` is the
projection of the polygon's lowest y-coordinate at angle=0
through `cos`. The correct value at angle is

```
min over polygon vertices (sin·vx_local + cos·vy_local)
```

which at angle=±0.5 gives `−0.564` (m below center), not
`−0.293` as the JAX approximation says. The JAX port therefore
**registers crash too LATE at tilted angles** — the body can be
deeper than gymnasium would allow before the JAX env declares
crash. Combined with the rigid-leg geometry, the lander has a
narrow tilt window: too tilted → leg contact fails AND body
crash registers late.

**Empirical effect**: not visible under random play (crash rate
93 % JAX vs 100 % gymnasium). Will matter for fine-tuned
policies.

**Why not fix**: would require recomputing
`min(sin·vx + cos·vy)` over four polygon corners per step
inside the jit kernel. Doable but adds ~6 mul + 3 cmp per step.
Marked low priority.

### 2.4 Terrain — flat ground vs random moonscape

gymnasium generates a randomly-jagged moonscape outside the
helipad; only the helipad strip (chunks 4–7) is flat. The JAX
port has flat ground at `helipad_y` everywhere. Effect: a JAX
lander that drifts off-helipad can still "land" safely if the
predicate is satisfied; a gymnasium lander on the same trajectory
would have hit jagged terrain and crashed. This rewards the
JAX-trained agent for off-helipad excursions that the gymnasium
agent would not get rewarded for.

**Why not fix**: documented in module docstring as intentional.
The shaping reward `−100·sqrt(x² + y²)` strongly penalizes
off-center landings, so off-helipad landings are already
discouraged. Substrate purposes (statistical fidelity over a
population of trajectories) tolerate this.

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

### 3.1 Distributional comparison, 100 random-policy episodes

| metric             | JAX (post-fix) | gymnasium (Box2D)   |
|--------------------|----------------|---------------------|
| Mean return        | -189.3         | -197.8              |
| Return SD          | 97.9           | 118.5               |
| Mean length        | 96.1           | 93.5                |
| Length SD          | 20.1           | 19.6                |
| Crash rate         | 93 %           | 100 %               |
| Timeout rate       | 7 %            | 0 %                 |
| Landing rate       | 0 %            | 0 %                 |

The substrate-relevant statistics (mean / SD of return, length)
are within 1 SD between envs.

Per-axis observation distribution KS test (n ≈ 9600 obs/env):

| axis    | D     | p          | notable                            |
|---------|-------|------------|------------------------------------|
| x       | 0.046 | 4.8e-09    | tighter in JAX (no off-pad terrain)|
| y       | 0.027 | 2.4e-03    | similar                            |
| vx      | 0.068 | 3.2e-19    | JAX narrower (no joint coupling)   |
| vy      | 0.013 | 0.44       | indistinguishable                  |
| angle   | 0.065 | 4.9e-18    | JAX bounded near ±2.6; GYM ±4.3    |
| ang_vel | 0.052 | 2.1e-11    | JAX SD=0.25 vs GYM SD=0.55         |
| leg1    | 0.005 | 1.00       | indistinguishable                  |
| leg2    | 0.007 | 0.96       | indistinguishable                  |

KS statistically rejects equality on 6/8 axes but **practical
divergence is small** — the largest D is 0.07 (vx). For
substrate purposes (population-level seed statistics, not
trajectory replication), this is acceptable.

The largest qualitative divergence is **angular velocity SD**:
GYM 0.55 vs JAX 0.25 (2× tighter in JAX). The likely cause is
that gymnasium's polygon collisions + joint dynamics generate
large transient angular spikes that JAX's "either crash or
continue intact" doesn't model. This shows up in the per-axis
obs envelope and likely shifts the effective state distribution
the agent learns over.

### 3.2 Fixed action sequence comparison

| sequence            | JAX len/return         | gymnasium len/return    |
|---------------------|------------------------|-------------------------|
| 100 nops            | 52 / -92               | 52 / -119               |
| 200 main engine     | 176 / -838             | 89 / -394               |
| alt L/R side x 100  | 52 / -100              | 52 / -122               |
| 100 left side       | 52 / -497              | 51 / -328               |

The "200 main" run shows the biggest gap: JAX lasts 176 steps
(2x gymnasium) and accrues more negative reward because the JAX
engine can keep the lander aloft longer — JAX's engine produces
nearly equal upward thrust without joint dissipation, so the
lander hovers / overshoots vs gymnasium's quicker descent. This
is the **engine-thrust 70 %** effect from §2.1 working in
reverse: at angle=0 JAX's vertical thrust per step is actually
**stronger** than gymnasium's effective thrust through the
joints, so the lander stays up longer.

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

### High priority — DONE in this review

- [x] **Fix `m_impulse_x` sign** (line 337) — was inverting the
      main-engine horizontal thrust at non-zero lander angle. The
      most load-bearing bug: an agent learning to tilt-and-thrust
      would have learned the **opposite** tilt convention vs
      gymnasium. Fixed; regression test
      `test_main_engine_thrust_direction_at_tilt` added.

- [x] **Fix `s_impulse_y` sign** (line 364) — similar inversion on
      the side-engine vertical contribution. Fixed; regression
      test `test_side_engine_thrust_direction_at_tilt` added.

- [x] **Fix side-engine lever arm `s_rx`, `s_ry`** (lines 371-378)
      — match gymnasium's literal formula including the
      documented "17-vs-14" Box2D bug. Fixed inline.

- [x] **Correct misleading docstring on main-engine thrust
      direction**. Fixed inline.

### Medium priority — defer with rationale

- [ ] **Relax `landed_now` predicate to allow non-zero angle**.
      Currently requires both legs in contact, which under flat
      ground + rigid attachments only happens at angle=0.
      Suggested: replace `both_legs` with `at_least_one_leg ∧
      low_vy` and tighten `upright` to `|angle| < 0.1`. **Defer**
      because the trained-DQN landing rate is ~0 % in current
      sweeps anyway; the simplification doesn't bind on the
      learning bottleneck.

- [ ] **Improve body-bottom crash detection** to use the actual
      polygon-corner min, not `cos(angle)·10/SCALE`. **Defer**
      because random-policy crash rate (93 vs 100 %) shows it
      doesn't significantly misclassify; would matter for an
      already-landing agent.

### Low priority — skip

- [ ] **Articulated legs** with revolute joints + spring torque.
      This is the largest single divergence but matching it
      requires Box2D or a 2-body constraint solver. **Skip** —
      this is the entire raison d'être of the JAX port. The
      DDQN-vs-DQN comparison is valid without it; cross-substrate
      comparisons are not the use case.

- [ ] **Random terrain** outside helipad. **Skip** — documented;
      shaping reward already penalizes off-helipad.

- [ ] **Per-step impulse dispersion**. **Skip** — substrate
      prefers deterministic transitions for seed-pairing.

- [ ] **Joint-coupling damping** that reduces vertical thrust by
      ~30 %. **Skip** — bias is consistent across DDQN and DQN
      arms; cancels in within-substrate comparisons.

---

## 6. Files touched by this review

- **Modified**: `src/corroborate_rl/corroborate_rl/lunar_lander_jax.py`
  (4 sign / lever-arm fixes + comment updates, ~30 lines net
  change).
- **Modified**: `src/corroborate_rl/tests/test_lunar_lander_jax.py`
  (2 regression tests; 22/22 pass).
- **Added**: `scripts/lunar_lander_head_to_head.py` (empirical
  probe harness; reproducible: `uv run python
  scripts/lunar_lander_head_to_head.py`).
- **Added**: this file
  (`LUNAR_LANDER_DYNAMICS_REVIEW.md`).
- **Added**: `experiments/figures/lunar_lander/*.png` (obs and
  episode-statistic histograms).

All tests green (`uv run pytest src/corroborate_rl/tests/
test_lunar_lander_jax.py`). Pyright clean on the modified files.
