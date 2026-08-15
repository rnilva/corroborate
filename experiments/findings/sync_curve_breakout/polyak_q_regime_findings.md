# Polyak Q-regime decomposition — and what was already known

**Date:** 2026-05-07
**Followup to:** `polyak_tau_findings.md`, `polyak_causal_panel.json`

## Context: most of this rediscovers established findings

Before claiming this as a new mechanism story, the relevant
already-documented knowledge:

- **`findings_ddqn_scope_synthesis`** — the DDQN scope is a
  conjunction of (Q-divergence, mech dormancy, bandit-tail,
  saturation, power) confounds, each env-level. There IS no
  single-variable scope predicate; cross-env partition reflects
  multi-feature env structure.
- **`findings_q_amplification_cartpole`** — already walks the
  multi-mediator (q_std, q_mag, q_cv) dead end, then collapses
  to "the existing dormancy bridge predicts everything".
- **`findings_dampened_alpha_envs`** — already documents
  `mech HELD ↛ link HELD` on DiscountingChain and similar.
- **`findings_target_staleness_mediator`** — Acrobot is "the
  lone NULL despite GOAL polarity", with mech-HELD frac only
  63% on Acrobot.
- **`findings_l2_acrobot_goldilocks`** — Acrobot mech fires
  strongly at γ=0.999 (jens 29→14) but γ=0.99 is below the
  amplifier-active range.
- Bridge `ddqn_refuted_when_dormancy_fires` already encodes the
  "premise dormant ⇒ DDQN doesn't help" reading.

The polyak sweep ran at γ=0.99, sync=100, where Acrobot's
mechanism is documented as marginal/dormant. The "Acrobot
ATE = −349 reversal" was rediscovering this.

## What was actually verified by the polyak sweep

1. **Per-stratum Q-regime split holds**:
   - r_min ≥ 0 envs ⇒ Q̄_late > 0 (1:1 in 8/8 envs from ddqn corpus)
   - r_min < 0 envs ⇒ Q̄_late < 0 (1:1 in 2/2 dense-penalty envs)
   - This is a structural identity, not a discovery — but a useful
     **endogenous predicate** (`q_late_mean > 0`) replacing the
     env-name-or-r_min predicate.

2. **Bridge scope refactored to portable predicates**:
   `staleness_amplifies_ddqn_outcome__sparse_goal_polyak` now
   scopes on `finite('target_sync.tau') & q_late_mean > 0 &
   env_reward_polarity < -0.5 & q_divergence_score < 100`.
   No corpus tag, no env_name. Generalizes to any future polyak
   sweep on a sparse-positive GOAL env.

3. **New trace instrumentation added**:
   `target_q_at_online_argmax_per_step` reduction in
   `corroborate_rl.dqn.trace_reductions`. Enables direct
   measurement of DDQN's per-step bootstrap value (vs vanilla's
   `max(target_q)`). The new measurable
   `ddqn_bootstrap_gap_late = mean(max(target_q) − target_q[
   argmax_online]) over late 50%` quantifies DDQN's
   per-step correction magnitude as a per-cell scalar.

## What the gap-decomposition test added

Mediation regression `Δ_o = α + β_s·stale + β_g·gap + ε` on the
new q_decomp data:

```
                   β_stale (direct)   p_s     β_gap     p_g
Acrobot           −299    n.s.        0.31    +157     0.009
FourRooms         +18     n.s.        0.26    −18      0.75 (collinear)
```

```
α-sweep on dense-penalty (n=30 each, total 300 cells):
  α → jensen_gap (MECHANISM):
    Acrobot:      slope = +0.44 ± 0.83, t=+0.53, p=0.60   NULL & non-monotone
    MountainCar:  slope = −2.22 ± 0.93, t=−2.40, p=0.018  HELD

  α → outcome (LINK):
    Acrobot:      slope = +0.32 ± 0.53, t=+0.60, p=0.55   NULL
    MountainCar:  slope = −0.62 ± 1.20, t=−0.52, p=0.61   NULL
```

**Three-class taxonomy** (already implicit in `findings_ddqn_scope_
synthesis`'s confound table — making it explicit per env):

| env | r_min | mech | link | already-documented diagnosis |
|---|---:|:---:|:---:|---|
| FourRooms-misc | 0 | HELD | HELD | sparse-positive GOAL, in-rescue band — chain fires |
| MountainCar-v0 | −1 | HELD | NULL | mech fires (α→jens slope significant) but bias correction doesn't translate to outcome (`mech HELD ↛ link HELD` pattern from `findings_dampened_alpha_envs`, just on a different env) |
| Acrobot-v1 | −1 | NULL | n/a | mech dormant at γ=0.99 (matches `findings_l2_acrobot_goldilocks`'s γ=0.999 finding inverted: γ=0.99 is BELOW the chain-amplifier-active range, so Acrobot's premise is dormant here) |

## What's NEW (small contribution)

- The `ddqn_bootstrap_gap_late` measurable is a per-step DDQN-
  correction-magnitude probe that didn't exist before. It
  decomposes the algorithmic step (vanilla bootstrap value vs
  DDQN bootstrap value at fixed τ).
- The bridge `staleness_amplifies_ddqn_outcome__sparse_goal_polyak`
  with `q_late_mean > 0` endogenous predicate is a new portable
  formulation. Earlier scope used corpus or env_name.

## What's NOT new

- The "mech vs link" distinction (CLAUDE.md vocabulary).
- The "DDQN scope is multi-feature conjunction" reading (`findings_
  ddqn_scope_synthesis`).
- The "Acrobot mech is dormant at low γ" pattern (`findings_l2_
  acrobot_goldilocks`).
- The "MountainCar mech HELD link NULL" pattern (`findings_target_
  staleness_mediator` panel).
- The "Hasselt's premise dormant ⇒ DDQN doesn't help" framing
  (`ddqn_refuted_when_dormancy_fires` bridge).

## Methodological lesson

Read the existing memory before launching new analyses. Almost
every "gotcha" encountered here was already documented — the
multi-mediator dead end, the per-env mech-vs-link decomposition,
the Acrobot dormancy, the |A|/r_min/polarity discriminator
search dead-ends. The corroborate framework is *designed* to
accumulate scope-and-mechanism knowledge in memory + bridges
specifically so future investigations don't re-walk the same
paths.

## Reproduction

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. uv run python \
    experiments/findings/sync_curve_breakout/run_q_decomp_mechanism.py
JAX_PLATFORMS=cpu PYTHONPATH=. uv run python \
    experiments/findings/sync_curve_breakout/run_alpha_dense_penalty_analysis.py
```
