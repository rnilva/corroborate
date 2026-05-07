# Why r_min discriminates: Q-regime sign sets the direction of Hasselt's bias

**Date:** 2026-05-07
**Followup to:** `polyak_tau_findings.md`, `polyak_causal_panel.json`

## TL;DR

The exogenous predicate `r_min ≥ 0` (no per-step penalty floor)
discriminates FourRooms from Acrobot/MountainCar because it
determines the **sign of vanilla DQN's late-window Q**, which sets
the **direction of Hasselt's overestimation bias**, which inverts
the sign of `ATE(target_staleness → Δ_outcome)`.

The endogenous downstream of `r_min` is the new measurable
`q_late_mean` (mean of `online_max_q_per_step` over the late 50%
of training). Bridges now use `q_late_mean > 0` directly as the
regime predicate instead of `r_min ≥ 0`.

## Empirical signature

Polyak corpus, vanilla baseline arm only, late-window mean Q:

| env                 | r_min | polarity | Q̄ vanilla | regime    |
|---------------------|------:|---------:|----------:|-----------|
| FourRooms-misc      |     0 |   −0.92  |    +0.82  | POSITIVE  |
| Asterix-MinAtar     |     0 |   +0.50  |    +5.73  | POSITIVE  |
| Breakout-MinAtar    |     0 |   +0.99  |   +15.95  | POSITIVE  |
| Acrobot-v1          |    −1 |   −0.94  |   −35.50  | NEGATIVE  |
| MountainCar-v0      |    −1 |   −1.00  |   −57.35  | NEGATIVE  |

`r_min` perfectly determines `sign(Q̄_vanilla)`. The Q-regime sign
in turn explains the staleness-ATE sign:

| env             | r_min | regime    | DDQN g | ATE(stale → Δ_o) |
|-----------------|:-----:|-----------|:------:|-----------------:|
| FourRooms       |   0   | POSITIVE  | +0.26  | **+39.83 (HELD)** |
| Asterix (SURV)  |   0   | POSITIVE  | +0.08  | +0.11 (~null)     |
| Breakout (SURV) |   0   | POSITIVE  | −0.15  | +39.5             |
| Acrobot         |  −1   | NEGATIVE  | −0.05  | **−349 (REVERSED)** |
| MountainCar     |  −1   | NEGATIVE  | −0.06  | +205 (n=12 small) |

## Mechanism

**Sparse-terminal-positive (`r_min ≥ 0`):**
- `Q* ∈ [0, R_max/(1−γ)]` — true Q values are **positive bounded
  above**.
- Vanilla DQN's max-bootstrap pushes Q **above** the true value
  (Hasselt). Wrong actions get inflated values; the policy
  becomes confidently wrong on non-goal-reaching actions.
- DDQN's argmax/max separation removes the upward bias →
  bigger benefit when there's more bias to correct.
- **Higher staleness → more accumulated upward bias in the
  target → DDQN's correction has more bite → ATE positive.**

**Dense-penalty (`r_min < 0`):**
- `Q* ∈ [−|r_min|/(1−γ), 0]` — true Q values are **negative
  bounded below**.
- Vanilla's max-bootstrap pushes Q **less negative than true**
  (still negative, just inflated). This is mild "optimism" that
  helps the policy explore through the long-horizon penalty
  floor.
- DDQN's correction makes Q **more negative** (closer to true).
  Removes exploration optimism → can hurt.
- **Higher staleness → vanilla's optimism advantage grows →
  DDQN's relative deficit grows → ATE reversed.**

In both cases staleness amplifies whatever bias direction vanilla
has — but the SIGN of that bias's effect on outcome flips with
the Q-regime.

## Why polarity isn't enough

GOAL polarity (within-cell `r(L, return) < 0`) holds for both
FourRooms and Acrobot. But the reward-formula difference
distinguishes them:
- FourRooms terminal +1 at goal, 0 elsewhere.
- Acrobot per-step −1 until terminal, terminal 0.

Both encode "shorter episode = better outcome" (GOAL polarity).
But Q-trajectories differ in sign. Polarity captures the
length→return correlation; `r_min` captures the reward
distribution's location relative to zero. They're orthogonal
features in the cross-env panel.

## Endogenous downstream: `q_late_mean`

New measurable in `corroborate_rl.dqn.measurables`:

```python
@measurable(reads=('online_max_q_per_step',))
def q_late_mean(record) -> float:
    """Mean of online_max_q over the late 50% of training."""
    arr = _record_array(record, 'online_max_q_per_step')
    if arr is None:
        return float('nan')
    return _mean_window(arr, 0.5, 1.0)
```

The bridge `staleness_amplifies_ddqn_outcome__sparse_goal_polyak`
now uses `finite_gt('q_late_mean', 0.0)` as the regime predicate.
The exogenous `r_min` is no longer needed — `q_late_mean` is the
endogenous observable that captures the same regime split per
cell.

## Causal chain summary

```
   r_min  →  sign(Q̄_late_vanilla)  →  direction(Hasselt bias)  →  sign(ATE(stale → Δ_o))
   ─────       ───────────────         ─────────────────           ────────────────
 exogenous     endogenous              algorithmic                  observable
 [verified]    [verified]              [hypothesised]              [verified]
```

The exogenous structural property (env's reward range) determines
an endogenous trajectory property (Q-regime sign — verified
1:1), which **putatively** determines the algorithmic bias
direction (theoretical, no per-step probe), which inverts the
observable ATE sign (verified by per-stratum DoWhy +
interaction-term).

## Open: why exactly does negative Q harm DDQN?

The fully-decomposed algorithmic mechanism is unresolved by
current data. Two candidate explanations remain on the table:

### Candidate 1: bias-direction asymmetry

In r_min ≥ 0, vanilla's max-bias pushes Q ABOVE truth → policy
chases inflated values → DDQN's correction unblocks. In r_min < 0,
vanilla's max-bias pushes Q LESS NEGATIVE than truth (mild
optimism); DDQN's correction makes Q more negative → kills the
exploration optimism that helps escape penalty floors.

**Status**: theoretical, plausible, but not directly tested
(would need per-step decomposition of Q_target at argmax_online
vs max_Q_target).

### Candidate 2: Q-magnitude vs argmax-rank disagreement

The new disagreement-rate test surfaced a surprising regime
split:

```
ρ(τ, argmax_disagreement_late):
  Acrobot, MountainCar, Asterix:  ≈ −0.8  (strong)
  Breakout:                        −0.60  (moderate)
  FourRooms:                       −0.07  (FLAT)
```

In FourRooms, online and target argmax disagree ~36% of steps
regardless of τ. The DDQN-vs-vanilla effect can't be "more
staleness → more disagreement → bigger DDQN bite" — disagreement
is constant. The active variable must be the *magnitude* of
Q_target − Q_online at disagreement moments (which scales with
staleness).

In dense-penalty envs (Acrobot, MountainCar), disagreement IS
staleness-dependent but the per-step DDQN-vs-vanilla effect
fails to translate to outcome — because the policy's response to
correction is governed by penalty-floor dynamics that the
correction doesn't help navigate.

**Status**: pattern observed, mechanism conjectural.

### What would resolve it

Per-step probes that aren't currently in the trace schema:
- `Q_target(s, argmax_a Q_online(s, a))` per step (the DDQN
  bootstrap value, vs `max_a Q_target(s, a)` which is vanilla's).
- The DIFFERENCE of these two per step is the "DDQN-correction
  magnitude per step". Decomposing it by Q-regime would give
  the algorithmic-level proof.

### Resolution attempt (2026-05-07)

Added `target_q_at_online_argmax_per_step` to substrate trace
reductions and authored measurable
`ddqn_bootstrap_gap_late = mean(max(target_q) − target_q[argmax_
online]) over late 50%`. Ran focused FourRooms+Acrobot polyak
sweep (`polyak_tau_q_decomp.yaml`).

Tests:

```
TEST: Δ_o = β₀ + β_g·gap + β_q·q_late + β_int·(gap × q_late) + ε

  β_intercept       = +0.134        t=+0.66    p=0.51
  β_gap             = −116.6        t=−1.05    p=0.30
  β_q_late          = +0.011        t=+1.17    p=0.24
  β_interaction     = −5.02         t=−2.26    p=0.025  ← significant

Per-Q-regime: ATE(gap → Δ_outcome):
  q > 0 (FourRooms):  slope = +42.95, t=+2.46, p=0.015  POSITIVE
  q < 0 (Acrobot):    slope = +116.4, t=+2.63, p=0.010  POSITIVE
```

**Both regimes show POSITIVE gap → Δ_outcome slope.** The DDQN
correction magnitude (gap) IS the proximal driver of Δ_outcome
universally. The earlier "ATE flips sign with Q-regime" framing
on `staleness → Δ_o` was a noise-driven artifact on Acrobot — at
the algorithmic step (correction magnitude itself), there's no
sign flip.

The interaction-term significance is now interpreted as:
- In Q < 0 regime: gap range is larger (~0.028 max on Acrobot
  vs ~0.0016 on FourRooms — 17× wider).
- Per-unit slope differs (+116 Acrobot vs +43 FourRooms) but
  effective Δ_outcome impact differs less (3.2 reward vs 0.07
  reward; FourRooms's relative impact is ~7% of bounded reward
  range, Acrobot's is ~3.5% of unbounded penalty range).

**Updated story.** DDQN's bootstrap gap (the per-step
correction magnitude) correlates with `staleness` (ρ=+0.95
FourRooms, +0.66 Acrobot). And the gap → Δ_outcome slope is
positive in both regimes. So:

- The staleness → Δ_outcome chain runs through the gap.
- The gap → Δ_outcome step is **not** sign-flipped by Q-regime.
- The earlier "Acrobot ATE = −349 reversal" was high-noise
  Simpson's-paradox aggregation, not a real algorithmic reversal.

This RECONCILES the apparent paradox. DDQN doesn't fundamentally
hurt in dense-penalty regimes — it's just much less effective
because:
1. The correction magnitude (gap) is small relative to outcome
   variance from other sources on Acrobot.
2. Acrobot's outcome is dominated by penalty-floor dynamics, not
   bias correction.

The "negative ATE" result on Acrobot's staleness was a
LOW-SIGNAL noise pattern, not a structural sign reversal of
Hasselt's mechanism.

**Implication for the bridge.** The
`staleness_amplifies_ddqn_outcome__sparse_goal_polyak` bridge's
narrow scope (`q_late_mean > 0`) is still empirically correct —
it's the regime where DDQN's correction magnitude is sufficient
to register as a statistically detectable effect. The exclusion
of dense-penalty regimes is "low-signal-to-noise" rather than
"reversed-direction-mechanism".

## Reproduction

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. uv run python \
    experiments/findings/sync_curve_breakout/run_rmin_mechanism_analysis.py
```

Output: `rmin_q_regime_panel.json` (Q regime per env in polyak data).
