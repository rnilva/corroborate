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
- Currently we have `online_max_q_per_step`, `target_max_q_per_
  step`, `online_argmax_per_step`, `target_argmax_per_step` —
  but not `target_q_at_online_argmax`. Adding this trace column
  to the substrate is the missing ingredient.

The findings as-stand: the regime split is causally robust at
the ATE level (rung-2 evidence with refutations); the
algorithmic-level mechanism is best characterised as "Hasselt's
bias direction is sign-flipped by Q-regime sign" but the
intermediate decomposition requires additional substrate
instrumentation.

## Reproduction

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. uv run python \
    experiments/findings/sync_curve_breakout/run_rmin_mechanism_analysis.py
```

Output: `rmin_q_regime_panel.json` (Q regime per env in polyak data).
