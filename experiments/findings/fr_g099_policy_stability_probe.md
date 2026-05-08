# FourRooms γ=0.99 — policy-stability mediator probe

## Question

Within FR γ=0.99 (where the textbook DDQN story fires: g_link=+1.11
per CLAIM 5), find a non-polarity-locked mediator carrying the link.

## Setup

- Two new measurables authored:
  - `argmax_persistence_late`: fraction of late-window step-pairs
    where `online_argmax_per_step[t] == online_argmax_per_step[t-1]`.
    Reduces over the *temporal* axis — distinct from existing
    `argmax_mode_freq_late` (categorical-action axis).
  - `q_max_temporal_cv_late`: temporal CV of `online_max_q_per_step`
    over the late 50%. Reduces over training time — distinct from
    `q_action_std_late` (cross-action at each step).
- Restored gamma_sweep tmp/arm000+arm003 (FR γ=0.99 vanilla + DDQN).
- n=30 paired seeds, mech-HELD on Δ_jens<0 (30/30 pairs qualify).

## Single-mediator linear proportion

| mediator | proportion | slope | r(Δ_Y, Δ_M) |
|---|---|---|---|
| `effective_horizon` (polarity-locked) | 0.762 | -0.0055 | -0.551 |
| `q_max_temporal_cv_late` | 0.221 | -6.13 | -0.536 |
| `argmax_persistence_late` | 0.156 | -0.61 | -0.264 |

## Joint OLS

`Δ_Y ~ Δ_eff_h + Δ_persist + Δ_CV + intercept`

- β_intercept = +0.022
- β_eff_h     = -0.004 (per-pair contribution +0.075, ≈82% of mean Δ_Y)
- β_persist   = +0.246 (per-pair contribution -0.006, slightly negative)
- β_CV        = -3.015 (per-pair contribution +0.010, ≈11% positive)
- R² = 0.34

## Surprises

1. **DDQN has LESS argmax persistence on FR γ=0.99**: Δ=-0.023, p=0.001
   paired-t. My hypothesis (DDQN stabilizes argmax → directed trajectory)
   is REFUTED. Reading: vanilla's bias FREEZES the argmax onto the
   over-inflated action; DDQN's correction lets online's argmax move
   freely batch-to-batch. On FR's small state space, this is
   mechanism-irrelevant or slightly outcome-negative (β_persist > 0
   in joint regression — more persistence → more outcome).

2. **q_max_temporal_cv_late points the predicted way but small**:
   DDQN reduces it (Δ=-0.003, p=0.02), joint β negative — small
   positive contribution to outcome (~10%).

3. **Joint R²=0.34, n=30**. With three mediators, two-thirds of variance
   in Δ_outcome remains unexplained. The "FR is fully mediated by
   eff_h" reading from the broader corpus panel (proportion=1.03,
   CI [0.96, 1.10] at n=535) does NOT survive once non-polarity-locked
   mediators are added — eff_h's solo proportion drops to 0.76
   on this slice and the joint regression doesn't have a clean
   single-channel story.

## What survives the "be careful" filter

- The new measurables compute correctly end-to-end from the per-arm
  trace parquets (workflow validated).
- `argmax_persistence_late` is a real signal but **opposite-signed**
  from the naive bias-correction → policy-stability hypothesis.
  Worth keeping as a measurable, but **NOT** as the channel.
- `q_max_temporal_cv_late` is suggestive (right sign, small effect)
  but n=30 isn't enough to corroborate.
- The polarity-tautology framing for `effective_horizon` stands —
  proportion≈1.0 in the broader pool is the polarity lock, not
  substantive mechanism content. Joint regression with proper
  mediators dilutes its dominance.

## What's still open

1. **What's between bias-correction and length-reduction on FR?**
   The polarity tautology fixes the *shape* (Δ_outcome co-moves with
   Δ_eff_h by polarity); the *magnitude* of Δ_eff_h is what DDQN
   actually delivers. We don't yet have a measurable that captures
   the upstream of *Δ_eff_h itself*.
2. **Per-state argmax persistence** would be a sharper proxy than
   batch-mean argmax persistence — but requires state_hash
   stratification + per-state Q tracking. Not in current trace
   schema.
3. **Counterfactual mediation primitive (Pearl NDE/NIE)**:
   FUTURE_WORKS §3a lift gate has fired empirically on the
   per-env panel (4/11 envs ASSUMP_FAIL). Would let us test
   treatment×mediator interaction directly and properly decompose
   when multiple correlated mediators are at play.

## Reproduction

- Measurables: `corroborate_rl/dqn/measurables.py:argmax_persistence_late`,
  `q_max_temporal_cv_late`.
- Trace files restored: `gamma_sweep/tmp/arm000__FourRooms-misc__vanilla_dqn_g099__traces.parquet`,
  `gamma_sweep/tmp/arm003__FourRooms-misc__ddqn_g099__traces.parquet` (~680MB).
- Cloud restore via `corroborate.corpus.cloud.restore` + `.env` AWS creds.
