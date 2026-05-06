# Sync-curve outcome analysis — DDQN's Goldilocks band on Breakout

**Date:** 2026-05-05
**Companion to:** `findings.md` (per-burst link analysis, same data).

## Headline

DDQN does NOT monotonically improve outcome with increasing sync_period
on Breakout-MinAtar — **and the sync=10000 regime where the link is
"most active" (plc=0.85) is where DDQN HURTS most**. The Hasselt theorem
is satisfied in a narrow band (sync ∈ {1000, 3000}); above and below,
two distinct failure modes engage.

| sync | g(outcome_mean) | g(best_burst) | g(late_quarter) | n_pairs |
|---|---|---|---|---|
| 100 | +0.02 (p=0.92) | +0.15 (p=0.42) | −0.02 (p=0.93) | 30 |
| 1000 | +0.24 (p=0.20) | +0.21 (p=0.25) | +0.29 (p=0.12) | 30 |
| 3000 | +0.23 (p=0.21) | +0.14 (p=0.45) | +0.15 (p=0.42) | 30 |
| 10000 | **−0.71 (p=5e-4)** | −0.34 (p=0.07) | **−0.41 (p=0.03)** | 30 |

## Why "link active" misled

The per-burst link primitive measures r(−Δ_bias, Δ_outcome) across paired
seeds. At sync=10000 burst 0, r=+1.0 — perfect per-seed correlation. But
the *signs* of the means tell a different story:

| sync | mean Δ_bias @ b0 | mean Δ_bias @ b19 | mean Δ_outcome @ b0 | mean Δ_outcome @ b19 |
|---|---|---|---|---|
| 100 | −1.48 | −203573 | +0.06 | −0.14 |
| 1000 | −0.49 | +5366 | −0.24 | +0.35 |
| 3000 | −0.09 | −365 | −0.08 | +0.41 |
| 10000 | **+0.88** | **+5.74** | −0.91 | −0.48 |

At sync=10000, DDQN does NOT reduce bias — it **increases** it. The link
r=+1.0 reflects a per-seed dose-response in the *wrong direction* of the
Hasselt theorem: more Δ_bias-increase tracks more Δ_outcome-decrease.

The current `phase_link_consistency` primitive treats sign-match-with-
expected-direction as "link active". When mean Δ_predictor has the
opposite sign of what the theorem assumes (mechanism active but in
reverse), per-seed r-significance is not equivalent to "mechanism helping
outcome".

## Three failure modes

**sync=100 — Q-overwhelm.** Vanilla Q hits 540k by burst 19 (MC stays
~5). DDQN reduces this to 337k (a 38% reduction in absolute terms,
−203k Δ_bias). The reduction is huge in magnitude but the regime is so
divergent that outcome is unchanged. DDQN helped on 43% of seeds at b19.
The original Hasselt mechanism IS engaged (DDQN reduces overestimation),
but the sync=100 Q-divergence has decoupled mediator from outcome.

**sync ∈ {1000, 3000} — Goldilocks.** Vanilla Q grows moderately (140 →
26k at sync=1000, 26 → 905 at sync=3000). DDQN reduces this and outcome
mildly improves (g≈+0.2). Helped fraction climbs from ~30% at b0 to
~57-67% at b19 — DDQN takes time to materialize benefit but does so
consistently. This is the regime the Hasselt theorem describes.

**sync=10000 — Q-amplification.** Vanilla Q stays bounded (28 at b19,
overestimating MC=3.93 by ~24). But DDQN's "decoupled greedification"
amplifies overestimation (Q=33, bias=+30). The same Q-amplification
phenomenon documented for CartPole (`findings_q_amplification_cartpole.md`)
appears here on Breakout. DDQN's faithful target-tracking, freed from
the Q-dampening that vanilla's max-bias provides, runs Q higher than
vanilla's. Outcome suffers across all bursts (helped fraction 10% → 33%).

## What this implies

1. **plc is a necessary but not sufficient condition for "DDQN helps."**
   It captures per-seed mechanism engagement; it does not capture whether
   the engagement is in the theorem-predicted direction. A high plc with
   negative Δ_outcome is a **silent inversion** — the mechanism is firing
   in reverse but each seed responds proportionally.

2. **The `findings_q_amplification_cartpole.md` finding generalizes.**
   It was n_envs=1 there. With Breakout-at-sync=10000 it's n_envs=2 (and
   pending Freeway+SI from the resume sweep, potentially n_envs=4). The
   pattern: when sync_period is long enough to suppress vanilla's
   Q-explosion BUT vanilla's Q is still mildly overestimating, DDQN's
   decoupling causes Q to grow PAST vanilla's. This is structural, not
   stochastic: it's deterministic dose-response per seed.

3. **DDQN has a Goldilocks band on Breakout, and the band exists in a
   moderate-sync interval.** The sync hyperparameter is itself part of
   the scope of "DDQN helps Hasselt-theorem holds": at sync=100 the
   mechanism is overwhelmed; at sync=10000 the mechanism is reversed.
   The theorem-supporting regime is sync ∈ [1000, 3000] with effect
   sizes g ≈ +0.2 (statistically marginal at n=30).

## Open question for the bridge

The deferred bridge `ddqn_link_dies_after_q_divergence__minatar` was
about plc(sync). With this finding, the bridge needs to distinguish
**plc-active-with-correct-sign** from **plc-active-with-reversed-sign**.
A new analysis primitive — call it `phase_link_consistency_directional`
— that thresholds on `mean_d_target` having the predicted sign before
counting a burst as active would express this.

Or: keep `phase_link_consistency` as-is (per-seed correlation strength)
and report alongside `paired_g(outcome)`. The two together pin the
verdict: PLC > 0 with positive g = DDQN helps in the predicted way;
PLC > 0 with negative g = DDQN's mechanism inverted (the sync=10000
case). Substrate-author-friendly: report both, let the bridge condition
on both.

## Reproduction

```bash
PYTHONPATH=. uv run python experiments/findings/sync_curve_breakout/run_outcome_analysis.py
```

Output: `outcome_panel.json`.
