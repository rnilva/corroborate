# Q-MC calibration: stronger within-seed mediator than Q-amplification

> ## ⚠️ REVISED 2026-05-05 (later same day)
>
> The headline "stronger mediator than Q-amplification" was based on
> **Fisher-z-pooled partial Spearman ρ=+0.71 across 4 sync strata**.
> 14 framework bridges (`experiments/findings/calibration_bridges.py`)
> evaluated the claim per-stratum across 12 env-regime cohorts.
>
> **Verdict:** Only **Breakout sync=100 HELD** (slope_y_on_m = +2.08,
> p = 0.041, n=30). All other cohorts NO_EFFECT or degenerate:
> - CartPole / DiscountingChain at action_dim_sweep regime: outcome
>   `eval_best_burst_mean` saturates at env ceiling (std=0); mediation
>   analysis is mathematically degenerate (NaN slope).
> - Other MinAtar envs at sync=100: weak positive slopes (+0.1 to +0.25,
>   all p > 0.4).
> - All MinAtar envs at sync=10000: NEGATIVE slopes (the "silent
>   inversion" signature — calibration improvement correlates with
>   outcome harm).
> - Breakout sync=10k specifically: slope = −0.013, p = 0.98 — within-
>   stratum the link is essentially zero, contradicting the pooled +0.71.
>
> **Methodological lesson:** Fisher-z pooling captured CROSS-stratum
> variation (different syncs have different mean Δ_calibration AND
> different mean Δ_outcome — the variation co-moves across syncs).
> That's NOT the same as "calibration mediates outcome within a sync."
> The bridge framework's per-scope verdict mechanically caught the
> over-extrapolation.
>
> Memory `findings_qmc_calibration_mediator.md` carries the corrected
> framing. The remainder of this document is the original analysis as
> written; treat it as a record of what the unconditional Spearman
> said before the per-stratum bridge tests.

**Date:** 2026-05-05
**Promoted to canonical:** `q_mc_calibration_pearson` measurable now
authored in `corroborate_rl.dqn.measurables`. Tests run on the same
per-pair panel (4 sync × 30 seeds = 120 paired cells, Breakout-MinAtar).

## What it measures

Per-cell scalar Pearson r between `predicted_q_at_start` and `mc_return`
over all 100 (burst × eval-episode) points. Measures Q's predictive
validity for the agent's own policy:
- r → 1: Q correctly ranks initial states by realized return
- r → 0: Q is uninformative as a value predictor
- r < 0: Q is inversely calibrated

## Per-arm per-sync calibration mean

| sync | arm | r_overall | r_early (b0-4) | r_late (b15-19) | r_burst_mean |
|---|---|---|---|---|---|
| 100 | vanilla | +0.003 | −0.187 | nan | +0.006 |
| 100 | ddqn | −0.045 | −0.199 | −0.115 | −0.077 |
| 1000 | vanilla | −0.040 | −0.319 | +0.156 | −0.031 |
| 1000 | ddqn | −0.001 | −0.198 | +0.042 | +0.008 |
| 3000 | vanilla | −0.018 | +0.066 | −0.009 | −0.026 |
| 3000 | ddqn | +0.062 | +0.139 | +0.033 | +0.064 |
| 10000 | vanilla | **+0.322** | +0.430 | +0.098 | +0.387 |
| 10000 | ddqn | **+0.351** | +0.420 | −0.010 | +0.402 |

Two mean-level observations:
1. **Calibration emerges only at sync=10000**. At sync ≤ 3000, r ≈ 0 —
   Q is essentially uncorrelated with return because Q is dominated by
   Q-explosion magnitudes, not return-relevant content.
2. **At sync=10000, both arms have r ≈ 0.34**. The mean DDQN-vs-vanilla
   calibration difference is small (Δ = +0.029, NS).

So per-sync paired_g on calibration is mostly null. The interesting
finding is at the within-seed Δ level.

## Per-sync paired_g on calibration measures

| sync | g(r_overall) | g(r_early) | g(r_late) | g(r_burst_mean) |
|---|---|---|---|---|
| 100 | −0.14 (p=0.45) | −0.02 (p=0.91) | **−0.41 (p=0.038)** | −0.19 (p=0.29) |
| 1000 | +0.11 (p=0.56) | +0.21 (p=0.26) | −0.24 (p=0.19) | +0.09 (p=0.64) |
| 3000 | +0.20 (p=0.28) | +0.13 (p=0.49) | +0.08 (p=0.67) | +0.17 (p=0.35) |
| 10000 | +0.10 (p=0.59) | −0.03 (p=0.86) | −0.18 (p=0.32) | +0.04 (p=0.81) |

Only sync=100 r_late is significant (DDQN slightly worse late calibration
when Q is exploding).

## The load-bearing finding: stratified partial Spearman

n=120 pooled across 4 sync strata.

**Marginal pooled** ρ(Δ_pearson_q_mc_overall, Δ_mc_late) = **+0.714, p ≈ 0**.

This is the strongest within-seed mediator we've found.

**Conditional partial** ρ(Δ_pearson_q_mc_<window>, Δ_mc_late \| Δ_q_b19):

| calibration window | ρ_partial | p | n |
|---|---|---|---|
| **overall** (100 pts) | **+0.701** | **≈ 0** | 120 |
| early (5 bursts × 5 eps) | +0.277 | 3.8e-3 | 120 |
| late (5 bursts × 5 eps) | −0.229 | 1.9e-2 | 118 |
| burst_mean (20 means) | +0.682 | ≈ 0 | 120 |

**Calibration adds independent variance over Q-amplification.** The
overall and burst_mean Pearsons preserve most of their predictive power
after controlling for late-Q gap.

(The late-window flips sign — small-n artifact of conditioning on Δ_q_b19
which is itself derived from late bursts; multi-collinearity.)

**Reverse partial** ρ(Δ_q_b19, Δ_mc_late \| Δ_pearson_q_mc_<window>):

| Z (calibration window) | ρ_partial | p |
|---|---|---|
| Δ_pearson_q_mc_overall | **−0.247** | 0.010 |
| Δ_pearson_q_mc_early | −0.278 | 3.6e-3 |
| Δ_pearson_q_mc_late | −0.331 | 5.2e-4 |
| Δ_pearson_q_mc_burst_mean | −0.244 | 1.1e-2 |

**Q-amplification's coefficient attenuates from −0.343 to −0.247** when
calibration is controlled. Still significant, but weakened. The two
mediators share variance — calibration is the stronger of the pair.

## Combined controls absorb the cross-sync trend

Multi-Z partial Spearman:
ρ(log_sync, Δ_mc_late \| Δ_q_b19, Δ_pearson_q_mc_late) = **−0.09, p = 0.34** (ns).

After conditioning on both mediators, log_sync has no residual effect.
Calibration + Q-amplification together explain the cross-sync outcome
trend.

## Synthesis: revised mediator hierarchy

Strongest → weakest within-seed mediator of Δ_outcome (sync-stratified):

1. **Δ q_mc_calibration_pearson_overall** (NEW): ρ = +0.701 \| Δ_q_b19, p ≈ 0
2. Δ_q_b19 (Q-amplification): ρ = −0.247 \| calibration, p = 0.010
3. Δ q_b0 (early Q-suppression): ρ = +0.123 \| Δ_q_b19, p = 0.21 (NS)
4. Δ_target_staleness: collinear with log_sync, no residual mediation

The Q-amplification finding from `findings_q_amplification_cartpole.md`
generalizes from CartPole to Breakout, but the **mechanism is more
sharply expressed as a calibration-validity problem** than as a
Q-magnitude divergence problem. The two are correlated (DDQN's
Q-magnitude divergence accompanies its calibration variation per seed)
but calibration is the more direct signal.

Why it's the better mediator: Q-magnitude (Δ_q_b19) varies by orders of
magnitude across syncs (mostly tracking the Q-explosion regime), so
within-stratum variance is small relative to absolute scale.
Calibration r is bounded in [−1, 1] — within-stratum variance is
comparable across syncs, so per-seed paired Δ is more interpretable.

## Bridge implication

The deferred `ddqn_link_dies_after_q_divergence__minatar` should consume
**`q_mc_calibration_pearson`** as the primary mediator (alongside
Δ_q_b19 as confirming evidence). Bridge schema:
- source = treatment vs vanilla DDQN
- target = paired_g on outcome
- mediator = q_mc_calibration_pearson
- holds_when: significant negative `proportion_mediated` of outcome
  through calibration when calibration drops

## Authored measurables

Added to `src/corroborate_rl/corroborate_rl/dqn/measurables.py`:
- `q_mc_calibration_pearson` — per-cell Pearson(Q_at_start, MC) over
  all (burst, episode) points
- `target_staleness_late` — relative gap of online vs target Q over
  late 50% of training
- `target_staleness_early` — same for early 25%

All three are now first-class framework measurables; bridges can declare
them in `Bridge.source` / `Bridge.mediator` slots.

## Reproduction

```bash
PYTHONPATH=. uv run python experiments/findings/sync_curve_breakout/run_qmc_calibration_analysis.py
```

Output: `qmc_calibration_panel.json`.
