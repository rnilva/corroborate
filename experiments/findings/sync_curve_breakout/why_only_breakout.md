# Why does the calibration mediator HELD only on Breakout sync=100?

**Date:** 2026-05-05
**Followup to:** `findings_qmc_calibration_mediator.md` revised (bridge framework
verdict: 1/12 cohorts HELD; Breakout sync=100 is the lone positive).

## The slope decomposition

`slope_y_on_m = ρ × σΔ_out / σΔ_cal`. Computing each factor per env-regime
on the cached `calibration_bridges.parquet`:

| scope | n | ρ(Δcal,Δout) | σΔ_cal | σΔ_out | slope | σout/seed (vanilla) | mean jensen_gap (vanilla) |
|---|---|---|---|---|---|---|---|
| Acrobot ADS | 60 | +0.07 | 0.32 | 2.01 | +0.46 | 1.44 | 8 |
| Asterix sync100 | 30 | +0.07 | 0.12 | 0.19 | +0.11 | 0.15 | **3,964,256** |
| **Breakout sync100** | 30 | **+0.37** | 0.33 | 1.85 | **+2.08** | 1.29 | 94,759 |
| Freeway sync100 | 30 | +0.15 | 0.15 | 0.26 | +0.25 | 0.19 | 381 |
| Asterix sync10k | 30 | −0.19 | 0.12 | 0.28 | −0.44 | 0.31 | (low Q) |
| Breakout sync10k | 30 | −0.01 | 0.28 | 0.77 | −0.01 | 0.51 | (low Q) |
| Freeway sync10k | 30 | −0.10 | 0.20 | 0.30 | −0.15 | 0.21 | (low Q) |
| SpaceInvaders sync10k | 30 | −0.23 | 0.40 | 0.82 | −0.47 | 0.68 | (low Q) |

## Three conditions, all required

The Breakout-sync=100 HELD is at the intersection of three conditions, each of
which fails for at least one other env:

### Condition 1: Mechanism engagement (Q-explosion regime active)

`jensen_gap` mean must be substantial — vanilla's Q must be diverging from
MC enough that DDQN's bootstrap-rule difference has variation to bite on.

- ✓ Asterix sync100, Breakout sync100, Freeway sync100 (Q-explosion)
- ✗ All sync=10k MinAtar (Q stable, mediator chain flips sign — see
  `findings_target_staleness_collinear.md`)
- ✗ Acrobot ADS (Q ~ 8, no explosion)

### Condition 2: Sufficient outcome variance (σΔ_out)

DDQN's effect on per-seed outcomes must be detectable above sampling noise.

- ✓ Breakout sync100 (σΔ_out = 1.85), Acrobot ADS (σΔ_out = 2.01)
- ✗ Asterix sync100 (σΔ_out = 0.19) — outcomes are tightly converged
  across seeds; no room for the mediator to express
- ✗ Freeway sync100 (σΔ_out = 0.26) — same, episode-length-bounded MC

### Condition 3: Per-seed Q→outcome coupling (ρ)

Even with conditions 1 & 2 satisfied, the per-seed correlation between
calibration and outcome must be strong enough.

- ✓ Breakout sync100 (ρ = +0.37) — uniquely high
- ✗ Acrobot ADS (ρ = +0.07) — mediator chain weak per seed despite high σ
- ✗ Asterix / Freeway sync100 (ρ = +0.07, +0.15) — weak

## What singles out Breakout sync=100 on Condition 3?

ρ(Δ_calibration, Δ_outcome) = 0.37 is 3-5× higher on Breakout than on
neighbouring envs. Hypothesis: **Breakout's reward is directly determined
by Q-greedy policy quality at every step**.

In Breakout-MinAtar the paddle position determines whether the ball is caught.
A well-calibrated Q ranks "move paddle toward ball" above other actions →
ball caught → brick broken → reward; mis-calibrated Q → wrong paddle
position → ball lost. The mapping from Q-quality to MC is essentially
**1-step deterministic**: every step, Q-quality decides reward.

Other envs decouple Q-quality from outcome:
- **Acrobot**: success requires a sequence of swings; Q-quality matters but
  early-trajectory chance dominates. Mapping is multi-step noisy.
- **Asterix**: dodge + collect; multiple successful strategies; Q-quality at
  any single state doesn't strictly determine survival.
- **Freeway**: walk forward; few decisions per step. Q-quality is over-
  determined (most actions are equivalent).

So Breakout's reward structure makes ρ(Q-quality, MC-return) per seed
naturally tighter than the other envs.

## What singles out Breakout sync=100 on Condition 1 vs Asterix?

Asterix sync=100 has BIGGER Q-explosion (mean jensen_gap = 4M) than Breakout
(95k). But Asterix's σout/seed is only 0.15 — outcomes converge across seeds
despite Q-divergence. Why? In Asterix the policy distribution is dominated by
"move to gold + away from enemy" decisions that aren't action-margin-
sensitive — even when Q values diverge to 4M, the argmax remains "the right
direction" for most states. Q-divergence doesn't translate to policy
divergence, hence not to outcome divergence.

In Breakout, the argmax is sensitive to small Q differences (paddle position
needs precision). Q-divergence DOES translate to outcome divergence. Hence
Breakout has both the Q-explosion regime AND the policy-fragility that
makes the mediator detectable.

## What singles out sync=100 vs sync=10k for Breakout?

Same env, different sync. At sync=10k:
- Q is bounded (no explosion)
- ρ(Δ_calibration, Δ_outcome) drops to 0.00 (essentially zero)
- σΔ_out drops from 1.85 to 0.77 (less outcome variance)
- mean Δ_outcome ≈ −0.27 (DDQN HURTS slightly)
- mean Δ_calibration ≈ +0.03 (DDQN's calibration slightly better)
- → opposite-sign means → linear-mediation degenerate (slope ≈ 0)

Two effects compose:
1. **Smaller mechanism amplitude** at sync=10k (no Q-explosion to bite on)
2. **Sign-flip in the mean direction** (silent inversion) — DDQN's calibration
   improves on average but outcome harm — so per-seed link, even if real, has
   inverted direction relative to Hasselt's prediction

The combined effect: Breakout sync=10k has neither the magnitude nor the
direction to surface as HELD on the bridge's positive-slope-test.

## Implication for the bridge family

The Breakout-sync=100 HELD is **mechanistically narrow but principled**:
- The bridge's slope-floor of 0.5 cleanly separates this env-regime from
  the other 11
- The 3-condition conjunction pins the scope: (Q-explosion) ∧ (policy-
  fragile env) ∧ (high per-seed Q→outcome coupling)

This is closer to the framework's intended "scope as causal bridge"
discipline than my original "calibration is the strongest mediator
universally" framing. The bridge HOLDs in a narrow but principled regime;
elsewhere the prerequisites fail and the mediator chain doesn't ignite.

## Reproduction

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. uv run python -c "..."  # the inline script in
                                                       # the chat history
```

Cache: `experiments/data/cache/calibration_bridges.parquet`.

## Followup questions

1. Does the 3-condition conjunction predict HELD on yet-unseen env-regimes
   that satisfy all three? (No new data; would need a synthesizable example.)
2. Is the "policy-fragility" condition expressible as a measurable? Maybe
   `paired_g(σ(eval_best_burst_mean) per arm, treatment vs baseline)` or a
   per-cell measure of how much policy quality varies under fixed Q.
3. Can the silent-inversion sign-flip at sync=10k be made explicit via a
   directional plc primitive (cf. `findings_sync_curve_goldilocks.md`)? A
   sign-aware mediator HELDs when EITHER positive slope (Hasselt direction)
   OR negative slope above |threshold| (inverted direction) — the inverted
   case is also informative even if not predicted.
