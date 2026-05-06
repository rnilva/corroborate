# Per-burst link analysis on expectile_3way: Strategy 2 result

**Date:** 2026-05-05
**Tested:** the unfired test from `findings_residual_unexplained.md` —
does a different bias-correction mechanism (expectile-greedify) show
the SAME per-burst link structure as DDQN-double-greedify on the same
sparse-reward envs?

## Per-arm per-burst panel (`paired_link_per_burst` × env, n=30 paired seeds)

| env | DDQN plc | EXP plc | DDQN slope range | EXP slope range | reading |
|---|---|---|---|---|---|
| Acrobot-v1 | **1.00** | **1.00** | +0.70 to +1.03 | +0.86 to +1.03 | identical link mechanism |
| Catch-bsuite | 0.00 | 0.00 | nan | nan | outcome saturated (sd_target=0) |
| DCC-bsuite | 0.50 | **0.80** | -0.02 to +0.93 | +0.04 to +0.94 | expectile sustains link longer |
| FourRooms-misc | **1.00** | **1.00** | +0.47 to +1.05 | +0.56 to +1.09 | identical link mechanism |
| MountainCar-v0 | 0.30 | 0.40 | -0.13 to +0.55 | -0.09 to +0.55 | similar weak link |

## What's preserved across DDQN and expectile

1. **plc is COMPARABLE OR HIGHER for expectile on every env** — the per-burst link is at least as active under expectile. Strategy 2 doesn't BREAK the link mechanism anywhere.

2. **Per-burst slopes are remarkably similar** — on FourRooms and Acrobot, both arms have β ≈ 0.85-1.05 across bursts. **The "per unit Δbias-reduction → Δoutcome" coupling is essentially identical.** Two distinct greedification rules produce the same per-burst causal arrow.

3. **The link signal is preserved per-seed** — r ≥ +0.94 at most bursts on FourRooms / Acrobot for both arms.

## What differs

**Mean Δ_predictor (= mean Δbias) magnitudes**, not direction:

- Acrobot bursts 5-9: DDQN Δbias ∈ [-2, +1] (sign-flipping); expectile Δbias ∈ [-30, -25] (huge consistent negative)
- FourRooms bursts: DDQN [-0.65, -0.07]; expectile [-0.92, -0.43] — ~2× more bias reduction
- MountainCar bursts: DDQN [-2.8, +9.9] sign-flipping; expectile [-10, -5.5] consistent

**Expectile = stronger and more consistent bias reducer at every burst** on every env.

**Mean Δ_target (= mean Δoutcome) sometimes differs in sign or magnitude:**

- Acrobot DDQN: positive at 6 of 10 bursts (+0.28 to +2.56); expectile NEGATIVE at every burst (−12.3 to −2.7) — silent inversion at burst level
- FourRooms: both positive every burst (similar magnitude)
- DCC: both positive early; expectile sustains positive longer
- MountainCar: both small/mixed

## Connecting to the scalar residual

The scalar paired_g earlier showed expectile had WORSE outcome (more negative g_link) on 4 of 5 envs while having STRONGER mech (more negative g_mech). The "stronger mech ↛ better link" trade-off was attributed to over-correction.

**Per-burst tells a different story:**
- The per-burst link mechanism is preserved (β similar, r positive every burst)
- Per-burst plc is comparable or HIGHER for expectile
- The scalar discrepancy lives in the MEAN Δ_target sign, not the per-seed coupling

**Acrobot is the cleanest illustration of the discrepancy:**
- Per-burst: r=+0.98, slope ≈ +1.0 every burst (link mechanism intact)
- Per-burst Δ_target: NEGATIVE at every burst for expectile (mean Δoutcome is uniformly worse)
- Scalar `eval_best_burst_mean`: g_link = -0.53 (DDQN harm)

The per-burst pattern says: **for each individual seed, more bias reduction → more outcome change in the same direction.** But the AVERAGE direction of Δ_outcome is wrong (negative). This is the silent-inversion signature at the per-burst panel level.

## Connecting back to FINDINGS.md ninth revision

The FourRooms case (FINDINGS.md ninth revision, lines 421-497) made the same observation: per-burst r negative every burst, scalar mean masks the effect. Same pattern here for expectile: the chain operates per-burst the same as DDQN; the scalar `eval_best_burst_mean` aggregation makes them look different by selecting different "best" bursts for different arms.

## Implication for the residual hunt

**The residual `bootstrap_fraction → g_link | g_mech` at the SCALAR level may be a within-env best-burst-selection-nonalignment artifact**, not a missing causal mediator.

The chain `mech → outcome` is preserved at the per-burst level across two distinct greedification rules. The framework's documented next-frontier candidates (gradient stability, network curvature, exploration coverage from `ddqn_universe_summary.md`) are still untested, but the per-burst evidence here suggests they may not be necessary — the chain may already be complete at the per-burst level on these envs, with scalar discrepancies being aggregation artifacts.

## Open questions left after this test

1. **Is the silent inversion at expectile's Acrobot real or aggregation artifact?** Per-burst Δ_target uniformly negative, but per-seed r positive. The mean is the puzzle.
2. **Catch's degenerate outcome** is a dead end here. Need a non-saturating outcome (mc_late or area-under-learning-curve) to test mediator on Catch.
3. **MountainCar's weak coupling** in both arms (plc 0.3/0.4) suggests the mediator chain is broken on this env regardless of greedification rule. Could be the dormancy regime — MountainCar's vanilla bias is small/sign-flipping.

## Reproduction

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. uv run python \
    experiments/findings/sync_curve_breakout/run_expectile_per_burst.py
```

Output: `expectile_per_burst_panel.json` and the inline table above.
Required cloud-restored data: `experiments/data/expectile_3way/traces.parquet` (5.3 GB).
