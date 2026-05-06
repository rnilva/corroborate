# Target staleness as upstream cause: collinear with log_sync, not separately identifiable

**Date:** 2026-05-05
**Tested:** target staleness (||online_max_q − target_max_q|| per step) as
upstream mediator of sync→harm. Two operationalizations: absolute and
relative-to-Q-magnitude. Same per-pair panel (4 sync × 30 seeds).

## Headline

Target staleness **is** the algorithmic mechanism by which sync controls
DDQN behavior, BUT empirically it is **not separately identifiable from
log(sync_period)** with a fixed-sync sweep:

- ρ(log_sync, mean_staleness_rel_late) = **+0.946, p < 1e-118** (n=240 cells)
- ρ(log_sync, mean_staleness_rel_b0) = **+0.968, p < 1e-144**

These are essentially deterministic functions of each other in this design.
Target staleness adds no information beyond log_sync, so we cannot
empirically distinguish "sync hurts via staleness" from "sync hurts via
some other sync-determined variable" without an intervention that varies
staleness independently of sync_period.

## What target staleness is, measured

For each cell × burst (50k steps), the per-step gap between max-Q under
online vs target networks. Two normalizations:

- **Absolute**: ``|online_max_q − target_max_q|`` — per-step raw gap
- **Relative**: ``|gap| / max(|online|, |target|, 1e-6)`` — fraction-of-Q-magnitude

The absolute gap behaves perversely with sync because absolute Q magnitude
explodes at low sync (Q-explosion):

| sync | abs gap @ b0 | abs gap @ b19 | rel gap @ b0 | rel gap @ b19 |
|---|---|---|---|---|
| 100 | 0.06 | **659** | 0.019 | 0.002 |
| 1000 | 0.07 | 151 | 0.083 | 0.007 |
| 3000 | 0.07 | 17 | 0.191 | 0.017 |
| 10000 | 0.05 | 0.79 | **0.416** | **0.027** |

Absolute gap goes the WRONG way (drops with sync) because at sync=100
both networks have huge Q values (~250k late), so absolute gaps grow
proportionally. The relative gap is the right operationalization — it
**increases monotonically by 22× across the sync range** at burst 0
(0.019 → 0.416).

## Within-seed mediation tests (well-powered)

Stratified partial Spearman, n=120 pooled across 4 sync strata:

| test | ρ_pooled | p |
|---|---|---|
| ρ(mean_staleness_rel_late, Δ_mc_late) marginal | −0.017 | 0.86 |
| ρ(mean_staleness_rel_late, Δ_mc_late \| Δ_q_b19) | −0.009 | 0.92 |
| **ρ(Δ_q_b19, Δ_mc_late \| mean_staleness_rel_late)** | **−0.368** | **8.3e-5** |
| ρ(log_sync, Δ_mc_late \| mean_staleness_rel_late) | +0.018 | 0.85 |
| ρ(log_sync, Δ_mc_late \| Δ_q_b19, mean_staleness_rel_late) | +0.028 | 0.76 |

The within-seed reading is unchanged: **Δ_q_b19 is the only mediator that
survives**. After controlling for staleness, late-Q gap STILL predicts
outcome with ρ=−0.368, p<0.001. After controlling for late-Q gap,
staleness predicts nothing (ρ=−0.009).

Staleness is a sync-level constant — across all seeds at a given sync,
the mean staleness is nearly identical (it depends on hyperparameters,
not seed-specific dynamics). So Δ_staleness across arms is near zero
within a stratum, and within-seed it has no predictive power.

## Cross-sync (meta-regression on 4 strata, severely underpowered)

| covariate | coefficient | 95% CI | p |
|---|---|---|---|
| staleness_rel_b0 | −1.33 | [−5.12, +2.46] | 0.27 |
| staleness_rel_late | −17.5 | [−70.3, +35.3] | 0.29 |
| staleness_rel_mean | −8.69 | [−29.1, +11.7] | 0.21 |
| q_b0_ratio + staleness_late: q_b0_ratio | +0.94 | [−6.9, +8.8] | 0.37 |
| q_b0_ratio + staleness_late: staleness_late | +8.14 | [−237, +254] | 0.75 |

All NS. With only 4 strata and three highly collinear covariates
(log_sync, q_b0_ratio, staleness all ρ > 0.9 with each other),
meta-regression gives no useful disambiguation. The CIs blow up wildly
when two correlated covariates are added jointly.

## What this implies

1. **Algorithmically:** target staleness IS what causes sync_period to
   matter. DDQN's bootstrap = `Q_target(s', argmax_online Q_online)`
   — when online and target diverge (high staleness), this diverges from
   vanilla's `max_a Q_target(s', a)`. The per-cell relative staleness
   grows monotonically from 1.9% (sync=100) to 41.6% (sync=10000) at
   burst 0 — that's the lever sync_period actually pulls on.

2. **Empirically (with a fixed-sync sweep):** target staleness and
   log_sync are nearly perfectly collinear (ρ=+0.97). They are the same
   upstream variable in this experimental design. We cannot empirically
   separate "sync matters" from "staleness matters" — there's only one
   degree of freedom across the 4 syncs.

3. **Within-seed:** the only mediator that adds independent variance
   over the regime variable is Δ_q_b19 (late-Q amplification). This
   captures the seed-specific cumulative divergence in Q values that
   the per-cell staleness measure averages out across seeds.

## To separate staleness from sync_period

The required intervention: **vary target update rule WITHOUT changing
sync_period**. Concrete designs:

- **Polyak averaging at varying τ** with sync_period fixed at 1000
  (e.g., target ← τ·online + (1−τ)·target every step). Different τ
  values produce different effective staleness without changing sync.
- **Random target perturbation**: every sync, sync target ← online +
  N(0, σ²·I). σ controls residual staleness independent of sync rate.
- **Asymmetric target lag**: vary how many steps target lags online
  on certain layers but not others, with sync_period fixed.

These would isolate target staleness as a separately identifiable cause.
A small Breakout-only sweep with 4 staleness levels at a single
sync_period (~3 hours GPU) would resolve this.

## Honest summary

The target-staleness hypothesis is **mechanistically correct** (the
algorithm clearly diverges via this path) but **empirically
indistinguishable from sync_period itself** in the current sweep.
The within-seed mediator that survives strict conditioning is still
Δ_q_b19 (Q-trajectory amplification), as in the previous statistical
proof. Adding staleness as a candidate didn't change that conclusion —
it just confirmed the cross-sync story is a single-degree-of-freedom
collinear system that 4-stratum meta-regression can't crack.

## Reproduction

```bash
PYTHONPATH=. uv run python experiments/findings/sync_curve_breakout/run_target_staleness_analysis.py
```

Output: `staleness_panel.json`.
