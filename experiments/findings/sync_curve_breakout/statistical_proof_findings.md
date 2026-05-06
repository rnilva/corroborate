# Statistical proof: what the framework primitives say

**Date:** 2026-05-05
**Replaces / corrects:** the "bootstrap conservatism is THE mechanism" claim
in `mechanism_findings.md` was based on cross-sync mean-level correlations,
not within-seed mediation. Rigorous tests show a more nuanced picture.

## What was tested

Four framework primitives applied to the per-pair panel (4 sync × 30 paired
seeds = 120 pairs on Breakout-MinAtar):

1. **`paired_g`** per sync (sanity / power baseline)
2. **`proportion_mediated`** per sync × 4 candidate mediators
3. **`stratified_partial_spearman_rho`** for the JCI form of the chain
4. **`meta_regress_panel`** on per-sync g vs stratum-level covariates

## Power baseline

n_pairs=30 per sync, paired-g SE ≈ 0.18:

| power | MDE for paired g |
|---|---|
| 0.50 | 0.51 |
| 0.80 | 0.72 |
| 0.95 | 0.93 |

The observed **g = −0.41 at sync=10000 is BELOW the 80% MDE** — so the
single-sync test is borderline (p=0.032 by p-value, ~50% power against a
true g=0.5 alternative). It replicates the qualitative direction but not
with overwhelming evidence.

The marginal Spearman ρ across all 120 pairs has detectable |ρ| ≈ 0.18 at
α=0.05; observed |ρ_marginal| = 0.128. **Pooled-marginal is also
underpowered.** Within-sync stratified tests have higher power because
stratification removes between-sync variance.

## Headline panel

| sync | g(mc_late) | p | Δ_mc_late | mc_b0 mediation | q_b0 mediation | q_b19 mediation |
|---|---|---|---|---|---|---|
| 100 | −0.015 | 0.93 | −0.03 | n/a (total≈0) | n/a | n/a |
| 1000 | +0.286 | 0.12 | +0.37 | 14% ✓ | over (×2.3) ✗ | n/a |
| 3000 | +0.148 | 0.42 | +0.18 | 1% ✓ | over (×1.1) ✗ | 57% ✓ |
| 10000 | **−0.409** | **0.032** | **−0.33** | **84% ✓** | over (×1.1) ✗ | 26% ✓ |

✓ = `in_unit_interval` (linear mediation assumptions satisfied);
✗ = proportion outside [0,1] → linear mediation assumptions broken
(direct and indirect effects have opposite signs).

**At sync=10000 the harm is interpretably mediated**: 84% of the late
outcome harm is mediated by `mc_b0` (DDQN's worse early policy). Late-Q
gap (`q_b19`) mediates only 26%.

## Stratified partial Spearman (JCI form, n=120 pooled, well-powered)

The cleanest within-seed test of the causal chain. Uses
`partial_spearman_rho` and `stratified_partial_spearman_rho`.

**Marginal** (across all 120 pairs): ρ(log_sync, Δ_mc_late) = −0.128, p = 0.16 ns.

**Conditional partial** ρ(log_sync, Δ_mc_late | mediator):
| mediator | ρ_partial | attenuation from marginal |
|---|---|---|
| Δ_q_b0 | −0.036 (p=0.70) | **72.1%** |
| Δ_mc_b0 | −0.099 (p=0.28) | 22.5% |
| Δ_online_max_q_b0 | −0.122 (p=0.19) | 4.6% |
| Δ_q_b19 | −0.115 (p=0.21) | 10.3% |

Δ_q_b0 attenuates the cross-sync log_sync→outcome correlation by 72% —
consistent with bootstrap-conservatism mediation. **But the marginal
correlation itself is underpowered (p=0.16)**, so the attenuation runs
from a low baseline.

**Stratified-pooled partial** (within-sync per-seed, Fisher-z pooled
across the 4 sync strata, df totals to 4×26=104 — well-powered):

| test | ρ_pooled | p | reading |
|---|---|---|---|
| ρ(Δ_q_b0, Δ_mc_late \| Δ_q_b19) | +0.123 | 0.21 ns | early-Q gap doesn't predict outcome after controlling for late-Q gap |
| ρ(Δ_q_b19, Δ_mc_late \| Δ_q_b0) | **−0.343** | **2.7e-4** | late-Q gap STRONGLY predicts outcome even controlling for early-Q gap |

This is the load-bearing test. **Within-seed across all syncs, the late-Q
gap is the predictor of outcome harm; the early-Q gap is not.** This
contradicts the "bootstrap conservatism is the mediator" reading.

## Meta-regression (4 strata, severely underpowered)

| covariate spec | coefficient | 95% CI | p |
|---|---|---|---|
| log_sync alone | −0.063 | [−0.49, +0.36] | 0.58 |
| mean argmax_disagree_b0 | −1.18 | [−4.15, +1.78] | 0.23 |
| mean q_b0_ratio (DDQN/van) | +2.76 | [−0.21, +5.72] | 0.057 |
| log_sync + q_b0_ratio: log_sync | +0.06 | [−0.29, +0.41] | 0.27 |
| log_sync + q_b0_ratio: q_b0_ratio | +3.55 | [−3.21, +10.32] | 0.095 |

`q_b0_ratio` coefficient is **positive** (when DDQN's early Q is closer
to vanilla's, g is higher) — direction consistent with bootstrap
conservatism. But p=0.057-0.095, not significant. **With only 4 strata,
the meta-regression is too underpowered to confirm bootstrap
conservatism as the cross-sync mechanism.** Adding sync values (5000,
7000) would improve this — currently a deferred experiment.

## Honest synthesis

Two stories survived contact with the data:

**(A) Within-seed mechanism (well-powered, ρ=−0.343, p=2.7e-4):**
At any given sync value, seeds where DDQN's late-Q diverged most from
vanilla's late-Q had the worst outcomes. Late-Q-trajectory amplification
is the within-seed signal. Early-Q-suppression has no within-seed
predictive power after controlling for late-Q.

**(B) Cross-sync mechanism (underpowered, suggestive):**
Across sync values, mean DDQN/vanilla early Q-ratio drops from 0.84
(sync=100) to 0.64 (sync=10000), and mean outcome g goes from null to
−0.41. proportion_mediated says 84% of sync=10000's harm is mediated by
DDQN's worse early policy (mc_b0). Meta-regression gives the q_b0_ratio
coefficient the right direction (+2.76) but only p=0.057. This is
consistent with bootstrap conservatism but not statistically confirmed
from 4 strata.

Both stories may be true at different scales. The previous
`mechanism_findings.md` overclaimed (B) as THE mechanism by reading
mean-level patterns. The within-seed primitives say (A) is the
explanatory variable that survives strict conditioning.

## Bridge implication (revised)

A bridge claiming "DDQN harms when sync is large" would be supported by
(A) alone:
- per-burst link active per seed at sync=10k (already in
  `findings_sync_curve_breakout.md`, plc=0.85)
- Δ_q_late strongly predicts Δ_outcome across all sync conditioned on
  Δ_q_early — so the failure mode IS Q-trajectory amplification,
  consistent with `findings_q_amplification_cartpole.md` generalizing
  from CartPole to Breakout

A bridge claiming "bootstrap conservatism is the cross-sync mechanism"
needs more sync values for the meta-regression slope on q_b0_ratio to
reach significance. Currently coefficient direction is right but
p=0.057-0.095. Extra sync ∈ {5000, 7000} would help — relatively cheap
follow-up sweep on Breakout only.

## What this changes about the active recommendation

- The earlier `findings_bootstrap_conservatism.md` memory should be
  rewritten to lead with (A) and frame (B) as cross-sync evidence
  needing more strata.
- The Q-amplification finding from CartPole (`findings_q_amplification_cartpole.md`)
  is corroborated on Breakout via stratified partial Spearman
  (ρ=−0.343, p=2.7e-4) — n_envs is now 2 with strong within-seed
  confirmation, not just observational suggestion.

## Reproduction

```bash
PYTHONPATH=. uv run python experiments/findings/sync_curve_breakout/run_statistical_proof.py
```

Output: `statistical_proof.json`.
