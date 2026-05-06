# Staleness causality vs inverse causality — observational defence

**Date:** 2026-05-06
**Question (user, Q1):** *"For each seed staleness affects DDQN
success but on average over seeds we cannot see — might just be like
inverse causality. Maybe you should run our causal inference tools.
Defend yourself."*

## TL;DR

**Observational paired-Δ data does NOT identify a staleness → outcome
causal arrow distinct from inverse-causal alternatives.** Forward
β(Δ_stale → Δ_o | Δ_jens) ≈ reverse β(Δ_o → Δ_stale | Δ_jens) at
z ≈ −0.22 each (within-env standardized, n=621 across 9 envs).
Direction is **observationally underdetermined**.

The 27% / 65% mediation findings (FourRooms-capacity, Breakout-
sync=100) are **narrow-scope results**: they activate only within
their specific endogenous regime. The cross-env pooled probe
washes them out. **This IS the endogenous-scope-predicate principle
in action** — staleness mediates outcome only when the regime
admits the mechanism.

The proper test is the Polyak-τ intervention sweep (`do(τ)` at
fixed sync_period) currently running on GPU. **Until that completes,
the staleness causal claim is mode-conditional, not universal.**

## Three causal-inference probes

### Check 1 — Per-env partial Spearman ρ(Δ_stale, Δ_o | Δ_jens)

Per env (mech-HELD: Δ_jens < 0), test whether the staleness-outcome
correlation survives conditioning on the upstream mech step:

| env | n | ρ_marg | p | ρ_part | p_part | reading |
|---|---|---|---|---|---|---|
| Acrobot-v1 | 62 | −0.11 | 0.40 | −0.02 | 0.89 | null |
| Asterix-MinAtar | 61 | +0.13 | 0.32 | +0.14 | 0.28 | null |
| Breakout-MinAtar | 67 | −0.14 | 0.28 | −0.10 | 0.43 | null |
| CartPole-v1 | 59 | −0.09 | 0.49 | −0.05 | 0.69 | null |
| **FourRooms-misc** | 114 | **−0.46** | **2e-7** | **−0.21** | **0.027** | **survives** |
| MetaMaze-misc | 97 | +0.09 | 0.36 | +0.16 | 0.12 | null |
| **MountainCar-v0** | 103 | **+0.47** | **6e-7** | **+0.47** | **5e-7** | **survives, opposite sign** |
| Pong-misc | 67 | NaN | – | NaN | – | constant outcome |
| SpaceInvaders-MinAtar | 58 | +0.19 | 0.15 | +0.14 | 0.30 | null |

Only **2/9 envs** survive partial-Spearman after mech adjustment, with
**opposite signs**: FourRooms is canonical (less staleness → better
outcome), MountainCar is anti-canonical (more staleness → better
outcome). MountainCar's ρ_part stays at +0.47 even after adjusting
for Δ_jens — it's not a mech-channel artifact. Likely a different
mechanism (exploration regime), not the bias-amplification chain.

### Check 2 — Stratified partial Spearman pooled across envs

`ρ_strat(Δ_stale, Δ_o | Δ_jens, strata=env_idx) = +0.081, p = 0.049`
(n=688). Borderline-significant, near-zero magnitude. Reading:
**within-env signal is mostly absorbed by env-stratification + mech
adjustment.** The two single-env survivors with opposite signs cancel
in the pooled Fisher-z.

### Check 3 — Within-env-standardized OLS + refutations

Standardize Δ_o, Δ_stale, Δ_jens within each env (z-score), then
pool. Eliminates env-scale artifacts.

```
β(Δ_stale → Δ_o | Δ_jens) = −0.009 ± 0.041   z = −0.22
β(Δ_jens  → Δ_o | Δ_stale) = −0.008 ± 0.041   z = −0.20
PLACEBO   β(shuffled Δ_stale)            = −0.035   (similar to base)
RCC       β(Δ_stale, jens noised)        = −0.008   (drift +0.001)

REVERSE: zs ~ α + β·zo + β·zj
β(Δ_o → Δ_stale | Δ_jens)               = −0.009 ± 0.040   z = −0.22
```

**Forward ≈ Reverse, both ~0.** The observational data is symmetric
in direction — we literally cannot distinguish "staleness causes
outcome" from "outcome causes staleness" or "both are downstream of
something else."

The placebo (within-env shuffled Δ_stale) gives β = −0.035 — a
similar-magnitude null effect, which means the apparent forward
coefficient is statistically indistinguishable from random.

## Why the bridge findings still stand at narrow scope

The `target_staleness_late_mediates_outcome__fourrooms` and
`__breakout_sync100` bridges hold within their declared scopes:

- **FourRooms (capacity_sweep)**: proportion=0.27, n_pairs=88. The
  narrow scope filters to capacity-sweep replicates only; the
  partial ρ check on the broader corpus's FourRooms cells (n=114)
  ALSO gives ρ_part=−0.21 with p=0.027, signed-consistent.

- **Breakout-MinAtar at sync=100**: proportion=0.65, n_pairs=16.
  The narrow scope filters to sync_period=100 only; the broader
  Breakout panel (n=67, all syncs) gives ρ_part=−0.10, p=0.43 —
  null. **The narrow-scope mediation IS sync=100-specific** (the
  Q-explosion regime). Pooling across syncs washes it out, exactly
  as `findings_minatar_link_attenuation` predicted: phases collapse
  the link signal.

This is the **endogenous-scope-predicate principle in action**:
narrow-scope claims survive their declared regime; they should NOT
be expected to generalize to all envs/configurations. The
observational washout at the broad scope is a feature, not a refutation.

## What the do(τ) intervention adds

Observational data lets us test:
1. Is the marginal correlation real? (yes, in FourRooms / sync=100)
2. Does it survive mech-step adjustment? (yes, in FourRooms)
3. Is it directionally identified? (**NO** — forward ≈ reverse)

Only `do(τ)` fixes the third question. By directly varying staleness
(via `polyak_update`) at fixed sync_period:

- If staleness causes outcome: sweep over τ ∈ {0.001, 0.01, 0.1, 1.0}
  produces a **monotone outcome curve** as τ grows (less staleness
  → better outcome in the FourRooms-direction envs).
- If outcome causes staleness: τ-variation is exogenous w.r.t.
  outcome, so outcome shouldn't change with τ at all.
- If both are downstream confounded: τ might still produce a curve,
  but it would have a different shape than the staleness-mediator
  prediction.

The `polyak_tau_intervention.yaml` sweep is 1440 cells (6 envs × 4 τ
× 30 seeds × 2 arms) currently running. **The decisive answer to Q1
lives in that sweep's results, not in this script.**

## Verdict

| claim | status |
|---|---|
| Δ_stale predicts Δ_o marginally | TRUE in FourRooms (n=114), MountainCar (n=103, opposite sign); null elsewhere |
| Survives Δ_jens conditioning | TRUE in FourRooms / MountainCar; absorbed elsewhere |
| Survives env-stratification + mech adjustment | borderline (ρ=+0.08, p=0.049) — opposite signs cancel |
| Forward effect distinguishable from reverse | **NO** — z=−0.22 each (symmetric) |
| Universal staleness → outcome causal arrow | **REFUTED at SLOPE > 0.5 floor** |
| Mode-conditional staleness mediation (FourRooms-capacity, Breakout-sync100) | TRUE within scope |
| Direction identified from observational data | **NO** — needs do(τ) |

## Reproduction

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. uv run python \
    experiments/findings/sync_curve_breakout/run_staleness_causal_inference.py
```

Output: `staleness_causal_inference.json` (per-env panel + pooled
OLS + reverse-causal probe).
