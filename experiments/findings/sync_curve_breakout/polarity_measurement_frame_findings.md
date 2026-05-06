# Polarity is a measurement-frame property, not a DDQN mechanism

**Date:** 2026-05-06
**Followup to:** `polarity_asymmetry_findings.md` and the target_staleness mediator panel.

## Headline

> **Polarity affects how DDQN's outcome benefit appears in length-space,
> NOT whether DDQN works mechanistically.** The earlier polarity-coupling
> finding (`r ≈ 0.5 × env_reward_polarity, R²=0.886`) was a near-tautology:
> eff_h IS length, polarity IS the env's r(length, outcome) — so the
> coupling correlation is forced by polarity's definition. The actual
> mechanism (jens → policy → outcome) is polarity-blind. The staleness
> mediator (`target_staleness_late`) is polarity-orthogonal.

## The decisive test

For each polarity-panel env in the canonical `ddqn` corpus (n=8 envs,
60+60 cells each), compute within-env Pearson r between Δ_outcome and
each candidate Δ_predictor across paired DDQN/baseline cells. Then
cross-env: Spearman ρ between the within-env r and env's polarity.

The mechanism-blind hypothesis predicts:
- `r(Δ_eff_h, Δ_o)` flips sign by polarity (forced by polarity definition)
- `r(Δ_staleness, Δ_o)` is polarity-orthogonal (true mechanism)
- `r(Δ_jens, Δ_o)` ≈ polarity-orthogonal (DDQN's bias correction is universal)

## Result

| predictor | ρ(polarity, r) | p | reading |
|---|---|---|---|
| `r(Δ_eff_h, Δ_o)` | **+0.833** | **0.010** | strongly polarity-coupled — measurement-frame |
| `r(Δ_bf, Δ_o)` | +0.857 | 0.007 | same channel as eff_h (collinear) |
| **`r(Δ_staleness, Δ_o)`** | **0.000** | 1.0 | **completely polarity-orthogonal** |
| `r(Δ_jens, Δ_o)` | +0.333 | 0.42 | weakly polarity-correlated, ns |

## Per-env panel

| env | polarity | r_jens | r_stale | r_eff_h |
|---|---|---|---|---|
| MountainCar-v0 | −0.99 | +0.05 | +0.28 | **−0.78** |
| Acrobot-v1 | −0.93 | −0.35 | −0.15 | **−0.40** |
| FourRooms-misc | −0.90 | −0.01 | −0.48 | **−0.78** |
| MetaMaze-misc | −0.33 | +0.12 | +0.10 | −0.10 |
| SpaceInvaders-MinAtar | +0.16 | −0.38 | +0.36 | −0.21 |
| Asterix-MinAtar | +0.52 | −0.11 | +0.15 | −0.11 |
| CartPole-v1 | +0.92 | +0.21 | +0.04 | **+0.40** |
| Breakout-MinAtar | +0.99 | +0.10 | −0.08 | **+0.33** |

The r_eff_h column tracks polarity sign strongly. The r_stale column does
not — staleness's relationship to outcome is uncorrelated with the env's
L→outcome direction.

## Reframing the polarity finding

Previous framing (in `polarity_asymmetry_findings.md`): "Polarity is the
load-bearing moderator for DDQN's link mediator. r ≈ 0.5 × polarity."

Sharper framing: **"eff_h IS the polarity channel BY DEFINITION."**
The polarity-coupling correlation was an inevitable consequence of choosing
length-related variables (`bf`, `eff_h`) as mediators in an analysis that
projects outcome onto L-space. If we use staleness instead, the polarity
correlation vanishes (ρ=0).

This is not a contradiction of the prior finding — it's a sharper reading.
The prior data is correct; the interpretation was upside-down. Polarity
is downstream of mechanism (a property of the env's L→outcome formula),
not upstream of it (a moderator of DDQN's behavior).

## Implication for endogenous-scope hunting

The methodological lesson: **scope predicates must be functionally
independent of the outcome's projection axis.** `env_reward_polarity` is
a perfectly valid scope predicate for **length-coupled** claims (e.g.,
"DDQN's outcome benefit appears as a length change with sign matching
polarity"). It is NOT a valid scope predicate for **mechanism-blind**
claims about DDQN's actual operating principle.

When chasing endogenous predictors of DDQN's outcome benefit, polarity
is not the right variable — it just describes the env's reward formula's
projection-of-L-onto-outcome. Better candidates (per the staleness panel):

- `q_divergence_score` — Q-trajectory dynamics; predicts Q-amplification regime
- `jensen_dormancy_gap` — bias premise active vs dormant
- vanilla's late-training staleness baseline (proposed) — DDQN's
  reduction-headroom

These are mechanism-side predicates, NOT measurement-projection predicates.

## What this changes about the DDQN study panel

The `eff_h_mediates_g_link__goal_envs` / `__survival_envs` bridges encode
the measurement-frame finding correctly (with `predicted_direction='null'`
asserting that eff_h is NOT a dominant mediator). They should remain —
the negative result IS the load-bearing finding.

The new staleness bridges (`target_staleness_late_mediates_outcome__*`)
should be the polarity-blind ones — but currently they're env-name scoped.
Refactoring to endogenous scope predicates is the next step.

## Reproduction

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. uv run python \
    experiments/findings/sync_curve_breakout/run_polarity_mechanism_v2.py
```

Output: `polarity_mechanism_v2.json` (per-env panel + cross-env tests).
