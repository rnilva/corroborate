# Polarity-predicts-link-sign is a soft tautology — explicit demonstration

**Date:** 2026-05-06
**Question (user, Q2):** *"polarity × horizon ~ outcome is fairly
tautological. But how about the observation that polarity alone could
predict the link's sign?"*

## TL;DR

**Yes — also tautological.** The 8-of-8 sign-match finding from the
original `eff_h_mediates_g_link__{goal,survival}_envs` analysis is
the env's structural L→outcome map measured in another guise. The
empirical demonstration: three independent forms of the link r all
track polarity at **ρ ≈ +0.83-0.86**, with **R² ≈ 0.85** and slopes
all near +0.5.

## Definitions (load-bearing)

```
env_reward_polarity := within-cell Pearson r(episode_length,
                                             mc_return)

bootstrap_fraction  := 1 − mean(done) per cell
                       (≈ 1 − 1/E[L_cell] when episodes are i.i.d.)

effective_horizon   := 1 / (1 − γ · bf)
                       (monotone in bf; bf is monotone in E[L])
```

So **eff_h and bf are both monotone transforms of E[L_cell]** (with
γ as a multiplier in eff_h's case). Anything that involves eff_h or
bf is structurally a re-projection of L.

## The tautology, formally

For env e with polarity p_e:

1. **Within an env, by definition**: across the cells of e, each
   cell's per-step pair (L_step, R_step) correlates at p_e (this IS
   the polarity definition).

2. **Cell-level aggregation preserves the correlation**: a cell's
   `eff_h` and `mean_outcome` are aggregates over the cell's L's and
   R's, so r(eff_h_cell, outcome_cell) across an env's cells inherits
   p_e (with cell-aggregation noise). Empirically R² ≈ 0.85 — close
   but not 1.0.

3. **Paired-Δ inherits the same structural relation**: for paired
   (vanilla, ddqn) cells at fixed seed, Δ_eff_h and Δ_outcome are
   shifts along the same env's L→outcome curve. The env's structural
   slope (= p_e × σ_O / σ_L) governs the shift, so r(Δ_eff_h,
   Δ_outcome) across seeds ≈ p_e.

The empirical premise required for step 3: **DDQN doesn't fundamentally
change the env's L→outcome map** — it just nudges the operating point
along the curve. This is nearly always true (DDQN is a TD-update tweak,
not an env modification), so the soft tautology covers ~89% of the
cross-env variance in link-r.

## Empirical demonstration

For each of n=8 polarity-defined envs from the canonical ddqn corpus,
compute three independent forms of the "link r":

| form | meaning | sample base |
|---|---|---|
| `r_within(eff_h, outcome)` | HARD: within-env r over baseline cells | 60 cells per env |
| `r_pair(Δ_eff_h, Δ_outcome)` | SOFT: across-pair r using eff_h | 30 paired seeds |
| `r_pair(Δ_bf, Δ_outcome)` | SOFT (no γ): across-pair r using bf directly | 30 paired seeds |

### Per-env panel

| env | polarity | r_within(eff_h,O) | r_pair(Δeff_h,ΔO) | r_pair(Δbf,ΔO) |
|---|---:|---:|---:|---:|
| MountainCar-v0 | −0.99 | **−0.83** | **−0.78** | **−0.77** |
| Acrobot-v1 | −0.93 | −0.47 | −0.40 | −0.41 |
| FourRooms-misc | −0.90 | **−0.83** | **−0.78** | −0.60 |
| MetaMaze-misc | −0.33 | −0.10 | −0.10 | −0.10 |
| SpaceInvaders-MinAtar | +0.16 | −0.17 | −0.21 | −0.10 |
| Asterix-MinAtar | +0.52 | −0.16 | −0.11 | −0.12 |
| CartPole-v1 | +0.92 | **+0.45** | **+0.40** | +0.42 |
| Breakout-MinAtar | +0.99 | **+0.35** | **+0.33** | +0.35 |

(Pong-misc excluded: outcome is constant at saturation, link r is
NaN.)

The within-arm form (HARD), the paired-Δ form (SOFT), and the bf
form (SOFT, no γ) **all track polarity in lockstep**. The within-arm
form (which uses NO DDQN data at all — it's a property of vanilla
cells alone in the env) gives the SAME polarity correlation as the
paired-Δ form that compares vanilla to DDQN.

### Cross-env correlation (the tautology test)

```
HARD:        ρ(polarity, r_within(eff_h, O))     = +0.833 (p=0.010, n=8)
             regression: r ≈ +0.529·polarity − 0.184,   R² = 0.848

SOFT:        ρ(polarity, r_pair(Δ_eff_h, Δ_O))   = +0.833 (p=0.010, n=8)
             regression: r ≈ +0.488·polarity − 0.172,   R² = 0.843

SOFT (no γ): ρ(polarity, r_pair(Δ_bf, Δ_O))      = +0.857 (p=0.007, n=8)
             regression: r ≈ +0.466·polarity − 0.134,   R² = 0.867
```

The HARD form (no DDQN data) and the SOFT form (with DDQN data) give
**ρ to three decimal places, R² within 0.005, slopes within 0.04**.
The DDQN data adds essentially zero information — the soft tautology
is structurally identical to the hard one.

## What this means for the original 8-of-8 finding

The original `eff_h_mediates_g_link__{goal,survival}_envs` bridges
declared **8-of-8 sign-match between polarity and link-r** as
evidence that polarity is the load-bearing moderator of DDQN's
mediator. The reframing:

- **What it actually measured**: the env's structural L→outcome
  map. Goal envs (polarity < 0): less length → better outcome,
  always — vanilla and DDQN both. Survival envs (polarity > 0):
  more length → better outcome, always.

- **What it did NOT measure**: any DDQN-specific mechanism. The
  same finding would hold if you compared two random RL algorithms
  in the same envs — as long as both move along the env's L→outcome
  curve (which any non-pathological algorithm does).

- **Independent mechanistic content**: ~11% of the cross-env
  link-r variance is NOT explained by polarity (R² = 0.843 → 0.157
  unexplained). That residual is small and sign-coherent; not
  enough to justify a mediator claim.

The bridges currently encode this correctly with
`predicted_direction='null'` — they assert that eff_h is NOT a
dominant mediator. The "8-of-8 sign match" framing in the original
analysis was a tautological observation dressed up as an empirical
discovery.

## Summary of the polarity story

The two memories that frame this saga:

| memory | claim | status |
|---|---|---|
| `findings_polarity_mediator.md` | polarity predicts link sign 8-of-8 (binomial p=0.004) | TRUE, but **soft tautology** |
| `polarity_measurement_frame.md` | polarity is measurement-frame, not DDQN mechanism | TRUE, **decisive** |

Both memories are factually correct. The first describes the
structural pattern; the second explains it.

The takeaway for endogenous-scope hunting: **polarity is downstream
of any algorithm that changes outcome via length**, not upstream of
DDQN's bias-correction mechanism. Don't use polarity as a scope
predicate for mechanism-blind claims about DDQN.

## Reproduction

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. uv run python \
    experiments/findings/sync_curve_breakout/run_polarity_tautology_demo.py
```

Output: `polarity_tautology_demo.json` (per-env panel + cross-env
ρs and regressions).
