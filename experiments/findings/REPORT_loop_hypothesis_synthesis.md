# Loop-reduction as a hidden channel of DDQN's benefit at γ→1 sparse-reward

**Status**: Draft synthesis, May 2026. **Not yet publication-grade.**
Workshop-readable today; needs additional experiments + formal
statistics for main-conference submission.

## 0. Abstract

We investigate the mechanism by which DDQN (Hasselt 2010, 2016)
improves outcomes over vanilla DQN at high discount factors with
sparse reward. The standard story credits DDQN's clip mechanism
with reducing max-of-K overestimation bias. We provide
correlational and intervention evidence that a previously
unrecognized **loop-reduction channel** accounts for ~70% of
DDQN's late-window benefit at FourRooms γ=0.999. The clip's
direct effect (bounded Q magnitude) is one of two channels: the
loop-reduction is downstream of bounded Q, but can also be
reached via a separate path (count-weighted gradient updates).

**Headline claim**: at FR γ=0.999, count-weighted gradient
updates (no DDQN clip) recover ~70% of DDQN's late_window
benefit through loop reduction alone. The remaining ~30% (plus
most of the eval_final-burst benefit) requires DDQN's
target-decoupling component.

**Major caveats**: single env for the intervention, n=30 per
arm, no formal pre-registration, single optimism magnitude in
the OPIQ-falsification arm, metric-specific recovery rate.

## 1. Background

### 1.1 The bias-chain amplifier

DDQN reduces per-update overestimation bias of `max_a Q(s', a)`.
Under standard noisy-Q estimates the per-update bias is positive
(Jensen's inequality). At γ→1, this small per-update bias
compounds through the bootstrap chain:

```
Q_{t+1}(s, a) ← E[r + γ · max_a' Q_t(s', a')]
             ≈ E[r] + γ · (E[max_a' Q_t(s', a')])
             = E[r] + γ · (E[Q_t(s', a*)] + bias)
```

At fixed point, accumulated bias = `b / (1 − γ)`. At γ=0.999,
this is **1000× per-update bias**. Tiny noise per step becomes
huge bias asymptote.

### 1.2 The chain amplifier across envs (Λ_m)

We summarize the cross-env susceptibility via

  Λ_m = γ · σ · φ(K) / (ρ · R_max)

where σ is per-step Q noise, K is action count, ρ is per-step
reward density, R_max is per-step reward magnitude. Sparse-
reward + γ→1 makes Λ_m large; chain dominates.

### 1.3 Loops as policy consequence

At over-visited states `s`, accumulated bias makes `max_a Q(s, a)`
inflate fastest. The greedy policy then preferentially returns
to `s` (greedy on the inflated Q), closing a self-reinforcing
trajectory loop. The loop attractor IS the policy consequence
of the chain amplifier at sparse-reward + γ→1.

## 2. Observational evidence

### 2.1 Cross-env diagnostic (5 envs, γ=0.999)

For each env, measure DDQN's effect on outcome (Δ_outcome) and
on episode-agnostic state-bucket revisit rate (Δ_rep_ea) over
the canonical 1M sweep. Sign-alignment table:

  | env       | Δ_outcome | Δ_rep_ea | sign-aligned? |
  | FR        | +0.96     | -0.07    | ✓            |
  | SI        | +14.5     | -0.06    | ✓            |
  | Breakout  | +2.88     | -0.01    | ✓            |
  | LL        | +3.99     | -0.004   | ✓ (small)    |
  | Asterix   | -2.45     | +0.029   | ✓ (inverted) |

5/5 envs sign-aligned. Asterix's inversion (DDQN harms; repeat
↑) is the falsifying case that distinguishes "DDQN universally
reduces loops" from "DDQN's repeat effect tracks its outcome
direction."

**Caveats**:
- n=30 cells per arm per env; small sample for cell-level partial
  correlation.
- Cross-env sign-alignment confounded by env-structure
  covariation (the 5 envs differ on many dimensions; loop-prone-
  ness is one of them but not the only one).
- The Asterix-inversion test only counts as falsification if you
  buy the "sign-aligned" framing; alternative read is that
  Asterix lives in a different regime entirely (informative-
  anisotropy bias).

### 2.2 Mediation magnitude — env-specific

Partial-Spearman ρ(arm, mc_at_49 | M) per env at γ=0.999:

  | env       | marg ρ  | best mediator | %absorbed |
  | SI        | +0.443  | rep_ea        | 66%       |
  | Asterix   | -0.398  | jens          | 92%       |
  | Breakout  | +0.356  | jens/q/rep    | 64-68%    |
  | LL        | +0.056  | jens/entropy  | 97%/86% (with caveats) |
  | FR        | +0.908  | distributed   | none > 37% |

Repeat-rate is the **dominant mediator only at SI**. Other envs
have alternative dominant mediators (jens at Asterix; entropy
at LL). The "loop reduction as universal mediator" claim is
unsupported; the "loop reduction as ONE channel that aligns
cross-env in sign" claim is supported.

**Caveats**:
- jens has a tautology risk at envs where Q variance is low
  (e.g., LL). The 97% absorption at LL is inflated by Q ≈ constant
  → jens ≈ constant - mc → conditioning on jens conditions
  on mc near-tautologically.
- FR's "distributed" pattern (all mediators <37%) is partly an
  effect-saturation artifact: at 1M steps DDQN's outcome saturates
  (std=0 across cells), kills within-DDQN mediation power.

### 2.3 Within-env γ contrast at Asterix

Asterix corpus at γ=0.99 (new cloud restore) vs γ=0.999
(existing). Same env, same intervention, only γ differs:

  | metric | γ=0.99 | γ=0.999 |
  | DDQN outcome | +0.283 (helps) | -0.398 (harms) |
  | Δ_repeat_ea  | -0.018 (DDQN ↓ rep) | +0.024 (DDQN ↑ rep) |
  | dominant mediator | q (89%), rep (55%) | jens (92%) |

Both outcome and repeat-Δ direction flip with γ. The cross-env
sign-alignment hypothesis survives the strongest single test:
holding env structure constant, changing γ flips both outcome
and repeat consistently.

**Caveat**: single env contrast. Replicating at SI/FR with
γ-sweeps would strengthen.

### 2.4 Per-burst trajectory (FR γ=0.999)

Direct trace inspection at FR γ=0.999 × 1M shows DDQN MC
crossover (first burst with MC≥0.95) at median burst 12 (~240k
steps), range burst 4-23. Vanilla 0/30 ever cross. Vanilla Q
peaks at burst 11 (~12.89, the Lemma 2 asymptote) then decays
to ~5 by burst 49 while MC stays near 0. DDQN's Q stays at ~0.96
throughout. The temporal anchoring story (DDQN policy anchors on
rare goal-reach before vanilla's Q saturates) is empirically
visible.

**Methodological note**: the canonical `eval_best_burst_raw_mean`
metric saturates at 1.000 for all 30 DDQN cells at FR γ=0.999
because the binary goal-reach reward → 1.0 if ALL 5 episodes in
any burst reach the goal. Use `eval_final_mean` for within-DDQN
variance.

## 3. Intervention evidence

### 3.1 Optimistic-init test (falsifies OPIQ at FR γ=0.999)

OPIQ (Rashid 2020) frames the loop attractor as pessimistic-init
under-exploration: Q(unvisited) too low → policy stuck. If
correct, optimistic Q init should rescue vanilla.

3-arm test at FR γ=0.999 × 100k × n=30:
- VANILLA (standard zero-bias init): eval_final 0.029, jens 6.42
- DDQN: eval_final 0.309, jens 0.89
- **OPT_INIT (output bias=20 > Lemma 2 asymptote of ~12.6)**:
  eval_final **0.004** (10× worse than vanilla), jens **19.68**
  (3× higher than vanilla, Cohen d=+8.2)

Optimistic init does NOT rescue; it makes things WORSE. The
chain amplifier ate the optimism: initial Q=20 fed into the
bootstrap chain inflated to jens=19.68 at the end. This
**empirically falsifies the OPIQ framing at FR γ=0.999** and
supports the bias-chain framing.

**Caveats**:
- Single optimism magnitude (bias=20). Lower values (5, 10)
  might rescue partially. Higher values (50) probably saturate.
  No dose-response done.
- Single env (FR). Cross-env not tested.
- Tests our adapter's `init_bias_offset` parameter, not OPIQ
  proper (which adds count-based bonuses during training in
  addition to optimistic init). OPIQ-proper might still rescue
  via the bonus mechanism. Our test only falsifies the init-
  alone interpretation.

### 3.2 Count-weighted intervention (loop-channel test)

Substrate addition: state-hash visit counter (`state_hash_count`)
on DQNState, incremented in rollout_phase at the visited
state-hash. In train_phase, per-sample TD loss is multiplied by

  weight_i = 1 / (1 + count[state_hash(next_obs_i)])^α

α=0 → standard uniform mean. α>0 → downweight gradient updates
at over-visited next-states (uniform-coverage training).

**Dose-response at FR γ=0.999 × 100k × n=30**:

  | arm      | eval_final | late_win | best_burst | jens |
  | V α=0    | 0.029      | 0.067    | 0.50       | 6.4  |
  | V α=1    | 0.019      | 0.089    | 0.54       | 5.6  |
  | V α=2    | 0.039      | 0.370    | 0.67       | 5.1  |
  | V α=3    | 0.052      | 0.442    | 0.77       | 5.6  |
  | V α=5    | 0.101      | 0.669    | 0.84       | 7.1  |
  | V α=10   | 0.016      | 0.161    | 0.82       | 194  |
  | DDQN α=0 | 0.309      | 0.836    | 0.91       | 0.9  |
  | DDQN+α=2 | 0.114      | 0.624    | 0.74       | 1.1  |

**At 200k (refined dose-response, with DDQN anchor)**:

  | arm    | eval_final | late_win | best_burst | jens |
  | V α=5  | 0.120      | 0.614    | 0.83       | 13.5 |
  | V α=6  | 0.124      | 0.698    | 0.90       | 41.4 |
  | V α=7  | 0.087      | 0.459    | 0.88       | 368  |
  | V α=8  | 0.052      | 0.188    | 0.81       | 416  |
  | DDQN   | 0.662      | 0.982    | 0.96       | 0.69 |

**Recovery rates (V α=6 at 200k vs DDQN)**:
- late_window: 0.698 / 0.982 ≈ **71%** of DDQN's improvement above
  vanilla
- best_burst: 0.90 / 0.96 ≈ **94%**
- **eval_final: 0.124 / 0.662 ≈ 19%** — most of DDQN's
  final-convergence benefit is NOT recovered by count-weighting

**Caveats**:
- Single env (FR).
- α=5/6/7/8 each n=30 — minimum sample for any effect detection.
  α=10 N=30 too at 100k.
- Inverted-U sharper at 200k: α=7,8 catastrophically over-
  regularize (jens 368, 416 — chain amplifier diverges when
  Q at low-visit states is undertrained).
- DDQN+α=2 combo HURTS DDQN (eval_final 0.31 → 0.11 at 100k).
  Interventions don't stack additively — combination interferes.
- No pre-registration of α sweep. α=5/6 chosen post-hoc as
  "where outcome peaks in our exploratory grid."
- High within-arm cell variance (std 0.18 on eval_final means
  ~0.10) — distribution is bimodal. Some seeds reach DDQN-level
  performance; others don't.

### 3.3 Mechanism check: does count-weighted reduce jens?

Per-burst trajectory plot
(`experiments/figures/fr_g999_alpha_perburst_200k.png`):

  | arm   | jens early (burst 0-5) | jens late (burst 45-49) |
  | V α=5 | 3.40                   | **25.2**                |
  | V α=6 | 3.48                   | **74.2**                |
  | V α=7 | 3.74                   | **203**                 |
  | DDQN  | 1.21                   | **0.12**                |

**Count-weighted does NOT reduce jens at any stage.** Early jens
is slightly lower than vanilla baseline but climbs without bound
through training. DDQN's jens converges to ~0.

This is a critical observation: count-weighted reduces loops
(repeat-rate matches DDQN by late training) WITHOUT reducing
bias (Q diverges). Two routes to "fewer loops":

- **DDQN route**: bound Q → low Q at visited states → less
  greedy attraction → fewer loops
- **Count-weighted route**: noisy/undertrained Q at over-visited
  states → unreliable greedy choice → less consistent attraction

Same endpoint (loop reduction → outcome benefit on
late_window/best_burst), different mechanism (bounded vs noisy
Q).

## 4. Two-channel decomposition

DDQN's mechanism at γ→1 sparse-reward decomposes into:

| Channel             | Mechanism                          | Recovered by count-weighted? |
|---------------------|------------------------------------|------------------------------|
| Loop-reduction      | Reduce policy-stickiness at        | YES (~70% of late_window     |
|                     | over-visited states                 | benefit)                     |
| Q-magnitude bound   | DDQN's clip removes max-of-K bias  | NO (jens stays high)         |
|                     | → bounded Q via Hasselt 2010       |                              |

Both channels reduce loops eventually, via DIFFERENT routes:
- Q-bound → bounded greedy attraction → fewer loops
- Noisy-Q at over-visit (count-weighted) → unreliable greedy
  choice → fewer loops

DDQN provides BOTH; count-weighted provides only loop-reduction.
At late_window/best_burst metrics, loop-reduction alone is
sufficient for most of DDQN's benefit. At eval_final (final-burst
convergence), Q-magnitude bound is required for stability.

## 5. Limitations and what would strengthen the claim

### 5.1 What's solid

- Cross-env diagnostic sign-alignment at 5 envs at γ=0.999
- Asterix within-env γ-contrast (strongest single test)
- OPIQ falsification at FR γ=0.999 (optimistic init makes things
  worse, not better)
- Count-weighted intervention recovers a measurable fraction of
  DDQN's benefit at FR γ=0.999 (causal evidence)
- Per-burst trajectory plot shows DDQN's bounded-Q vs count-
  weighted's diverging-Q (mechanism articulation backed by data)

### 5.2 What's not solid

1. **Single env for the intervention**: count-weighted only tested
   at FR γ=0.999. SI/Asterix replication would strengthen
   causality. SI's existing 66%-mediation finding suggests the
   intervention should transfer; Asterix's inverted-direction
   could go either way (loop-channel-reduction at Asterix γ=0.999
   should HARM, matching DDQN's harm direction).

2. **No formal statistical tests**: only means + Cohen d
   reported. Need t-tests, confidence intervals, multiple-
   comparison correction. Within-arm partial-Spearman ρ at n=30
   is underpowered (all NS).

3. **eval_final gap is named but not characterized**: we say "the
   remaining 74% is bias-bound" but don't directly test that.
   The symmetric intervention (bound Q without bound loops) hasn't
   been run.

4. **No pre-registration**: α=5/6 chosen post-hoc.

5. **Single optimism magnitude**: OPIQ-falsification test used
   bias=20 only. Dose-response (5, 10, 50) untested. And we tested
   optimistic-init alone, not OPIQ-proper (which adds count
   bonuses during training).

6. **α=10 catastrophic explosion**: known empirical failure but
   no quantitative model. Reviewer would ask why.

7. **High seed variance** at the intervention sweet spot: eval_
   final at α=5 has std 0.18 on mean 0.10. Distribution is
   bimodal — some seeds reach DDQN-level, others don't.
   Counted as "partial recovery" but is really "some-seeds-yes,
   others-no."

### 5.3 What would convince a pedantic reviewer

Approximate effort estimates (single-developer):

1. **Cross-env intervention replication**: count-weighted sweep
   at SI γ=0.999 + Asterix γ=0.999. (~1 day GPU + analysis.)

2. **Symmetric intervention**: Q-magnitude bound without
   count-weighting. Simplest: hard-clip the bootstrap target at
   q_clip. Tests whether the bias-channel residual is real.
   (~2 days substrate work + sweep + analysis.)

3. **Bootstrap-side count modification**: per-state target damp
   `target = r + γ · max Q(s') − β · f(count(s'))`. Different
   from loss-weighting; closer DDQN analog. (~2 days.)

4. **Formal stats**: bootstrap 95% CIs, two-sided t-tests for
   pairwise arm differences, Bonferroni-corrected. (~1 day.)

5. **OPIQ-proper baseline**: implement OPIQ's count bonus on
   action selection + Q init together. Run as an arm in the
   intervention sweep. (~2 days substrate + 1 day sweep.)

6. **α dose-response with pre-registration**: declare α ∈
   {1,2,5,10}, predicted outcomes before running. (Re-running
   what we already have but with the pre-reg discipline.)

7. **Mechanism explanation for α=10 explosion**: derive Q
   divergence under count-weighted training. (~1 week theory.)

8. **Cross-env γ-flip replication**: SI γ-sweep (0.95, 0.99,
   0.995, 0.999) with new state_hash registered. (~1 day GPU.)

Total: roughly 2-3 weeks of careful additional work to reach
publication-grade for a top-tier conference. Without these
additions: workshop-level (NeurIPS workshop, ICLR blog).

## 6. Honest framing of what we have

Two-channel decomposition with one channel directly tested:

> "We provide intervention evidence that DDQN's outcome benefit
> at FR γ=0.999 has at least two channels: (1) a loop-reduction
> channel (~70% of late_window benefit, recovered by count-
> weighted gradient updates without DDQN's clip) and (2) a Q-
> magnitude-bound channel (residual ~30% of late_window, ~80% of
> eval_final convergence, requires DDQN's target-decoupling).
> Cross-env sign-alignment at 5 γ=0.999 envs (including an
> Asterix γ=0.99-vs-0.999 within-env flip) supports the loop-
> reduction channel's existence across the regime. Optimistic-init
> at FR γ=0.999 makes outcomes worse, falsifying the OPIQ frame
> of init-asymmetry under-exploration as the underlying
> mechanism."

This is the cleanest defensible single-paragraph claim. Scope
limits are clear. Walks-back from earlier overclaims are honest.

## 7. References to underlying memory notes

- `findings_cross_env_per_burst_panel_g999.md` (5-env panel)
- `findings_cross_env_mediation_magnitude_g999.md` (mediation %s)
- `findings_asterix_g099_vs_g0999_sign_flip.md` (within-env γ)
- `findings_optimistic_init_test_falsifies_opiq.md` (OPIQ test)
- `findings_count_weighted_recovers_ddqn_benefit_at_fr_g999.md`
  (intervention test)
- `findings_loop_reduction_as_hidden_channel.md` (previous
  framing)
- `findings_loop_hypothesis_lit_positioning.md` (lit review)

## 8. Code + data references

Sweeps:
- `experiments/configs/fr_g999_loop_test.yaml` (FR 1M baseline)
- `experiments/configs/fr_g999_100k_loop.yaml` (FR 100k)
- `experiments/configs/fr_g999_optimistic_init_test.yaml`
- `experiments/configs/fr_g999_count_weighted.yaml`
- `experiments/configs/fr_g999_count_weighted_aggressive.yaml`
- `experiments/configs/fr_g999_alpha_5to8_200k.yaml`

Substrate diffs (count-weighted intervention):
- `src/corroborate_rl/corroborate_rl/dqn/state.py`
  (`state_hash_count` field on DQNState)
- `src/corroborate_rl/corroborate_rl/dqn/dqn.py`
  (`count_weight_alpha` param + cardinality threading)
- `src/corroborate_rl/corroborate_rl/dqn/phases.py`
  (rollout-side counter increment + train-side weighted loss)
- `src/corroborate_rl/corroborate_rl/cell_runner.py`
  (cardinality plumbing from EnvSpec)

Optimistic-init substrate diff:
- `src/corroborate_rl/corroborate_rl/dqn/claims/q_network.py`
  (`init_bias_offset` field on MLP, threaded to `mlp_init`)

Figures:
- `experiments/figures/per_burst_5env_g999.png`
- `experiments/figures/per_burst_cross_env_g999.png`
- `experiments/figures/repeat_we_vs_ea_g999.png`
- `experiments/figures/fr_g999_alpha_perburst_200k.png`

Cached analysis panels (60-row per-env):
- `/tmp/panel_cache/{FR,SI,Asterix,Breakout,LL,Asterix_g099,Snake_g0999}.parquet`
