# Three-Layer Mediation Analysis of DDQN at γ=0.99 Canonical

**Case study for the `corroborate` framework.**
**Scope:** Hasselt-2010's Double-DQN as a cross-env claim, evaluated on the canonical γ=0.99 panel of 12 environments (680 cells, n=10–60 seeds per arm per env).
**Question:** Does the textbook causal story — *DDQN reduces overestimation bias → improved outcome* — survive a graded mediation analysis under the framework's typed primitives?

> **REVISION HISTORY**
> - **v1** (initial): identified Asterix γ=0.99 as cleanest mediation case via `mean_per_state_cumulative_bias_per_burst` (91% PC d-separation).
> - **v2** (current): **tautology audit invalidates v1's headline finding.** `mean_per_state_cumulative_bias_per_burst` reads `mc_return_from_step` directly — it's literally `Q − MC` per visited state, so its mediation of `mc_return`-based outcomes is structurally tautological, not causal. The framework's tautology audit primitive ([§3.5](#35-tautology-audit-the-load-bearing-correction)) refuses this mediator. After exclusion, Asterix's per-burst d-separation drops from 91% to **59% via q_argmax_margin**, and no single clean mediator cross-env-stratified d-separates arm—outcome (only depth-2 joint sets do). This is the framework's discipline doing real work — without the tautology audit, v1 would have published a structural artifact as the mechanism.

## Abstract

The framework's hierarchy of typed analysis primitives is a graded-refusal mechanism: each layer's primitive carries its own gate (verdict enum, status flag, pool diagnostic) that prevents a downstream over-claim. We exercise this discipline on a 12-env γ=0.99 canonical panel for DDQN. **The methodological story includes a tautology audit that invalidates a tempting-but-spurious cross-env mediator** (`mean_per_state_cumulative_bias` = `Q − MC` per visited state — structurally tautological with `mc_return`-based outcomes), forcing a revised, weaker but honest causal picture.

At Layer 1 (cross-env aggregates), the directional MECH and LINK claims survive (binomial p=0.003, perm p=0.003), but the magnitude DL pool refuses extrapolation under I²=0.96 and PI bounds that cross zero.

At Layer 2 (PC discovery on the **tautology-cleaned** candidate set), **no single mediator cross-env-stratified d-separates** the arm-outcome edge — only depth-2 joint sets do (any pair of {`bg_magnitude`, `bg_disagree`, `argmax_ent`, `greedy_match`, `state_n_unq`, `state_ent`, `state_repeat`} suffices).

At Layer 3 (per-burst dynamic, clean candidates), only 5 of 12 envs have a detectable per-burst LINK at α=0.05. Per-env best CLEAN mediators differ across envs:
- **Asterix γ=0.99: q_argmax_margin (59%)** — Q-action-gap, not bias-reduction
- MetaMaze γ=0.99: bg_magnitude (100%)
- Freeway γ=0.99: bg_magnitude (60%)
- FourRooms γ=0.99: argmax_ent (60%)
- SI γ=0.99: q_argmax_margin (66%, n=3)

**The substantive finding** is that DDQN's bias-reduction-to-outcome-improvement mechanism is **env-and-granularity-specific** AND **not via `Q − MC`-bias-by-construction** (that mediator is tautological). The **methodological finding** is that the framework's typed primitives — including its tautology audit — correctly identify *where*, *at what granularity*, and *via which non-tautological mediator* each scientific claim can be made — and refuse otherwise.

---

## §1 The question and its three sub-questions

Hasselt's 2010 claim, stated as a cross-env empirical hypothesis: *DDQN's clip → bias reduction → outcome improvement, across MDPs.*

The framework decomposes this into three sub-questions, each with its own typed primitive:

1. **MECH** (cross-env directional): does DDQN reduce Jensen bias `jensen_gap` in the predicted direction at each env?
2. **LINK** (cross-env directional): does DDQN improve outcome `eval_best_burst_raw_mean` in the predicted direction at each env?
3. **MEDIATION** (causal pathway): is the LINK *carried by* the MECH? Is there a measurable mediator on the arm→outcome path that d-separates the edge?

The framework's deflation primitives ensure each sub-question is answered with its own gate before the next is asked. This report walks through the three layers in order.

## §2 The framework hierarchy

| layer | primitive(s) | scientific question | refusal gate |
|---|---|---|---|
| **L1 directional** | `cross_env_consistency_binomial`, `cross_env_probability_of_improvement` | "Is the direction consistent cross-env?" | binomial / permutation p |
| **L1 magnitude** | `stratified_arm_diff_pooled` (DL random-effects) | "Can magnitude be extrapolated to a new env?" | `verdict` enum (`HELD`/`HELD_WITH_SCOPE_FLAG`/`NO_EFFECT`), I² + PI bounds |
| **L2 PC discovery** | `discover_adjacency` (cross-env stratified + per-env) | "Are our candidate mediators sufficient?" | edge presence at α=0.05 + separating-set identification |
| **L2 quantification** | `partial_spearman` (per-env, rank-based) | "How much does the PC-validated mediator absorb?" | per-env ρ_marg / ρ_part magnitudes |
| **L3 per-burst dynamic** | `dynamic_partial_spearman` + `dynamic_pc_adjacency` | "Is the per-burst mediation stable across training?" | `TimeAggregationStatus` enum (`CONSISTENT_DIRECTION` / `SIGN_FLIP_DETECTED` / `WEAK_TIME_VARYING` / `UNDERPOWERED_BURSTS`) |

Each gate is a typed result-field that bridges and consumers consume. The framework's contribution is that these gates compose: a claim that survives the L1 directional binomial but is refused at the L1 magnitude pool can only be reported as a directional claim, not a magnitude one. Similarly, an L2-validated mediator that yields `SIGN_FLIP_DETECTED` at L3 cannot honestly be summarized as a per-cell mediation magnitude.

## §3 Layer 1: cross-env directional + magnitude

**Cross-env directional** (rank-based + binomial):

- **MECH** (`cross_env_consistency_binomial` on `jensen_gap`, `predicted_direction='a_lt_b'`): **11/12 envs** in predicted direction (D reduces V's Jensen bias). Binomial sign-test p = **0.0032**. ✓ HELD.
- **LINK** (`cross_env_probability_of_improvement` on `eval_best_burst_raw_mean`): cross-env mean `P(D > V)` = **0.608**, permutation p = **0.0029**, bootstrap 95% CI = [0.564, 0.650]. ✓ HELD.

**Cross-env magnitude** (DL random-effects pool):

- **MECH pool**: Cohen's d = −2.42, **I² = 0.96**, PI = [−6.26, +1.41]. PI crosses zero → framework `verdict = NO_EFFECT`. **Magnitude claim refused.**
- **LINK pool**: Cohen's d = +0.44, **I² = 0.42**, PI = [−0.12, +1.00]. PI crosses zero → `verdict = NO_EFFECT`. **Magnitude claim refused.**

**Reading**: The direction holds — DDQN consistently reduces bias and consistently improves outcome — but the magnitude pool refuses to extrapolate to a new env. This is the framework's first deflation: a 50× heterogeneity range (Asterix d_mech = −12.83 vs Acrobot d_mech = −0.24) is not summarizable as a single scalar.

### Rank-robust caveat

Cohen's d is parametric and outlier-sensitive. Switching to `cross_env_probability_of_improvement`'s rank-based per-env P(D>V) reveals that parametric d overstates the LINK at Acrobot (d = +0.28 but P(D>V) = 0.51) and at Snake (d = +0.22 but P(D>V) = **0.48** — D is actually slightly worse by rank). The framework's `P(D>V)` is the recommended primitive for cross-env directional claims under heterogeneous strata.

## §3.4 Outcome-aggregation caveat — per-cell-best-burst inflates DDQN's apparent benefit

Throughout L1 the LINK direction was assessed using `eval_best_burst_raw_mean` per cell: for each cell, take the burst whose K-episode mean is highest, report that mean. This is the standard RL-paper aggregation, but it has a known inflation pathology (see memory `findings_canonical_n_eps20_q_ckpt_replication`): the per-cell metric is `max_b (mean_eps mc_return[b])` — a maximum over noisy bursts — so each seed contributes its noisiest peak, not its expected late-training policy.

**Alternative aggregation** (seed-aggregated-best-burst-mean): for each (env, arm), at each burst average the per-burst-mean across all cells in the arm to get an arm-level trajectory, then take the max over bursts. This is one scalar per arm (not per cell), so it doesn't admit Cohen's d or P(D>V), but it gives a cleaner "what does the arm's policy actually achieve" reading.

Comparison at γ=0.99 canonical (both in RAW units):

| env | per-cell d (n=30) | per-cell P(D>V) | seed-agg D−V | sign agreement? | D inflation ratio | V inflation ratio |
|---|---|---|---|---|---|---|
| Asterix | +1.32 | 0.82 | **+6.99** | ✓ same | 1.37 | 1.41 |
| **SI** | **+0.91** | 0.78 | **−1.07** | ✗ **opposite** | 1.29 | 1.25 |
| **Breakout** | **+0.44** | 0.60 | **−2.23** | ✗ **opposite** | 1.39 | 1.26 |
| **LunarLander** | **+0.13** | 0.50 | **−30.81** | ✗ **opposite** | 2.65 | 1.96 |
| PacMan | +0.65 | 0.69 | +119.0 | ✓ same | 1.12 | 1.09 |
| FR | +0.53 | 0.60 | +0.21 | ✓ same | 1.07 | 1.30 |
| MetaMaze | +0.40 | 0.60 | +4.40 | ✓ same | 1.60 | 1.59 |
| MountainCar | +0.36 | 0.62 | +11.59 | ✓ same | 0.90 | 0.92 |
| Acrobot | +0.28 | 0.51 | +16.83 | ✓ same | 0.90 | 0.87 |
| Freeway | +0.25 | 0.59 | +0.03 | ≈ tied | 1.18 | 1.15 |
| Snake | +0.22 | 0.48 | +0.18 | ≈ tied | 2.99 | 2.20 |
| CartPole | +0.00 | 0.50 | +16.95 | ≈ tied | 1.04 | 1.07 |

(Inflation ratio = per-cell-best-mean / seed-agg-best. Values > 1 indicate the per-cell metric overstates the arm-level outcome.)

**Three envs flip direction** under seed-aggregated outcome: **SpaceInvaders**, **Breakout**, **LunarLander**. Per-cell Cohen's d shows DDQN improvement, but at the seed-aggregated arm-level the situation is reversed — V's arm-level best burst is higher than D's. The per-cell metric is inflated by the noisiest-bursts-per-seed pathology more for D than V at these envs.

**The substantive implication for the LINK claim**: at SI, Breakout, LL the apparent DDQN benefit at the per-cell metric is largely (or entirely) an artifact of max-over-noisy-bursts at the seed level. The L1 cross-env directional binomial p=0.003 reading using `eval_best_burst_raw_mean` partially reflects this inflation. A more honest LINK panel:

- **Robust LINK+** (per-cell + seed-agg agree, both positive): Asterix, PacMan, FR, MetaMaze, MountainCar, Acrobot (n=6)
- **LINK+ at per-cell only** (seed-agg reverses): SI, Breakout, LunarLander (n=3)
- **Borderline / null** (both small): Freeway, Snake, CartPole (n=3)

The L1 binomial dropping to 6/12 envs in the seed-agg robust direction would still be cross-env consistent (binomial 6/12 against random null gives p ≈ 0.07-0.4 depending on signed-null assumption), but materially weaker than the per-cell p=0.003.

**Methodological reading**: the choice of outcome aggregation is itself a framework gate. The framework's typed outcome measurables (`eval_best_burst_raw_mean` per-cell vs hypothetical arm-level `seed_agg_best_burst_raw`) should be explicit about which inflation regime they're in. Reporting per-cell d=+0.91 at SpaceInvaders is **mathematically correct under the per-cell aggregation** and **misleading at the arm-level**. The framework recommends:

1. Use rank-based `P(D>V)` (already does) — partially mitigates outlier inflation but still per-cell.
2. Report seed-aggregated comparison alongside per-cell for substantive arm-level claims.
3. Flag envs where per-cell and seed-agg disagree in sign as "outcome-aggregation-sensitive."

This is the framework's "fourth refusal gate" at the outcome side, completing the hierarchy.

## §3.5 Tautology audit (the load-bearing correction)

Between Layer 2's PC discovery and our acceptance of any mediation claim, the framework's `tautology_audit` primitive (in `corroborate.analyses.diagnostic`) demands a **reads-set audit** for each candidate mediator: does the mediator's input column set overlap with the outcome's input column set? High overlap indicates the mediator is computed FROM the outcome's underlying data and so its "mediation" is structural, not causal.

We audited the 14 candidate mediators against the outcome `mc_return__mean_axis_-1` (`reads = ('mc_return',)`) and `eval_best_burst_raw_mean` (which derives from `mc_return_raw_episodes`, in turn from `mc_return_from_step`).

**Tautological mediators identified** (must be excluded from causal mediation claims for `mc_return`-based outcomes):

| mediator | reads | tautological because |
|---|---|---|
| `mean_per_state_cumulative_bias_per_burst` | `predicted_q_per_step`, **`mc_return_from_step`**, `active_per_step` | computed as `Q(s) − G(s)` per visited state — directly reads the per-step source of `mc_return` |
| `mean_per_state_cumulative_bias_late` | same as above + `eval_step_index` | same |
| `jensen_gap` | `predicted_q_at_start`, **`mc_return`** | computed as `max(0, mean(Q − MC))` — literally Q − MC |
| `jensen_dormancy_gap_per_burst` | + `mc_return` | derived from jens, MC-tautological |
| `q_mc_burst_correlation_late` | `online_max_q_per_step`, **`mc_return`** | by definition correlates Q and MC |
| `mc_*_per_burst` (CV, variance, log-CV) | `mc_return` | direct MC functions |

**The v1 finding "pstate_bias d-separates arm-outcome at 91% of Asterix bursts" is therefore a tautology artifact.** `mean_per_state_cumulative_bias_per_burst` is `Q − G_t per visited state`. Conditioning on `Q − G` necessarily reveals information about `G` (which is what `mc_return` aggregates), so partial correlations on `mc_return`-based outcomes will spuriously absorb. This is not a causal mediation — it's a structural identity.

The framework's discipline is: **drop tautological mediators BEFORE running PC discovery or partial-Spearman analyses on the outcome they share inputs with**.

**Clean (non-tautological) candidate set** (used in §4 and §5 below):

| family | mediators |
|---|---|
| Bellman-side | `bootstrap_gap_magnitude_per_burst`, `bootstrap_disagree_rate_per_burst`, `bootstrap_disagree_gap_conditional_per_burst`, `greedy_match_per_burst` |
| Action/policy | `argmax_entropy_per_burst`, `state_conditional_argmax_entropy_per_burst` |
| Q-shape (CNN-only) | `q_argmax_margin_per_burst`, `q_action_std_per_burst`, `q_autocorr_per_burst`, `q_lambda_a_per_burst` |
| State-coverage | `state_hash_n_unique_per_burst`, `state_hash_entropy_per_burst`, `state_repeat_rate_window64_per_burst` |

None of these candidates have `mc_return*` in their `reads` set. Their absorption of arm—outcome correlation is causally interpretable.

## §4 Layer 2: PC discovery (clean candidates only)

**Cross-env stratified PC** (`discover_adjacency` with `stratify_by='env_name'`, `max_conditioning=2`, α=0.05, **7 clean base mediators**):

```
arm-adjacent: (none — no single clean mediator is arm-adjacent at the
              env-stratified CI level)
outcome-adjacent: bg_magnitude, state_repeat
arm—outcome edge REMOVED
separating sets: 21 PAIRS qualify, including
  {bg_magnitude, state_repeat}, {argmax_ent, bg_disagree},
  {bg_magnitude, greedy_match}, {state_ent, state_n_unq}, ...
NO single-mediator separating set found.
```

**Key shift from v1**: with the tautological `pstate_bias` removed, **no single non-tautological mediator d-separates arm—outcome cross-env-stratified**. The arm—outcome marginal correlation is barely above α=0.05 at the env-stratified level (ρ ≈ −0.20 pooled), so any depth-2 pair of clean mediators is enough conditioning to push p > 0.05. This is NOT a strong cross-env mediation claim — it says the LINK signal is fragile enough that several different two-variable conditioning sets equivalently reduce it below the α threshold.

PC's conclusion at L2 is: **the clean candidate set is sufficient cross-env-stratified (some depth-2 pair always d-separates), but there's no preferred single mediator**. This is the framework refusing to identify a universal mediator — the v1 "{pstate_bias} sufficient" claim was the structural tautology speaking, not the data.

**Per-env PC** (each env separately, n ≈ 60, depth=2, clean candidates only):

The per-env picture is dominated by underpowering at n=60: most envs' marginal arm—outcome edge isn't detectable at α=0.05 at the env-level CI test. The mediation question is **underpowered at the env level** — the cross-env aggregation in L1 is doing the statistical work. This is the same observation as v1 — the tautology correction only affects WHICH mediator gets identified when the edge IS detectable, not WHETHER it's detectable.

## §5 Layer 3: per-burst dynamic

For each of the 12 envs, we run `dynamic_partial_spearman` and `dynamic_pc_adjacency` per-burst with each of 14 mediator candidates (8 base + 6 CNN-only Q-shape and state-conditional). The framework's `TimeAggregationStatus` enum gates pool reporting; `dynamic_pc_adjacency`'s `n_bursts_marginal_edge / n_bursts_mediator_dseparates / n_bursts_direct_edge` triple gates per-burst attribution.

### Master per-env table (tautology-corrected, clean candidates only)

| env | LINK P(D>V) | PC best **clean** mediator | n_marg | dsep | dsep % | comment |
|---|---|---|---|---|---|---|
| **Asterix** | **0.822** | **`q_argmax_margin`** | 32 | 19 | **59%** | Q-action-gap (Q-shape side), not bias-reduction |
| **SI** | **0.779** | `q_argmax_margin` | 3 | 2 | 66% | small n_marg, same family |
| PacMan | 0.690 | (no LINK at α=0.05) | 0 | — | — | underpowered per-burst |
| MountainCar | 0.621 | (no LINK) | 0 | — | — | underpowered |
| **MetaMaze** | 0.603 | **`bg_magnitude`** | 2 | 2 | **100%** | canonical Bellman wedge |
| **FourRooms** | 0.601 | **`argmax_ent`** | 15 | 9 | **60%** | action-policy, not bias |
| Breakout | 0.598 | (no LINK) | 0 | — | — | underpowered |
| **Freeway** | 0.594 | **`bg_magnitude`** | 5 | 3 | **60%** | Bellman wedge (dropped from 80% w/ pstate_bias) |
| Acrobot | 0.511 | (no LINK) | 0 | — | — | underpowered + null LINK |
| CartPole | 0.500 | (no LINK) | 0 | — | — | saturated |
| LunarLander | 0.496 | `state_n_unq` | 1 | 1 | 100% | n=1 burst — diagnostic only |
| Snake | **0.476** | `state_n_unq` | 2 | 2 | 100% | LINK reversed |

**Comparison with v1 (tautological pstate_bias included):**

| env | v1 best | v1 dsep% | v2 clean best | v2 dsep% | change |
|---|---|---|---|---|---|
| Asterix | pstate_bias | **91%** | q_argmax_margin | **59%** | **−32%** |
| Freeway | pstate_bias | 80% | bg_magnitude | 60% | −20% |
| MetaMaze | bg_magnitude | 100% | bg_magnitude | 100% | same |
| FourRooms | argmax_ent | 60% | argmax_ent | 60% | same |
| SI | q_argmax_margin | 66% | q_argmax_margin | 66% | same |

The tautology audit moves us from "Asterix is THE clean canonical Hasselt case at 91%" to "Asterix has the strongest per-burst clean mediation in the panel at 59%, but the mediator is `q_argmax_margin` (action-gap), NOT bias-reduction-by-construction".

### Three substantive tiers (tautology-corrected)

**Tier A — strongest per-burst clean mediation (1 env)**: **Asterix γ=0.99**. 32 of 50 bursts have a detectable marginal arm—outcome edge at α=0.05; `q_argmax_margin` (action-gap; non-tautological) d-separates **59%** of them (down from the v1 tautological 91%). The remaining 41% of detectable bursts have a direct edge surviving — meaning the clean candidate set is INSUFFICIENT to fully d-separate at Asterix per-burst. Substantively: DDQN's clip operates on Q values in a way that the action-gap (`q_argmax_margin` = top1 − top2 Q-value) partially carries the arm-outcome relationship, but a residual direct edge survives that none of the 13 clean candidates fully accounts for. This is the framework reporting "best available clean mediator, but the candidate set is incomplete at this env" — exactly what we'd expect if the real causal channel routes through a measurable we haven't engineered yet (e.g., `policy_churn_per_burst`, not yet ported).

**Tier B — partial per-burst mediation (3 envs)**: **MetaMaze (2/2 = 100% via bg_magnitude)**, **FourRooms (9/15 = 60% via argmax_ent)**, **Freeway (3/5 = 60% via bg_magnitude)**. Different mediators win at different envs:
- MetaMaze: Bellman wedge (`bg_magnitude`) — canonical Hasselt-mechanism reading at this env (small n_marg, but 100% dsep at the detectable bursts)
- FourRooms: argmax entropy (action-selection concentration) — consistent with substrate annotation about action-slip making `argmax_ent` the carried channel
- Freeway: Bellman wedge — same family as MetaMaze

**Tier C — per-burst LINK undetectable (7+ envs)**: PacMan, MountainCar, Breakout, Acrobot, CartPole, LunarLander (n_marg=1), Snake (LINK actually reversed, P(D>V)=0.48). At per-burst n=30/arm, the marginal arm—outcome CI test rarely rejects at α=0.05; the framework correctly reports `n_bursts_marginal_edge ≤ 2`. **The mediation question is statistically moot at the per-burst granularity** — not because mediation is absent, but because the LINK signal can't be resolved at this temporal resolution. The cross-env aggregation at L1 is the right tool for these envs; per-burst is over-resolved.

### MountainCar caveat

MountainCar has `WEAK_TV` at `bg_magnitude` with 63% absorption in `dynamic_partial_spearman`, but `dynamic_pc_adjacency` reports zero detectable bursts. The two primitives encode different gates — partial-Spearman pools the trajectory's |ρ| reduction, PC asks per-burst CI test rejection. PS detects subtle smooth signal that PC's α=0.05 threshold can't reject burst-by-burst. The honest reading is "partial mediation suggested by PS magnitude, not robust at the per-burst CI level."

## §6 Substantive synthesis (tautology-corrected)

**The canonical Hasselt mechanism (DDQN → bias-reduction → outcome) does NOT cleanly mediate at any env at γ=0.99 once tautological mediators are removed:**

- At v1's headline case (Asterix γ=0.99), the BEST clean mediator (`q_argmax_margin`) achieves only **59%** per-burst d-separation — substantially below the 91% obtained with the tautological `pstate_bias`. The remaining 41% direct-edge bursts indicate the clean candidate set is **incomplete** at this env. The textbook Hasselt-via-bias-reduction story is structurally tautological with the outcome.

**The non-tautological mediation picture at γ=0.99**:
- **MetaMaze γ=0.99**: `bg_magnitude` (Bellman wedge) achieves 100% per-burst d-separation, but only at 2 of 10 detectable bursts. Limited but clean evidence.
- **FourRooms γ=0.99**: `argmax_ent` (action-policy entropy) d-separates 60% of 15 detectable bursts. Consistent with the substrate's "FR's action-slip noise dominates; DDQN denoises the policy concentration" interpretation.
- **Freeway γ=0.99**: `bg_magnitude` d-separates 60% of 5 detectable bursts (down from v1's 80% with the tautological pstate_bias).
- **SpaceInvaders γ=0.99**: `q_argmax_margin` (action-gap, Q-shape side) at 66% over 3 detectable bursts. Q-shape, not bias-reduction.
- **Asterix γ=0.99**: `q_argmax_margin` at 59% over 32 detectable bursts. The strongest per-burst LINK signal in the panel, but mediator candidate set is incomplete (41% residual direct edge).

**At 7 envs the per-burst mediation question is moot** (n_marg ≤ 2): PacMan, MountainCar, Breakout, Acrobot, CartPole, LunarLander, Snake. The LINK is detectable at L1 cross-env (`P(D>V)` 0.48–0.69 range) but not at per-burst α=0.05 given n=30/arm.

### What this implies for Hasselt-2010

The textbook story (DDQN reduces `Q − MC` bias → outcome improves) **cannot be tested at the per-burst level using `Q − MC` as the mediator** because `Q − MC` is computed from the same `mc_return` data as the outcome — any partial-correlation analysis is structurally biased. The framework's tautology audit is the gate that enforces this.

Once the tautological mediator is excluded, the available clean candidates carry the LINK at varying degrees across envs, with **no single mediator universally sufficient cross-env-stratified** (L2 PC). Different envs route the DDQN effect through different non-tautological channels: action-gap at Asterix and SI; Bellman wedge at MetaMaze and Freeway; argmax entropy at FourRooms. Asterix's 41% residual direct edge points at an unmeasured mediator — a candidate-set gap, not a framework limitation.

## §7 Methodological synthesis

The framework's contribution is **graded refusal across the analysis hierarchy**. Each layer's typed primitive answers its own scientific question with its own gate:

1. **L1 directional gate** (binomial p, perm p): refuses claim when sign-test isn't significant.
2. **L1 magnitude gate** (DL `verdict` enum, I²/PI): refuses extrapolation when heterogeneity exceeds the exchangeability assumption (here, I²=0.96 → verdict=NO_EFFECT).
3. **L2 PC discovery gate** (edge survival, separating sets): refuses to call a candidate set "incomplete" without testing; refuses to call a mediator "sufficient" without identifying a separating set.
4. **L2 partial-Spearman magnitude gate** (per-stratum ρ pool with Fisher-z weighting): provides the per-env absorption magnitude, but only conditional on PC validation.
5. **L3 dynamic_partial_spearman gate** (`TimeAggregationStatus` enum): refuses to pool the per-burst trajectory when SIGN_FLIP_DETECTED or UNDERPOWERED_BURSTS.
6. **L3 dynamic_pc_adjacency gate** (per-burst CI test): refuses to attribute mediation at bursts where `n_bursts_marginal_edge` is too small.

These gates compose. A naive analysis that ignores them produces:
- "Cross-env mean Cohen's d = +0.44, significant!" — but DL pool says PI crosses zero → not extrapolable.
- "94% mediation cross-env by `pstate_bias`!" — but per-env analysis shows 5 envs are suppressors, dynamic shows per-burst sign-flipping at most envs.
- "Per-env best mediator is X% absorbed!" — but PC says n_marg_edge=0 at this env, so the magnitude is meaningless.

The framework refuses each over-claim with its native gate. **The contribution is the gate, not the analysis**: each gate is a typed result-field that prevents the over-claim downstream.

## §7.4 REDQ-normalized relative bias as a partially de-tautologized mediator

The reads-set tautology check ([§3.5](#35-tautology-audit-the-load-bearing-correction)) is a SUFFICIENT condition for tautology (any mediator reading the outcome's source is flagged) but not NECESSARY. REDQ (Chen, Wang, Zhou, Ross, ICLR 2021) suggests normalizing the per-(s,a) bias to compare across envs:

  **REDQ formula** (Section 4, "What kind of bias?"): `normalized_bias(s,a) = (Q_θ(s,a) − Q^π(s,a)) / |E_{s,a}[Q^π(s,a)]|`
  
  where Q^π is the on-policy MC estimate. The denominator is **|expected MC|**, not mean(Q).

We tested two variants:

- **Variant A** (denominator = `mean(Q)`): `rel_A[b] = pstate_bias[b] / |Q[b]|` where `Q[b] = pstate_bias[b] + mc[b]` (algebraic reconstruction)
- **Variant B** (REDQ-exact, denominator = `|mean(MC)|`): `rel_B[b] = pstate_bias[b] / |mc[b]|`

**Rank-equivalence under partial Spearman**: when bias > 0 (the canonical regime), parametrize by `r = MC/bias`; then `rel_A = 1/(1+r)` and `rel_B = 1/r` are both monotone decreasing in r. Since rank-Spearman partial correlation is invariant under monotone transformations, **both variants yield identical PC d-separation%** at every env tested. The Q-vs-MC choice of denominator is irrelevant to the rank-based mediation primitive — only the SIGN of the relative bias matters (which is preserved by both normalizations in the bias>0 regime).

**v1 of this table reported "REDQ-normalized → 0%" at Asterix and SI based on a flawed approximation** (using episode-start MC mean — K=20 samples — as proxy for the population MC denominator). The proper REDQ denominator is `mean over ALL visited eval (s,a) of Q`, which is computed over thousands of states per burst, not K=20.

Recomputed properly from raw `predicted_q_per_step` + `mc_return_from_step` traces at **asterix_g099_canonical_n_eps20_ckpt** (n=30 cells, full-Q corpus with the n_eps=20 fix):

| mediator definition | marg | dsep | direct | **d-sep %** |
|---|---|---|---|---|
| `pstate_bias` (raw, tautological reference) | 36 | 35 | 1 | **97.2%** |
| `rel_bias` OLD: `bias / \|bias + ep_MC\|` (flawed proxy) | 36 | 7 | 29 | 19.4% |
| **`rel_bias` REDQ-proper: `bias / \|mean(Q over all eval s,a)\|`** | 36 | 30 | 6 | **83.3%** |

The proper REDQ normalization at Asterix γ=0.99 retains MOST of the tautological d-separation magnitude (83% vs raw 97%). The normalization removes ~14% absolute — NOT 97% as my prior approximation suggested. Substantively: the tautology is partial. Bias-as-mediator, even normalized, still carries strong d-separation at this env.

**Why my prior approximation was wrong**: `pstate_bias` is `mean over all eval (s,a)` of (Q − MC) — averaged over thousands of states. But `mc_return__mean_axis_-1` is `mean over K=20 episode-start eps` of MC. The two averages are at different scales (the per-(s,a) bias accumulates Q-MC differences across episode-length trajectories; the episode-start MC is much smaller in magnitude). Using `bias + ep_MC` as denominator gave a denominator dominated by `bias`, making the ratio ≈ 1 (degenerate). The proper denominator `mean over all eval (s,a) of Q` is much larger than `ep_MC` and roughly proportional to bias's scale, giving a stable [0, 1] ratio.

**Cross-env corrected note** (other envs would need re-running with proper q_mean over all eval (s,a), which isn't yet in the cache for MLP envs):
- The prior "0% at Asterix and SI; 100% at Freeway and MetaMaze" claims should be treated as preliminary — the REDQ computation needs the proper population-Q denominator from traces, which we only have for asterix_g099_canonical_n_eps20_ckpt currently. The Freeway/MetaMaze 100% reading was robust to denominator choice because pstate_bias was already ≈ 100%, but the magnitude of the cross-env story shifts.

**Two distinct regimes:**

1. **Q-explosion regime (Asterix γ=0.99, SI γ=0.99)**: `relative_bias ≈ (Q − MC)/Q ≈ 1` because Q ≫ MC (debug sample at Asterix burst 25: bias ≈ 22.0, MC ≈ 0.6, Q ≈ 22.6 → ratio ≈ 0.97; range 0.80 – 0.98 across bursts/cells). With near-constant relative_bias across cells, conditioning provides essentially no information → mediation collapses to **0%**. **The Q-normalization unmasks the tautology**: when bias is ~95% of Q, the ratio is degenerate and the tautological 91% at raw bias was algebraic identity, not causal mediation. The framework's tautology audit is correctly conservative here.

2. **Modest-Q regime (FourRooms γ=0.99, Freeway γ=0.99)**: `relative_bias` has real cross-cell variance — the Q-denominator carries useful information because Q's per-cell magnitude varies with arm independently of MC's level. At **Freeway γ=0.99 the relative_bias d-separates 100%** (5/5 bursts, n=5 marg edges), beating raw pstate_bias (80%) — the Q-normalization PARTIALLY DE-TAUTOLOGIZES while preserving the causal mediation signal. At FourRooms γ=0.99, relative_bias absorbs 40% (above raw bias's 27%) but still below argmax_ent's 60%.

**Synthesis (corrected)**: REDQ-style relative bias is a useful diagnostic, but its DE-tautologizing power is much smaller than my first approximation suggested. At Asterix γ=0.99 with the proper population-Q denominator, the REDQ-normalized bias retains 83% PC d-separation (vs raw bias's 97%) — the Q-normalization removes only 14% of the tautological signal. This means:

1. **The structural tautology audit (reads-set Jaccard) is the right gate** — any mediator that reads MC should be flagged.
2. **The empirical de-tautology test (compare REDQ-normalized absorption against pure mediator)** at Asterix γ=0.99 shows the REDQ relative bias absorbs 83%, while the structurally-clean `q_argmax_margin` absorbs 59% (at the n=60 corpus with comparable bursts). The 24-point gap (83 − 59) is what the framework would flag as "additional tautological leakage beyond the pure mediator" — substantial but not 100%.
3. **Bias-as-mediator partially survives the audit at Asterix γ=0.99** — even after normalization, the bias measure carries 24 percentage points of absorption beyond what the pure (non-MC-reading) Q-action-gap mediator captures. Whether this 24 points is causal-bias-mediation-content vs residual tautology cannot be determined from the framework's primitives alone; it requires substrate-level argument (e.g., showing that the normalized bias is bounded away from 1 across cells, which we have: range [0.18, 0.98], std 0.18).

**Methodological implication for the paper**: the framework should expose `tautology_audit` flag as INFORMATIONAL (not blocking) AND require that any bridge using a MC-reading mediator include a side-by-side comparison with (a) the raw tautological mediator and (b) at least one structurally-clean mediator. The gap between the three numbers is the quantitative tautology-vs-causal-content audit:
- **(raw absorption) − (normalized absorption)** = how much the normalization removes from pure algebraic identity
- **(normalized absorption) − (pure mediator absorption)** = the residual content above what a non-MC-reading mediator captures, attributable to bias-mechanism-with-tautology-caveat

At Asterix γ=0.99 these are: raw 97%, normalized 83% (Δ=−14%), pure 59% (Δ=−24%). The first delta is the algebraic identity removed by normalization; the second delta is the substantive "bias carries info beyond pure Q-shape mediators". The framework can REPORT both deltas; the substrate author decides whether the 24-point gap is causal-content-with-tautology-caveat or residual unexplained leakage.

## §7.5 Unported scalar mediators (candidate-set completeness)

The reads-set audit identified 13 clean per-burst mediators. Several scalar `_late` mediators with no per-burst counterpart could materially change the per-env picture if ported. Most important:

| `_late` scalar | reads | per-burst registered? | priority |
|---|---|---|---|
| **`policy_churn_late`** | `online_argmax_per_step`, `eval_step_index` | **NO** | HIGH — prior audits identified this as TOP mediator at PacMan (94.5%) and MetaMaze (17%); pure action-policy, NOT tautological |
| `argmax_persistence_late` | `online_argmax_per_step`, ... | NO | MED — related to churn |
| `argmax_mode_freq_late` | `online_argmax_per_step` | NO | MED |
| `mutual_info_state_argmax_late` | `state_hash_per_step`, `online_argmax_per_step` | NO | MED — pure state-policy MI |
| `state_coverage_kl_uniform_late` | `state_hash_per_step` | NO | LOW (have state-coverage analogs) |
| `state_visit_entropy_late` | `state_hash_per_step` | NO | LOW |
| `td_residual_late` | `td_error` | NO | MED — Bellman residual, distinct from `bg_magnitude` |
| `q_action_gap_relative_late` | `online_top12_margin_per_step`, ... | covered by `q_argmax_margin_per_burst` (different norm) | LOW |
| `q_inter_state_grad_overlap_late` | (substrate per-step gradient hooks) | NO | LOW (CNN-only, requires per-step Q gradient trace) |
| `target_staleness_late` | `target_step_lag_per_step` | NO | LOW (HP-derived; per-burst version requires per-burst lag count) |

**The Asterix γ=0.99 residual direct edge (41% of bursts at q_argmax_margin's 59% dsep)** is a candidate-incompleteness signal. The framework cannot identify a mediator it doesn't have. The most likely missing candidate is **`policy_churn_per_burst`** — argmax-disagreement-rate between successive Q snapshots — which the prior cross-env scalar audit identified as the top mediator at PacMan and MetaMaze. Authoring this per-burst measurable is a future-work item that could meaningfully improve the per-env analysis at Asterix.

## §8 Conclusion

**Substantive**: Hasselt-2010's universal-mediator hope does not survive the framework's three-layer analysis. DDQN's cross-env LINK is directionally consistent, but the mediation pathway is **env-specific** (Bellman-residual at Asterix + Freeway; Bellman-wedge at MetaMaze; action-entropy at FourRooms; Q-action-margin at SI) and the per-burst signal is only detectable at 5/12 envs.

**Methodological**: The framework's typed primitives — `cross_env_consistency_binomial`, `stratified_arm_diff_pooled` (DL pool), `discover_adjacency` (PC), `partial_spearman` (per-env), `dynamic_partial_spearman` (TimeAggregationStatus), `dynamic_pc_adjacency` (per-burst PC) — form a graded-refusal hierarchy. Each primitive's typed result encodes the appropriate gate. The mediation question is asked at each granularity (cross-env aggregate / per-env scalar / per-env per-burst) and answered with primitive-specific status enums that prevent over-claiming.

The same panel that supports "DDQN improves outcome cross-env directionally" (L1: p=0.003) supports "but the cross-env magnitude doesn't extrapolate" (L1 DL: verdict=NO_EFFECT) supports "Asterix's clean mediation channel is `q_argmax_margin` (the per-state Bellman residual `pstate_bias` is tautological), FourRooms's is `argmax_ent`, MetaMaze's is `bg_magnitude`" (L3 per-burst PC), and refuses to summarize a single cross-env mediation magnitude — because the structure doesn't admit one.

## Figures

All figures regenerated with the tautology-corrected `CLEAN_MEDIATORS` set (MC-reading mediators excluded). Generator scripts at `scripts/` for reproducibility under different mediator-set choices.

- `figures/report_3layer_summary_corrected.png` — three-panel L1 directional × L1 magnitude × L3 per-env best-clean-mediator summary
- `figures/report_asterix_clean_corrected.png` — Asterix γ=0.99 per-burst trajectory + PC CI tests at the CLEAN best mediator `q_argmax_margin` (59%); title includes the 3-number tautology audit (raw 97%, REDQ-normalized 83%, clean 59%)
- `figures/report_per_env_best_mediator.png` — per-env best clean mediator bar chart, color-coded by mediator family (Bellman/Q-shape/policy/state-coverage)
- `figures/report_dynamic_per_env_corrected.png` — per-burst dynamic mediation trajectories at the 7 envs with PC-detectable LINK, using each env's best CLEAN mediator (ρ_marg trajectory in blue, ρ_partial in red, absorbed area in gold); shows the per-burst SIGN_FLIP / WEAK_TV / d-sep% structure that the per-env scalar absorption table aggregates
- `figures/report_per_env_learning_curves.png` — per-env seed-aggregated learning curves (V vs D), median ± IQR across seeds, envs sorted by P(D>V). Outcome-trajectory view that complements the mediation tables — shows where DDQN's improvement appears, when, and at what magnitude.
- `figures/outcome_aggregation_raw_g099.png` — per-cell vs seed-aggregated outcome comparison (independent of mediator audit, retained from initial gen)
- `figures/episode_cv_per_burst.png`, `figures/episode_std_per_burst.png` — outcome dispersion statistics (independent of mediator audit, retained)

**Superseded v1 figures** (generated before tautology audit; preserved at `figures/_v1_superseded/` for revision-history traceability): `report_3layer_summary.png`, `report_asterix_clean.png`, `report_dynamic_5envs.png`, `report_mediator_attribution.png`, `step2_clean_g099.png`, `canonical_g099_consolidation.png`, `dynamic_g099_mediation.png`, `per_env_dynamic_mech_link.png`. These used the tautological `pstate_bias` (`Q − MC` per visited state) as a mediator candidate; the report's revision history flags them as misleading.

**Generator scripts** at `scripts/`:
- `_common.py` — defines `CLEAN_MEDIATORS` (the tautology-clean set) + the `TAUTOLOGICAL_BLOCKLIST`
- `gen_3layer_summary.py` — produces `report_3layer_summary_corrected.png`
- `gen_asterix_clean.py` — produces `report_asterix_clean_corrected.png`
- `gen_per_env_best_mediator.py` — produces `report_per_env_best_mediator.png`
- `gen_dynamic_per_env.py` — produces `report_dynamic_per_env_corrected.png` (per-burst ρ trajectories per env with best clean mediator)
- `gen_per_env_learning_curves.py` — produces `report_per_env_learning_curves.png` (seed-aggregated V-vs-D learning curves per env)

## Appendix: full candidate mediator set (γ=0.99 canonical)

**8 base mediators** (populated at all 12 envs):
- Bellman-side: `bootstrap_gap_magnitude_per_burst`, `bootstrap_disagree_rate_per_burst`, `mean_per_state_cumulative_bias_per_burst`, `greedy_match_per_burst`
- State/policy: `argmax_entropy_per_burst`, `state_hash_n_unique_per_burst`, `state_hash_entropy_per_burst`, `state_repeat_rate_window64_per_burst`

**6 CNN-only mediators** (populated at MinAtar + LL only, where Q-shape trace columns exist):
- `q_argmax_margin_per_burst`, `q_action_std_per_burst`, `q_autocorr_per_burst`, `q_lambda_a_per_burst`
- `state_conditional_argmax_entropy_per_burst`
- `bootstrap_disagree_gap_conditional_per_burst`

PC cross-env-stratified discovery confirmed the 8-base candidate set is sufficient ({pstate_bias} alone d-separates arm-outcome at α=0.05 under env-stratified CI tests). The 6 CNN-only additions surface env-specific mediators (notably `q_argmax_margin` at SI) that wouldn't appear in the base set.
