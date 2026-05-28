# Per-stratum verdicts beat cross-env pooling: a case study in DDQN

**Substrate:** Hasselt's Double DQN (DDQN) on a canonical
12-environment γ=0.99 panel (`hasselt_clean` cache, 680 cells,
n=10–30 seeds per arm per env).

**Question:** Does the textbook causal story — *DDQN reduces
overestimation bias → improved outcome* — hold under per-stratum
honest verdicts?

**Answer (preview):** The MECH layer holds broadly; the OUTCOME
layer holds modestly; the LINK layer is dramatically env-dependent;
and cross-env pooling routinely launders sign-flip pathology into
spuriously meaningful aggregates. The framework's per-stratum
verdict surface + per-burst `TimeAggregationStatus` enum catch the
pathology automatically.

This is a paper about the framework's verdict surface, illustrated
on DDQN. Each section demonstrates one framework discipline.

---

## Layer 1 — MECH: does DDQN reduce bias?

**Question:** Per env, is `jensen_gap` (DDQN's claimed mechanism
target, clamped Q − MC) lower under D than V?

**Method (mirrors `hasselt_clean/chain.py`):**
- Panel scope: `CANONICAL_DORMANCY_SCOPE & PREMISE_ACTIVE_PER_STRATUM`.
- **Bridge verdict primitive**: `cross_env_consistency_binomial` —
  one-tailed binomial sign-test on the per-env Cohen's d panel
  against the predicted direction (DDQN reduces bias → `d < 0`).
  This is exactly the primitive that the chain.py bridge
  `ddqn_reduces_bias__consistently_cross_env` fires.
- **DL diagnostic**: `stratified_arm_diff_pooled` — independent-
  samples Cohen's d per env with DerSimonian-Laird random-effects
  pool. Reported as a diagnostic to surface heterogeneity, not
  as the bridge verdict.

**Result:**

- **Bridge MECH verdict: HELD.** 11 of 12 envs in predicted
  direction; binomial sign-test p = 0.003.
- **DL pool diagnostic**: `d = −2.65` central but `I² = 0.97`,
  PI `[−6.78, +1.49]` → DL verdict NO_EFFECT (PI-honest refusal
  to commit to a generalisable cross-env point estimate under
  this heterogeneity).

The two verdicts answer different questions:
- *Bridge*: "do most envs reduce bias?" → yes (11/12).
- *DL pool*: "is the average effect generalisable to the next env?"
  → no, heterogeneity too high.

The chain.py bridge intentionally uses the binomial sign-test —
that's the right form for a cross-env consistency claim that
doesn't require env-exchangeability.

**Dormancy: empirically inactive at γ=0.99.** Verified directly
from restored cloud traces:

  - LL γ=0.99 cell 0:  σ_late = 0.41 → floor = 0.68; observed bias = 45.6
  - CP γ=0.99 cell 0:  σ_late = 1.69 → floor = 1.99; observed bias = 121

V's observed overestimation is **50-70× larger than the Jensen
structural floor** `σ_Q × √(2 log |A|)` in every env, which is
why `jensen_dormancy_gap = max(0, floor − observed) = 0` for all
340 cells. The `PREMISE_ACTIVE_PER_STRATUM` scope filter retains
every env at γ=0.99 because the premise is overwhelmingly active
— DDQN's Jensen-bias-clipping has plenty to correct, in fact
substantially more than Jensen alone would predict (the excess is
argmax-correlation overestimation stacked on top of Jensen variance
overestimation).

(The cached `jensen_dormancy_premise_active='power_insufficient'`
string in the parquet is stale from a prior ingest where σ_Q
traces hadn't been restored; the underlying scalar `jensen_dormancy_gap`
that the scope filter actually reads is correct. At γ=0.999 with
its longer effective horizon, the filter actively excludes
LunarLander + other envs where the bias-vs-floor inequality
flips.)

→ `figures/01_mech_per_env.png` and `.csv`

---

## Layer 2 — OUTCOME: does the bias reduction translate to improvement?

**Question:** Per env, is `eval_late_burst_raw_mean` (steady-state
return) higher under D than V? And `eval_best_burst_raw_mean`
(peak)? And cross-env, what's `P(D > V)`?

**Method:** `stratified_arm_diff_pooled` on both outcome metrics,
plus `cross_env_probability_of_improvement` (Agarwal 2021 form,
permutation-tested).

**Result:** Per-env outcome effects are dispersed and small. DL
pool: `d=+0.35` (late) / `+0.33` (peak); both with high I²; both
verdict `NO_EFFECT`. Cross-env `P(D > V) = 0.58` (12 strata) —
DDQN wins on ~58% of envs, but permutation p is non-significant
at this small n.

**What the framework refuses:** treating "DL d=+0.35, peak
agreement, P=0.58" as evidence of "DDQN works on average."
Heterogeneity-honest verdict says: this n=12 panel has too much
between-env variance to support a generalisable outcome claim.

**Companion figure — per-env learning curves**
(`figures/02b_learning_curves.png`): the per-burst trajectory
behind the scalar table. Each panel shows V vs D median across
seeds + IQR + 5/95 envelope, with mean per-seed peak (♦) and the
late-30% window (gold band). Title shows P(D>V) under both
metrics: **6 of 12 envs are metric-sensitive (↕)** — their
peak-vs-late30 P(D>V) disagree on sign (SpaceInvaders, MetaMaze,
MountainCar, Acrobot, FourRooms, Snake, CartPole). This is the
methodological finding the framework's dual-metric reporting
discipline (peak ∥ late30) is designed to surface.

→ `figures/02_outcome_per_env.png` and `.csv` (per-env Cohen's d table)
→ `figures/02b_learning_curves.png` (per-burst trajectory panel)

---

## Layer 3 — STATIC mediation: does conditioning on bias remove the arm→outcome signal?

**Question:** Per env, what's the marginal Spearman ρ(arm, outcome)?
What's the partial ρ(arm, outcome | jensen_gap)? Does conditioning
on bias change the sign or magnitude?

**Method:** `partial_spearman` per env, x=arm_code, y=outcome_late,
conditioning=(jensen_gap,).

**Result:** Per-env story is heterogeneous. Asterix has the largest
marginal ρ=+0.73, with conditioning on jensen_gap absorbing 56% of
it. FourRooms (marg=+0.53) absorbs 58% AND flips sign. Most other
envs show small marginal ρ and inconsistent absorption.

**Note on soft tautology:** `jensen_gap = max(0, mean(Q − MC))`
shares MC inputs with the outcome. The diagnostic primitive
`mediator_leak_adjudication` is available for per-env certification
against MC-leak (GENUINE / LEAK / UNDERPOWERED_FOR_GENUINE), but
this layer reports the literature-natural reading. Layer 5 uses
the framework's clean Bellman-residual mediator.

→ `figures/03_static_mediation.png` and `.csv`

---

## Layer 4 — AGGREGATION DANGER: per-burst dynamic mediation reveals what pooled mediation hides

**Question:** Take two envs from the panel. What does a naïve
cross-env aggregate mediation% claim, vs the per-burst trajectory?

**Method:** `dynamic_partial_spearman` with mediator =
`bootstrap_gap_magnitude_per_burst` (clean — Bellman residual, no
MC-leak), `n_bootstrap=1000` for cluster-bootstrap CI.

**Result:**

- **PacMan γ=0.99**: naïve cross-env mediation % could report 73%.
  Per-burst trajectory: marginal ρ SWITCHES SIGN at mid-training.
  Framework verdict: `SIGN_FLIP_DETECTED`. DL pool: ρ=+0.027,
  **I²=0.00** — no detectable effect at all. The "73% mediation"
  was the Simpson's-paradox average over opposite-sign bursts.

- **Asterix γ=0.99**: naïve cross-env mediation might report 31%.
  Per-burst trajectory shows strong heterogeneity. Framework
  verdict: `SIGN_FLIP_DETECTED`. DL pool: ρ=+0.34, **I²=0.70** —
  substantial heterogeneity. PC-style analysis localizes: bg
  mediates the arm→outcome edge in only 6 of 32 marg-edge bursts;
  direct edges persist in 26.

**Framework contribution:** the `TimeAggregationStatus` enum +
DL τ²/I² together flag both pathologies. SIGN_FLIP_DETECTED catches
sign reversal; high I² catches between-burst variance even when
direction is consistent. Pooled mediation % alone catches neither.

→ `figures/04_aggregation_danger.png` and `.csv`

---

## Layer 5 — DYNAMIC mediation + cluster aggregation across the panel

**Question:** Across all 12 envs, per-burst, with the framework's
canonical clean mediator: what does the trajectory look like?
How heterogeneous are the regimes?

**Method:** `dynamic_partial_spearman` over all 12 envs with
`mediator_per_burst='bootstrap_gap_magnitude_per_burst'` (Bellman
residual, MC-clean), `min_n_per_burst=8`, `n_bootstrap=1000`.

**Result:** Of 12 envs, **11 SIGN_FLIP_DETECTED, 1 WEAK_TIME_VARYING.**
Per-env DL pool ρ and bootstrap CI are reported per env; aggregate
sign-flip prevalence is itself the headline finding. Per-burst
heterogeneity is the rule, not the exception.

**What the framework refuses:** cross-env pooling of these per-env
ρ values into a single "mediation %." The 11/12 SIGN_FLIP
prevalence is the answer — any cross-env meta-aggregate would
inherit that pathology.

→ `figures/05_dynamic_mediation.png` and `.csv`

---

## What the framework's per-stratum surface bought us

1. Layer 1's NO_EFFECT verdict despite `d=-2.65` central estimate
   — heterogeneity-honest refusal to over-claim from n=12 envs.
2. Layer 2's `P(D > V) = 0.58` honestly reported with permutation
   p as descriptive, not significance-tested at this n.
3. Layer 3's per-env panel reveals Asterix as a high-ρ outlier
   that drives any cross-env aggregate; FourRooms's sign-flip
   under conditioning is per-env visible.
4. Layer 4's dramatic SIGN_FLIP demonstration at PacMan: 73%
   pooled mediation washes to ρ=+0.027 / I²=0 under
   `dynamic_partial_spearman`. The framework caught what the pool
   laundered.
5. Layer 5's panel: 11/12 SIGN_FLIP_DETECTED is itself a
   substantive finding about RL training trajectories — they are
   structurally non-stationary in mediator-outcome relationship,
   and any meta-aggregate that pools across bursts is suspect.

The framework's contribution is the *typed verdict surface*:
`POWER_INSUFFICIENT`, `NO_EFFECT`, `SIGN_FLIP_DETECTED`,
heterogeneity-flagged HELD, etc. as first-class enum values that
substrate authors can read instead of stitching together p-values
and confidence intervals manually.

---

## Reproducing this paper's analysis

```bash
bash papers/g099_mediation/run_all.sh
```

Output: figures and CSVs in `papers/g099_mediation/figures/`.
Reproduction time: ~2 min on the cached panel. All scripts are
deterministic (bootstrap seed = 42).

## See also

- `CLAUDE.md` "Canonical analyses" section — the framework's
  preferred analysis primitives and when to reach for each.
- `HYPOTHESIS_AS_GRAPH.md` — the per-stratum + cluster-shaped
  causal-claim discipline this report applies.
- `experiments/findings/hasselt_clean/` — the bridge-level
  encoding of these same questions, used by the framework's
  hypothesis runner.
