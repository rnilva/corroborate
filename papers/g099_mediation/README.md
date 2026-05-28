# DDQN at γ=0.99: a 5-layer framework demonstration

This directory contains the figures and scripts for a case study
of Hasselt's Double DQN (DDQN) on a 12-env γ=0.99 canonical panel,
structured as five layers that each surface a different framework
discipline. The paper's contribution is the *framework's verdict
surface* and its per-stratum honesty, not adjudicating DDQN
specifically.

## Layout

```
papers/g099_mediation/
├── README.md          ← you are here
├── report.md          ← paper draft skeleton (5 sections, one per layer)
├── run_all.sh         ← reproduce every figure end-to-end (~2 min)
├── scripts/
│   ├── _common.py     ← shared panel loader, env catalogue, constants
│   ├── 01_mech_per_env.py
│   ├── 02_outcome_per_env.py
│   ├── 03_static_mediation.py
│   ├── 04_aggregation_danger.py
│   ├── 05_dynamic_mediation.py
│   └── _deprecated/   ← retired adjudication forensics + earlier drafts
└── figures/
    ├── 01_mech_per_env.{png,csv}
    ├── 02_outcome_per_env.{png,csv}
    ├── 03_static_mediation.{png,csv}
    ├── 04_aggregation_danger.{png,csv}
    ├── 05_dynamic_mediation.{png,csv}
    └── _deprecated/   ← prior iterations
```

## The 5 layers

1. **MECH per-env** — Does DDQN reduce overestimation bias?
   Per-env independent-samples Cohen's d on `jensen_gap`,
   DerSimonian-Laird random-effects pool. The framework refuses to
   commit to a cross-env pool point estimate when between-env
   heterogeneity is large (I²=0.97 here).
   *Primitive*: `stratified_arm_diff_pooled`.

2. **OUTCOME per-env** — Does the bias reduction translate to
   outcome improvement? Per-env d on `eval_late_burst_raw_mean`
   (steady-state) and `eval_best_burst_raw_mean` (peak), plus
   cross-env `P(D > V)` via Mann-Whitney aggregation.
   *Primitives*: `stratified_arm_diff_pooled`,
   `cross_env_probability_of_improvement`.

3. **STATIC mediation** — Per-env partial Spearman:
   `ρ(arm, outcome | jensen_gap)`. Asks the literature's natural
   mediation question. Soft-tautology note: `jensen_gap` shares
   MC inputs with the outcome — clean Bellman-residual mediator
   is used at Layer 5.
   *Primitive*: `partial_spearman`.

4. **AGGREGATION DANGER** — Two dramatic per-env examples (PacMan
   + Asterix γ=0.99) where per-burst dynamic mediation reveals
   what pooled mediation hides. The framework's
   `TimeAggregationStatus` enum + DL τ²/I² catch SIGN_FLIP_DETECTED
   pathology where naïve pooled mediation reports a meaningful
   percent (Simpson's-paradox artifact).
   *Primitive*: `dynamic_partial_spearman` with `n_bootstrap=1000`.

5. **DYNAMIC mediation + cluster aggregation** — Per-env trajectories
   across all 12 envs with the framework's clean (Bellman-residual)
   mediator `bootstrap_gap_magnitude_per_burst`. Each env tagged with
   its TimeAggregationStatus; cluster status counts surface how
   many envs share each regime.
   *Primitive*: `dynamic_partial_spearman` with `n_bootstrap=1000`.

## Reproduction

```bash
bash papers/g099_mediation/run_all.sh
```

Requires the cached panel `experiments/data/cache/hasselt_clean.parquet`
(and its `.sources.json` sidecar). Scripts are deterministic
(bootstrap seed = 42).

## Why these primitives

Per CLAUDE.md "Canonical analyses" guide:

- `stratified_arm_diff_pooled` is the principled cross-env
  aggregator for paired-arm contrasts. Independent-samples
  Cohen's d per stratum, DL pool — heterogeneity-honest by
  construction.
- `cross_env_probability_of_improvement` is the framework's
  Mann-Whitney-based cross-env aggregator that doesn't assume
  Gaussian effect-size pooling (Agarwal et al. 2021 P(X>Y) form).
- `partial_spearman` is the canonical mediation primitive
  (unified per-cell + per-burst, depth-0 marginal / depth-1
  closed-form / depth-k OLS-residual).
- `dynamic_partial_spearman` is the canonical RL-substrate
  per-burst mediation primitive (CLAUDE.md vocabulary section).
  Its `TimeAggregationStatus` enum + DL `i2` are the typed
  surface for trajectory heterogeneity.
- Cluster-bootstrap CI (`n_bootstrap > 0`) is the framework's
  assumption-free alternative to DL's parametric PI bounds when
  within-cell autocorrelation matters.

`mediator_leak_adjudication` (the soft-tautology adjudication
primitive) is intentionally absent from this paper's headline.
It remains in the framework as a diagnostic tool for substrate
authors; its forensic application to the bias mediator is in
`scripts/_deprecated/` as historical context.
