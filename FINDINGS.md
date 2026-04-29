# Findings — DDQN case study

Empirical findings from the DDQN acceptance corpus. Dated entries
to track when claims were authored vs. observed.

---

## 2026-04-29 (third revision) — Tautology-audit reveals most "solve predictors" on the CartPole HP corpus are HP-shadow false-positives.

### Methodology

`redundancy_check.audit_mediator_panel` was extended with three
independent checks:

1. **Outcome-tautological** (`flagged_outcome`): structural reads-
   set jaccard between the mediator and the outcome's source
   columns. Flagged when ≥ 0.5.
2. **HP-deterministic** (`flagged_hp`): per-axis OLS R² of mediator
   on each HP. Flagged when ≥ 0.95.
3. **HP-shadow / no-residual-signal** (`flagged_no_residual_signal`):
   stratified Spearman ρ(mediator, outcome | HP-stratum) using
   `causal_discovery.stratified_spearman_rho`. Flagged when |ρ| <
   0.1 AND p ≥ 0.05 — within each HP stratum the mediator doesn't
   correlate with the outcome, so the marginal correlation is
   purely HP-mediated.

The third check replaced an earlier partial-Spearman version that
was systematically biased toward large ρ on small samples (rank
artifacts when within-stratum noise was small relative to
between-stratum signal).

### Result on CartPole HP corpus (180 cells, 36 configs × 5 seeds, vanilla DQN)

Outcome path: `outcome.eval_final_mean`. HP stratum: `replay.capacity`.

| Mediator | jaccard | strat ρ | strat p | Flags |
|---|---|---|---|---|
| `learning_curve_auc` | 1.00 | +0.52 | <0.001 | **OUTCOME** |
| `plateau_slope_late` | 1.00 | +0.44 | <0.001 | **OUTCOME** |
| `return_at_25pct_steps` | 1.00 | +0.05 | 0.524 | **OUTCOME / SHADOW** |
| `greedy_match_late` | 0.00 | −0.08 | 0.285 | **SHADOW** |
| `q_gap_late` | 0.00 | +0.05 | 0.507 | **SHADOW** |
| `q_max_growth` | 0.00 | −0.01 | 0.908 | **SHADOW** |
| `v_vs_max_delta_late` | 0.00 | +0.05 | 0.507 | **SHADOW** |
| `td_residual_late` | 0.00 | +0.10 | 0.168 | clean (borderline) |
| `td_within_batch_var_late` | 0.00 | +0.10 | 0.179 | clean (borderline) |
| `state_coverage_kl_uniform_late` | 0.00 | +0.19 | **0.011** | **clean ✓** |

### Reading

1. **Most "solve predictors" are mechanical, not causal.** Three
   are outcome-tautological (read from `mc_return`, the same
   trace column the outcome aggregates from); five more
   (including `greedy_match_late`, which I had earlier framed as
   the strongest scale-free predictor) are HP-shadow — their
   marginal correlation with solving comes entirely from the
   HP, with no residual within-capacity-stratum signal.

2. **`greedy_match_late`'s sign-flip across HP regimes** (which
   I'd noted earlier as a "wild interaction") is exactly the
   signature of HP-shadow. Within each capacity, ρ ≈ 0; the
   marginal cross-capacity correlation comes from the HP-driven
   regime change, not from greedy-match per se.

3. **`state_coverage_kl_uniform_late` is the only mediator with
   significant residual signal** (ρ=+0.19, p=0.011). Within each
   capacity stratum, agents whose late-training state visits are
   more concentrated relative to uniform are more likely to solve.
   Theoretical reading: solving cells have converged to a focused
   policy region; non-solving cells are still flailing.

4. **A new training-stability candidate** (`td_within_batch_var_late`
   — within-batch std of |TD-error|, added to the substrate) passes
   all three checks but has only borderline within-stratum signal
   (ρ=+0.10, p=0.18). Promising but underpowered at n=180.

### Implication for prior FINDINGS entries

The "consistent positive predictors of solving" listed in the
2026-04-29 morning entry (`learning_curve_auc`, `plateau_slope_late`,
`return_at_25pct_steps`, `greedy_match_late`) are **all false
positives** by the corrected three-check standard. The morning
entry's claim that `learning_curve_auc` was the strongest predictor
is now superseded — it was reading from `mc_return` (the outcome's
source), so the high g was a re-encoding, not a mediator.

This is the framework working as intended: the audit primitive
catches an analyst (me) conflating outcome-restatement with
causation, and HP-conditioning with causation.

### Honest scope

- **Within-capacity power is thin.** With 90 cells per capacity
  level and 5 seeds per config, the within-stratum ρ has wide
  CIs. Borderline mediators (`td_*` with ρ ≈ 0.10) need a fuller
  sweep before we can call them clean *or* shadow.
- **The audit is necessary, not sufficient.** Surviving all three
  checks means the mediator might be causal — it doesn't prove
  it. Only an interventional study that varies the mediator
  directly can establish that. For `state_coverage_kl_uniform_late`,
  a future study could add an exploration-bonus intervention to
  test whether forcing higher coverage improves solving.
- **The check at the corpus level depends on the HP grid spanning
  enough variation.** A corpus with one HP setting can't surface
  HP-shadow mediators by definition.

### Reproduction

```
# Compute mediators on the CartPole HP corpus:
uv run python experiments/cartpole_hp_sweep.py
uv run python -c "import compute_mediators; ..."  # path-overridden

# Apply the three-check audit:
from corroborate.redundancy_check import audit_mediator_panel
reports = audit_mediator_panel(
    panel, runs,
    outcome_reads=frozenset({'mc_return'}),
    hp_axes=('replay.capacity', 'replay.batch_size', ...),
    outcome_path='outcome.eval_final_mean',
    hp_stratum_axis='replay.capacity',
)
```

---

## 2026-04-29 (fourth revision) — DoWhy backdoor passes for state_coverage_kl, but direction of causation remains unresolved.

### Setup

Applied `bridges_dowhy.{backdoor_ate, placebo_refutation,
random_common_cause_refutation}` on:

- treatment: `mediator.state_coverage_kl_uniform_late`
- outcome: `outcome.eval_final_mean`
- DAG (caller-posited): every HP → both SCV and outcome
  (confounders); SCV → outcome (the hypothesis).
- HP backdoor adjustment set: `{capacity, batch_size, lr,
  sync_period}`.

### Result

| Check | Result | Verdict |
|---|---|---|
| `backdoor_ate` (SCV range [0.96, 2.93]) | ATE = +8.82 / SCV unit | **HELD** |
| `placebo_refutation` (permuted treatment) | placebo ATE = +0.12 (1.4% of real) | **HELD** |
| `random_common_cause_refutation` | drift = 0.0075 (synthetic confounder doesn't move estimate) | **HELD** |

CausalGraph after `promote_bridged_evidence`: all three edges
upgrade to `INTERVENTIONAL / causal_bridged`. State_coverage_kl
is the first mediator on the CartPole HP corpus that survives
*every* check the framework currently has:

1. Reads-set jaccard with outcome's source (= 0; not tautological).
2. HP R² (< 0.95; not deterministic in any HP).
3. Stratified Spearman within capacity stratum (ρ=+0.19, p=0.011).
4. Backdoor-adjusted ATE under the posited DAG (+8.82, HELD).
5. Placebo refutation (placebo ATE / real ATE = 1.4%).
6. Random-common-cause refutation (drift << tolerance).

### The caveat — direction of causation is not resolved

The backdoor adjustment is **rung-2-conditional-on-DAG**. The
posited DAG says SCV → outcome. The reverse DAG (outcome → SCV)
would produce identical observed correlations:

- **Forward**: low SCV (concentrated state visits) → agent stays
  near goal-region → outcome high.
- **Reverse**: outcome high (agent solves) → trajectory by
  construction stays in goal-region → SCV low.

For CartPole specifically, the reverse interpretation is
plausible: a successful 500-step pole-balancing episode naturally
produces a state-visit distribution concentrated near the
upright-pole region. SCV is then a *signature* of solving rather
than a *cause*.

Backdoor adjustment cannot distinguish forward from reverse — by
construction, observational data under either DAG produces the
same correlation matrix. The framework should be honest about
this: the dowhy verdict is "consistent with mediating ATE +8.82
*under this DAG*" not "we have proven SCV → outcome."

### What would distinguish the directions

1. **Temporal precedence**: does SCV stabilize *before* outcome
   stabilizes during training? Mid-training trajectories aren't
   currently logged at sufficient resolution to test this; would
   require per-eval-burst SCV computation.

2. **Interventional**: an intervention that forces SCV down
   (concentrated states) *without* directly improving the policy.
   Candidates: entropy regularization on policy, intrinsic-
   curiosity bonus that *spreads* state visits (and check whether
   the spread reduces solve rate), action-repetition / frame-skip
   modifications.

3. **Held-out generalization**: does the SCV → outcome ATE
   replicate on a different corpus (different env or different HP
   regime)? If so, structural; if not, idiosyncratic.

### Implication for the framework's gist

The framework's audit + dowhy chain is now end-to-end runnable:
audit selects clean candidates, dowhy backdoor verifies under the
DAG. What's still missing — and what the SCV story illustrates —
is the **direction-of-causation gap** between rung-2-conditional
and true causal claims. The framework can flag a mediator as
"survives all checks" but should explicitly mark this as
"observationally indistinguishable from reverse" until an
intervention or temporal evidence resolves it.

### Reproduction

```
uv run python experiments/dowhy_state_coverage.py
```

---

## 2026-04-29 (revised same day) — Substrate units bug + HP-conditioning of solve verdicts

### Bug

The framework's eval (`rl/dqn/eval.py:60`) records the *discounted*
Monte-Carlo return: `mc_return = Σ_t γ^t * r_t`. Literature solve
thresholds (gymnasium docs, Y&T 2019, Osband 2019) are *raw*
episodic return. Initial `SOLVE_THRESHOLDS` table used raw values
against the framework's discounted output — units mismatch.

Concrete symptom: CartPole-v1 baseline `outcome.eval_best_burst_mean
= 99.343` across all 200k corpus runs and across 178/180 cells in
a 36-config × 5-seed HP sweep. The exact value is `100 *
(1 - 0.99^500)` — the discounted return for a 500-step max-length
CartPole episode at γ=0.99. The agent IS reaching the optimal
policy at peak; it's just reported in discounted units. The raw
threshold of 475 (gymnasium docs) was unreachable in discounted
units (max ≈ 99.34).

### Fix

`rl/env_solve_thresholds.py` updated: each Tier-1 / Tier-2 entry
now carries the discounted-equivalent threshold at γ=0.99 with
the conversion documented in `source`:
- CartPole-v1: 475 → 99.0
- Acrobot-v1: -100 → -63.4
- MountainCar-v0: -110 → -67.3
- MinAtar (4): Y&T 50%-baseline × discount factor (≈ 0.2 for L=500)
- bsuite envs: slight downward adjustment (γ^L attenuation)

### Convergence-class shifts (200k corpus, baseline arm)

| Env | Old class | New class |
|---|---|---|
| Breakout-MinAtar | unsolved | **solved** (best=1.0, final=1.0) |
| CartPole-v1 | unsolved | **partial** (best=0.80, final=0.20) |
| MountainCar-v0 | solved | **unsolved** (-76.7 < tighter -67.3) |
| Acrobot-v1 | solved | solved |
| Catch / DeepSea / DiscountingChain / UmbrellaChain | solved | solved |
| Asterix / Freeway / MNISTBandit | unsolved | unsolved |

Solved subset is now 6 envs (was 6 — different composition).

### CartPole HP sweep — vanilla DQN solves CartPole with right HPs

Pre-registered hypothesis: a different HP regime should solve
CartPole at 200k *without* any new mechanism (DDQN, PER, n-step,
etc.). Tested on the 36-config × 5-seed sweep:

- **2 of 36 configs achieve final_solve = 1.0** — every seed
  stably solves at end of training. Both: `lr=1e-3, capacity=50k,
  sync=100`, varying batch_size ∈ {32, 64}.
- **11 of 36 configs have final_solve ≥ 0.5** — stable solve.
- **19 of 36 configs have best_solve ≥ 0.8** — peak is solved
  but degrades by final eval.
- The original corpus's HP config (`capacity=10000`,
  `batch_size=32`, `lr=1e-3`, `sync=100`) gets best_solve=1.0,
  final_solve=0.0 — **the agent solves CartPole at peak but
  catastrophically forgets**. Increasing replay capacity to 50k
  fixes the stability problem. Mechanism-level unchanged.

### Reading

1. **Vanilla DQN at appropriate HPs solves CartPole at 200k.**
   No new mechanism (DDQN, PER, n-step, dueling) is needed.
   Capacity=50k is the load-bearing HP for stability; the
   original corpus used 10k.

2. **The earlier convergence audit was contaminated by two bugs**:
   raw-vs-discounted units mismatch + HP-undertuning. The "5 of 13
   thresholdable envs are unsolved" finding from 2026-04-29 morning
   was over-stated. Corrected: 4 of 13 are unsolved (Asterix,
   Freeway, MNISTBandit, MountainCar), and at least CartPole is
   HP-fixable without mechanism changes.

3. **The §3 verdict pattern survives the correction.** On the
   new solved subset (6 envs), DDQN's mechanism g = -0.925, outcome
   g = -0.032 — directionally identical to the morning's finding.
   The headline conclusion (DDQN's bias-reduction activates but
   doesn't propagate to outcome) is robust to the threshold
   correction.

4. **Implication for any future DDQN claim**: the existing 200k
   corpus's results are conditional on `capacity=10k` HPs that
   don't reach stable solve on multiple envs. A re-run of DDQN at
   the better HPs (capacity=50k, etc.) might shift the outcome
   verdict. Future intervention studies should sweep HPs first to
   establish a converging baseline before testing mechanism-level
   interventions.

### Reproduction

```
# Convergence audit with corrected discounted thresholds:
uv run python experiments/convergence_audit.py --total-steps 200000

# CartPole HP sweep (15 min on CPU):
uv run python experiments/cartpole_hp_sweep.py
```

---

## 2026-04-29 — DDQN's bias-reduction mechanism activates on
## converged envs, but does not propagate to outcome.

### Setup

- Corpus: `runs_with_mediators.parquet`, 1080 cells at total_steps
  = 200000, 18 envs × 30 seeds × 2 arms (vanilla / DDQN).
- Outcome path: `outcome.eval_best_burst_mean` (Hasselt-convention
  reporting).
- Mechanism path: `mechanism.jensen_gap` (Hasselt 2010 §3 —
  empirical positive Jensen bias `max(0, mean(predicted_q_at_start
  − mc_return))` over eval-burst arrays).

### Convergence audit

Per-env solve thresholds from `rl/env_solve_thresholds.py`
(literature: gymnasium docs + Osband 2019 / bsuite scoring;
derived: 50% of Young & Tian 2019 MinAtar-DQN baseline). Applied
via `rl/convergence.classify_envs` to the baseline arm at 200k.

| Class | n | Envs |
|---|---|---|
| solved (final-mean ≥ thresh in ≥50% of cells) | 6 | Acrobot-v1, Catch-bsuite, DeepSea-bsuite, DiscountingChain-bsuite, MountainCar-v0, UmbrellaChain-bsuite |
| partial (0 < solve rate < 0.5) | 2 | MemoryChain-bsuite, SpaceInvaders-MinAtar |
| unsolved (final-mean ≥ thresh in 0 cells) | 5 | **CartPole-v1**, all 4 MinAtar-Atari (Asterix, Breakout, Freeway, MNISTBandit-bsuite) |
| absent (no defensible threshold) | 5 | BernoulliBandit, FourRooms, GaussianBandit, MetaMaze, Pong |

Critically, **5 of 13 thresholdable envs are not converged at
200k**, including CartPole-v1 (vanilla DQN reaches mean 98 / 500,
threshold 475). Catastrophic forgetting is visible in many envs
(Freeway 0.93 → 0.14, GaussianBandit 9.1 → -15.5).

### §3 verdict pattern shifts under convergence-conditioning

| Edge | All 18 envs | Solved 6 envs |
|---|---|---|
| Mechanism g (Δ jensen_gap) | -0.349 (I²=0.86) | **-0.985 (I²=0.93)** |
| Outcome g (Δ best_burst_mean) | +0.086 (I²=0.68) | **-0.025 (I²=0.41)** |

### Reading

1. **Hasselt's bias-reduction mechanism is partly vindicated.**
   On the converged subset, DDQN reduces the Jensen overestimation
   gap by approximately one standard deviation (g = -0.985) — the
   mechanism activates strongly. The framework's gap-grounded
   Scope was right that the gap is the load-bearing scope-defining
   feature; it just had to be conditioned on convergence to surface.

2. **The link to outcome is empirically broken.** Even on envs
   where the bias-reduction activates, the outcome effect is
   essentially zero (g = -0.025, slightly negative). Bias
   reduction does not propagate to better return on this corpus.

3. **The "DDQN modestly helps" finding from the unrestricted
   analysis is a convergence artifact.** The +0.086 outcome g
   comes from underconverged envs where DDQN's "help" reflects
   noisier mid-trajectory dynamics rather than improved final
   policy. Once we restrict to envs where vanilla DQN has reached
   a learned policy, DDQN's outcome contribution disappears.

### Honest scope of this claim

- **Corpus-conditional.** The finding is for total_steps=200k, on
  this 18-env grid, with this set of HPs. We do not claim DDQN's
  bias-reduction-without-outcome-propagation generalizes to
  longer training horizons or different env distributions.
- **Pearl-rung 2 unconditional for arm-effect verdicts** (paired
  Hedges' g across seeds), Pearl-rung 1 for the link (Pearson r
  across envs of per-env effect sizes). The link is observational;
  the dowhy backdoor estimates of mediator → outcome (in the
  `dowhy_ddqn` smoke) confirm at rung-2-conditional that
  jensen_gap → outcome is below threshold even with backdoor
  adjustment.
- **No held-out validation**, since the existing corpus has been
  fully explored. Any sharper claim would need a fresh sweep on
  the held-out env subset (12 of 18 envs reserved per
  `rl/env_splits.py` once that ships).

### Implications for next steps

- The framework's gist ("find scope, find causal chain") survives
  this case study. The framework correctly finds that the gap is
  the scope-defining feature *when scope is properly conditioned*.
  Where the framework's verdict diverges from the literature
  reading is on the chain: the link `gap → outcome` is empirically
  broken even when the upstream `arm → gap` mechanism activates.
- "What's DDQN's actual gap?" — closed (or shelved) by accepting
  this terminal verdict: the bias-reduction mechanism IS what DDQN
  does, but on this corpus that mechanism doesn't translate to
  outcome.
- The next study should set DDQN as the default arm and intervene
  on other components (sync period, n-step return, PER) to
  characterize the effects that DO propagate to outcome at this
  training horizon.

### Reproduction

```
uv run python experiments/convergence_audit.py --total-steps 200000
```
