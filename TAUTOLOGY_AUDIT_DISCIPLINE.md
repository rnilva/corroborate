# Tautology audit: when reads-overlap ≠ outcome-leak

The framework's `tautology_audit` primitive flags a mediator as
**outcome-tautological** when its `reads` set overlaps with the
outcome's `reads`. This is a CONSERVATIVE binary verdict — it
catches the *risk* of structural redundancy but cannot tell apart:

- **Hard tautology**: the mediator is functionally determined by
  outcome inputs alone. Reject as a mediator (purely structural).
- **Soft tautology**: the mediator combines outcome inputs with
  *independent* inputs. The reads overlap is real but the mediator
  may still carry genuine causal information beyond the outcome
  leak.

This document records the **empirical adjudication discipline**
established at γ=0.99 canonical for handling soft-tautological
mediators, and how it propagates into bridge authoring. The
discipline went through a critic-driven rigor upgrade on
2026-05-28; the empirical findings should be read against the
**v2 (rigorous)** verdicts, not the v1 (ad-hoc) ones that
preceded the upgrade.

## The empirical adjudication test (v3 — paired McNemar, post-second-critic-review)

A second critic review of v2 identified that:
- **noise-padding was theatre** — adding N(0,1) ε to a conditioning
  set adds rank but no information, so the v2 "df-matching" was
  hand-waving over an asymmetric test;
- **independent-binomial Wald SE was wrong direction** for paired
  data (same n_marg bursts across runs) — under-stated SE means
  v2's LEAK verdicts at MetaMaze/SI were biased toward false LEAK;
- **UNDERPOWERED ≠ LEAK** — collapsing them into "LEAK" launders
  insufficient evidence as evidence of absence;
- **no multiplicity correction** — 12 strata × α=0.05 gives FWER
  ≈ 0.46;
- **"linear span" was technically wrong** — partial-Spearman
  conditions on rank-monotone, not linear;
- **HURTS was sampling noise** — population monotonicity rules it
  out; finite-sample HURTS verdicts were just variance.

v3 fixes all six:

For a soft-tautological mediator `M` whose reads include outcome
inputs `O_inputs`:

**Step 1 — Author the outcome-input sibling.** Construct a
measurable `M_sibling` that depends ONLY on `O_inputs`. The sibling
covers the SAME outcome-overlap as `M` but NONE of `M`'s
independent-input contribution.

  Example: `mean_per_state_cumulative_bias_per_burst` (bias) reads
  both `predicted_q_per_step` (Q, independent) and
  `mc_return_from_step` (MC, in outcome's chain). The sibling is
  `mean_mc_per_state_per_burst` — same MC reduction, no Q.

**Step 2 — Run two PC tests (no noise-pad).** Via
`dynamic_pc_adjacency`:

  | conditioning set | depth | what it estimates |
  | --- | --- | --- |
  | `{sibling}` | 1 | rank-monotone outcome-input leak |
  | `{mediator, sibling}` | 2 | joint conditioning |

The depths differ. The depth-0 marg test is mediator-independent
in the population, so the per-burst d-sep booleans are
**approximately paired** across runs. v5 enforces the cell-set
match per-burst via `n_per_burst >= min_n_per_burst` intersection
on both runs (handles cases where mediator and sibling have
different NaN patterns at some bursts; only bursts with adequate
sample size in BOTH runs contribute to McNemar).

*Note on residual asymmetry*: under per-burst NaN-coupling, the
joint run's cell set at burst b is a subset of the sibling run's
cell set (joint drops cells with NaN in EITHER column; sibling
drops only on `sibling`). The intersection-on-n_per_burst gate
mitigates but does not eliminate this — the two booleans at burst
b can be estimated from slightly different cell subsets. For
dense, NaN-rare per-burst columns (the canonical bias and MC
mediators) this asymmetry is negligible; for sparse columns it
may bias the McNemar comparison slightly. A primitive-architecture
fix would prune to the burst-wise NaN intersection before BOTH
runs see the cells; v5 is rigorous enough for the canonical
mediator pairs but the architectural fix is on the roadmap.

**Step 3 — McNemar paired test on discordant pairs.**
Among marg-edge bursts, count discordant pairs:

  - `n_01` = bursts where joint d-separates but sibling does NOT
    (joint > sibling evidence)
  - `n_10` = bursts where sibling d-separates but joint does NOT
    (anomaly; population monotonicity says this is pure noise)

  Continuity-corrected McNemar z:

  ```
  z = (n_01 − n_10 − sign(n_01 − n_10)) / sqrt(n_01 + n_10)
  ```

  Disposition logic (uses EXACT one-sided binomial for the decision;
  reported `z_mcnemar` is the Edwards continuity-corrected normal):

  - exact-binomial `p ≤ 1 − Φ(z_genuine)` → **GENUINE**
  - `n_01 + n_10 < min_discordant` (default 5) → **UNDERPOWERED_FOR_GENUINE**
    (cannot distinguish small effect from null; v2 collapsed this
    into LEAK — wrong)
  - exact-binomial `p > 1 − Φ(z_genuine)` AND `n_01 + n_10 ≥ min_discordant`
    → **LEAK** (adequately powered, no effect)
  - `n_marg_edge < min_marginal_edges` → **UNDERPOWERED**

  **Doc note**: the reported `z_mcnemar` (Edwards normal) and the
  disposition (exact binomial) may disagree in direction at small
  `n_disc` (< ~15). When `n_disc` is small, prefer comparing
  `disposition` to the threshold rather than `z_mcnemar` directly —
  the exact-binomial p is the rigorous gate; the reported z is the
  well-behaved summary statistic.

  Note: there is NO "HURTS" disposition — population monotonicity
  ensures conditioning on more variables cannot reduce conditional
  independence in expectation. Finite-sample violations are noise
  contributing to `n_10`; the McNemar test absorbs them naturally.

**Step 4 — Multiplicity correction.** Pass
`n_strata_for_multiplicity` to switch `z_genuine` to the
Bonferroni-adjusted one-sided z at α / n_strata. Without
correction, FWER across 12 strata at α=0.05 is ~0.46.

**Step 4 — Report env-conditionally.** The same mediator can be
GENUINE at one env and LEAK at another. Do not pool the
adjudication verdict across stratum.

## Interpretation caveat (load-bearing)

**A GENUINE verdict does NOT confirm the mediator's mechanism
claim.** It establishes that the mediator's signal extends beyond
the **rank-monotone** span of sibling (partial-Spearman conditions
on rank residuals, not linear residuals — earlier doc text said
"linear span" which was technically wrong). Higher moments
(variance, sign concentration of the per-burst pattern), nonlinear
non-monotone couplings, and alternative causal channels are not
adjudicated by this test.

At γ=0.99 with `bias = Q − MC`, GENUINE means **mean-Q contains
predictive info beyond mean-MC**. That could equally well be:

- (a) The bias-clip mechanism (Hasselt's claim) is the real
  causal channel — bias-as-mediator.
- (b) Q has independent causal effects on outcome via OTHER
  channels (state visitation, exploration drive, etc.) that the
  bias mediator picks up because it contains Q.

The adjudication test **cannot distinguish (a) from (b)**. It is
necessary-but-not-sufficient for the bias-clip mediation claim.
Bridges using this primitive should state the disposition AND
explicitly note what the test does not adjudicate.

## The empirical case study (γ=0.99 canonical, v3 verdicts)

`mean_per_state_cumulative_bias_per_burst` paired with
`mean_mc_per_state_per_burst`. Outcome:
`mc_return__mean_axis_-1`. Stratum: env.

Bonferroni-adjusted `z_genuine = 2.642` for 12 strata. Asterix's
z=3.02 still passes (margin 0.38σ).

  | env | n_marg | `sib_dsep` | `joint_dsep` | n_01 | n_10 | z | disposition |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Asterix | 32 | 47% | 81% | **11** | **0** | **+3.02** | **GENUINE** |
  | FR | 44 | 84% | 86% | 2 | 1 | — | UNDERPOWERED_FOR_GENUINE |
  | MetaMaze | 12 | 67% | 83% | 2 | 0 | — | UNDERPOWERED_FOR_GENUINE |
  | SI | 3 | 67% | 100% | 1 | 0 | — | UNDERPOWERED_FOR_GENUINE |
  | Freeway | 5 | 40% | 40% | 0 | 0 | — | UNDERPOWERED_FOR_GENUINE |
  | (other 7 envs) | <3 | — | — | — | — | — | UNDERPOWERED |

**Only Asterix passes** the McNemar test under Bonferroni
correction. All 11 discordant pairs at Asterix favor joint
(n_01=11, n_10=0) — the joint conditioning adds d-sep at 11 of
the 32 marg-edge bursts, sibling adds nothing the joint doesn't
already give. This is the clean signature of a mediator that
carries information beyond the sibling.

**4 envs are UNDERPOWERED_FOR_GENUINE** (discordant pairs < 5).
This is the v2→v3 walk-back of the v2 "LEAK" verdicts: we
cannot conclude the mediator IS just leak at these envs — the
sample is too small to detect a real effect. FR's 44 marg-edge
bursts had only 3 discordant pairs (n_01=2, n_10=1) so almost
all marg-edge bursts agree on d-sep between {sib} and {m,s}; the
underlying agreement is high but the discordant evidence is thin.
Freeway's 0 discordant pairs is the strongest evidence that
mediator and sibling give literally identical d-separation
verdicts there — but still labeled UNDERPOWERED_FOR_GENUINE
because 0 evidence cannot statistically distinguish from a real
small effect.

**The substantive finding**: at γ=0.99 Asterix, bias carries
predictive information beyond mean-MC at the 99.9%-Bonferroni
level. Whether this is the bias-clip mechanism or a
Q-via-other-channel effect remains undistinguished — the test
does not adjudicate mechanism. At 4 other PC-detectable envs,
the test is underpowered; "consistent with MC-leak alone" cannot
be claimed from these data.

## How to bake this into bridge authoring

Bridges using a soft-tautological mediator should:

1. Call `mediator_leak_adjudication` (in
   `corroborate.analyses.diagnostic`) over the bridge's scope.

2. State the per-stratum disposition (GENUINE / LEAK /
   UNDERPOWERED_FOR_GENUINE / UNDERPOWERED) alongside the bridge
   verdict. Distinguish UNDERPOWERED_FOR_GENUINE ("insufficient
   evidence") from LEAK ("adequately powered, no effect").

3. Pair the mediator-of-interest with its sibling in
   REQUIRED_MEASURABLES so both columns are cached for the
   adjudication.

4. Explicitly note that GENUINE is **necessary-but-not-sufficient**
   for the bridge's mechanistic claim — additional evidence
   (alternative-channel falsification, dose-response, intervention
   studies) is needed to attribute mediation to a specific
   mechanism.

## REDQ note (correction)

A prior version of this doc claimed `normalized_bias_redq` would
adjudicate identically to raw bias because REDQ is "a monotone
transform of the raw bias when the denominator is positive." That
is **incorrect**. REDQ divides each cell by its own |E[MC]|, which
is per-cell (per-burst), not stratum-constant. A per-burst
rescaling does NOT preserve cross-burst rank correlations — which
is what the partial-Spearman CI tests run on. REDQ may genuinely
shift the adjudication relative to raw bias and should be
adjudicated independently if it's the bridge's mediator-of-record.

## Limitations (v5 — what the primitive does NOT yet support)

- **Burst non-independence**: McNemar assumes independent paired
  observations; adjacent bursts within one cell share training
  trajectory. The Asterix "11/11 discordant favor joint" pattern
  is more consistent than 11 independent coin flips even under a
  real population effect, suggesting strong within-cluster
  correlation. The primitive's z-score is therefore conditional
  on the per-burst CI-test sequence; for a within-cluster-robust
  confidence interval, pass `n_bootstrap > 0` (default 0) — the
  primitive forwards this to `dynamic_pc_adjacency`'s cluster
  bootstrap on cells, exposed via `bootstrap_marginal` /
  `bootstrap_partial` / `bootstrap_edge_counts` on the
  underlying result. McNemar's z itself is not bootstrapped;
  consumers should read those fields for the cluster-robust CI
  on d-sep rate.
- **Multi-input siblings**: a 3-component mediator like
  `(Q − target_Q − MC)` needs two siblings + a depth-3 joint test.
  The primitive currently supports only single mediator + single
  sibling.
- **Bonferroni denominator counts all candidate strata** (not just
  the subset with marg-edge bursts). This is conservative-correct
  because the testable-stratum set is data-dependent — you don't
  know which strata have ≥`min_marginal_edges` until PC runs. A
  caller wanting tighter control can pass
  `n_strata_for_multiplicity=<subset>` to focus on testable strata
  at the cost of losing the data-dependent-selection adjustment.
- **Measurable-instance mediators**: must currently pass column
  names (strings) for the NaN-coupling pre-filter to recover per-
  cell array finiteness. Bare `Measurable` instances would need
  transitive-reads walking to recover lengths.
- **Perfect concordance** (n_01 = n_10 = 0) lands in
  UNDERPOWERED_FOR_GENUINE; a future revision could expose a
  separate `PERFECT_CONCORDANCE` disposition for the descriptively
  load-bearing case where bias and sibling agree at every testable
  burst (Freeway in the case study). For now, inspect
  `n_discordant_*` fields directly.

## Cross-references

- `feedback_tautology_audit_is_conservative.md` — agent-memory entry
- `paper_g099_mediation_mc_leak_finding.md` — empirical findings memory
- `src/corroborate/analyses/diagnostic/mediator_leak_adjudication.py` — primitive
- `papers/g099_mediation/scripts/gen_mc_leak_adjudication.py` — visual report
