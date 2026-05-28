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

## The empirical adjudication test (v2 — rigorous)

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

**Step 2 — Run three df-matched conditioning sets.** Via
`dynamic_pc_adjacency` with noise-padding at depth-2 so all three
runs cost the same df (this is the v2 fix — the v1 implementation
compared depth-1 single-mediator runs against depth-2 joint runs
and conflated info gain with df cost):

  | conditioning set | depth | what it estimates |
  | --- | --- | --- |
  | `{sibling, ε}` | 2 | outcome-input leak alone (ε ~ N(0,1)) |
  | `{mediator, ε}` | 2 | full mediator alone |
  | `{mediator, sibling}` | 2 | joint conditioning |

**Step 3 — Binomial Wald z-test, NOT an ad-hoc Δ threshold.**
Compute `Δ = dsep({mediator, sibling}) − dsep({sibling, ε})` and
its Wald SE under binomial:

  ```
  SE_Δ ≈ sqrt(p_j(1−p_j)/n_j + p_s(1−p_s)/n_s) × 100   (pp-scale)
  z    = Δ / SE_Δ
  ```

  Disposition:
  - `z ≥ +z_genuine` (default 1.65, 95% one-sided) → **GENUINE**
  - `z ≤ −z_hurts`   (default 1.65, symmetric)       → **HURTS**
  - `|z| < z_genuine`                                  → **LEAK**
  - `n_marg < min_marginal_edges` (default 3)          → **UNDERPOWERED**

This is sample-size-aware: a +30pp Δ at n=3 is z≈1.2 (LEAK), while
the same +30pp at n=32 is z≈2.8 (GENUINE). The v1 implementation's
fixed +10pp threshold inflated small-n positives.

**Step 4 — Report env-conditionally.** The same mediator can be
GENUINE at one env and LEAK at another. Do not pool the
adjudication verdict across stratum.

## Interpretation caveat (load-bearing)

**A GENUINE verdict does NOT confirm the mediator's mechanism
claim.** It establishes that the mediator's signal extends beyond
the LINEAR span of mean-sibling. Higher moments (variance, sign
concentration), nonlinear couplings, and alternative causal
channels are not in the sibling-span.

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

## The empirical case study (γ=0.99 canonical, v2 verdicts)

`mean_per_state_cumulative_bias_per_burst` paired with
`mean_mc_per_state_per_burst`. Outcome:
`mc_return__mean_axis_-1`. Stratum: env.

  | env | n_marg | `{sib,ε}` | `{med,ε}` | `{m,s}` | Δ | SE_Δ | z | disposition |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | Asterix | 32 | 50% | 91% | 81% | +31pp | 11.2 | **+2.79** | **GENUINE** |
  | MetaMaze | 12 | 58% | 75% | 83% | +25pp | 17.8 | +1.40 | LEAK |
  | SI | 3 | 67% | 33% | 100% | +33pp | 27.2 | +1.22 | LEAK |
  | FR | 44 | 80% | 89% | 86% | +7pp | 8.0 | +0.85 | LEAK |
  | Freeway | 5 | 40% | 80% | 40% | +0pp | 31.0 | +0.00 | LEAK |
  | (other 7 envs) | <3 | — | — | — | — | — | — | UNDERPOWERED |

**Only Asterix passes the rigorous threshold.** MetaMaze and SI
were called GENUINE under v1's ad-hoc +10pp threshold; under
sample-size-aware z, both are LEAK. The +33pp Δ at SI's n=3 was
the v1 protocol's worst-case false positive — corrected here.

The substantive finding: at γ=0.99 Asterix, bias carries
predictive information beyond mean-MC at the 95% level. Whether
this is the bias-clip mechanism or a Q-via-other-channel effect
remains undistinguished — the test does not adjudicate that.
At the other 4 PC-detectable envs, the bias-mediator's apparent
d-separation power is consistent with the MC-leak alone.

## How to bake this into bridge authoring

Bridges using a soft-tautological mediator should:

1. Call `mediator_leak_adjudication` (in
   `corroborate.analyses.diagnostic`) over the bridge's scope.

2. State the per-stratum disposition (GENUINE / LEAK / HURTS /
   UNDERPOWERED) alongside the bridge verdict.

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

## Limitations (what the primitive does NOT yet support)

- **Multi-input siblings**: a 3-component mediator like
  `(Q − target_Q − MC)` needs two siblings + a depth-3 joint test
  (df = n − 3 − 3). The primitive currently supports only single
  mediator + single sibling at depth-2.
- **McNemar/paired test**: SE_Δ uses the independent-binomial form
  (slightly conservative for paired data — same n_marg cells).
  Replace with McNemar discordant-pair test when per-burst d-sep
  booleans are exposed by `dynamic_pc_adjacency` (currently it
  exposes counts only).
- **Measurable-instance mediators**: must currently pass column
  names (strings) so the noise-padding can use the column's
  per-cell array lengths. Bare `Measurable` instances would need
  transitive-reads walking to recover lengths.

## Cross-references

- `feedback_tautology_audit_is_conservative.md` — agent-memory entry
- `paper_g099_mediation_mc_leak_finding.md` — empirical findings memory
- `src/corroborate/analyses/diagnostic/mediator_leak_adjudication.py` — primitive
- `papers/g099_mediation/scripts/gen_mc_leak_adjudication.py` — visual report
