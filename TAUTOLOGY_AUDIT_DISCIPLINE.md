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
established at γ=0.99 canonical (commit `bd1c4e2`) for handling
soft-tautological mediators, and how it should propagate into
future bridge authoring.

## The empirical adjudication test

For a soft-tautological mediator `M` whose reads include outcome
inputs `O_inputs`:

**Step 1 — Author the outcome-input sibling.** Construct a
measurable `M_sibling` that depends ONLY on `O_inputs` (no
independent inputs). The sibling has the same outcome-overlap as
`M` but none of `M`'s independent-input contribution.

Example: `mean_per_state_cumulative_bias_per_burst` (the bias
mediator) reads both `predicted_q_per_step` (Q, independent) and
`mc_return_from_step` (MC, also in outcome's chain). Its sibling
is `mean_mc_per_state_per_burst` — same MC reduction, no Q.

**Step 2 — Multi-mediator d-separation comparison.** Via
`dynamic_pc_adjacency` (or static `partial_spearman_rho_multi`),
run three conditioning sets and report d-sep%:

  | conditioning set | what it tells you |
  | --- | --- |
  | `{M_sibling}` | what the outcome-input leak alone explains |
  | `{M}` | the full mediator's apparent power (may include leak) |
  | `{M, M_sibling}` | does conditioning on BOTH improve over sibling alone? |

**Step 3 — Verdict by margin.** Let `Δ = dsep({M, sibling}) − dsep({sibling})`.

- `Δ ≥ +10pp` → `M` carries genuine independent-input information
  at this stratum. The mediator is doing real work beyond the
  leak. Disposition: **GENUINE**.
- `−5pp ≤ Δ < +10pp` → `M` is mostly outcome-leak. The bulk of
  its d-separation power is the sibling's contribution. The strict
  tautology audit was right at this stratum. Disposition:
  **LEAK**.
- `Δ < −5pp` → conditioning on `M` HURTS d-separation
  (`M` is contaminating the conditioning set). Disposition:
  **HURTS** — refuse to use `M` here.

**Step 4 — Report env-conditionally.** Do NOT pool the
adjudication verdict across stratum. The same mediator can be
GENUINE at one env and LEAK at another. The discipline is
per-stratum, not global.

## The empirical case study: bias-mediator at γ=0.99 canonical

`mean_per_state_cumulative_bias_per_burst` paired with
`mean_mc_per_state_per_burst`. Outcome:
`mc_return__mean_axis_-1`. Stratum: env (12 envs in canonical
panel, 5 with PC-detectable marginal edge n ≥ 3).

  | env | n_marg | `{sibling}` | `{M, sibling}` | Δ | disposition |
  | --- | --- | --- | --- | --- | --- |
  | Asterix | 32 | 47% | **81%** | +34pp | GENUINE |
  | MetaMaze | 12 | 67% | **83%** | +16pp | GENUINE |
  | SI | 3 | 67% | 100% | +33pp | GENUINE (small n) |
  | FourRooms | 44 | 84% | 86% | +2pp | LEAK |
  | Freeway | 5 | 40% | 40% | +0pp | LEAK (pure) |

So the bias mediator is **GENUINE at the envs where Hasselt's
bias-clip mechanism mechanistically operates** (Asterix and
MetaMaze are the canonical "DDQN-helps via bias reduction" cases)
and **pure outcome-leak at FR / Freeway**. The strict tautology
verdict would have rejected bias as a mediator globally — but the
adjudication shows that's the wrong disposition at 2 of 5 envs.

(This is also the empirical answer to the question "is the
bias-mediator real or is it just the MC bleeding through?": at
some envs it's real, at others it's bleeding.)

## How to bake this into bridge authoring

Bridges using a soft-tautological mediator should:

1. Reference (or compute) the joint d-sep adjudication for the
   bridge's scope. Don't just cite a global "M mediates X" result.

2. State the per-stratum disposition (GENUINE / LEAK / HURTS)
   alongside the bridge verdict. A bridge that holds via a
   LEAK-disposition mediator is reporting an outcome-leak, not a
   causal mediation.

3. Pair the mediator-of-interest with its sibling in
   REQUIRED_MEASURABLES. Both columns must be cached for the
   adjudication to run.

4. Treat REDQ-style normalization (`(Q − MC) / |E[MC]|`) the same
   way — REDQ is a monotone transform of the raw bias when the
   denominator is positive, so its adjudication should match the
   raw bias's adjudication at every stratum. Don't claim REDQ
   "fixes" the tautology — it doesn't, it just rescales.

## Framework-level next step

The current `tautology_audit` primitive is binary (clean vs
flagged). A future upgrade should extend the verdict enum:

- `clean` — no reads overlap.
- `HARD_TAUTOLOGY` — full functional determination by outcome
  inputs (refuse).
- `SOFT_TAUTOLOGY_GENUINE` — adjudication finds Δ ≥ +10pp on the
  scope of interest (mediator carries genuine info).
- `SOFT_TAUTOLOGY_LEAK` — adjudication finds Δ < +10pp
  (mediator is mostly leak, but disposition is per-stratum).
- `SOFT_TAUTOLOGY_UNADJUDICATED` — overlap detected, sibling
  available, but adjudication not yet run (caller should run it).

Until that primitive lands, the `gen_mc_leak_adjudication.py`
script under `papers/g099_mediation/scripts/` is the reference
implementation. Wire its logic into a typed analysis primitive
(`mediator_leak_adjudication`) when the next bridge needs to
adjudicate a soft tautology.

## Cross-references

- `feedback_tautology_audit_is_conservative.md` — agent-memory entry
- `paper_g099_mediation_mc_leak_finding.md` — empirical findings memory
- `papers/g099_mediation/scripts/gen_mc_leak_adjudication.py` — reference test
- `papers/g099_mediation/figures/report_mc_leak_adjudication.png` — visual
