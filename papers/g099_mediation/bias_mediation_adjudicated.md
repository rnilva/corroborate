# Adjudicating the bias-mediation claim at Asterix γ=0.99

DDQN's published mechanism is *bias reduction → outcome improvement*.
We test this empirically on the 12-env γ=0.99 canonical panel using
the framework's `mediator_leak_adjudication` primitive. **Only
Asterix γ=0.99 passes a stringent statistical test; the other
PC-detectable envs are inconclusive (underpowered).** This is, to
our knowledge, the first cross-env mediation analysis of DDQN
explicitly controlling for the structural entanglement between the
bias mediator and the outcome.

## Why the naïve test is biased

The bias mediator is

  bias(s, a) = Q(s, a) − G(s, a)

where G is the γ-discounted realised return from (s, a) onward
(the on-policy MC estimate of Q^π). The cross-env outcome is the
per-burst mean of the γ-discounted return at the episode start.

The bias and the outcome **share an input**: G. Regressing the
outcome on bias is partly a regression of the outcome on its own
input. Any apparent mediation signal that comes from this shared
input is structural, not causal. The MC-leak inflates the
mediation effect-size in a direction that has nothing to do with
the agent's Q-function.

This concern is not specific to DDQN — it applies to *any* bias-
style mediator that subtracts the outcome's underlying realised
return from a model prediction. We didn't find a published cross-
env DDQN mediation analysis that controls for it; readers who know
of one should write in.

## The test, in plain language

For each env, we ask three conditional independence questions
about the arm→outcome edge per burst:

1. **What does MC alone explain?** Condition on the per-(s, a) MC
   summary (the realised return averaged over all visited states
   in that burst). This is the structural MC-leak baseline — what
   the bias mediator would explain *even if it carried no
   information beyond MC*.

2. **What does bias alone explain?** Condition on the full bias
   mediator. This includes both the MC-leak component and the
   Q-component.

3. **Does conditioning on BOTH (bias + MC) improve over MC alone?**
   If yes, the Q-component is doing something MC alone doesn't —
   the bias is not just a restatement of the outcome's own input.

The third question is the load-bearing one. We compare the
per-burst d-separation rate under condition (3) to condition (1)
via a paired test on the bursts where the arm→outcome marginal
edge is detected. A burst contributes only if both conditions
fire on enough cells; a stratum contributes a verdict only if it
has at least a minimum number of marginal-edge bursts.

A stratum can come back **GENUINE** (Q-component is doing real
work beyond the MC-leak), **LEAK** (adequately powered but no
effect — bias is consistent with MC-leak alone), or
**UNDERPOWERED** (not enough data to tell). The verdict is
per-env; the multiplicity across 12 envs is controlled by
Bonferroni.

(Methodology details — McNemar's discordant-pair test, exact
binomial at small discordant counts, Edwards continuity
correction for the reported z, all-candidate-strata Bonferroni
denominator — are in the framework primitive's docstring and
the discipline doc `TAUTOLOGY_AUDIT_DISCIPLINE.md`. The verdicts
below are robust to all four reviewer-driven methodology
revisions documented there.)

## Empirical result

  | env | marg-edge bursts | bias is...                  |
  | --- | ---------------- | --------------------------- |
  | **Asterix γ=0.99** | 32 | **GENUINE** — 11 of 11 informative bursts favor adding bias over MC alone; none favor the reverse. |
  | FourRooms γ=0.99 | 44 | underpowered — 3 informative bursts, can't tell |
  | MetaMaze γ=0.99 | 12 | underpowered — 2 informative bursts |
  | SpaceInvaders γ=0.99 | 3 | underpowered — 1 informative burst |
  | Freeway γ=0.99 | 5 | underpowered — 0 informative bursts (bias and MC give identical verdicts) |
  | 7 other envs | <3 | no test possible — marginal arm→outcome edge not detected |

Only Asterix γ=0.99 rigorously exhibits the DDQN-promised
mediation under cross-env multiplicity control. This is a
stronger claim than prior cross-env DDQN reports: it survives
the structural-tautology concern by construction.

## What GENUINE does and does NOT mean

The test confirms bias carries information beyond the structural
MC-leak. It does **not** identify *which* causal channel that
information runs through. Two non-exclusive interpretations:

- **(a) Bias-clip mechanism** — DDQN's clip targets the bias
  directly, and bias-reduction is the proximate cause of outcome
  improvement.
- **(b) Q-via-state-visitation** — DDQN changes Q, Q changes the
  policy's state visitation, state visitation changes outcome.
  Bias contains Q, so it predicts outcome via this longer
  pathway.

The next falsification step is to add the panel's state-coverage
measurables (`state_hash_n_unique_per_burst`,
`state_hash_entropy_per_burst`, `state_repeat_rate_window64_per_burst`)
as multi-input siblings: if conditioning on state visitation
ALSO absorbs bias's mediation power, then (b) is the more
parsimonious story; if not, (a) survives. The current primitive
supports single-mediator + single-sibling; the multi-input
extension is the recommended next development.

## Channel separation at Asterix

Asterix has TWO candidate Q-side mediators:

- `bias` (tautology-audited above)
- `q_argmax_margin` (clean — depends only on Q, not on MC)

The relationship between them at Asterix:

- *Does bias add information beyond q_argmax?* — Yes: at 10 of
  the 11 informative bursts, joint conditioning d-separates
  where q_argmax alone does not. One burst goes the other way.
  This is borderline-significant under Bonferroni (z=+2.41
  against the 12-stratum threshold of 2.64), clearly significant
  without correction.
- *Does q_argmax add information beyond bias?* — No: at zero
  bursts does q_argmax-conditioning d-separate where bias alone
  does not.

So bias *subsumes* q_argmax at Asterix — these are not parallel
channels; bias contains everything q_argmax does and more. But
the *source* of bias's surplus is ambiguous: it could be richer
Q-information that q_argmax misses (the action-margin is a
coarse summary), OR residual MC-leak the adjudication couldn't
fully exclude. Resolving this needs the multi-input sibling
extension flagged above.

## What survives

After all the methodological refinement, the substantive claim
about DDQN at γ=0.99 canonical is:

- **Asterix** is the only env where bias rigorously mediates
  outcome after controlling for structural MC-leak.
- The mediation is robust to multiplicity correction across the
  panel.
- The mechanism interpretation (bias-clip vs Q-via-state-
  visitation) remains open and requires additional siblings to
  adjudicate.
- Other PC-detectable envs (FourRooms, MetaMaze, SpaceInvaders,
  Freeway) are *underpowered* — the data cannot distinguish a
  small mediation effect from zero.

## See also

- `TAUTOLOGY_AUDIT_DISCIPLINE.md` — the discipline document with
  methodology details and v1→v6 revision history.
- `src/corroborate/analyses/diagnostic/mediator_leak_adjudication.py`
  — the typed primitive.
- `figures/report_mc_leak_adjudication.png` — the v1 visual.
- `figures/report_asterix_bias_adjudication.png` — accompanying
  panel for this report (to be authored).
