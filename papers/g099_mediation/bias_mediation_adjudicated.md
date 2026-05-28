# Adjudicating the bias-mediation claim at Asterix γ=0.99

The DDQN literature reports two empirical regularities together —
DDQN reduces overestimation bias AND improves outcome — and
*intuitively* suggests the first explains the second. Hasselt
(2016) writes that bias-reduction "leads to" better performance
and "can benefit" stability; REDQ (Chen et al. 2021) argues stable
normalized bias is the signature of good algorithms. **Neither
explicitly tests whether bias-reduction mediates outcome-
improvement causally.** That mediation claim is the field's
folk reading, not the papers' formal claim.

We formalize the folk reading as an explicit mediation hypothesis
and test it on the 12-env γ=0.99 canonical panel using the
framework's `mediator_leak_adjudication` primitive. **Only
Asterix γ=0.99 passes a stringent statistical test; the other
PC-detectable envs are inconclusive (underpowered).** To our
knowledge, no published cross-env DDQN analysis controls for the
structural entanglement between the bias mediator and the outcome.

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
MC-leak. It does NOT pin down *which* Q-side property is the load-
bearing channel.

### The analytical-tautology concern

`bias = Q − MC`. Under successful Q-learning, Q tracks MC (Bellman
contraction). When learning succeeds:

  - MC rises (better policy → higher returns)
  - Q catches up to MC (Bellman contraction → bias shrinks)
  - both are downstream signatures of "the algorithm is learning"

So "bias-reduction co-occurs with outcome-improvement" might be,
in part, an *analytical fact about Q-learning succeeding* rather
than a separately-testable causal claim about *bias-as-channel*.

### Why the cross-env pattern partially defuses this

If bias-reduction were *purely* a definitional shadow of successful
Q-learning, every env where DDQN learns at all would show bias→
outcome mediation. **Empirically, that's not what we observe.**
The mech bridge (`ddqn_reduces_bias__consistently_cross_env`) is
HELD broadly across the panel — DDQN reduces bias in most envs.
But the link bridge under the stability outcome (`ddqn_helps_
outcome__consistently_cross_env__late30`) comes back POWER_
INSUFFICIENT — outcome improvement does not reliably follow.

Bias-reduction-without-outcome-movement is *common* in our cache.
The mechanisms vary by env:

- **Q-magnitude scaling without policy change.** DDQN's clip
  damps Q magnitude. If MC is small or stable, `Q−MC` shrinks
  but the argmax policy is unchanged → outcome flat. Memory
  `findings_m1_rate_vs_magnitude_decomposition` documents this
  for the MinAtar envs at γ=0.99 (rate_d ≈ 0 while Q_d ≪ 0).
- **Saturation.** CartPole hits the cap on both arms; bias
  variation is real but outcome can't move.
- **Sometimes harm.** Asterix γ=0.999 reduces bias but harms
  outcome (different env, different memory).

So bias-reduction is empirically *not* equivalent to outcome
improvement. They're separable. The mediation question is
therefore non-trivial — at which env (if any) is bias's
per-burst pattern *temporally aligned* with outcome's per-burst
pattern across the arm intervention?

**Asterix γ=0.99 is the only env that passes.** Asterix is not
where DDQN reduces bias most (MinAtar envs are comparable), nor
where DDQN helps outcome most (PacMan has a larger raw effect).
It is the unique env where the *temporal coupling* between bias-
reduction and outcome-improvement is detectable across training.
That coupling is what mediation actually tests — it is the
substantive content of the GENUINE verdict, even after the
analytical-tautology caveat.

### What the verdict CAN'T pin down

The test cannot distinguish:

- **(a) Bias-clip intuition** — DDQN's clip targets bias directly;
  bias-reduction is the proximate cause of outcome improvement.
  (The field's folk reading of Hasselt.)
- **(b) Q-via-state-visitation** — DDQN changes Q; Q changes
  policy; policy changes state visitation; state visitation
  changes outcome. Bias contains Q, so it predicts outcome via
  this longer pathway.
- **(c) "Q is what DDQN modifies"** — DDQN is by construction a
  Q-side intervention, so any Q-side mediator (bias, q_argmax,
  Q-magnitude, …) will mediate by construction to the extent
  that DDQN's effect lives on Q. The test confirms the channel
  is Q-side at Asterix, not e.g. replay-side or schedule-side,
  but doesn't single out *which* Q-property carries the load.

The substantive paper-honest claim is therefore narrower than
"DDQN's bias-clip mechanism is corroborated at Asterix":

> DDQN's effect on outcome at Asterix γ=0.99 runs through Q-side
> properties, of which bias-reduction is one plausible per-burst
> summary. The Q-channel is non-trivially active here (not pure
> MC-leak, not pure analytical tautology). Distinguishing
> bias-reduction from other Q-side properties (action-gap
> sharpening, magnitude damping, trajectory coherence) and from
> non-Q channels (state visitation, exploration) is the next
> falsification step.

### Next falsification step

Run the adjudication with state-visitation siblings:
`state_hash_n_unique_per_burst`, `state_hash_entropy_per_burst`,
`state_repeat_rate_window64_per_burst`. If conditioning on
state-visitation ALSO d-separates the arm→outcome edge at
Asterix γ=0.99, then non-Q channels are doing parallel work and
the Q-channel isn't unique. If only Q-side mediators work and
state-visitation doesn't, the Q-channel is genuinely special and
bias is one of multiple admissible Q-summaries inside it.

The current primitive supports single-mediator + single-sibling.
Multi-input siblings (needed for the proper Q-vs-non-Q contrast)
are queued as the next development.

## Multiple Q-side summaries — bias subsumes q_argmax_margin

We have TWO candidate Q-side mediators at Asterix:

- `bias` (Q − MC, tautology-audited above)
- `q_argmax_margin` (clean — depends only on Q, not on MC)

When we pair them as mediator/sibling pairs:

- Adding bias to q_argmax adds d-separation at 10 of 11 informative
  bursts; the reverse at 1 (borderline-significant under
  Bonferroni; clearly significant without).
- Adding q_argmax to bias adds d-separation at zero bursts.

So bias is the broader Q-side summary at Asterix — it contains
everything q_argmax captures and more. But this doesn't single
out bias-reduction as *the* mechanism; q_argmax is one specific
Q-summary among many possible ones (action-gap, magnitude,
trajectory autocorrelation, …), and bias picks up the richer
combination. We've shown the Q-channel is active and that bias
is one admissible summary inside it — we haven't shown
bias-reduction is the load-bearing Q-property.

Resolving "which Q-property carries the load" needs systematic
multi-input sibling adjudication across the Q-summary space.

## What survives

After all the methodological refinement, the substantive claim
about DDQN at γ=0.99 canonical is narrower than the literature's
folk reading but non-trivial:

- **Asterix is the only env where the temporal coupling between
  bias-reduction and outcome-improvement is rigorously detectable
  across training**, after controlling for the structural MC-leak
  and after Bonferroni correction across the 12-env panel. The
  literature's two regularities (mech HELD, link HELD) co-occur
  at Asterix at the same per-burst granularity, which is what
  mediation actually tests.
- This is **not** "DDQN's bias-clip mechanism is confirmed."
  It is "the Q-channel is active and bias is one admissible
  per-burst summary inside it." The Q-vs-non-Q channel
  question, and the bias-vs-other-Q-property question, both
  remain open.
- **Other envs are underpowered, not refuted.** The data cannot
  distinguish a small mediation effect from zero at FourRooms /
  MetaMaze / SpaceInvaders / Freeway. The framework's
  POWER_INSUFFICIENT verdict is explicit on this.
- **The cross-env pattern itself is informative**: bias-reduction
  occurs broadly across envs while outcome-improvement does not
  reliably follow. This empirical separability defuses the
  pure-analytical-tautology concern (bias-reduction is not
  definitionally equivalent to outcome-improvement) but does
  not eliminate the Q-as-channel-by-construction concern.

The framework's contribution here is making the literature's
implicit causal claim explicit, testable, and per-stratum honest
— turning "DDQN reduces bias and improves performance" into a
question with a typed verdict surface that distinguishes
"adjudicated" from "underpowered" from "confounded."

## See also

- `TAUTOLOGY_AUDIT_DISCIPLINE.md` — the discipline document with
  methodology details and v1→v6 revision history.
- `src/corroborate/analyses/diagnostic/mediator_leak_adjudication.py`
  — the typed primitive.
- `figures/report_mc_leak_adjudication.png` — the v1 visual.
- `figures/report_asterix_bias_adjudication.png` — accompanying
  panel for this report (to be authored).
