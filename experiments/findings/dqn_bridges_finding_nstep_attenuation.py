"""DDQN's benefit attenuates as n_step grows — the causal chain.

Bias-compounding theory predicts: as `n_step` grows, the
bootstrap target shifts toward Monte Carlo, positive bias
(E[max_a Q] > max_a E[Q]) compounds less, so DDQN's
de-biasing has less to do, so DDQN's benefit attenuates.

The slope-form bridges
(`ddqn_{jensen, outcome, final_outcome}_slope_attenuates_with_
log_nstep__fourrooms`) test this moderation pattern via
meta-regression over n_step ∈ {1, 2, 3, 5, 10}. Each on its
own is a panel coefficient; jointly they're a causal chain
claim — and the chain has an implicit premise the slope
bridges' analysis CANNOT carry: mech must be ACTIVE at the
n_step floor (else the slope is testing in a regime where the
bias-compounding theory makes no claim).

Per `HYPOTHESIS_AS_GRAPH.md`, an implicit premise IS a node.
This Finding materializes that node as two bookend bridges
(`jensen_premise_active__fourrooms_n1` /
`jensen_premise_dormant__fourrooms_n10`) and composes the
five-edge cluster:

  premise(low n active)  ──────  cohort baseline mean(jens) > 0.05
        │                        on FR × n_step=1
        │
  mech slope    ────────────  Δ_jens shrinks with log_n_step
        │
  outcome slope (burst)  ─────  Δ_eval_best_burst_mean shrinks
        │                        with log_n_step
        │
  outcome slope (final)  ─────  Δ_eval_final_mean shrinks
        │                        with log_n_step
        │
  premise(high n dormant)  ──   cohort baseline mean(jens) ≤ 0.05
                                on FR × n_step=10

**Current verdict: REFUTED.**

Four edges admit (premise at n=1, mech slope, final-mean
outcome slope, premise at n=10). The fifth — the BURST-MEAN
outcome slope — resolves NO_EFFECT. The dissociation is the
finding: bias-compounding attenuation holds on STEADY-STATE
outcome but fails on PEAK outcome.

The methodological note: best-burst-mean is a peak metric that
compresses long-run differences once both arms hit their per-
arm ceiling. Final-mean is sensitive to long-run learning
quality and exposes attenuation cleanly. At n=10 vanilla
actually beats DDQN on final-mean (the slope's predicted sign-
flip endpoint), consistent with bias-compounding theory's
prediction that DDQN's edge inverts when bootstrap-bias is no
longer the dominant failure mode. This dissociation is the
substrate's signal — not noise — and the REFUTED cluster
verdict honestly surfaces it instead of cherry-picking the
outcome metric that aligns with theory.

The premise bookends matter independently: they say "the
TRAJECTORY between n=1 active and n=10 dormant is the
attenuation the slope bridges quantify." Without them the
slope's "attenuates" claim would silently rest on
unsubstantiated endpoints.

**Why cohort-form premise bridges (not the per-cell tally form
the other envs use)**: see the cluster header in `dqn_bridges.
py` near the slope bridges. FR's per-cell `jensen_dormancy_
premise_active` test power-fails on baseline cells (sigma_Q
small relative to per-cell test power → 100% POW_INSUF). The
cohort mean is conclusive (0.37 at n=1 vs 0.013 at n=10) and
that's the analysis these bookend bridges use.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.dqn_bridges import (
    ddqn_final_outcome_slope_attenuates_with_log_nstep__fourrooms,
    ddqn_jensen_slope_attenuates_with_log_nstep__fourrooms,
    ddqn_outcome_slope_attenuates_with_log_nstep__fourrooms,
    jensen_premise_active__fourrooms_n1,
    jensen_premise_dormant__fourrooms_n10,
)


# EMPIRICAL state on the canonical dqn_bridges cache:
#   premise(n=1)  = HELD (baseline mean jens=0.37, 7× floor)
#   mech slope    = HELD (β positive, CI excludes zero)
#   burst slope   = NO_EFFECT (β negative but CI spans zero —
#                   peak metric compresses long-run differences)
#   final slope   = HELD (β≈−0.27, p≈0.025 — steady-state
#                   attenuation as theory predicts)
#   premise(n=10) = HELD (baseline mean jens=0.013, below floor)
#
# Cluster = REFUTED via burst-slope NO_EFFECT. The dissociation
# between peak and steady-state outcome is the scientific
# takeaway (theory's attenuation prediction holds at steady-
# state, fails at peak); see module docstring.
EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    jensen_premise_active__fourrooms_n1,
    ddqn_jensen_slope_attenuates_with_log_nstep__fourrooms,
    ddqn_outcome_slope_attenuates_with_log_nstep__fourrooms,
    ddqn_final_outcome_slope_attenuates_with_log_nstep__fourrooms,
    jensen_premise_dormant__fourrooms_n10,
)
