"""Bias-correction magnitude does NOT predict DDQN outcome gain
on the G1-active DDQN-relevant scope — REFUTED on the current
corpus.

The bootstrap_gap_magnitude → Δ_outcome cluster (MC-free
predictor) fires REFUTED under G1 gating: ATE = −29 (placebo
holds, RCC drift exceeds threshold). Pre-G1-gate diagnostic
(no premise-active filter, n=29 strata) gives β = +244,
p < 10⁻⁴, CI = [+157, +332] — strong positive cross-env signal
— but the 9 low-bias configs that anchored the slope drop out
when G1 fires, and the remaining 20 G1-active strata don't
reproduce the positive relationship in aggregate.

Within-env, the relationship still looks strong (r ≥ +0.82 in
4/5 envs: FourRooms, SpaceInvaders, Asterix, Breakout; only
Acrobot disagrees at r=−0.60). The cross-env null isn't
"no link anywhere" — it's "no consistent slope after env
intercepts are absorbed." More configs per env in the
G1-active scope (currently 3-7) could flip this.

The jens-based companion cluster
(`bias_premise_jens_predicts_outcome_*`) is also REFUTED, but
that one is expected because `jens = Q − MC` shares its MC
term with the outcome — the framework correctly refuses to
corroborate that tautological measurement.

Renamed from `finding_reach_bias_link` (REACH-cohort jens→outcome
test, retired 2026-05-13 because tautological).

Three-bridge cluster form per CLAUDE.md's cluster-shaped causal
claims principle: backdoor (adjustment-identified ATE), placebo
(instrument validity), RCC (omitted-confound sensitivity).
Finding-level `composed_verdict` AND-aggregates."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.bias_correction import (
    bias_correction_clip_predicts_outcome_backdoor,
    bias_correction_clip_predicts_outcome_placebo,
    bias_correction_clip_predicts_outcome_rcc,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    bias_correction_clip_predicts_outcome_backdoor,
    bias_correction_clip_predicts_outcome_placebo,
    bias_correction_clip_predicts_outcome_rcc,
)
