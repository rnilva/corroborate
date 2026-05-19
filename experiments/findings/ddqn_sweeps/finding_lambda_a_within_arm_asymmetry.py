"""Vanilla-side Λ_a → outcome channel (within-arm HELD).

At γ=0.999 cross-env (8 envs, n=570 vanilla cells), per-cell Λ_a
inversely predicts outcome WITHIN VANILLA cells after
conditioning on jens: ρ_partial = −0.169, p = 7.7e-5. The
bias-asymmetry index Λ_a = σ_clip · √(2 ln K) / Δ_v carries
predictive content for vanilla performance — high per-cell Λ_a
signals the argmax-preservation inequality is closer to being
violated → outcome degrades. This is the WITHIN-CELL companion
to `finding_lambda_a_mediation`'s cross-env σ_Λ_a moderation
claim (which is POWER_INSUFFICIENT at n=8).

The framework currently lacks a way to encode "DDQN's matching
null result corroborates the channel-neutralization claim" as a
co-cluster member (NO_EFFECT under predicted_direction='null'
stamps as refuted via the verdict→evidentiary-level mapping,
which then refutes the cluster). The DDQN-side null bridge
(`ddqn_lambda_a_does_not_predict_outcome__within_arm_g0999`,
ρ=+0.006 p=0.96) lives in `ddqn_sweeps.BRIDGES` and fires its
own verdict at hypothesis run-time as **independent
corroborating evidence**: in DDQN cells the Λ_a → outcome
relationship is abolished, NOT just attenuated. The arm-asymmetry
together with the vanilla HELD is the substantive per-cell
evidence for the bias Type-A/B framing — DDQN's clip neutralizes
the bias-asymmetry → outcome channel that vanilla suffers from.

A single-bridge cluster is unusual but conscious — see
`docs/HYPOTHESIS_AS_GRAPH.md` for the framework's stance that
clusters need not be uniform-extent; the within-arm asymmetry
**is** the load-bearing finding even though it surfaces as a
single bridge here pending framework support for typed null-
prediction admit-equivalent verdicts."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.lambda_a_mediation import (
    vanilla_lambda_a_predicts_outcome__within_arm_g0999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    vanilla_lambda_a_predicts_outcome__within_arm_g0999,
)
