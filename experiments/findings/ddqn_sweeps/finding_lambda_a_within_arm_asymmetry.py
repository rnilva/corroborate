"""Vanilla-side Λ_a → outcome channel + DDQN-side null (within-arm asymmetry).

At γ=0.999 cross-env (8 envs, n=570 vanilla cells), per-cell Λ_a
inversely predicts outcome WITHIN VANILLA cells after
conditioning on jens: ρ_partial = −0.169, p = 7.7e-5. The
bias-asymmetry index Λ_a = σ_clip · √(2 ln K) / Δ_v carries
predictive content for vanilla performance — high per-cell Λ_a
signals the argmax-preservation inequality is closer to being
violated → outcome degrades. The DDQN-side null bridge
(`ddqn_lambda_a_does_not_predict_outcome__within_arm_g0999`,
ρ=+0.006 p=0.96) corroborates the matching null: in DDQN cells
the Λ_a → outcome relationship is abolished, NOT just
attenuated. The arm-asymmetry together with the vanilla HELD is
the substantive per-cell evidence for the bias Type-A/B framing
— DDQN's clip neutralizes the bias-asymmetry → outcome channel
that vanilla suffers from. This is the WITHIN-CELL companion
to `finding_lambda_a_mediation`'s cross-env σ_Λ_a moderation
claim (which is POWER_INSUFFICIENT at n=8).

The DDQN bridge declares `predicted_direction='null'` and its
NO_EFFECT (NULL_EFFECT) verdict is stamped as corroboration by
the framework — both bridges admit their respective predictions,
cluster composes to SUPPORTED."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.lambda_a_mediation import (
    ddqn_lambda_a_does_not_predict_outcome__within_arm_g0999,
    vanilla_lambda_a_predicts_outcome__within_arm_g0999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    vanilla_lambda_a_predicts_outcome__within_arm_g0999,
    ddqn_lambda_a_does_not_predict_outcome__within_arm_g0999,
)
