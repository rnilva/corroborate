"""σ_Λ_a is a moderator, NOT a within-cell mediator (γ=0.999).

The two bridges form a scope-cluster that jointly characterises
where the Λ_a structural predictor sits in DDQN's causal graph
at γ=0.999:

  Bridge 1 — σ_Λ_a moderates DDQN's d_out cross-env (n=8).
  Bridge 2 — Per-cell Δ_Λ_a does NOT mediate Δ_outcome (NULL).

The publishable empirical claim is the JOINT pattern:
σ_Λ_a operates at the env-aggregate level (env feature predicts
DDQN's sign/strength), distinct from "DDQN reduces Λ_a, which
mediates the outcome." This is the moderation-vs-mediation
distinction CLAUDE.md §"Moderation vs mediation" makes
load-bearing in the bias-geometry analysis.

EXPECTED is pinned to UNDERPOWERED because at the current cache
state (commit `f471913`) the cross-env moderation Spearman
ρ=−0.643 p=0.086 misses the rho_threshold_held=0.6 + p≤0.05
calibration (n=8, two-sided critical |r|≈0.71). Memory's prior
ρ=−0.78 cleared HELD; this one doesn't. The pre-registered
DRIFT prediction: when the k=4 minatar sweep ingests + the panel
extends to n=12+ strata, σ_Λ_a's cross-env signal should
sharpen and Bridge 1 will fire HELD."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.lambda_a_mediation import (
    joint_bias_geometry_mediates_arm_outcome__cross_env_g0999,
    lambda_a_does_not_mediate_outcome__cross_stratum_g0999,
    sigma_lambda_a_moderates_ddqn_outcome__cross_env_g0999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'Bridge 1 (σ_Λ_a moderation cross-env): current 8-env panel '
    'gives ρ=−0.643 p=0.086, below the rho_threshold_held=0.6 + '
    'p≤0.05 calibration at n=8 (critical |r|≈0.71). Memory '
    'snapshot ρ=−0.78 cleared HELD; current cache cohort is '
    'weaker due to FR contribution (σ_Λ_a=1.02 + d_out=+0.09 '
    'breaks monotone). When the running k=4 minatar sweep '
    'ingests + Asterix/Breakout/Freeway/SI strata land at k=2 '
    'and k=4, panel will extend to n=12-16 strata and the '
    'Bridge 1 signal should sharpen → HELD. Predicted post-k=4 '
    'EXPECTED: SUPPORTED. Bridge 2 (within-cell null) predicted '
    'to remain NULL_EFFECT regardless — the moderator-not-'
    'mediator framing is the load-bearing pattern.'
)


BRIDGES: tuple[Bridge, ...] = (
    sigma_lambda_a_moderates_ddqn_outcome__cross_env_g0999,
    lambda_a_does_not_mediate_outcome__cross_stratum_g0999,
    joint_bias_geometry_mediates_arm_outcome__cross_env_g0999,
)
