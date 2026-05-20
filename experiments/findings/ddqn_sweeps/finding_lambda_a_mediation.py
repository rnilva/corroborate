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
    'Bridge 1 (σ_Λ_a moderation cross-env): HELD post-T3a panel '
    'extension. After ingesting LunarLander γ=0.999 + Snake γ=0.999, '
    'panel grows from n=8 to n=10 strata; ρ=−0.745 p=0.0133 clears '
    'the rho_threshold_held=0.6 + p≤0.05 calibration. The original '
    'BLOCKED_ON predicted k=2/k=4 strata would extend the panel — '
    'walked back per `findings_k_axis_gamma_regime_map`: Λ_a has '
    'K_eff dependency, so per-K_eff strata are non-comparable on '
    'the cross-env σ_Λ_a panel. The actual extension came via more '
    'k=1 envs (LL, Snake) at γ=0.999. Bridge 2 (within-cell null) '
    'still POWER_INSUFFICIENT — moderator-not-mediator framing '
    'remains the load-bearing pattern. Cluster verdict awaits '
    'Bridge 2 + Bridge 3 resolution; Bridge 1 substantively HELDs.'
)


BRIDGES: tuple[Bridge, ...] = (
    sigma_lambda_a_moderates_ddqn_outcome__cross_env_g0999,
    lambda_a_does_not_mediate_outcome__cross_stratum_g0999,
    joint_bias_geometry_mediates_arm_outcome__cross_env_g0999,
)
