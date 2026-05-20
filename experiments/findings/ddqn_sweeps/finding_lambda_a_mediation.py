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


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None  # Post-T3a + symmetric Bridge 2 verdict logic: REFUTED.

# Cluster history (2026-05-20):
# - Bridge 1 (σ_Λ_a moderation cross-env): HELD at ρ=−0.745 p=0.0133,
#   n=10 after T3a panel extension (LL + Snake γ=0.999).
# - Bridge 2 (Δ_Λ_a does not mediate Δ_out): REFUTED via SIGN_FLIP.
#   Symmetric verdict logic now triggers on |ρ|≥0.5 in either
#   direction. Actual ρ=+0.567 (cross-env partial controlling for
#   Δ_jens) — POSITIVE direction, opposite of the anticipated
#   "DDQN reduces Λ_a → outcome improves" mediator. Substantively
#   the moderator-not-mediator framing partly survives: there's NO
#   canonical negative-direction mediation, but cross-env Δ_Λ_a
#   POSITIVELY tracks Δ_outcome. Envs where DDQN keeps/raises Λ_a
#   (LL, MetaMaze, Breakout) are the bigger-help envs; envs where
#   DDQN aggressively reduces Λ_a (FR Δ_Λ_a=−0.47, Asterix −0.09)
#   don't see outcome benefit. See
#   `findings_lambda_a_mediation_cluster_refuted` memo.
# - Bridge 3 (joint bias-geometry triplet mediates within-cell):
#   POWER_INSUFFICIENT. ρ_pooled=−0.186 in PI band (0.10, 0.30).
#   Env-cohort-dependent absorption (66% on MinAtar subset, 17% on
#   full panel) — needs scope refinement or more strata to escape.
#
# Cluster verdict: REFUTED (Bridge 2 NO_EFFECT/SIGN_FLIP) per the
# composed-verdict aggregator. Substantively the σ_Λ_a moderation
# finding (Bridge 1) stands; the cluster framing as
# "moderator-not-mediator" needs revision to reflect the
# positive-direction cross-env tracking.


BRIDGES: tuple[Bridge, ...] = (
    sigma_lambda_a_moderates_ddqn_outcome__cross_env_g0999,
    lambda_a_does_not_mediate_outcome__cross_stratum_g0999,
    joint_bias_geometry_mediates_arm_outcome__cross_env_g0999,
)
