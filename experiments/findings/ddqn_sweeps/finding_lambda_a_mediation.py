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
    'Cluster verdict shifted from REFUTED to UNDERPOWERED on '
    '2026-05-21 when σ_Λ_a converted from hardcoded `MappingProxyType` '
    'constant to `DerivedCovariateSpec` (std of lambda_a_late on '
    'baseline cells per env, computed in scope). Previously hardcoded '
    'values were aggregated across HP-mixed cohorts and inflated '
    'σ_Λ_a 2-17× per env (Breakout 17×, SI 15×, FR 7×, ...). The '
    'cross-env Spearman ρ collapses from −0.745 p=0.013 (HELD) to '
    '−0.283 p=0.46 (PI/NO_EFFECT) under canonical-corpus-per-env '
    'scoping. See memory '
    '`findings_sigma_lambda_a_hp_artifact_walkback`. The Λ_a-cluster '
    'as a publishable cross-env finding needs walking back. Bridges '
    '2-4 also rederive: Bridge 2 (Δ_Λ_a NULL prediction) now HELDs '
    'at p=0.934 instead of refuting via SIGN_FLIP — the positive '
    'cross-env signal that previously fired SIGN_FLIP was also an '
    'HP-mixing artifact. Bridge 3 (joint bias-geometry triplet) '
    'still PI at ρ=−0.21 on the 9-env single-corpus panel (40% '
    'absorption). MetaMaze excluded — older corpus traces lack '
    '`online_top12_margin_per_step` so lambda_a_late is NaN.'
)

# Cluster history:
#
# 2026-05-20 (REFUTED state, now superseded):
#   Bridge 1 hardcoded σ_Λ_a → ρ=−0.745 p=0.013 HELD; Bridge 2
#   Δ_Λ_a cross-stratum ρ=+0.567 fired NO_EFFECT (SIGN_FLIP) under
#   the symmetric verdict logic. Cluster verdict REFUTED via
#   Bridge 2.
#
# 2026-05-21 (canonical-scope rework):
#   - Investigation triggered by user observation "FourRooms have
#     too many points; definitely corrupted." Audit confirmed FR
#     γ=0.999 cohort was 660 cells from 6 probe corpora (linear vs
#     MLP architecture, shaped/unshaped reward, varying ε, varying
#     lr). Other MLP envs similar.
#   - Restored cloud-evicted traces for fr_g999_loop_test (1M),
#     ddqn_axis_probes_mc_1m (1M); recomputed
#     `q_argmax_margin_late` + `q_action_std_late` for FR + MC
#     locally. MetaMaze excluded — older `metamaze_g0999_1M_postfix`
#     corpus lacks `online_top12_margin_per_step` in traces (older
#     substrate version, can't recompute).
#   - Canonical-corpus allowlist: one canonical-HP-shape corpus
#     per env at γ=0.999, n=60 each (MC subsampled to 30 seeds).
#     9-env balanced panel n=540.
#   - σ_Λ_a hardcoded values shrink 2-17× per env on canonical
#     pool (memory `findings_sigma_lambda_a_hp_artifact_walkback`).
#     Hardcoded constant converted to `DerivedCovariateSpec` —
#     σ_Λ_a is now computed at bridge-resolution time from
#     baseline cells in scope, so future scope changes auto-
#     redrive.
#   - Bridge verdicts on canonical pool:
#     * Bridge 1 σ_Λ_a moderation: ρ=−0.28 p=0.29 →
#       POWER_INSUFFICIENT (walk-back of the n=10 HP-mixed HELD)
#     * Bridge 2 Δ_Λ_a does not mediate Δ_out: ρ near zero p=0.93
#       → HELD (the prior SIGN_FLIP at ρ=+0.567 was also an
#       HP-mixing artifact)
#     * Bridge 3 joint (σ_clip, Δ_v, jens) within-cell: rho_pooled
#       =−0.21 p=0.001 → PI. 40% absorption. FR (−0.65) and SI
#       (−0.62) carry large residual; their established mediator
#       is loop-revisit-rate (not in triplet).
#     * Bridge 4 vanilla Λ_a within-arm: PI (p=0.18)
#   - Cluster: UNDERPOWERED (no bridge refutes; Bridge 2 HELDs,
#     others PI).
#
# Substantive case-study implication: the σ_Λ_a / Λ_a axis as a
# publishable cross-env discriminator at γ=0.999 needs walked
# back. What remains: per-env single-cell observations (Asterix
# anchors high-σ_Λ_a + harm corner; FR anchors low-σ_Λ_a + strong
# help corner). These are descriptive, not a tested cross-env
# moderation.


BRIDGES: tuple[Bridge, ...] = (
    sigma_lambda_a_moderates_ddqn_outcome__cross_env_g0999,
    lambda_a_does_not_mediate_outcome__cross_stratum_g0999,
    joint_bias_geometry_mediates_arm_outcome__cross_env_g0999,
)
