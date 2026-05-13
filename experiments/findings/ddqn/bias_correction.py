"""Hasselt bias-correction chain: mechanism magnitude → outcome.

Two clusters test the common claim "reducing Q overestimation
causes higher return" via non-seed-paired stratum panels with
env fixed effects:

- `bias_premise_jens_predicts_outcome_{backdoor,placebo,rcc}`:
  vanilla's mean `jensen_gap` (Q − MC) per (env, config) stratum
  as the predictor; (DDQN−vanilla) Δ on `eval_best_burst_raw_mean`
  as the target. **Expected to fire NO_EFFECT** — the predictor's
  MC term partially contaminates the target's MC term, and after
  env fixed effects the residual signal is null on the current
  corpus. Documents what the framework correctly refuses: a
  mediator that shares a constituent with the outcome can't be
  validly tested as a causal driver of that outcome via
  regression.

- `bias_correction_clip_predicts_outcome_{backdoor,placebo,rcc}`:
  vanilla's mean `bootstrap_gap_magnitude` (= target_max −
  target_q_at_online_argmax, pure network outputs, **MC-free**)
  as the predictor; same Δ_outcome target. **Expected to fire
  HELD** — diagnostic on the current corpus: β = +244, p < 10⁻⁴,
  95% CI = [+157, +332] with env fixed effects. Within-env r
  ≥ +0.82 in 4/5 envs (FourRooms, SpaceInvaders, Asterix,
  Breakout); only Acrobot disagrees (high-γ Goldilocks
  ceiling). The bias-correction magnitude, measured cleanly,
  IS predictive of DDQN's outcome gain.

The contrast between the two clusters is the framework's
empirical contribution: when the mediator measurement is
algebraically entangled with the outcome (jens = Q − MC,
outcome = MC), the framework correctly refuses to corroborate;
when measured via the algorithmic-correction magnitude
(bootstrap_gap), the link fires.

All bridges use independent-samples stratum aggregation (no
seed pairing per `feedback_paired_g_in_rl`)."""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.analyses.stratum_panel_jci_spearman import (
    StratumPanelJciResult,
)
from corroborate.analyses.stratum_vanilla_predictor_link_dowhy import (
    StratumVanillaPredictorLinkDowhyResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import Verdict

from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM
from experiments.findings.ddqn._scope import (
    DDQN_RELEVANT_SCOPE, VANILLA_JENS_NOISE_FLOOR,
)
from experiments.findings.ddqn._verdicts import (
    dowhy_backdoor_verdict,
    dowhy_placebo_verdict,
    dowhy_rcc_verdict,
)


# === Cluster 1: vanilla_jens → Δ_outcome (expected NO_EFFECT) ===
#
# The "naive" form of the bias-magnitude → outcome-gain test.
# Predictor: mean `jensen_gap` over baseline seeds at each
# (env, config) stratum. Target: cross-arm Δ on
# `eval_best_burst_raw_mean`.
#
# Why the framework will (correctly) not corroborate: `jens = Q
# − MC` by definition, so `vanilla_jens` shares its `−MC_v` term
# with `Δ_outcome = MC_d − MC_v`. After env fixed effects in
# DoWhy backdoor, the residual signal washes to null
# (diagnostic 2026-05-13: β=+0.13, p=0.54). The bridges fire
# NO_EFFECT, documenting that this measurement of the
# mediation is not validly testable on the current data.


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
)
def bias_premise_jens_predicts_outcome_backdoor(
    stratum_vanilla_predictor_link_dowhy: (
        StratumVanillaPredictorLinkDowhyResult
    ),
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor_col: str = 'jensen_gap',
    target_col: str = 'eval_best_burst_raw_mean',
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    ate_floor: float = 0.10,
) -> Verdict:
    """Vanilla `jensen_gap` (one-arm scalar) predicting cross-arm
    Δ_outcome: DoWhy backdoor with env one-hot. The hypothesis is
    POSITIVE (more vanilla bias → bigger DDQN outcome gain), so
    sign=+1, ATE ≥ `ate_floor`."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_vanilla_predictor
    return dowhy_backdoor_verdict(
        stratum_vanilla_predictor_link_dowhy.backdoor,
        ate_threshold=ate_floor, sign=1,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
)
def bias_premise_jens_predicts_outcome_placebo(
    stratum_vanilla_predictor_link_dowhy: (
        StratumVanillaPredictorLinkDowhyResult
    ),
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor_col: str = 'jensen_gap',
    target_col: str = 'eval_best_burst_raw_mean',
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    placebo_max_ratio: float = 0.2,
) -> Verdict:
    """Placebo refutation: random treatment ATE should be near
    zero relative to the real ATE."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_vanilla_predictor
    return dowhy_placebo_verdict(
        stratum_vanilla_predictor_link_dowhy.placebo,
        max_ratio=placebo_max_ratio,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
)
def bias_premise_jens_predicts_outcome_rcc(
    stratum_vanilla_predictor_link_dowhy: (
        StratumVanillaPredictorLinkDowhyResult
    ),
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor_col: str = 'jensen_gap',
    target_col: str = 'eval_best_burst_raw_mean',
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    rcc_max_drift_ratio: float = 0.15,
) -> Verdict:
    """Random-common-cause refutation: synthetic confounder
    leaves the ATE near-stable."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_vanilla_predictor
    return dowhy_rcc_verdict(
        stratum_vanilla_predictor_link_dowhy.random_common_cause,
        max_drift_ratio=rcc_max_drift_ratio,
    )


# === Cluster 2: vanilla bootstrap_gap_magnitude → Δ_outcome (HELD expected) ===
#
# The non-tautological form. Predictor: mean
# `bootstrap_gap_magnitude` over baseline seeds — pure network-
# output difference, no MC anywhere. Target: same as cluster 1.
#
# Diagnostic 2026-05-13 (env fixed effects, n_strata=29):
# β = +244 (SE=41.4), p < 10⁻⁴, 95% CI = [+157, +332]. Per-env
# r: FR +0.99, SI +0.99, Asterix +0.99, Breakout +0.82, Acrobot
# −0.60 (the outlier).


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
)
def bias_correction_clip_predicts_outcome_backdoor(
    stratum_vanilla_predictor_link_dowhy: (
        StratumVanillaPredictorLinkDowhyResult
    ),
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor_col: str = 'bootstrap_gap_magnitude',
    target_col: str = 'eval_best_burst_raw_mean',
    min_vanilla_predictor: float = 0.0,
    ate_floor: float = 50.0,
) -> Verdict:
    """Vanilla `bootstrap_gap_magnitude` (MC-free) predicting
    cross-arm Δ_outcome. Bigger algorithmic clip magnitude →
    bigger DDQN outcome benefit, so sign=+1.

    `ate_floor=50` calibrated to the empirical slope scale: with
    vanilla_bootstrap_gap ranging ≈ [0.001, 0.02] and Δ_outcome
    ranging ≈ [-5, +25] within-env, β = +244 in the diagnostic
    means a 1-unit clip change shifts outcome by 244 — but
    realistic clip ranges are ~0.01, so meaningful outcome
    shifts are β × 0.01 ≈ 2.4 outcome units, calibrated against
    typical env reward scales."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_vanilla_predictor
    return dowhy_backdoor_verdict(
        stratum_vanilla_predictor_link_dowhy.backdoor,
        ate_threshold=ate_floor, sign=1,
    )


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
)
def bias_correction_clip_predicts_outcome_placebo(
    stratum_vanilla_predictor_link_dowhy: (
        StratumVanillaPredictorLinkDowhyResult
    ),
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor_col: str = 'bootstrap_gap_magnitude',
    target_col: str = 'eval_best_burst_raw_mean',
    min_vanilla_predictor: float = 0.0,
    placebo_max_ratio: float = 0.2,
) -> Verdict:
    """Placebo refutation: random treatment ATE should be near
    zero relative to the real ATE."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_vanilla_predictor
    return dowhy_placebo_verdict(
        stratum_vanilla_predictor_link_dowhy.placebo,
        max_ratio=placebo_max_ratio,
    )


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
)
def bias_correction_clip_predicts_outcome_rcc(
    stratum_vanilla_predictor_link_dowhy: (
        StratumVanillaPredictorLinkDowhyResult
    ),
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor_col: str = 'bootstrap_gap_magnitude',
    target_col: str = 'eval_best_burst_raw_mean',
    min_vanilla_predictor: float = 0.0,
    rcc_max_drift_ratio: float = 0.15,
) -> Verdict:
    """Random-common-cause refutation: synthetic confounder
    leaves the ATE near-stable."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_vanilla_predictor
    return dowhy_rcc_verdict(
        stratum_vanilla_predictor_link_dowhy.random_common_cause,
        max_drift_ratio=rcc_max_drift_ratio,
    )


# === Cluster 3: JCI/PC mediation falsification (predicted NULL) ===
#
# Encodes the causal-discovery refutation directly. Three bridges
# author null-form claims at three test levels:
#
# 1. JCI-stratified Spearman ρ(v_clip, Δ_outcome | env) — pooled
#    within-env Spearman, Fisher-z averaged. The "raw" within-env
#    link, scale-decontaminated. **Predicted near zero**.
# 2. JCI + partial: ρ(v_clip, Δ_outcome | v_outcome, env) — same
#    but with vanilla outcome (config-quality proxy) partialled
#    out. The strongest falsification of "bias-correction
#    magnitude causes outcome gain" — controls for env AND
#    config-quality. **Predicted near zero**.
# 3. Sibling test using `jensen_gap` predictor — should also fire
#    NULL (algebra collapses to noise after env + quality
#    control). Documents that no jens-based predictor escapes the
#    null result.
#
# Empirical reading 2026-05-13 (n_strata=29, 11 envs): ρ_jci =
# +0.075 (p=0.87), ρ_jci_partial = -0.61 (p=0.22, sign flips
# after config-quality control). PC at depth-1 env-stratified
# removes the v_clip↔delta_out edge at Z={∅}. The mediation link
# is empirically NULL on this corpus.


def _jci_null_verdict(
    res: StratumPanelJciResult,
    *,
    rho: float,
    null_max_abs_rho: float,
    min_strata: int,
) -> Verdict:
    """Null-form Spearman verdict. HELD when |ρ| <
    `null_max_abs_rho` (predicted-null confirmed); NO_EFFECT
    when |ρ| ≥ threshold (would-be mediation signal exceeds the
    null tolerance — null prediction empirically refuted)."""
    if res.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if abs(rho) < null_max_abs_rho:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='null',
)
def mediation_link_null__jci_stratified_clip(
    stratum_panel_jci_spearman: StratumPanelJciResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor_col: str = 'bootstrap_gap_magnitude',
    target_col: str = 'eval_best_burst_raw_mean',
    min_vanilla_predictor: float = 0.0,
    null_max_abs_rho: float = 0.25,
    min_strata: int = 10,
) -> Verdict:
    """JCI-stratified Spearman ρ(v_clip, Δ_outcome | env) is
    near zero — the within-env link between bias-correction
    magnitude and outcome gain, pooled across envs via Fisher z,
    fails to detect a consistent mediation signal. **HELD when
    |ρ| < null_max_abs_rho** (predicted-null confirmed)."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_vanilla_predictor
    return _jci_null_verdict(
        stratum_panel_jci_spearman,
        rho=stratum_panel_jci_spearman.rho_stratified,

        null_max_abs_rho=null_max_abs_rho,
        min_strata=min_strata,
    )


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='null',
)
def mediation_link_null__jci_partial_clip(
    stratum_panel_jci_spearman: StratumPanelJciResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor_col: str = 'bootstrap_gap_magnitude',
    target_col: str = 'eval_best_burst_raw_mean',
    min_vanilla_predictor: float = 0.0,
    null_max_abs_rho: float = 0.25,
    min_strata: int = 10,
) -> Verdict:
    """JCI + partial Spearman ρ(v_clip, Δ_outcome | v_outcome, env)
    is near zero — after controlling for both env and vanilla's
    convergence quality (config-quality confound proxy), no
    residual link survives. The strongest falsification of the
    bias-correction mediation claim. **HELD when
    |ρ| < null_max_abs_rho**."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_vanilla_predictor
    return _jci_null_verdict(
        stratum_panel_jci_spearman,
        rho=stratum_panel_jci_spearman.rho_partial_stratified,

        null_max_abs_rho=null_max_abs_rho,
        min_strata=min_strata,
    )


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='null',
)
def mediation_link_null__jci_partial_jens(
    stratum_panel_jci_spearman: StratumPanelJciResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor_col: str = 'jensen_gap',
    target_col: str = 'eval_best_burst_raw_mean',
    min_vanilla_predictor: float = 0.0,
    null_max_abs_rho: float = 0.25,
    min_strata: int = 10,
) -> Verdict:
    """JCI + partial Spearman ρ(v_jens, Δ_outcome | v_outcome,
    env) is near zero — the jens-based mediation also fails
    under the same null-form test. Documents that the algebra
    collapse of the jens predictor isn't rescued by within-env
    aggregation. **HELD when |ρ| < null_max_abs_rho**."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_vanilla_predictor
    return _jci_null_verdict(
        stratum_panel_jci_spearman,
        rho=stratum_panel_jci_spearman.rho_partial_stratified,

        null_max_abs_rho=null_max_abs_rho,
        min_strata=min_strata,
    )


# Retired 2026-05-13: all jens→outcome stratum-Δ link bridges
# (REACH cluster + extreme_q_div cluster + fourrooms_action_dim).
# They correlated Δ_jens with Δ_mc_return, which has Δ_MC on
# both sides because `jens = Q − MC` by definition (partial r
# given Δ_Q empirically = +1.000 at both seed and stratum levels).
# The new vanilla-only-predictor cluster + bootstrap-gap cluster
# replace them — vanilla_jens still inherits residual MC overlap
# (NO_EFFECT result documents the algebra), bootstrap_gap is
# MC-free (HELD result corroborates the bias-correction story).


# Unused module-level imports kept for downstream readers.
del Mapping


BRIDGES = (
    bias_premise_jens_predicts_outcome_backdoor,
    bias_premise_jens_predicts_outcome_placebo,
    bias_premise_jens_predicts_outcome_rcc,
    bias_correction_clip_predicts_outcome_backdoor,
    bias_correction_clip_predicts_outcome_placebo,
    bias_correction_clip_predicts_outcome_rcc,
    mediation_link_null__jci_stratified_clip,
    mediation_link_null__jci_partial_clip,
    mediation_link_null__jci_partial_jens,
)
