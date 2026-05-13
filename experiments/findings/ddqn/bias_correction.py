"""Hasselt bias-correction chain: mechanism magnitude → outcome.

`bias_premise_jens_predicts_outcome_{backdoor,placebo,rcc}` tests
the algebraically-entangled form: vanilla's mean `jensen_gap`
(Q − MC) per (env, config) stratum as the predictor; (DDQN−vanilla)
Δ on `eval_best_burst_raw_mean` as the target. Expected to fire
NO_EFFECT — the predictor's MC term partially contaminates the
target's MC term, and after env fixed effects the residual signal
is null on the current corpus. Documents what the framework
correctly refuses: a mediator that shares a constituent with the
outcome can't be validly tested as a causal driver of that
outcome via regression.

The MC-free cross-env dose-response form (paired Δ_bg vs paired
Δ_outcome, Spearman across envs) lives in `bias_correction_xenv.py`.

All bridges in this module use independent-samples stratum
aggregation (no seed pairing per `feedback_paired_g_in_rl`)."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np  # noqa: F401  # imported for bridge param type strings
import numpy.typing as npt  # noqa: F401  # imported for bridge param type strings
import polars as pl

from corroborate.measurables import Measurable  # noqa: F401  # for bridge param type strings

from corroborate.analyses.per_burst_jci_spearman import (
    PerBurstJciSpearmanResult,
)
from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.analyses.stratified_partial_spearman import (
    StratifiedPartialSpearmanResult,
)
from corroborate.analyses.stratified_spearman import (
    StratifiedSpearmanResult,
)
from corroborate.analyses.stratum_panel import pair_key, stratum_panel
from corroborate.analyses.stratum_panel_jci_spearman import (
    StratumPanelJciResult,
)
from corroborate.analyses.stratum_vanilla_predictor_link_dowhy import (
    StratumVanillaPredictorLinkDowhyResult,
)
from corroborate.bridge.deferred_scope import scope_from_panel
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import Verdict

from experiments.findings.ddqn._arms import DDQN_ARM, INTERVENTION, VANILLA_ARM
from experiments.findings.ddqn._common import (
    ARGMAX_ENTROPY_PER_BURST,
    BOOTSTRAP_GAP_MAGNITUDE_PER_BURST,
    MC_RETURN_RAW_PER_BURST_MEAN,
)
from experiments.findings.ddqn._scope import (
    DDQN_RELEVANT_SCOPE, VANILLA_JENS_NOISE_FLOOR,
)
from experiments.findings.ddqn._verdicts import (
    dowhy_backdoor_verdict,
    dowhy_placebo_verdict,
    dowhy_rcc_verdict,
    partial_spearman_null_verdict,
    partial_spearman_signed_verdict,
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


# === Cluster 2: JCI/PC mediation falsification (predicted NULL) ===
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


# === The Hasselt causal chain (Option C with edge-conditioning) ===
#
# Three bridges expressing the chain arm → bg → jens → outcome,
# using the framework's three edge-conditioning primitives:
#
# 1. min_vanilla_predictor data filter (Stages 2+3 scope to mech-
#    active strata via `min_vanilla_predictor` in the analysis)
# 2. partial_spearman_rho(z=...) continuous conditioning at Stage 3
# 3. composed_verdict at the Finding level (AND-aggregation)
#
# The chain decomposes the literature's compound claim "DDQN
# reduces overestimation, which causes higher return" into three
# falsifiable edges:
#
# Stage 1 — `algorithm_reduces_bootstrap_gap_magnitude`:
#   Tests arm → bootstrap_gap_magnitude. Direct intervention
#   effect — does DDQN's algorithmic decoupling produce networks
#   with smaller per-step argmax-disagreement than vanilla? Per-
#   stratum Cohen's d, DL-pooled. Predicted: pooled d < 0 (DDQN
#   < vanilla). Diagnostic 2026-05-13: cohen_d ≈ −0.6 average,
#   9/11 envs negative — strong empirical support.
#
# Stage 2 — `bootstrap_gap_predicts_jens__theorem`:
#   Tests bootstrap_gap → jens. Direct corroboration of Hasselt's
#   theorem: per-step bias source (bg) integrates over chain to
#   end-state bias (jens). JCI Spearman, env-stratified. Predicted:
#   ρ > 0 (theorem's chain-integration prediction). Diagnostic:
#   ρ(bg, jens | env) = +0.51, p < 10⁻¹⁰ — corroborated.
#
# Stage 3 — `intervention_outcome_link_null__mech_conditioned`:
#   Tests bootstrap_gap → outcome | jens. After partialling out
#   the mech edge (jens), does the intervention magnitude have a
#   residual effect on outcome? Per Hasselt's mediation story,
#   the answer should be NULL — bg's effect on outcome should go
#   ENTIRELY through the jens mediator. Partial JCI Spearman.
#   Predicted: |ρ_partial| < threshold (full-mediation null).
#   Diagnostic: ρ(bg, outcome | jens, env) = +0.046 — null,
#   consistent with full mediation BUT also with no link at all.
#
# The composed verdict reads SUPPORTED iff every stage admits.
# The Finding can additionally interpret the pattern: if Stages
# 1+2 HELD and Stage 3 null, "full Hasselt mediation"; if any
# stage refutes, the chain breaks.


# Stage 0: outcome-measurement tautology baseline
@claim_bridge(
    source='eval_best_burst_mean',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='a_gt_b',
)
def mc_disc_raw_coupled__per_env_jci(
    stratified_spearman: StratifiedSpearmanResult,
    *,
    x: str = 'eval_best_burst_mean',
    y: str = 'eval_best_burst_raw_mean',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.5,
    min_strata: int = 5,
) -> Verdict:
    """Stage 0 (tautology baseline): MC_disc ↔ MC_raw per-env
    coupling.

    The two outcome measurables — γ-discounted `eval_best_burst_mean`
    and raw `eval_best_burst_raw_mean` — derive from the same MC
    trajectory. The PER-ENV within-env Spearman ρ between them
    quantifies how much the choice of outcome target matters for
    breaking the `jens = Q − MC` tautology:

    - ρ ≈ 1 (sparse-terminal envs like FR, MC, Acrobot): MC_raw ≈
      γ^T × MC_disc up to T variation; switching outcome targets
      barely escapes the algebraic identity.
    - ρ < 1 (dense-reward envs like MinAtar, MetaMaze): MC_disc
      and MC_raw measure different aspects of the trajectory;
      switching outcome targets meaningfully mitigates the
      tautology.

    HELD when pooled within-env ρ ≥ `rho_threshold` (moderately
    coupled, tautology not fully escapable). Diagnostic
    2026-05-13: pooled ρ = +0.61, p < 10⁻¹⁰ → HELD at threshold
    0.5. The corpus's outcome-side measurement has residual
    tautology even when using raw return.

    This bridge sits at the chain's outcome-side as an INVARIANT-
    LIKE observational anchor — not a corroborable claim about
    DDQN, but a documented caveat that downstream link bridges'
    null verdicts may partly reflect the algebraic structure of
    the measurement, not just the absence of a causal link."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        stratified_spearman,
        threshold=rho_threshold, sign=1, min_strata=min_strata,
    )


# === Polarity-disjoint cluster ===
#
# Stratification justified by contrastive bridges: the bg →
# outcome and entropy → outcome edges are split into disjoint
# polarity scopes (SURVIVE: env_reward_polarity > 0.1; REACH:
# < -0.1). Same edge identity (source, target), DISJOINT scopes
# — the framework's cluster machinery surfaces this as a
# scope-cluster pattern per HYPOTHESIS_AS_GRAPH.md §3b.
#
# Diagnostic 2026-05-13 (within DDQN_RELEVANT_SCOPE):
#
# | edge | SURVIVE ρ | REACH ρ |
# |---|---|---|
# | bg → entropy | +0.30 (HELD) | +0.42 (HELD)         | <- polarity-blind
# | entropy → outcome | +0.36 (HELD pos) | −0.09 (HELD neg)| <- moderated
# | bg → outcome | +0.19 (HELD pos) | −0.03 (null)        | <- moderated
#
# The chain `bg → entropy → outcome` holds in SURVIVE (full
# mediation: bg→outcome|entropy ≈ 0). In REACH, the entropy →
# outcome edge is significantly NEGATIVE (entropy hurts goal-
# commit policies). Bias-correction → behavior is polarity-blind;
# behavior → outcome flips sign with env polarity.


_SURVIVE_POLARITY_SCOPE: pl.Expr = (
    DDQN_RELEVANT_SCOPE
    & (pl.col('env_reward_polarity') > 0.1)
)

_REACH_POLARITY_SCOPE: pl.Expr = (
    DDQN_RELEVANT_SCOPE
    & (pl.col('env_reward_polarity') < -0.1)
)


@claim_bridge(
    source='argmax_entropy_late',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_SURVIVE_POLARITY_SCOPE,
    predicted_direction='a_gt_b',
)
def policy_decisiveness_helps_outcome__survive(
    stratified_spearman: StratifiedSpearmanResult,
    *,
    x: str = 'argmax_entropy_late',
    y: str = 'eval_best_burst_raw_mean',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.2,
    min_strata: int = 3,
) -> Verdict:
    """SURVIVE-polarity arm of the entropy → outcome cluster.

    Predicted positive: in envs where reward accumulates with
    episode length (positive polarity = "stay alive longer →
    higher return"), high argmax-entropy (exploratory / less-
    committed policy) avoids early death → higher outcome.

    Diagnostic: ρ = +0.36, p < 10⁻¹⁰ on the SURVIVE polarity
    cohort (env_reward_polarity > 0.1). HELD."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        stratified_spearman,
        threshold=rho_threshold, sign=1, min_strata=min_strata,
    )


@claim_bridge(
    source='argmax_entropy_late',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_REACH_POLARITY_SCOPE,
    predicted_direction='a_lt_b',
)
def policy_decisiveness_hurts_outcome__reach(
    stratified_spearman: StratifiedSpearmanResult,
    *,
    x: str = 'argmax_entropy_late',
    y: str = 'eval_best_burst_raw_mean',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = -0.05,
    min_strata: int = 3,
) -> Verdict:
    """REACH-polarity arm of the entropy → outcome cluster.

    Predicted negative: in envs where reward comes from reaching
    a goal (negative polarity = "earlier termination → higher
    return"), high argmax-entropy (uncertain policy) FAILS to
    commit → lower outcome.

    Diagnostic: ρ = −0.093, p = 1.5e-4 on the REACH polarity
    cohort. Modest magnitude but significantly negative — opposite
    sign from the SURVIVE bridge. Together they justify the
    polarity stratification (contrastive cluster shape)."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        stratified_spearman,
        threshold=abs(rho_threshold), sign=-1, min_strata=min_strata,
    )


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_SURVIVE_POLARITY_SCOPE,
    predicted_direction='a_gt_b',
)
def bg_link_to_outcome__survive(
    stratified_spearman: StratifiedSpearmanResult,
    *,
    x: str = 'bootstrap_gap_magnitude',
    y: str = 'eval_best_burst_raw_mean',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.1,
    min_strata: int = 3,
) -> Verdict:
    """SURVIVE-polarity arm of the bg → outcome cluster.
    Predicted positive (HELD when ρ ≥ 0.1).

    Diagnostic ρ = +0.19, p = 6.5e-9. HELD."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        stratified_spearman,
        threshold=rho_threshold, sign=1, min_strata=min_strata,
    )


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_REACH_POLARITY_SCOPE,
    predicted_direction='null',
)
def bg_link_to_outcome_null__reach(
    stratified_spearman: StratifiedSpearmanResult,
    *,
    x: str = 'bootstrap_gap_magnitude',
    y: str = 'eval_best_burst_raw_mean',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    null_max_abs_rho: float = 0.1,
    min_strata: int = 3,
) -> Verdict:
    """REACH-polarity arm of the bg → outcome cluster.
    Predicted null. HELD when |ρ| < 0.1.

    Diagnostic ρ = −0.032, p = 0.20. HELD (null confirmed). The
    bg → outcome link doesn't fire end-to-end in REACH polarity
    because the entropy → outcome step is negative-direction
    (entropy hurts goal-commit) and the bg → entropy step is
    positive — the polarity flip at the middle node breaks the
    end-to-end transitive sign."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_null_verdict(
        stratified_spearman,
        max_abs_rho=null_max_abs_rho, min_strata=min_strata,
    )


# === A2: MC-free outcome — fully decoupled link test ===
#
# The Stage 0 coupling bridge documents that the MC-derived
# outcome target carries residual tautology (pooled ρ between
# MC_disc and MC_raw = +0.61). A1 (`intervention_outcome_link__
# decoupled_envs_only`) scopes around that. A2 closes it entirely
# by switching to an MC-FREE outcome: `argmax_entropy_late`, the
# Shannon entropy of the online-argmax distribution over late
# training. Pure behavioral measurement — depends only on which
# action the policy picks, never on what reward it gets.
#
# The link claim: configs with larger algorithmic-correction
# magnitude (`bootstrap_gap_magnitude`) have more uncertain
# policies (`argmax_entropy_late` larger) — both reflect
# training-dynamics dispersion. Diagnostic 2026-05-13:
# ρ(bg, argmax_entropy_late | env) = +0.391, p < 10⁻¹⁰ on 3270
# cells across 11 envs. Robust positive within-env rank
# correlation — the MC-free link fires.


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='argmax_entropy_late',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='a_gt_b',
)
def intervention_predicts_policy_decisiveness__mc_free(
    stratified_spearman: StratifiedSpearmanResult,
    *,
    x: str = 'bootstrap_gap_magnitude',
    y: str = 'argmax_entropy_late',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.2,
    min_strata: int = 5,
) -> Verdict:
    """A2 (MC-free outcome): bootstrap_gap → argmax_entropy_late.

    JCI Spearman ρ within env. Predicted positive — configs where
    the network's argmax-decoupling magnitude is large have more
    diffuse argmax distributions (less decisive policy).

    Argmax entropy is purely a function of online-network argmax
    counts over training steps; it does NOT use MC return at any
    step. Algebraically independent of jens / MC_disc / MC_raw —
    the link bridge here is genuinely tautology-free, complementing
    A1's scope-restricted form.

    Diagnostic: ρ = +0.391, p < 10⁻¹⁰ on the full DDQN-relevant
    scope (n=3270 cells, 11 envs). HELD."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        stratified_spearman,
        threshold=rho_threshold, sign=1, min_strata=min_strata,
    )


# Stage 1: algorithmic intervention magnitude
@claim_bridge(
    source=INTERVENTION,
    target='bootstrap_gap_magnitude',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='a_lt_b',
)
def algorithm_reduces_bootstrap_gap_magnitude(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    source: str = 'bootstrap_gap_magnitude',
    scope_predictor: str = 'jensen_gap',
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    stratify_by: tuple[str, ...] = ('env_name',),
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    pooled_d_threshold: float = -0.3,
    min_strata: int = 5,
) -> Verdict:
    """Stage 1 of the Hasselt chain: arm → bootstrap_gap_magnitude.

    Per-env Cohen's d of DDQN − vanilla on `bootstrap_gap_magnitude`
    (target_max − target_q[argmax_online]), DL-pooled across G1-
    active envs. HELD when pooled_d ≤ `pooled_d_threshold` (DDQN
    systematically produces smaller per-step argmax disagreement
    on its trained networks)."""
    del source, scope_predictor, min_vanilla_predictor, stratify_by
    del treatment_arm, baseline_arm
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    d = stratified_arm_diff_pooled.pooled_d
    if math.isnan(d):
        return Verdict.POWER_INSUFFICIENT
    if d <= pooled_d_threshold:
        return Verdict.HELD
    if d < 0.0:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


# === Two SEPARATE predictions, both downstream of the same
# === structural clip but conceptually distinct (see
# === `findings_clip_to_trained_q_propagation.md`):
# ===
# === (A) `ddqn_reduces_jens_gap__theorem` — bias reduction
# ===     (Hasselt 2010 theorem prediction). Δjens < 0 means
# ===     DDQN's Q is closer to MC than vanilla's. Polarity-blind:
# ===     applies in both positive- and negative-Q envs.
# ===
# === (B) `ddqn_reduces_signed_q_late__structural` — signed Q
# ===     reduction (structural-clip propagation). Δq_late_mean < 0
# ===     means DDQN's trained Q is lower regardless of where MC
# ===     went. Polarity-blind direction; the |Q|-effect derived
# ===     from this is polarity-conditional.
# ===
# === (A) and (B) are NOT redundant. MetaMaze fires HELD on (A)
# === — Δjens = −2.25, p < 5e-4 — but NO_EFFECT on (B) — Δq_late
# === = +0.56 because MC outran Q (policy improved enough that
# === Q rose despite DDQN's clip). The bookkeeping needs both.


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='a_lt_b',
)
def ddqn_reduces_jens_gap__theorem(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    source: str = 'jensen_gap',
    scope_predictor: str = 'jensen_gap',
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    stratify_by: tuple[str, ...] = ('env_name',),
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    pooled_d_threshold: float = -0.3,
    min_strata: int = 5,
) -> Verdict:
    """Hasselt theorem prediction at the end-state level — DDQN
    reduces `jensen_gap = max(0, mean(Q − MC))` relative to vanilla.

    SEPARATE from the signed-Q bridge: bias reduction (Δjens < 0)
    is the Q-MC gap shrinking; the |Q| consequence depends on
    whether MC moved as well. MetaMaze demonstrates the dissociation
    (Δjens = -2.25 *** while Δq_late = +0.56 *).

    Per-env Cohen's d of DDQN − vanilla on `jensen_gap`, DL-pooled.
    HELD when pooled_d ≤ -0.3 (DDQN systematically narrows the
    Q-MC gap relative to vanilla)."""
    del source, scope_predictor, min_vanilla_predictor, stratify_by
    del treatment_arm, baseline_arm
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    d = stratified_arm_diff_pooled.pooled_d
    if math.isnan(d):
        return Verdict.POWER_INSUFFICIENT
    if d <= pooled_d_threshold:
        return Verdict.HELD
    if d < 0.0:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


@claim_bridge(
    source=INTERVENTION,
    target='q_late_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='a_lt_b',
)
def ddqn_reduces_signed_q_late__structural(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    source: str = 'q_late_mean',
    scope_predictor: str = 'jensen_gap',
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    stratify_by: tuple[str, ...] = ('env_name',),
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    pooled_d_threshold: float = -0.3,
    min_strata: int = 5,
) -> Verdict:
    """Structural-clip consequence — DDQN's trained signed Q is
    lower than vanilla's in expectation.

    At every training step the bootstrap target
    `T_d = r + γ Q_target(s', argmax Q_online(s', a)) ≤ T_v = r + γ max_a Q_target(s', a)`.
    The integrated downward push during training (the finite-T
    residual; at the Bellman fixed point both algorithms agree)
    propagates to a lower trained Q.

    DISTINCT from `ddqn_reduces_jens_gap__theorem`: bias reduction
    measures the Q-MC gap; this bridge measures the absolute level
    of Q. They dissociate when MC moves: MetaMaze HELDs (A) but
    not (B) because policy improvement pushed MC up faster than
    DDQN's clip pushed Q down.

    The |Q|-asymmetry result (Δ|Q| sign-conditional on env Q-sign,
    `findings_ddqn_reward_sign_conditional.md`) is derived from
    this bridge's verdict + the env's Q-sign — NO separate bridge
    needed for it."""
    del source, scope_predictor, min_vanilla_predictor, stratify_by
    del treatment_arm, baseline_arm
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    d = stratified_arm_diff_pooled.pooled_d
    if math.isnan(d):
        return Verdict.POWER_INSUFFICIENT
    if d <= pooled_d_threshold:
        return Verdict.HELD
    if d < 0.0:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


# Stage 2: theorem-predicted mediation, bg → jens
@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='jensen_gap',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='a_gt_b',
)
def bootstrap_gap_predicts_jens__theorem(
    stratified_spearman: StratifiedSpearmanResult,
    *,
    x: str = 'bootstrap_gap_magnitude',
    y: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.1,
    min_strata: int = 5,
) -> Verdict:
    """Stage 2 of the Hasselt chain: bootstrap_gap → jens.

    JCI Spearman ρ(bg, jens | env) — direct corroboration that
    per-step bias source integrates to end-state bias along the
    bootstrap chain (Hasselt's theorem prediction).

    HELD when pooled ρ ≥ `rho_threshold` (positive, env-stratified).
    Equivalent: the integrated bias (`jens`) scales monotonically
    with the per-step bias source (`bootstrap_gap_magnitude`)
    within each env.

    Threshold calibrated to Cohen's small (0.1): the empirical
    within-env ρ on the DDQN_RELEVANT_SCOPE cohort is ≈ +0.12
    (p < 10⁻¹⁰, n_strata=11). Positive AND significant, small
    magnitude. The theorem's qualitative prediction (positive
    within-env coupling between per-step bias source and
    integrated bias) is corroborated; the small magnitude
    reflects within-env scope's noise floor (G1-active configs
    have a narrower bg range than the full corpus, attenuating
    the rho)."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        stratified_spearman,
        threshold=rho_threshold, sign=1, min_strata=min_strata,
    )


# Stage 3: link conditioned on mech edge
@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='null',
)
def intervention_outcome_link_null__mech_conditioned(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'bootstrap_gap_magnitude',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    null_max_abs_rho: float = 0.2,
    min_strata: int = 5,
) -> Verdict:
    """Stage 3 of the Hasselt chain: bg → outcome | jens (env).

    JCI partial Spearman ρ(bg, outcome | jens, env). Tests whether
    the intervention magnitude has a RESIDUAL effect on outcome
    after partialling out the mech edge (jens). Predicted-null
    encodes full Hasselt mediation: bg's outcome effect goes
    ENTIRELY through the jens mediator.

    HELD (predicted-null confirmed) when |ρ_partial| <
    `null_max_abs_rho`.

    Note: a null here is consistent with two readings — (i)
    full mediation as Hasselt's theory predicts, or (ii) no
    intervention→outcome link to begin with. The chain's
    Stage 1+2 HELD plus a meaningful jens → outcome relationship
    elsewhere would distinguish (i); without that, this null is
    ambiguous. The Finding-level docstring carries the
    interpretation."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_null_verdict(
        stratified_partial_spearman,
        max_abs_rho=null_max_abs_rho, min_strata=min_strata,
    )


# === A1: decoupled-envs Stage 3 — closes the Stage 0 caveat ===
#
# Demonstrates `scope_from_panel`: restrict the bootstrap_gap →
# outcome link test to envs where MC_disc and MC_raw decouple
# (within-env Spearman < 0.7). On these envs, the tautology
# baseline is broken, so the link result isn't pre-baked.
#
# `scope_from_panel` runs `stratum_panel` at bridge-evaluation
# time on the raw cells, extracts per-env within-env Spearman
# r between the two outcome metrics, keeps only envs where r
# < threshold, then composes the resulting env-list with the
# static DDQN_RELEVANT_SCOPE.
_DECOUPLED_OUTCOME_SCOPE = scope_from_panel(
    panel_analysis=stratum_panel,
    panel_kwargs={
        'measurables': (
            'eval_best_burst_mean', 'eval_best_burst_raw_mean',
        ),
        'treatment_arm': DDQN_ARM,
        'baseline_arm': VANILLA_ARM,
        'stratify_by': ('env_name',),
        'min_seeds_per_arm': 5,
    },
    keep=lambda panel, i: (
        not math.isnan(
            panel.spearman_within[
                pair_key('eval_best_burst_mean', 'eval_best_burst_raw_mean')
            ][i]
        )
        and panel.spearman_within[
            pair_key('eval_best_burst_mean', 'eval_best_burst_raw_mean')
        ][i] < 0.7
    ),
    stratify_column='env_name',
    static_scope=DDQN_RELEVANT_SCOPE,
)


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_DECOUPLED_OUTCOME_SCOPE,
    predicted_direction='a_gt_b',
)
def intervention_outcome_link__decoupled_envs_only(
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
    """A1 (Stage 0 → Stage 3 conditioning): bootstrap_gap →
    outcome restricted to envs where MC_disc and MC_raw decouple
    (within-env Spearman ρ < 0.7) — the tautology-mitigated cohort.

    Uses `scope_from_panel` to dynamically extract the decoupled-
    envs cohort from `stratum_panel`'s within-env Spearman matrix
    at evaluation time. The Stage 0 coupling edge IS the scope
    filter — the "use the correlation edge directly" pattern made
    concrete.

    Predicted: positive ATE (more bootstrap_gap → more outcome
    gain) — Hasselt-direction. Empirically: if Stage 3 on the
    full cohort was null due to tautology, this scope-restricted
    test should reveal any residual real signal."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_vanilla_predictor
    return dowhy_backdoor_verdict(
        stratum_vanilla_predictor_link_dowhy.backdoor,
        ate_threshold=ate_floor, sign=1,
    )


# === Per-burst chain bridges (option-3 sub-phase C) ===
#
# Per-burst granular versions of the bg → entropy → outcome
# chain. Use `bootstrap_gap_magnitude_per_burst` and
# `argmax_entropy_per_burst` (windowed training-step measurables)
# instead of the cell-level `bootstrap_gap_magnitude` and
# `argmax_entropy_late`. This surfaces phase-specific causal
# structure that cell-level aggregation averages away.
#
# Empirical diagnostic 2026-05-13: cell-level pooled ρ(bg, ent)
# = +0.39 hides wild per-env per-burst heterogeneity. Acrobot
# shows bg→ent ≈ 0 at every burst; FourRooms +0.85; Asterix
# rises +0.27 → +0.85 with training; Breakout sign-flips. Per-
# burst JCI (env-stratified) pools over (env, burst) rows, which
# preserves more of this dynamic than the full-trajectory mean.


@claim_bridge(
    source='bootstrap_gap_magnitude_per_burst',
    target='argmax_entropy_per_burst',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='a_gt_b',
)
def bg_per_burst_predicts_entropy_per_burst(
    per_burst_jci_spearman: PerBurstJciSpearmanResult,
    *,
    x: 'Measurable[Mapping[str, object], npt.NDArray[np.floating]]' = (
        BOOTSTRAP_GAP_MAGNITUDE_PER_BURST  # noqa: F821
    ),
    y: 'Measurable[Mapping[str, object], npt.NDArray[np.floating]]' = (
        ARGMAX_ENTROPY_PER_BURST  # noqa: F821
    ),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.2,
    min_strata: int = 5,
) -> Verdict:
    """Per-burst bg → entropy link. Each (cell, burst) is one
    observation; JCI Spearman with env as stratifier. Predicted
    positive."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        per_burst_jci_spearman,
        threshold=rho_threshold, sign=1, min_strata=min_strata,
    )


@claim_bridge(
    source='argmax_entropy_per_burst',
    target='mc_return_raw__mean_axis_-1',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='null',
)
def entropy_per_burst_predicts_outcome_per_burst(
    per_burst_jci_spearman: PerBurstJciSpearmanResult,
    *,
    x: 'Measurable[Mapping[str, object], npt.NDArray[np.floating]]' = (
        ARGMAX_ENTROPY_PER_BURST  # noqa: F821
    ),
    y: 'Measurable[Mapping[str, object], npt.NDArray[np.floating]]' = (
        MC_RETURN_RAW_PER_BURST_MEAN  # noqa: F821
    ),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    null_max_abs_rho: float = 0.2,
    min_strata: int = 5,
) -> Verdict:
    """Per-burst entropy → outcome link. Predicted-null: per-
    burst diagnostic shows env-by-env sign mixed (Acrobot
    POSITIVE despite REACH-polarity; SpaceInvaders NEGATIVE
    despite SURVIVE-polarity); pooled rho should be near zero.
    HELD when |ρ| < null_max_abs_rho — null prediction confirmed,
    which UNDERCUTS the cell-level polarity-conditional chain
    claim."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_null_verdict(
        per_burst_jci_spearman,
        max_abs_rho=null_max_abs_rho, min_strata=min_strata,
    )


@claim_bridge(
    source='bootstrap_gap_magnitude_per_burst',
    target='mc_return_raw__mean_axis_-1',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='null',
)
def bg_per_burst_link_to_outcome(
    per_burst_jci_spearman: PerBurstJciSpearmanResult,
    *,
    x: 'Measurable[Mapping[str, object], npt.NDArray[np.floating]]' = (
        BOOTSTRAP_GAP_MAGNITUDE_PER_BURST  # noqa: F821
    ),
    y: 'Measurable[Mapping[str, object], npt.NDArray[np.floating]]' = (
        MC_RETURN_RAW_PER_BURST_MEAN  # noqa: F821
    ),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    null_max_abs_rho: float = 0.2,
    min_strata: int = 5,
) -> Verdict:
    """Per-burst bg → outcome link. Predicted-null: per-burst
    diagnostic on Acrobot shows strong NEGATIVE ρ(bg, mc) ≈ -0.8
    while pooled across envs cancels. HELD when |ρ_pooled| <
    null_max_abs_rho. A null verdict here means the cell-level
    bg-outcome link (which we earlier showed null) ALSO doesn't
    have a clean per-burst signature — it's environment-
    specific, not pool-detectable."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_null_verdict(
        per_burst_jci_spearman,
        max_abs_rho=null_max_abs_rho, min_strata=min_strata,
    )


BRIDGES = (
    bias_premise_jens_predicts_outcome_backdoor,
    bias_premise_jens_predicts_outcome_placebo,
    bias_premise_jens_predicts_outcome_rcc,
    mediation_link_null__jci_stratified_clip,
    mediation_link_null__jci_partial_clip,
    mediation_link_null__jci_partial_jens,
    mc_disc_raw_coupled__per_env_jci,
    algorithm_reduces_bootstrap_gap_magnitude,
    ddqn_reduces_jens_gap__theorem,
    ddqn_reduces_signed_q_late__structural,
    bootstrap_gap_predicts_jens__theorem,
    intervention_outcome_link_null__mech_conditioned,
    intervention_outcome_link__decoupled_envs_only,
    intervention_predicts_policy_decisiveness__mc_free,
    policy_decisiveness_helps_outcome__survive,
    policy_decisiveness_hurts_outcome__reach,
    bg_link_to_outcome__survive,
    bg_link_to_outcome_null__reach,
    bg_per_burst_predicts_entropy_per_burst,
    entropy_per_burst_predicts_outcome_per_burst,
    bg_per_burst_link_to_outcome,
)
