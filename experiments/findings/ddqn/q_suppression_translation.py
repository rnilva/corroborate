"""Q-suppression vs outcome-translation cross-env bookkeeping.

Empirical observation (2026-05-15, cache `ddqn.parquet` at canonical
1M): for three envs (Asterix-MinAtar, SpaceInvaders-MinAtar,
MetaMaze-misc) the DoWhy mediation through `jensen_gap`
*over-absorbs* to 400-977% (memory
`findings_ddqn_mediator_heterogeneity`) — DDQN reduces jens
strongly, but observed `Δ eval_best_burst_raw_mean ≈ 0`. The
linear-mediation model interprets the residual as an
opposite-direction direct effect; the diagnostic in
`scripts/mech_no_translation_diagnostic.py` shows the
non-translating cohort has DDQN suppressing `q_late_mean` 5-13×
harder than the translating cohort.

The interpretation captured here: **the magnitude of DDQN's
Q-suppression has multiple drivers, not only jens-bias
correction**. Vanilla's `q_late_mean` is set by reward scale,
deadly-triad Q-explosion, polarity structure, and the jens-bias
overhang — DDQN's bootstrap clip suppresses |Q| globally without
selectively isolating the jens-bias component. In high-|Q| envs
where vanilla's bias is observationally large but causally inert
for outcome (the Hasselt-2016-non-universal cohort per memory
`findings_canonical_scope_reverification`), the surplus Q-
suppression doesn't translate to outcome benefit and may carry
side-effects (target staleness up by d≈+2.7 in Asterix/SI).

The bridge cross-env-tests whether `Δ q_late_mean` POSITIVELY
predicts `Δ eval_best_burst_raw_mean` — i.e., less-suppression
envs improve outcome more, more-suppression envs improve outcome
less. Empirical pre-author (n=10 non-saturating): ρ=+0.515
p=0.128. Below the standard HELD gate at n=10 (critical |r|≈0.65
at p=0.05); the LOO sibling drops the PacMan outlier
(Δ_Q=+34, Δ_out=+166) which dominates the rank.

Expected verdict: POWER_INSUFFICIENT at canonical's n=10. The
bridge is **bookkeeping for the observation**; promoting to HELD
needs either (a) more envs, or (b) a designed intervention
(e.g., do(γ) sweep) that isolates Q-magnitude without bias-
correction. Author when verifying with a designed sweep.

Direction: DIRECT — less Q-suppression (Δ_Q closer to 0 or
positive) co-moves with bigger Δ_outcome. Predicted ρ ≥ +0.6 for
HELD at canonical.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import scipy.stats as stats

from corroborate.analyses.link.cross_stratum_arm_diff_slope import (
    CrossStratumArmDiffSlopeResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM
from experiments.findings.ddqn._scope import DDQN_RELEVANT_SCOPE


_PREDICTOR = 'q_late_mean'
_TARGET = 'eval_best_burst_raw_mean'

# CartPole saturated on its canonical (both arms 500-step cap);
# rank-tie noise.
_SATURATING_OUTCOME_ENVS: tuple[str, ...] = ('CartPole-v1',)
_NONSATURATED_SCOPE: pl.Expr = (
    ~pl.col('env_name').is_in(_SATURATING_OUTCOME_ENVS)
)

_SCOPE: pl.Expr = DDQN_RELEVANT_SCOPE & _NONSATURATED_SCOPE


def _spearman_loo_min(
    xs: tuple[float, ...], ys: tuple[float, ...],
) -> float:
    """Min Spearman ρ across all n leave-one-out subsets. NaN when
    any subset's ρ is NaN (tie-degenerate after LOO)."""
    n = len(xs)
    if n < 5:
        return float('nan')
    rhos: list[float] = []
    for i in range(n):
        xs_loo = np.asarray(xs[:i] + xs[i + 1:], dtype=np.float64)
        ys_loo = np.asarray(ys[:i] + ys[i + 1:], dtype=np.float64)
        rho_raw, _ = stats.spearmanr(xs_loo, ys_loo)
        rho_i = float(rho_raw)
        if math.isnan(rho_i):
            return float('nan')
        rhos.append(rho_i)
    return min(rhos)


@claim_bridge(
    source=_PREDICTOR,
    target=_TARGET,
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_SCOPE,
    predicted_direction='a_gt_b',
)
def ddqn_q_suppression_tracks_outcome_translation__xenv(
    cross_stratum_arm_diff_slope: CrossStratumArmDiffSlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor: str = _PREDICTOR,
    target: str = _TARGET,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
    min_strata: int = 8,
    rho_threshold_held: float = 0.6,
    p_threshold: float = 0.05,
    sign_flip_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """Per-env Δ_Q vs Δ_outcome_raw cross-env Spearman.

    Predicts: more Q-suppression (Δ_Q more negative) → smaller
    Δ_outcome. Encoded as positive ρ since both arm-diffs encode
    DDQN−vanilla in the same orientation. The bookkeeping claim
    here: outcome-translation tracks how much of the |Q|
    reduction lands on the jens-bias component versus other
    sources of vanilla's |Q| magnitude. Pre-author empirical
    (canonical 1M, n=10 non-saturating): ρ=+0.515 p=0.128.

    Verdict matrix (mirrors `bias_correction_dose_response`):
      HELD              : ρ ≥ +0.6 AND p ≤ 0.05
      NO_EFFECT (NULL)  : |ρ| < 0.3
      NO_EFFECT (SIGN_FLIP) : ρ ≤ −0.3
      POWER_INSUFFICIENT : otherwise / n_strata < 8
    """
    del (
        treatment_arm, baseline_arm, predictor, target,
        stratify_by, min_seeds_per_arm,
    )

    if cross_stratum_arm_diff_slope.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None

    rho = cross_stratum_arm_diff_slope.rho
    p = cross_stratum_arm_diff_slope.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None

    if rho >= rho_threshold_held and p <= p_threshold:
        return Verdict.HELD, None
    if rho <= -sign_flip_threshold:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(rho) < sign_flip_threshold:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


@claim_bridge(
    source=_PREDICTOR,
    target=_TARGET,
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_SCOPE,
    predicted_direction='a_gt_b',
)
def ddqn_q_suppression_tracks_outcome_translation__xenv_loo_robust(
    cross_stratum_arm_diff_slope: CrossStratumArmDiffSlopeResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor: str = _PREDICTOR,
    target: str = _TARGET,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_seeds_per_arm: int = 5,
    min_strata: int = 8,
    loo_min_rho_threshold: float = 0.3,
    sign_flip_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """LOO-robustness sibling: PacMan-outlier check.

    PacMan (Δ_Q=+34, Δ_out=+166) dominates the cross-env rank.
    Dropping it should leave ρ above the LOO threshold for the
    anchor's HELD reading to be robust.

    Verdict matrix:
      HELD              : min(ρ_LOO) ≥ +0.3 across all n LOO subsets
      NO_EFFECT (SIGN_FLIP) : min(ρ_LOO) ≤ −0.3
      NO_EFFECT (NULL)  : |min(ρ_LOO)| < 0.3
      POWER_INSUFFICIENT : otherwise / n_strata < 8
    """
    del (
        treatment_arm, baseline_arm, predictor, target,
        stratify_by, min_seeds_per_arm,
    )

    if cross_stratum_arm_diff_slope.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None

    loo_min = _spearman_loo_min(
        cross_stratum_arm_diff_slope.arm_diff_predictor,
        cross_stratum_arm_diff_slope.arm_diff_target,
    )
    if math.isnan(loo_min):
        return Verdict.POWER_INSUFFICIENT, None

    if loo_min >= loo_min_rho_threshold:
        return Verdict.HELD, None
    if loo_min <= -sign_flip_threshold:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(loo_min) < sign_flip_threshold:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None
