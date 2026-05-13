"""Cross-env dose-response bridges for DDQN's bootstrap clip.

Substantive claim: "Across envs, do envs where DDQN induces a
bigger Δ_clip (the algorithmic dose) see a bigger Δ_outcome?"

Each env contributes ONE point:
  x = mean(DDQN bg_magnitude) − mean(vanilla bg_magnitude)
  y = mean(DDQN outcome)      − mean(vanilla outcome)

Both arm means are computed independently — no seed-pairing
semantics. Spearman ρ across n=12 canonical envs (10 after
saturation-scope guard drops CartPole / FourRooms).

**Why independent-samples per-arm means, not paired Δ**: in RL
substrate, DDQN(seed=N) and vanilla(seed=N) share only initial
RNG state — trajectories diverge immediately. Seed pairing
carries no causal pairing semantics. The primitive consumed here
(`cross_stratum_arm_diff_slope`) makes the independent-samples
form explicit; the bridge isn't relying on a pairing the
framework doesn't earn. Mathematically equivalent to the
seed-paired form on corpora where each seed appears in both arms
(canonical), but the name no longer misleads about what's being
tested.

**Why predictor is arm-diff Δ (substrate-level dose), not vanilla
mean**: the substantive DOSE is the algorithmic intervention's
effect on the clip (DDQN − vanilla on bg_magnitude), not the
env's baseline clip magnitude. Per-env vanilla bg_mean would
confound env-identity (reward scale, polarity, density —
`findings_scope_density.md`) with the substrate-level dose. See
critic review history for the prior-draft mis-scoping.

**Saturation guard**: CartPole-v1 and FourRooms-misc both arms
saturate at the env reward ceiling (range_outcome ≈ 0 across
seeds), making per-env Δ_outcome structurally zero. These envs
contribute nothing but tie-noise to a rank correlation; the
scope predicate excludes them. See `cell_classification.py` for
the substrate's broader saturated-cell concept.

**Cluster shape**: the bridge ships with a leave-one-out
robustness sibling (`__loo_robust`). The first reports the raw
Spearman; the second gates HELD on `min(ρ_LOO) ≥ +0.3` —
robust to any single-env removal. Together they form the
canonical anchor + robustness-refutation cluster shape from
HYPOTHESIS_AS_GRAPH.md §3b.

**Verdict semantics**: framework's idiom for "tested + wrong-sign
refutation" is `(Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP)` —
NO_EFFECT is the refutation verdict, refinement says how. There
is no separate `Verdict.REFUTED`."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import scipy.stats as stats

from corroborate.analyses.cross_stratum_arm_diff_slope import (
    CrossStratumArmDiffSlopeResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM
from experiments.findings.ddqn._scope import DDQN_RELEVANT_SCOPE


_PREDICTOR = 'bootstrap_gap_magnitude'
_TARGET = 'eval_best_burst_raw_mean'


# Envs whose outcome saturates at the env reward ceiling for both
# arms at canonical 1M training (range_outcome ≈ 0 across seeds).
# Including them in the cross-env Spearman injects rank-tie noise
# without information — their per-env Δ_outcome is structurally
# zero regardless of policy quality.
_SATURATING_OUTCOME_ENVS: tuple[str, ...] = (
    'CartPole-v1',     # max episode steps = 500; both arms hit 500
    'FourRooms-misc',  # goal reward = 1.0; both arms reach it
)

_NONSATURATED_OUTCOME_SCOPE: pl.Expr = (
    ~pl.col('env_name').is_in(_SATURATING_OUTCOME_ENVS)
)


_XENV_DOSE_RESPONSE_SCOPE: pl.Expr = (
    DDQN_RELEVANT_SCOPE & _NONSATURATED_OUTCOME_SCOPE
)


def _spearman_loo_min(
    xs: tuple[float, ...], ys: tuple[float, ...],
) -> float:
    """Leave-one-out minimum Spearman ρ over (xs, ys). Returns
    NaN if any LOO subset has fewer than 4 points."""
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
    scope=_XENV_DOSE_RESPONSE_SCOPE,
    predicted_direction='a_gt_b',
)
def bias_correction_dose_response__xenv_arm_diff(
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
    """Cross-env Spearman of per-env arm-mean Δ dose vs arm-mean
    Δ outcome. Predicted positive (bigger Δ_clip → bigger Δ_outcome).

    Verdict matrix:
      HELD              : ρ ≥ +0.6 AND p ≤ 0.05
      NO_EFFECT (NULL)  : |ρ| < 0.3 (cleanly null both directions)
      NO_EFFECT (SIGN_FLIP) : ρ ≤ −0.3 (wrong-direction non-null)
      POWER_INSUFFICIENT : otherwise, or n_strata < 8

    Calibration: at n=10 (canonical 12 minus 2 saturating),
    two-sided critical |r| at p=0.05 is ≈0.648; HELD gate at 0.6
    sits just below — paired with the p ≤ 0.05 cut, the joint
    gate requires modest signal AND statistical significance.
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
    scope=_XENV_DOSE_RESPONSE_SCOPE,
    predicted_direction='a_gt_b',
)
def bias_correction_dose_response__xenv_arm_diff_loo_robust(
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
    """Robustness sibling: HELD requires the cross-env Spearman to
    stay ≥ +0.3 under every leave-one-out removal. Refutes the
    anchor bridge when its HELD reading is driven by a single
    outlier env (e.g. one Atari env where Δ_outcome scales to the
    reward scale while others contribute tie-noise).

    Verdict matrix:
      HELD              : min(ρ_LOO) ≥ +0.3 across all n LOO subsets
      NO_EFFECT (SIGN_FLIP) : min(ρ_LOO) ≤ −0.3 (robustly wrong sign)
      NO_EFFECT (NULL)  : |min(ρ_LOO)| < 0.3 across the subsets
      POWER_INSUFFICIENT : otherwise, or n_strata < 8

    The "drop the strongest contributor and ρ still holds" criterion
    is the substrate-author's discipline against single-stratum-
    driven cross-stratum claims (`findings_n3_pearson_brittle.md`)."""
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
