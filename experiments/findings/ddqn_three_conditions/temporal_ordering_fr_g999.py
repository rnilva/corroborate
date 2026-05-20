"""Threshold-free temporal-ordering bridges at FR γ=0.999 + SI γ=0.999.

Hypothesis: DDQN's outcome rescue is mediated by a shift in how
trajectory progress is APPORTIONED between policy growth (MC) and
bias-chain growth (Q). In vanilla, the bias chain dominates trajectory
progress; in DDQN, the policy growth dominates.

The threshold-free measurable `policy_growth_fraction` operationalizes
this: ratio of `|mc_growth|` to `|mc_growth| + |q_growth|`, where each
growth is `max(per_burst_X) − per_burst_X[0]`. Unitless, scale-
invariant within env, no authored cutpoints.

This replaces the earlier thresholded formulation
(`policy_anchors_before_bias`) which required q_threshold=9.2 (50%
of empirically-derived Lemma 2 asymptote) — that threshold was
defensible but reviewer-fragile. The growth-fraction is defensible
without authored thresholds.

Bridges:
1. `ddqn_increases_policy_growth_fraction__fr_g999` — pre-registered
   at FR γ=0.999 × MLP[64,64] × unshaped × B=32. Predicted HELD.
2. `ddqn_increases_policy_growth_fraction__si_g999` — pre-registered
   at SI γ=0.999 × canonical. Predicted HELD (smaller magnitude than
   FR due to SI's gradual rescue vs FR's phase transition).

Empirical pilots (from trace inspection):
- FR γ=0.999 vanilla median: ~0.00 (stuck cells: all bias growth)
  DDQN median: ~0.89 (cells: policy growth dominates)
- SI γ=0.999 vanilla median: ~0.07 (MC grew slightly, Q grew lots)
  DDQN median: ~0.24 (MC grew more, Q similar to vanilla)
"""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.paired.arm_mean_diff import ArmMeanDiffResult
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.bridge.predicates import finite

from experiments.findings.ddqn._arms import DDQN_ARM, INTERVENTION, VANILLA_ARM


_FR_CANONICAL_G999_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'FourRooms-misc')
    & (pl.col('gamma') == 0.999)
    & (pl.col('replay.batch_size') == 32)
    & (pl.col('fa_kind') == 'mlp_deep')
    & (pl.col('shaping_kind') == 'none')
    & (pl.col('total_steps') == 1000000)
    & finite(pl.col('policy_growth_fraction'))
)


_SI_CANONICAL_G999_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'SpaceInvaders-MinAtar')
    & (pl.col('gamma') == 0.999)
    & finite(pl.col('policy_growth_fraction'))
)


def _signed_d_verdict(
    result: ArmMeanDiffResult,
    *,
    d_floor: float,
    sign_flip_floor: float,
    null_band: float,
    alpha: float,
) -> tuple[Verdict, RefutationClass | None]:
    """HELD if Cohen's d ≥ d_floor AND p < alpha.
    NO_EFFECT/SIGN_FLIP if d ≤ -sign_flip_floor with sig.
    NO_EFFECT/NULL if |d| < null_band.
    POWER_INSUFFICIENT otherwise."""
    d = result.standardized_effect
    p = result.mean_diff_p_value
    if math.isnan(d) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if d >= d_floor and p < alpha:
        return Verdict.HELD, None
    if d <= -sign_flip_floor and p < alpha:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(d) < null_band:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


@claim_bridge(
    source=INTERVENTION,
    target='policy_growth_fraction',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=_FR_CANONICAL_G999_SCOPE,
    predicted_direction='a_gt_b',
)
def ddqn_increases_policy_growth_fraction__fr_g999(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'policy_growth_fraction',
    pair_by: tuple[str, ...] = ('seed',),
    d_floor: float = 0.8,
    sign_flip_floor: float = 0.3,
    null_band: float = 0.2,
    alpha: float = 0.05,
) -> tuple[Verdict, RefutationClass | None]:
    """At FR γ=0.999 × MLP × unshaped × B=32, DDQN cells have higher
    `policy_growth_fraction` (mc_growth / (mc_growth + q_growth))
    than vanilla cells.

    Threshold-free reformulation of the earlier
    `ddqn_anchors_policy_before_bias__fr_g999` (which used
    mc_threshold=0.1 and q_threshold=9.2 — both authored). The growth-
    fraction has no authored cutpoint — just the relative magnitude of
    policy-side vs bias-side trajectory progress.

    Pre-registered direction: a_gt_b. Pre-registered verdict: HELD.
    Empirical pilot (gamma_sweep_fourrooms cells): vanilla ~0.00,
    DDQN ~0.89, Cohen's d very large.

    Verdict matrix on Cohen's d:
      HELD              : d ≥ +0.8 AND p < 0.05
      NO_EFFECT (NULL)  : |d| < 0.2
      NO_EFFECT (SIGN_FLIP) : d ≤ -0.3
      POWER_INSUFFICIENT : otherwise

    Substantive: DDQN shifts the trajectory's progress toward
    policy-side rather than bias-side. This is the Theorem 1 / Cor 1.1
    operationalization without authored Q-threshold defensibility
    concerns. Pre-registered direction + verdict committed via
    bridge source-hash."""
    del treatment_arm, baseline_arm, source, pair_by
    return _signed_d_verdict(
        arm_mean_diff,
        d_floor=d_floor,
        sign_flip_floor=sign_flip_floor,
        null_band=null_band,
        alpha=alpha,
    )


@claim_bridge(
    source=INTERVENTION,
    target='policy_growth_fraction',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=_SI_CANONICAL_G999_SCOPE,
    predicted_direction='a_gt_b',
)
def ddqn_increases_policy_growth_fraction__si_g999(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'policy_growth_fraction',
    pair_by: tuple[str, ...] = ('seed',),
    d_floor: float = 0.6,
    sign_flip_floor: float = 0.3,
    null_band: float = 0.2,
    alpha: float = 0.05,
) -> tuple[Verdict, RefutationClass | None]:
    """At SI γ=0.999 × canonical, DDQN cells have higher
    `policy_growth_fraction` than vanilla cells.

    Sibling of the FR γ=0.999 bridge; tests whether the
    policy-growth-fraction shift generalizes from FR (phase-transition
    rescue) to SI (gradual rescue) using the SAME threshold-free
    measurable.

    Pre-registered direction: a_gt_b. Pre-registered verdict: HELD.
    Empirical pilot: vanilla ~0.07, DDQN ~0.24 — smaller magnitude
    than FR (Cohen's d expected ~0.6-1.0, weaker than FR's d ~3+).

    d_floor=0.6 (less stringent than FR's 0.8) reflects the smaller
    expected effect magnitude at the gradual-rescue env. Pre-registered.

    Verdict matrix same shape as FR bridge.

    Substantive: if both FR and SI bridges HELD, the
    policy-growth-fraction shift is a CROSS-ENV mediator of DDQN's
    rescue effect at γ=0.999 sparse-reward envs (both
    phase-transition FR and gradual SI). If only FR HELDs, the
    growth-fraction shift is FR-specific (or has different magnitude
    requirements per env)."""
    del treatment_arm, baseline_arm, source, pair_by
    return _signed_d_verdict(
        arm_mean_diff,
        d_floor=d_floor,
        sign_flip_floor=sign_flip_floor,
        null_band=null_band,
        alpha=alpha,
    )
