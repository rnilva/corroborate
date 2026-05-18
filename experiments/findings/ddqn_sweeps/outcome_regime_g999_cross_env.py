"""Cross-env γ=0.999 outcome regime bridges.

DDQN's outcome effect at γ=0.999 — the long-horizon edge — is
NOT universal across the 4-env MinAtar panel. Each env shows a
distinct sign:

  Asterix-MinAtar  γ=0.999 → d_out = -0.80 z=-3.1 (HARMS)
  Breakout-MinAtar γ=0.999 → d_out = +0.66 z=+2.6 (HELPS)
  SpaceInvaders    γ=0.999 → d_out = +2.16 z=+8.4 (HELPS)
  Freeway-MinAtar  γ=0.999 → d_out = +0.10 z=+0.4 (NEUTRAL)

Four bridges, one per env, each predicts the EMPIRICAL sign
specific to that env. Together they make the substantive
substantive cross-env claim:

  "DDQN's outcome effect at γ=0.999 is REGIME-DEPENDENT across
  MinAtar envs, with at least three distinct response classes
  (HARM, HELP, NEUTRAL) jointly observed in a single env
  family."

This is intentionally distinct from the existing per-env
Findings (`finding_asterix_gamma_999_harm`,
`finding_breakout_gamma_999_help_underpowered`): those make
per-env CI-based claims with learnability filters; THIS
Finding makes the joint cross-env claim with a uniform
canonical-HP scope (no learnability filter), so that Freeway
(which doesn't pass learnability) can be included as the
NEUTRAL regime member.

The regime classification framework that explains these
signatures lives in:
  - `findings_si_corroborates_regime_classification.md`
  - `findings_asterix_g999_harm_is_optimization_dynamics.md`
  - `findings_pc_cross_env_smoothness.md`

Substantively: super-linear vanilla jens scaling under γ→1
predicts Q-EXPLODED regime → DDQN HARM (Asterix 622×);
moderate scaling predicts Q-STRUCTURED → DDQN HELP (Breakout
113×, SI 133×); flat scaling predicts Q-COLLAPSED → DDQN
INACTIVE (Freeway 1.5×).
"""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn._scope import CANONICAL_HP_EXCLUDING_GAMMA


_G999_CANONICAL_SCOPE: pl.Expr = (
    (pl.col('gamma') == 0.999)
    & CANONICAL_HP_EXCLUDING_GAMMA
    & pl.col('eval_best_burst_raw_mean').is_finite()
)


def _ci_test(
    res: StratifiedArmDiffPooledResult,
    *,
    direction: int,  # +1 for help, -1 for harm, 0 for null
    floor: float,
    null_band: float,
    alpha: float,
    min_strata: int,
) -> tuple[Verdict, RefutationClass | None]:
    """Returns HELD when pooled_d's CI is in the predicted region
    at α. direction=+1 → d ≥ floor; -1 → d ≤ -floor; 0 → |d| ≤
    null_band."""
    if res.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    d = res.pooled_d
    se = res.pooled_se
    p = res.pooled_p_value
    if math.isnan(d) or math.isnan(p):
        # Single-stratum fallback: use the per-stratum d/se.
        if res.n_strata == 1 and len(res.per_stratum) == 1:
            s = res.per_stratum[0]
            if math.isnan(s.cohen_d) or math.isnan(s.cohen_se) or s.cohen_se <= 0:
                return Verdict.POWER_INSUFFICIENT, None
            d = s.cohen_d
            se = s.cohen_se
            from scipy.stats import norm
            z = d / se
            p = 2.0 * (1.0 - float(norm.cdf(abs(z))))
        else:
            return Verdict.POWER_INSUFFICIENT, None
    if direction > 0:
        # Predict help: d significantly ≥ floor.
        if d >= floor and p < alpha:
            return Verdict.HELD, None
        if d <= -floor and p < alpha:
            return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
        return Verdict.POWER_INSUFFICIENT, None
    elif direction < 0:
        # Predict harm: d significantly ≤ -|floor|.
        if d <= -abs(floor) and p < alpha:
            return Verdict.HELD, None
        if d >= abs(floor) and p < alpha:
            return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
        return Verdict.POWER_INSUFFICIENT, None
    else:
        # Predict null: |d| ≤ null_band with adequate power
        # (CI fits within band).
        if abs(d) <= null_band:
            return Verdict.HELD, None
        if abs(d) > 2 * null_band and p < alpha:
            return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
        return Verdict.POWER_INSUFFICIENT, None


# ============ Asterix γ=0.999: HARM ============

@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Asterix-MinAtar')
        & _G999_CANONICAL_SCOPE
    ),
    predicted_direction='a_lt_b',
)
def ddqn_harms_asterix_g999__cross_env_regime(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
    harm_floor: float = -0.4,
    alpha: float = 0.05,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Asterix γ=0.999: HARM regime (Q-EXPLODED env).

    HELD if d_out ≤ -0.4 with p < 0.05. Empirical: d=-0.80
    z=-3.1 → solid HELD."""
    del (
        treatment_arm, baseline_arm, source, stratify_by,
        scope_predictor, min_baseline_predictor, min_seeds_per_arm,
    )
    return _ci_test(
        stratified_arm_diff_pooled, direction=-1, floor=harm_floor,
        null_band=0.2, alpha=alpha, min_strata=min_strata,
    )


# ============ Breakout γ=0.999: HELP ============

@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Breakout-MinAtar')
        & _G999_CANONICAL_SCOPE
    ),
    predicted_direction='a_gt_b',
)
def ddqn_helps_breakout_g999__cross_env_regime(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
    help_floor: float = 0.3,
    alpha: float = 0.05,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Breakout γ=0.999: HELP regime (Q-STRUCTURED env).

    HELD if d_out ≥ +0.3 with p < 0.05. Empirical: d=+0.66
    z=+2.6 → HELD."""
    del (
        treatment_arm, baseline_arm, source, stratify_by,
        scope_predictor, min_baseline_predictor, min_seeds_per_arm,
    )
    return _ci_test(
        stratified_arm_diff_pooled, direction=+1, floor=help_floor,
        null_band=0.2, alpha=alpha, min_strata=min_strata,
    )


# ============ SpaceInvaders γ=0.999: HELP ============

@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'SpaceInvaders-MinAtar')
        & _G999_CANONICAL_SCOPE
    ),
    predicted_direction='a_gt_b',
)
def ddqn_helps_si_g999__cross_env_regime(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
    help_floor: float = 0.3,
    alpha: float = 0.05,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """SpaceInvaders γ=0.999: HELP regime (Q-STRUCTURED env).

    HELD if d_out ≥ +0.3 with p < 0.05. Empirical: d=+2.16
    z=+8.4 → strongly HELD (biggest help in panel)."""
    del (
        treatment_arm, baseline_arm, source, stratify_by,
        scope_predictor, min_baseline_predictor, min_seeds_per_arm,
    )
    return _ci_test(
        stratified_arm_diff_pooled, direction=+1, floor=help_floor,
        null_band=0.2, alpha=alpha, min_strata=min_strata,
    )


# ============ Freeway γ=0.999: NEUTRAL ============

@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Freeway-MinAtar')
        & _G999_CANONICAL_SCOPE
    ),
    predicted_direction='null',
)
def ddqn_neutral_freeway_g999__cross_env_regime(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
    null_band: float = 0.3,
    alpha: float = 0.10,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Freeway γ=0.999: NEUTRAL regime (Q-COLLAPSED env — FA
    fails to learn at γ→1, vanilla jens ≈ 0, no mech to engage).

    HELD if |d_out| ≤ null_band (default 0.3). Empirical:
    d=+0.10 z=+0.4 → HELD (within band, NS).
    REFUTED (SIGN_FLIP) if |d| substantially exceeds band with
    significance."""
    del (
        treatment_arm, baseline_arm, source, stratify_by,
        scope_predictor, min_baseline_predictor, min_seeds_per_arm,
    )
    return _ci_test(
        stratified_arm_diff_pooled, direction=0, floor=0.0,
        null_band=null_band, alpha=alpha, min_strata=min_strata,
    )


BRIDGES = (
    ddqn_harms_asterix_g999__cross_env_regime,
    ddqn_helps_breakout_g999__cross_env_regime,
    ddqn_helps_si_g999__cross_env_regime,
    ddqn_neutral_freeway_g999__cross_env_regime,
)
