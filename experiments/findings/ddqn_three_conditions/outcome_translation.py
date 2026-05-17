"""Outcome-translation cluster: shaping moderates bias → outcome.

Hasselt's bound describes WHY DDQN's mech fires (the bias side);
this module's two bridges describe WHEN that mech translates to
outcome. The cluster pattern: positive arm at unshaped FA-dense
+ null arm at shaped reads as "potential-based shaping decouples
DDQN's bias-reduction from its outcome effect at FR γ ∈ {0.99,
0.999} × MLP".

- positive arm `ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel`:
  the "Hasselt factors all active + dense FA" reference cell.
  DDQN's outcome > vanilla's outcome at every k_eff stratum.
- null arm `shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel`:
  the "alternative policy signal" perturbation. Under
  PotentialReward(Manhattan-to-goal), DDQN's outcome effect is
  never appreciably positive — the dense Φ-gradient signal
  overrides Q-noise on argmax, so bias-reduction doesn't matter.

Memo cross-ref: `findings_shaping_decouples_bias_from_outcome`
documents the substantive mechanism (three-condition scope
discriminator for DDQN's outcome benefit)."""
from __future__ import annotations

import polars as pl

from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import finite
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn_three_conditions._arms import INTERVENTION
from experiments.findings.ddqn_three_conditions._verdicts import (
    per_stratum_d_threshold_verdict,
    per_stratum_upper_bound_verdict,
)


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('gamma') == 0.999)
        & (pl.col('fa_kind') == 'mlp_deep')
        & (pl.col('shaping_kind') == 'none')
        & finite(pl.col('eval_best_burst_raw_mean'))
    ),
    predicted_direction='a_gt_b',
)
def ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('k_eff',),
    min_strata: int = 3,
    min_baseline_predictor: float = float('-inf'),
    per_stratum_d_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """DDQN's outcome > vanilla's outcome uniformly across
    k_eff ∈ {4, 8, 12, 16} at FourRooms γ=0.999 × MLP[64,64] ×
    no-shaping.

    Per-k_eff independent-samples Cohen's d on
    `eval_best_burst_raw_mean` (canonical γ-invariant target —
    cf. `findings_units_bug` memo). Source cells from
    `experiments/probes/action_dup_mismatch_probe_g999_1M/`
    (k_eff ∈ {4, 8, 12}) and `..._FR_k4_only/` (k_eff = 16). 30
    seeds per arm per stratum.

    HELD iff every stratum's d ≥ +`per_stratum_d_threshold`
    (= 0.3). NO_EFFECT/SIGN_FLIP if any stratum shows d < -0.3
    (DDQN substantially HURTS). NO_EFFECT (null) if any stratum
    sits in the indeterminate band (-0.3, 0.3).

    Pairs with `shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel` —
    both HELD reads as "DDQN's outcome benefit is real at
    unshaped MLP but vanishes under potential-based shaping at
    the same env/FA"."""
    del stratify_by, min_baseline_predictor
    return per_stratum_d_threshold_verdict(
        stratified_arm_diff_pooled,
        threshold=per_stratum_d_threshold,
        sign=+1,
        min_strata=min_strata,
    )


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('shaping_kind') == 'potential_manhattan')
        & pl.col('gamma').is_in([0.99, 0.999])
        & finite(pl.col('eval_best_burst_raw_mean'))
    ),
    predicted_direction='a_lt_b',
)
def shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('fa_kind', 'gamma'),
    min_strata: int = 3,
    min_baseline_predictor: float = float('-inf'),
    per_stratum_d_upper_bound: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """No POSITIVE outcome benefit at FR × shaped across
    (fa_kind, γ) panel.

    Per-stratum independent-samples Cohen's d on
    `eval_best_burst_raw_mean` (the canonical γ-invariant target
    for cross-γ analysis). HELD iff every stratum's d ≤
    `per_stratum_d_upper_bound` (= 0.3) — DDQN never appreciably
    HELPS the outcome under shaping. NEGATIVE effects (DDQN
    hurts) are consistent with the decoupling claim and don't
    refute it.

    Substantive mechanism (memo entry
    `findings_shaping_decouples_bias_from_outcome`): under
    potential-based shaping, vanilla's optimization signal is
    dense enough that the bias→behavior chain breaks. DDQN's
    bias-reduction is mechanistically real (jens still drops)
    but doesn't translate to outcome gains because vanilla
    already converges from the dense shaped signal. The
    empirical reading on this corpus is stronger than null at
    one cell (mlp × γ=0.99 has d=-1.5: DDQN actively HURTS),
    consistent with shaping INVERTING rather than just decoupling
    the translation at lower γ.

    Refutations:
    - INVARIANT_VIOLATION: any stratum's CI fully > +0.3
      (DDQN meaningfully helps under shaping — would refute
      decoupling).
    - POWER_INSUFFICIENT: any stratum's CI straddles +0.3
      without crossing fully above.

    Pairs with `ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel` —
    that bridge HELD (DDQN helps unshaped) + this bridge HELD
    (DDQN doesn't help shaped) = "shaping moderates the outcome
    translation"."""
    del stratify_by, min_baseline_predictor
    return per_stratum_upper_bound_verdict(
        stratified_arm_diff_pooled,
        upper_bound=per_stratum_d_upper_bound,
        min_strata=min_strata,
    )
