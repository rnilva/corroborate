"""Outcome-translation cluster: where DDQN's bias-reduction
translates to outcome and where it does not.

Two bridges, both on FourRooms, with non-matched stratification
axes:

- positive arm `ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel`:
  at FR γ=0.999 × MLP × unshaped, DDQN's outcome > vanilla's
  outcome at every k_eff stratum.
- null arm `ddqn_no_positive_outcome_under_shaping__fr_shaped_fa_x_gamma_panel`:
  at FR × {linear, MLP} × {γ=0.99, 0.999} × PotentialReward,
  DDQN's outcome effect is never appreciably POSITIVE.

What the cluster does NOT claim:
- That the two arms test matched scopes. The positive arm pins
  γ=0.999 + MLP and varies k_eff; the null arm pools γ +
  fa_kind and fixes k_eff. The honest cluster reading is
  "DDQN helps at A, doesn't help anywhere in B" where A and B
  are non-overlapping scopes of FR. The matched-stratum claim
  (positive vs null at FR γ=0.999 × MLP × {shaped, unshaped} ×
  k_eff sweep) would require a sweep that doesn't yet exist.
- That shaping CAUSALLY DECOUPLES bias from outcome. The
  empirical pattern is "no positive benefit". One stratum
  (γ=0.99 × MLP × shaped) shows d ≈ −1.5 — DDQN actively
  HURTS. The `predicted_direction='a_lt_b'` framing admits
  both "decouples to ~0" and "inverts to −1.5" as consistent;
  these are different mechanisms collapsed by the asymmetric
  upper-bound verdict.

Memo cross-ref: `findings_shaping_decouples_bias_from_outcome`
gives the substantive narrative; this module's bridges
corroborate the no-positive-benefit empirical content
(the narrative's mechanism story is partly tested, partly
asserted)."""
from __future__ import annotations

import polars as pl

from corroborate.analyses.panel.stratified_arm_diff_pooled import (
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
    (= 0.3). NO_EFFECT/SIGN_FLIP if any stratum shows d < -0.3.
    NO_EFFECT (null) if any stratum sits in (-0.3, 0.3).

    Pairs with `ddqn_no_positive_outcome_under_shaping__fr_shaped_fa_x_gamma_panel`
    on non-matched scopes (this arm: γ=0.999 × MLP × k_eff sweep
    × unshaped; null arm: γ ∈ {0.99, 0.999} × {linear, MLP} ×
    shaped). Both HELD reads as "DDQN's outcome benefit is real
    at A and absent at B" — NOT "shaping moderates the outcome
    at the same scope as A". The matched-scope claim requires
    a shaped × k_eff sweep that does not yet exist."""
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
def ddqn_no_positive_outcome_under_shaping__fr_shaped_fa_x_gamma_panel(
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
    `eval_best_burst_raw_mean` (canonical γ-invariant target).
    HELD iff every stratum's d ≤ `per_stratum_d_upper_bound`
    (= 0.3) — DDQN never appreciably HELPS the outcome under
    shaping. NEGATIVE effects (DDQN hurts) are consistent with
    the verdict and do not refute it.

    Refutations:
    - INVARIANT_VIOLATION: any stratum's CI fully > +0.3
      (DDQN meaningfully helps under shaping).
    - POWER_INSUFFICIENT: any stratum's CI straddles +0.3
      without crossing fully above.

    What this bridge does NOT claim:
    - That shaping DECOUPLES bias from outcome. The bridge's
      asymmetric `predicted_direction='a_lt_b'` admits BOTH:
      (a) "shaping decouples to ~0" (the narrative in
      `findings_shaping_decouples_bias_from_outcome`), and
      (b) "shaping INVERTS the translation" (the γ=0.99 × MLP
      × shaped stratum has d ≈ −1.5 — DDQN actively HURTS).
      These are different mechanisms; the upper-bound verdict
      collapses them. The honest empirical content is "no
      positive benefit", which is weaker than "decouples".
    - Cross-env generalisation. Scope is FR-only; no other env
      has been run with PotentialReward shaping at this
      writing.

    Pairs with `ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel`
    on non-matched scopes (positive arm: γ=0.999 × MLP × k_eff
    sweep; null arm: γ ∈ {0.99, 0.999} × {linear, MLP}). The
    matched-stratum claim — positive vs null at the SAME
    (γ=0.999, MLP) cell — needs a shaped × k_eff sweep that
    does not yet exist."""
    del stratify_by, min_baseline_predictor
    return per_stratum_upper_bound_verdict(
        stratified_arm_diff_pooled,
        upper_bound=per_stratum_d_upper_bound,
        min_strata=min_strata,
    )
