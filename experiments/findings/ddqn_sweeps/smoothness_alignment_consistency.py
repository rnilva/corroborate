"""Cross-env sign-alignment: DDQN's outcome effect direction
mirrors its Q-smoothness effect direction at every canonical
γ=0.999 env.

**Pre-registered claim REFUTED on 8-env adjudicating panel**
(2026-05-21). Sign-alignment held at the 4-env eyeball panel
(FR/SI/Asterix/Breakout γ=0.999, 3/4 aligned) used to motivate
the claim, but did NOT generalize: 4/8 envs aligned at the full
canonical pool (p=0.637, indistinguishable from random).

Substantive claim (PC discovery 2026-05-21, /tmp/pc_around_gradient.py):
DDQN's clip damps inter-state Q-gradient overlap (smoothness)
proportionally to chain depth × reward sparsity. Where DDQN
INCREASES smoothness (FR, SI γ=0.999), it HELPS outcome; where
it DECREASES smoothness (Asterix γ=0.999), it HARMS outcome.
The sign of Δ_smoothness was predicted to track the sign of
Δ_outcome SAME-direction across canonical envs.

Companion to `loop_channel_consistency`'s
`ddqn_outcome_opposes_loop_rate__canonical_g0999` bridge —
both register cross-env sign-alignment claims; they differ
on which Y-axis is paired with the outcome Δ:

  loop bridge:           alignment='opposite' against Δ_rep_ea   → HELD 7/8 p=0.035
  smoothness bridge (this): alignment='same'  against Δ_smoothness → REFUTED 4/8 p=0.637

The asymmetric verdict is informative: loop reduction is a
universal channel; smoothness preservation is env-specific (real
at Asterix γ=0.999 per the mechanism-active bridge in
`q_smoothness_harm_mechanism`, jens-shadow at Breakout, etc.).
See `finding_smoothness_alignment_consistency_g0999` and
`pc_cross_env_smoothness` for the per-env structural-role
verdicts.

Caveats:
- The cross-env Δ_smoothness/Δ_outcome ordinal correlation
  hypothesis turned out NOT to track. Reasons (post-hoc):
  Breakout + Snake γ=0.999 both CUT smoothness modestly while
  HELPING outcome via other channels (notably loop reduction);
  LL + MC have near-null Δ_smoothness so their alignment is
  effectively coin-flip noise.
- The single-env smoothness-harm chain at Asterix γ=0.999
  remains real (the mechanism-active bridge HELDs there);
  this Finding closes only the cluster-level sufficient-
  condition question.
"""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.panel.cross_env_sign_alignment_binomial import (
    CrossEnvSignAlignmentBinomialResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn_sweeps.lambda_a_mediation import (
    CANONICAL_G0999_CORPORA,
)


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('gamma') == 0.999)
        & (
            pl.col('action_duplicate_k').is_null()
            | (pl.col('action_duplicate_k') == 1)
        )
        & pl.col('corpus').is_in(CANONICAL_G0999_CORPORA)
        & pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('q_inter_state_grad_overlap_late').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def ddqn_outcome_aligns_with_q_smoothness__canonical_g0999(
    cross_env_sign_alignment_binomial: CrossEnvSignAlignmentBinomialResult,
    *,
    source_x: str = 'eval_best_burst_raw_mean',
    source_y: str = 'q_inter_state_grad_overlap_late',
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    stratify_by: tuple[str, ...] = ('env_name',),
    alignment: str = 'same',
    null_floor_x: float = 0.0,
    null_floor_y: float = 0.0,
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
    p_threshold_held: float = 0.05,
    p_threshold_pi: float = 0.15,
    min_strata: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """Cross-env sign-test: at each canonical γ=0.999 env, does
    sign(d_outcome) align SAME-direction with sign(d_smoothness)?

    Verdict matrix:
      HELD                 : binomial p ≤ 0.05 (at n=10, ≥9/10 aligned)
      POWER_INSUFFICIENT   : 0.05 < p ≤ 0.15 OR n_strata < min_strata
      NO_EFFECT (NULL)     : p > 0.15, aligned fraction ≤ 0.6
      NO_EFFECT (SIGN_FLIP): wrong-direction fraction ≥ 0.7

    On the 4-env eyeball panel (FR/SI/Asterix/Breakout γ=0.999):
    3/4 aligned same-direction (Breakout breaks alignment with
    d_grad=-0.52 d_out=+1.01). At n=10 canonical pool: predicted
    ≥9/10 aligned → p ≤ 0.011.

    Pre-registration: bridge committed before
    `q_inter_state_grad_overlap_late` is backfilled cluster-wide
    (probe added 2026-05-13; pre-existing corpora NaN). Fires PI
    until enough corpora populate the measurable."""
    del (
        source_x, source_y, treatment_arm, baseline_arm,
        stratify_by, alignment, null_floor_x, null_floor_y,
        scope_predictor, min_baseline_predictor, min_seeds_per_arm,
    )
    if cross_env_sign_alignment_binomial.n_strata_total < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    p = cross_env_sign_alignment_binomial.p_value
    if math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    n_total = cross_env_sign_alignment_binomial.n_strata_total
    n_aligned = cross_env_sign_alignment_binomial.n_strata_aligned
    if p <= p_threshold_held:
        return Verdict.HELD, None
    if p <= p_threshold_pi:
        return Verdict.POWER_INSUFFICIENT, None
    n_wrong = n_total - n_aligned
    if n_total > 0 and n_wrong / n_total >= 0.70:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if n_aligned / max(n_total, 1) <= 0.60:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


__all__ = [
    'ddqn_outcome_aligns_with_q_smoothness__canonical_g0999',
]
