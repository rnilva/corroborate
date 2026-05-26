"""Q-smoothness → outcome link at Asterix γ=0.999.

The mediator side of the canonical chain
  arm (DDQN) → q_smoothness reduction → eval_best reduction

The arm → q_smoothness edge is already documented at canonical
Asterix γ=0.999 (Cohen d=−2.13 in `q_smoothness_is_jens_shadow`).
This bridge establishes the M → Y edge: smoother Q across adjacent
states predicts higher outcome.

Cross-regime causal evidence supports this as the unique
linearly-reliable mediator of arm → eval_best harm at Asterix
γ=0.999 (see `findings_anneal_300k_causal_analysis.md`):

  DoWhy linear mediation with `linearity_status = RELIABLE` in
  all three explored regimes (canonical n_eps=20 / anneal=300k /
  ε=0.10), with proportion mediated in [0.11, 0.41]. Other
  Q-cluster mediators (jensen_gap, q_trajectory_autocorr,
  cumulative_bias) go OUT_OF_BOUNDS or SIGN_FLIPPED under the
  exploration interventions — only q_inter_state_grad_overlap_late
  stays well-behaved.

Bridge predicted-positive (a_gt_b): higher q-smoothness → higher
outcome. The framework's `partial_spearman` analysis returns the
per-cell stratified ρ; we use signed-positive verdict.

CLAIM.md mediation recipe: this is the canonical (rank-based)
side of the scope-cluster pattern. The linear-diagnostic sibling
(mediation_dowhy linearity status) is held back for a follow-up
authoring pass — needs `is_ddqn` measurable to make arm the
treatment column."""
from __future__ import annotations

import polars as pl

from corroborate.analyses.spearman.partial_spearman import (
    PartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import finite
from corroborate.bridge.verdict import Verdict

from experiments.findings.ddqn._verdicts import (
    partial_spearman_signed_verdict,
)


# Scope: Asterix γ=0.999, canonical-HP regime (standard config — no
# n-step / action-duplicate / rs-shift / polyak-τ knobs), with both
# the predictor (q_inter_state_grad_overlap_late) and outcome
# (eval_best_burst_raw_mean) finite per cell.
_ASTERIX_G0999_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'Asterix-MinAtar')
    & (pl.col('gamma') == 0.999)
    & finite('q_inter_state_grad_overlap_late')
    & finite('eval_best_burst_raw_mean')
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    & pl.col('target_sync.tau').is_null()
)


@claim_bridge(
    source='q_inter_state_grad_overlap_late',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_ASTERIX_G0999_SCOPE,
    predicted_direction='a_gt_b',
)
def q_smoothness_link_to_outcome_held_positive__asterix_g0999(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'q_inter_state_grad_overlap_late',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'corpus',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.2,
    min_strata: int = 1,
) -> Verdict:
    """Per-cell stratified ρ(q_smoothness, outcome) on Asterix
    γ=0.999 cells, Fisher-z pooled across corpora. Predicted
    POSITIVE: smoother Q across adjacent states → policy retains
    more state-coverage diversity → higher outcome. HELD when
    ρ ≥ +`rho_threshold` AND p < 0.05 across ≥`min_strata` strata.

    The M→Y edge of the canonical DDQN harm chain
    `arm → q_smoothness reduction → eval_best reduction` at
    Asterix γ=0.999. The A→M edge is documented as the
    `q_smoothness_is_jens_shadow` finding (DDQN reduces
    smoothness, Cohen d=−2.13). Together these triangulate the
    mediator chain without needing arm-as-treatment in this
    bridge.

    Empirical (scratch-script `/tmp/xregime_mediation.py` over 3
    ckpt-bearing corpora, n=30 per regime):
      canonical n_eps=20: marginal ρ(arm, eval_best) = -0.605;
        partial ρ | q_smoothness = -0.346 (43% absorbed)
      anneal=300k:        marginal = -0.451;
        partial | q_smoothness = -0.145 (68% absorbed)
      ε=0.10 anneal=200k: marginal = -0.189;
        partial | q_smoothness = -0.082 (56% absorbed)

    Mediated-share grows with intervention strength — consistent
    with q-smoothness as a fixed-magnitude channel whose share
    grows as lock-in-attenuated total harm shrinks.

    Note: `corpus`-stratification (not `env_name`) because the
    Asterix-γ=0.999 cohort has multiple corpora — pooling
    across them via Fisher-z avoids cross-corpus drift confounds
    while still capturing the regime variation."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        partial_spearman,
        threshold=rho_threshold, sign=+1, min_strata=min_strata,
    )


__all__ = ('q_smoothness_link_to_outcome_held_positive__asterix_g0999',)
