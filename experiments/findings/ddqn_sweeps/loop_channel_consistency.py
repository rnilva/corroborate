"""Cross-env sign-alignment: DDQN's outcome effect direction
mirrors its loop-rate effect direction at every canonical
γ=0.999 env.

Substantive claim (REPORT_loop_hypothesis_synthesis.md §2.1):
where DDQN helps outcome (Δ_outcome > 0) it REDUCES loops
(Δ_rep_ea < 0); where DDQN harms (Asterix γ=0.999 → Δ_outcome < 0)
it INCREASES loops (Δ_rep_ea > 0). The report's eyeball
sign-alignment table (5/5 envs aligned) becomes a framework-
typed bridge.

Predicted alignment: OPPOSITE (sign of Δ_outcome opposite to
sign of Δ_rep_ea). At n=10 canonical envs, binomial sign-test
power gates:
  10/10 opposite → p = 0.0010 → HELD
   9/10          → p = 0.0107 → HELD
   8/10          → p = 0.0547 → marginal
   7/10          → p = 0.1719 → PI

Pre-registration status: this bridge is committed BEFORE the
`state_repeat_rate_window64_late` (Δ_rep_ea) measurable is
backfilled cluster-wide. The measurable reads
`state_hash_per_step` (trace col); 2 of 10 canonical-pool
corpora have traces locally (Freeway, SI) — the rest need
restore + recompute. Bridge will fire PI until the backfill
completes, then automatically yields the verdict on the
sign-alignment claim. This is the framework's
pre-registration discipline: typed claim committed to git
ahead of the data; verdict comes when data lands.

Caveats from the report's §5.2 that this bridge does NOT
address:
- The sign-alignment claim is necessary but not sufficient for
  "loop-reduction is a causal channel" — supports the
  correlation but not the mechanism. Causal evidence
  (intervention) is in the count-weighted bridge family
  (deferred to a separate bridge).
- The report's panel uses `state_repeat_rate_window64_late`
  (episode-agnostic). The within-episode variant
  (`state_repeat_rate_within_episode_window64_late`) is
  related but distinct; their sign-correlation hasn't been
  separately tested in this bridge.
- State-hash availability varies per env; some envs may have
  degenerate state-hash (constant) producing rep_rate = 1.0
  artifacts (CLAUDE.md flag).
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
        & pl.col('state_repeat_rate_window64_late').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def ddqn_outcome_opposes_loop_rate__canonical_g0999(
    cross_env_sign_alignment_binomial: CrossEnvSignAlignmentBinomialResult,
    *,
    source_x: str = 'eval_best_burst_raw_mean',
    source_y: str = 'state_repeat_rate_window64_late',
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    stratify_by: tuple[str, ...] = ('env_name',),
    alignment: str = 'opposite',
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
    sign(d_outcome) oppose sign(d_rep_ea)?

    Verdict matrix:
      HELD                 : binomial p ≤ 0.05 (at n=10, ≥9/10 aligned)
      POWER_INSUFFICIENT   : 0.05 < p ≤ 0.15 OR n_strata < min_strata
      NO_EFFECT (NULL)     : p > 0.15, aligned fraction ≤ 0.6
      NO_EFFECT (SIGN_FLIP): wrong-direction fraction ≥ 0.7

    On report's 5-env eyeball panel (FR, SI, Breakout, LL, Asterix
    at γ=0.999): 5/5 envs aligned in opposite direction (where
    DDQN helps, repeat ↓; where DDQN harms, repeat ↑). Binomial
    p ≈ 0.031 at n=5. At n=10 canonical pool: predicted ≥9/10
    aligned → p ≤ 0.011.

    Pre-registration: this bridge is committed before the
    `state_repeat_rate_window64_late` measurable is backfilled
    cluster-wide (2 of 10 canonical corpora have local traces).
    Bridge fires PI until cells with the measurable populate."""
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
    'ddqn_outcome_opposes_loop_rate__canonical_g0999',
]
