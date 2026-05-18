"""DoWhy 3-bridge cluster — HP-variance dose-response of vanilla
bootstrap_gap_magnitude → DDQN Δ_outcome.

Salvaged from the deleted canonical-scope cluster
`bias_correction_clip_predicts_outcome_{backdoor,placebo,rcc}`
(commit 868a7d1 removed them from `ddqn/bias_correction.py`). At
canonical n_strata=12 the cross-stratum OLS slope SE caps out →
POWER_INSUFFICIENT. Here the scope is RELAXED to admit HP-axis
variation (n_step, reward_scale, action_duplicate_k, polyak τ)
— deliberately producing predictor variance via configured
intervention axes. n_strata grows accordingly (pre-canonical
diagnostic gave n=29 strata with β=+244 p<1e-4).

The substantive question is DIFFERENT from the canonical
`bias_correction_dose_response__xenv_arm_diff` (REFUTED at
canonical n=10 envs). Here we ask "does HP-induced variation in
vanilla bg_magnitude predict HP-induced variation in DDQN
Δ_outcome?" The signal — if it survives the canonical→sweeps
re-test — argues that HP-axis perturbations of bg do translate
to outcome benefit even when env-axis perturbations don't.

Caveat: the pre-canonical diagnostic on HP-mixed pool also
contained cross-corpus contamination (`findings_dqn_bridges_regime_mixing`),
so if the slope dissolves in the clean HP-variance pool, the
prior result was a HP/corpus confound — itself an empirical
finding worth recording.

Three-bridge anchor + placebo + RCC cluster shape per
HYPOTHESIS_AS_GRAPH.md §3b: backdoor identifies ATE,
placebo validates instrument, RCC bounds omitted-confound
sensitivity. Composed verdict at the extent: SUPPORTED iff all
three admit; REFUTED if any refutes.

**Cache state 2026-05-14**: the sweeps parquet was ingested before
these bridges existed — `bootstrap_gap_magnitude` is all-NaN
(5930/5930 cells). The bridges' `transitive_reads` will queue
backfill on the next `--ingest` run; until then bridges fire
POW_INSUF. The pre-canonical diagnostic (β=+244 p<1e-4 n=29) is
the expected reading after backfill.
"""
from __future__ import annotations

import polars as pl

from corroborate.analyses.dowhy.stratum_baseline_predictor_link_dowhy import (
    StratumBaselinePredictorLinkDowhyResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import finite
from corroborate.bridge.verdict import Verdict

from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM
from experiments.findings.ddqn._scope import (
    G1_VANILLA_CONFIG_PREMISE_ACTIVE,
)
from experiments.findings.ddqn._verdicts import (
    dowhy_backdoor_verdict,
    dowhy_placebo_verdict,
    dowhy_rcc_verdict,
)


# Relaxed-scope: G1 (premise active) + G2 (argmax-vulnerable) only.
# Deliberately admits n_step / reward_scale / action_duplicate_k /
# polyak τ variation — those HP axes are what produce the per-
# stratum predictor variance the OLS slope is measured on.
_HP_VARIANCE_SCOPE: pl.Expr = (
    G1_VANILLA_CONFIG_PREMISE_ACTIVE
    & finite('n_actions')
    & (pl.col('n_actions') >= 3)
)


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_HP_VARIANCE_SCOPE,
    predicted_direction='a_gt_b',
)
def bias_correction_clip_predicts_outcome__hp_variance__backdoor(
    stratum_baseline_predictor_link_dowhy: (
        StratumBaselinePredictorLinkDowhyResult
    ),
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor_col: str = 'bootstrap_gap_magnitude',
    target_col: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = 0.0,
    ate_floor: float = 50.0,
) -> Verdict:
    """DoWhy backdoor on per-stratum independent-samples panel of
    vanilla `bootstrap_gap_magnitude` → DDQN Δ_outcome. HELD when
    ATE > `ate_floor` with sign=+1. The HP-variance pool's
    predictor range spans HP-axis-induced bg variation; β slope
    captures the dose-response across that range."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_baseline_predictor
    return dowhy_backdoor_verdict(
        stratum_baseline_predictor_link_dowhy.backdoor,
        ate_threshold=ate_floor, sign=1,
    )


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_HP_VARIANCE_SCOPE,
    predicted_direction='a_gt_b',
)
def bias_correction_clip_predicts_outcome__hp_variance__placebo(
    stratum_baseline_predictor_link_dowhy: (
        StratumBaselinePredictorLinkDowhyResult
    ),
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor_col: str = 'bootstrap_gap_magnitude',
    target_col: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = 0.0,
    placebo_max_ratio: float = 0.2,
) -> Verdict:
    """Placebo refutation: random treatment ATE should be near
    zero relative to the real ATE."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_baseline_predictor
    return dowhy_placebo_verdict(
        stratum_baseline_predictor_link_dowhy.placebo,
        max_ratio=placebo_max_ratio,
    )


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_HP_VARIANCE_SCOPE,
    predicted_direction='a_gt_b',
)
def bias_correction_clip_predicts_outcome__hp_variance__rcc(
    stratum_baseline_predictor_link_dowhy: (
        StratumBaselinePredictorLinkDowhyResult
    ),
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    predictor_col: str = 'bootstrap_gap_magnitude',
    target_col: str = 'eval_best_burst_raw_mean',
    min_baseline_predictor: float = 0.0,
    rcc_max_drift_ratio: float = 0.15,
) -> Verdict:
    """Random-common-cause refutation: synthetic confounder
    leaves the ATE near-stable."""
    del treatment_arm, baseline_arm, predictor_col, target_col
    del min_baseline_predictor
    return dowhy_rcc_verdict(
        stratum_baseline_predictor_link_dowhy.random_common_cause,
        max_drift_ratio=rcc_max_drift_ratio,
    )


BRIDGES = (
    bias_correction_clip_predicts_outcome__hp_variance__backdoor,
    bias_correction_clip_predicts_outcome__hp_variance__placebo,
    bias_correction_clip_predicts_outcome__hp_variance__rcc,
)
