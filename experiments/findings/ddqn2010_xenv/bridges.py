"""Bridges for the cross-env DDQN-2010 study (CONSISTENCY shape).

The treatment is a PROGRAM swap, so we contrast `paired_dqn` vs `dqn`
on the typed `program` column (`arm_field='program'`); the paired-vs-
vanilla scope drops the ddqn2016 arm (so `program='dqn'` ⟺ vanilla).

Claim SHAPE = cross-env DIRECTION consistency (sign-test), NOT a
pooled magnitude. Across these 4 envs the per-env effect MAGNITUDES are
wildly heterogeneous (vanilla jensen_gap 289 / 57 / 34 / 0.12; outcome
d +5.7 … −2.9), so a DL magnitude pool's prediction interval brackets
zero (I²≈0.98) and `stratified_arm_diff_pooled` honestly returns
NO_EFFECT — the population magnitude is unidentified at n=4. The
DIRECTION question ("does paired move the measurable the predicted way
at every env?") is the well-posed shape here, via the binomial
sign-test (`cross_env_consistency_binomial`), exactly as
`ddqn_sweeps.jens_reduction_consistency` does at ~10 envs.

Both bridges defer the sign-test → verdict mapping to the framework's
`cross_env_consistency_binomial_verdict` (no hand-rolled gate).

POWER NOTE — only 4 strata, below that helper's `min_strata=5` floor
(the primitive's own "don't use < 5": an 8/8 is still only p=0.0039,
and a perfect 4/4 is p=0.0625 > 0.05). So both bridges return
POWER_INSUFFICIENT regardless of alignment — the honest underpowered
state at n=4. SUPPORTED needs ≥5 same-direction envs (5/5 → p=0.031).
Per-env directions today: mechanism 4/4 reduce jensen_gap; outcome 3/4
improve (Breakout flips to harm) — the mechanism ↛ outcome dissociation,
visible but not yet certifiable.
"""
from __future__ import annotations

import polars as pl

from corroborate.analyses.panel.cross_env_consistency_binomial import (
    CrossEnvConsistencyBinomialResult,
    cross_env_consistency_binomial_verdict,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn2010_xenv._scope import (
    BASELINE_PROGRAM, PAIRED_VS_VANILLA_SCOPE, TREATMENT_PROGRAM,
)


@claim_bridge(
    source='jensen_gap',
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=PAIRED_VS_VANILLA_SCOPE & pl.col('jensen_gap').is_finite(),
    predicted_direction='a_lt_b',
)
def paired_reduces_jens_consistently__minatar4(
    cross_env_consistency_binomial: CrossEnvConsistencyBinomialResult,
    *,
    treatment_arm: str = TREATMENT_PROGRAM,
    baseline_arm: str = BASELINE_PROGRAM,
    arm_field: str = 'program',
    source: str = 'jensen_gap',
    stratify_by: tuple[str, ...] = ('env_name',),
    predicted_direction: str = 'a_lt_b',
    null_floor: float = 0.0,
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
    min_strata: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """MECHANISM (direction-consistency): paired_dqn reduces jensen_gap
    at every env. 4/4 here → binomial p=0.0625 → POWER_INSUFFICIENT
    (one same-direction env short of HELD; the de-biasing is real per
    env but n=4 can't certify cross-env scope-invariance)."""
    del (treatment_arm, baseline_arm, arm_field, source, stratify_by,
         predicted_direction, null_floor, scope_predictor,
         min_baseline_predictor, min_seeds_per_arm)
    return cross_env_consistency_binomial_verdict(
        cross_env_consistency_binomial, min_strata=min_strata)


@claim_bridge(
    source='eval_late_burst_mean',
    target='eval_late_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=PAIRED_VS_VANILLA_SCOPE & pl.col('eval_late_burst_mean').is_finite(),
    predicted_direction='a_gt_b',
)
def paired_improves_outcome_consistently__minatar4(
    cross_env_consistency_binomial: CrossEnvConsistencyBinomialResult,
    *,
    treatment_arm: str = TREATMENT_PROGRAM,
    baseline_arm: str = BASELINE_PROGRAM,
    arm_field: str = 'program',
    source: str = 'eval_late_burst_mean',
    stratify_by: tuple[str, ...] = ('env_name',),
    predicted_direction: str = 'a_gt_b',
    null_floor: float = 0.0,
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.0,
    min_seeds_per_arm: int = 5,
    min_strata: int = 5,
) -> tuple[Verdict, RefutationClass | None]:
    """OUTCOME (direction-consistency): does paired_dqn improve the
    greedy late-eval return at every env? It does NOT — Breakout flips
    to harm (3/4, p=0.31 → NO_EFFECT). Contrast with the mechanism's
    4/4: identical mechanism, inconsistent outcome — the mechanism ↛
    outcome dissociation (neither terminal at n=4)."""
    del (treatment_arm, baseline_arm, arm_field, source, stratify_by,
         predicted_direction, null_floor, scope_predictor,
         min_baseline_predictor, min_seeds_per_arm)
    return cross_env_consistency_binomial_verdict(
        cross_env_consistency_binomial, min_strata=min_strata)


__all__ = [
    'paired_reduces_jens_consistently__minatar4',
    'paired_improves_outcome_consistently__minatar4',
]
