"""Dense-eval Acrobot k-sweep — within-k full-AUC outcome bridges.

The dense-eval corpus (`acrobot_dense_eval_k_sweep`, 40 bursts ×
5k steps, n=30 per arm) reveals the **bimodal** mech→outcome
translation in k:
  - k=1: modest early-burst speedup that decays as both arms
    converge (full-AUC d=+0.26).
  - k=4: near-null effect (full-AUC d=+0.04). The "sweet spot"
    where |A|=8's Hasselt floor σ_Q·√(2 log 8)≈2.04σ is too small
    for DDQN's clip to find correctable bias.
  - k=16: persistent monotone-decaying advantage (full-AUC
    d=+1.13). |A|=32's Hasselt floor ≈ 2.63σ is large enough that
    vanilla suffers optimism noise and DDQN's clip sustains the
    policy.

Memory: `findings_dense_eval_acrobot_transient`. Bridges below
use `eval_full_auc_raw_mean` (γ-invariant integrated outcome).

Scope is pinned to the dense-eval corpus only
(`total_steps=200000`, `eval_every=5000`) so these bridges don't
fire on the canonical 1M cells. Lives in `ddqn_sweeps` because
`action_duplicate_k != null` falls outside the canonical
`DDQN_CANONICAL_REGIME` of the parent `ddqn` module.
"""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.arm_mean_diff import ArmMeanDiffResult
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import DDQN_ARM, INTERVENTION, VANILLA_ARM


# Scope: dense-eval Acrobot corpus, γ=0.99, MLP[64,64].
_DENSE_EVAL_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'Acrobot-v1')
    & (pl.col('total_steps') == 200000)
    & (pl.col('eval_every') == 5000)
    & (pl.col('gamma') == 0.99)
    & pl.col('eval_full_auc_raw_mean').is_finite()
)


def _signed_verdict(
    result: ArmMeanDiffResult,
    *,
    d_threshold: float,
    sign: int,
    p_threshold: float = 0.05,
    sign_flip_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """Single-stratum HELD verdict matrix on Welch's-t mean diff.

    Predicted direction: sign=+1 → d ≥ d_threshold AND p ≤
    p_threshold; sign=−1 mirrored. NO_EFFECT (NULL) if |d| <
    sign_flip_threshold; NO_EFFECT (SIGN_FLIP) if d has the
    wrong sign with |d| ≥ sign_flip_threshold; POWER_INSUFFICIENT
    otherwise (e.g., d in the right direction but below threshold).
    """
    d = result.standardized_effect
    p = result.mean_diff_p_value
    if math.isnan(d) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if sign > 0:
        if d >= d_threshold and p <= p_threshold:
            return Verdict.HELD, None
        if d <= -sign_flip_threshold:
            return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    else:
        if d <= -d_threshold and p <= p_threshold:
            return Verdict.HELD, None
        if d >= sign_flip_threshold:
            return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(d) < sign_flip_threshold:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


@claim_bridge(
    source=INTERVENTION,
    target='eval_full_auc_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(_DENSE_EVAL_SCOPE & (pl.col('action_duplicate_k') == 16)),
    predicted_direction='a_gt_b',
)
def ddqn_full_auc_helps_at_acrobot_k16_dense(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_full_auc_raw_mean',
    pair_by: tuple[str, ...] = ('seed',),
    d_threshold: float = 0.6,
    p_threshold: float = 0.05,
    sign_flip_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """At Acrobot γ=0.99 k=16 dense-eval (200k steps, 40 bursts),
    DDQN's full-trajectory AUC outcome is higher than vanilla's.
    Pre-author empirical d=+1.13.

    Direction=DIRECT: more DDQN → more full-AUC outcome. The
    full-AUC integrator captures both early-burst speedup AND
    sustained late-burst advantage (memory
    `findings_dense_eval_acrobot_transient`).

    Verdict matrix (Welch's t on independent-samples Cohen's d):
      HELD              : d ≥ +0.6 AND p ≤ 0.05
      NO_EFFECT (NULL)  : |d| < 0.3
      NO_EFFECT (SIGN_FLIP) : d ≤ −0.3
      POWER_INSUFFICIENT : otherwise / NaN inputs
    """
    del treatment_arm, baseline_arm, source, pair_by
    return _signed_verdict(
        arm_mean_diff, d_threshold=d_threshold, sign=1,
        p_threshold=p_threshold,
        sign_flip_threshold=sign_flip_threshold,
    )


@claim_bridge(
    source=INTERVENTION,
    target='eval_full_auc_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(_DENSE_EVAL_SCOPE & (pl.col('action_duplicate_k') == 4)),
    predicted_direction='null',
)
def ddqn_full_auc_null_at_acrobot_k4_dense(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_full_auc_raw_mean',
    pair_by: tuple[str, ...] = ('seed',),
    null_ceiling: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """The k=4 "sweet spot" claim: DDQN's clip has nothing to
    correct at |A|=8 (Hasselt floor ≈ 2.04σ — too small for
    vanilla's bias to hurt). Predicted NULL; pre-author empirical
    d=+0.04.

    Verdict matrix:
      HELD              : |d| < null_ceiling=0.3 (NULL confirmed)
      INVARIANT_VIOLATION : d > +0.3 (sweet-spot falsified — DDQN
        actually helps at k=4)
      NO_EFFECT (SIGN_FLIP) : d < −0.3 (DDQN cleanly hurts; would
        refine the sweet-spot story to "DDQN harms exploration in
        low-bias regime")
      POWER_INSUFFICIENT : NaN inputs only (a single 30+30 stratum
        should always resolve to one of the above)
    """
    del treatment_arm, baseline_arm, source, pair_by
    d = arm_mean_diff.standardized_effect
    if math.isnan(d):
        return Verdict.POWER_INSUFFICIENT, None
    if abs(d) < null_ceiling:
        return Verdict.HELD, None
    if d > null_ceiling:
        return Verdict.INVARIANT_VIOLATION, None
    return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
