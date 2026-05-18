"""Bridges for the action_duplicate FA-gradient-density mechanism.

`ActionDuplicate(k=K)` inflates the *declared* action space K-fold
(`|A_decl| = K · |A_inner|`), keeping dynamics, reward, and optimal
Q* unchanged. Each declared action gets its own output-layer
weight row in the Q-network; per-action gradient signal splits by
K. Per-action Q-estimation noise σ_Q rises ~√K. Empirically
(FR/MC k=1..4 sweeps at γ=0.99) vanilla jens amplifies far beyond
Hasselt's √(2 ln K|A_inner|) floor.

Three bridges encode the substantive empirical claims:

1. `vanilla_jens_amplifies_with_k` — within-env Spearman
   ρ(action_duplicate_k, jensen_gap) > 0 on vanilla cells.
   Tests that vanilla bias scales with K. Empirical
   preview: FR ρ ≈ +1, MC ρ ≈ +1; pooled HELD expected.

2. `vanilla_q_late_drifts_with_k` — within-env Spearman
   ρ(K, q_late_mean) > 0 on vanilla cells. Tests the Q-drift
   channel (one of two routes for jens amplification). MC's
   Q rises from -60 to -50 across k=1..4 → ρ > 0 expected.
   FR's q_late is roughly stable (different env regime — see
   the MC-collapse channel which we don't measure here).

3. `vanilla_sigma_q_scales_with_k` — within-env Spearman
   ρ(K, q_action_std_late) > 0 on vanilla cells. Tests the
   per-action noise amplification directly. Currently the cache
   doesn't carry q_action_std_late for k-sweep corpora; this
   bridge fires POWER_INSUFFICIENT until the cache picks up the
   column (planned: add to ddqn_sweeps REQUIRED_MEASURABLES and
   re-ingest).

The shared scope: cells with non-null `action_duplicate_k`,
vanilla arm only (we're characterising vanilla's response to
the |A|-inflation, not arm-diff).

See `findings_action_duplicate_fa_mechanism.md` for the full
empirical context including the FR-vs-MC env-regime dependence
of the downstream channel (MC-collapse on solvable envs vs
Q-drift on unsolvable).
"""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.spearman.partial_spearman import (
    PartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import Verdict


_K_SWEEP_VANILLA_SCOPE: pl.Expr = (
    pl.col('action_duplicate_k').is_not_null()
    & (pl.col('arm_key') == 'baseline')
)


# ============ Bridge 1: vanilla jens scales with K ============

@claim_bridge(
    source='action_duplicate_k',
    target='jensen_gap',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _K_SWEEP_VANILLA_SCOPE
        & pl.col('jensen_gap').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def vanilla_jens_amplifies_with_k(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'action_duplicate_k',
    y: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 8,
    min_rho: float = 0.5,
    alpha: float = 0.05,
    min_strata: int = 1,
) -> Verdict:
    """Vanilla jens scales positively with K within k-sweep envs.

    Hasselt predicts ρ > 0 from the √log scaling. The empirical
    observation (FR/MC k=1..4) is that ρ is essentially +1 within
    env — vanilla jens grows monotonically with K. This bridge
    HELDs when the within-env Spearman is strongly positive
    (≥0.5) and Fisher-z-pooled across envs is significant.

    HELD: pooled ρ ≥ min_rho with p < alpha.
    REFUTED (NULL): |pooled ρ| ≤ min_rho/2 with adequate strata
    (the framework's NO_EFFECT shape — could happen if action_
    duplicate doesn't move jens at all)."""
    del x, y, stratify_by, min_stratum_size
    if partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = partial_spearman.rho_pooled
    p = partial_spearman.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT
    if rho >= min_rho and p < alpha:
        return Verdict.HELD
    if abs(rho) < min_rho / 2 and p > alpha:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


# ============ Bridge 2: Q drifts with K (Q-drift channel) ============

@claim_bridge(
    source='action_duplicate_k',
    target='q_late_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _K_SWEEP_VANILLA_SCOPE
        & pl.col('q_late_mean').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def vanilla_q_late_drifts_with_k(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'action_duplicate_k',
    y: str = 'q_late_mean',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 8,
    min_rho: float = 0.3,
    alpha: float = 0.10,
    min_strata: int = 1,
) -> Verdict:
    """Vanilla q_late_mean drifts upward with K — the Q-drift
    channel of jens amplification (active on unsolvable envs).

    Empirical preview: MC q_late from -60 → -50 across k=1..4
    (ρ ≈ +1 within MC). FR q_late stays roughly stable (different
    channel: MC drops, not Q rises). Pooling these gives a
    weakly-positive pooled ρ.

    HELD: pooled ρ ≥ 0.3 with p < 0.10 (allowing weaker bound
    because the Q-drift channel only activates on some envs).
    """
    del x, y, stratify_by, min_stratum_size
    if partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = partial_spearman.rho_pooled
    p = partial_spearman.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT
    if rho >= min_rho and p < alpha:
        return Verdict.HELD
    if abs(rho) < min_rho / 2:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


# ============ Bridge 3: per-action noise σ_Q grows with K ============

@claim_bridge(
    source='action_duplicate_k',
    target='q_action_std_late',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _K_SWEEP_VANILLA_SCOPE
        & pl.col('q_action_std_late').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def vanilla_sigma_q_scales_with_k(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'action_duplicate_k',
    y: str = 'q_action_std_late',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 8,
    min_rho: float = 0.3,
    alpha: float = 0.10,
    min_strata: int = 1,
) -> Verdict:
    """Per-action Q-estimation noise σ_Q grows with K — the
    proximate mechanism.

    Empirical (raw measurements.parquet): FR σ_Q 0.0050→0.0070
    (k=1→4), MC σ_Q 0.115→0.167. Within-env Spearman ρ ≈ +1.
    Currently the ddqn_sweeps cache doesn't carry q_action_std_late
    on k-sweep corpora, so this bridge fires POWER_INSUFFICIENT
    until the column is uplifted (re-ingest with REQUIRED_
    MEASURABLES update).

    HELD: pooled ρ ≥ 0.3 with p < 0.10."""
    del x, y, stratify_by, min_stratum_size
    if partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = partial_spearman.rho_pooled
    p = partial_spearman.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT
    if rho >= min_rho and p < alpha:
        return Verdict.HELD
    if abs(rho) < min_rho / 2:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


BRIDGES = (
    vanilla_jens_amplifies_with_k,
    vanilla_q_late_drifts_with_k,
    vanilla_sigma_q_scales_with_k,
)
