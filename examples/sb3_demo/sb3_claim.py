"""Executable claim test for the SB3 study.

This module knows the estimand and the decision rule.  It deliberately knows
nothing about the bundle, adapter, producer, or producer-specific arm labels;
those are bound from the verified record only when the bridge is evaluated.
"""
from __future__ import annotations

import polars as pl

from corroborate.analyses.paired.paired_directional import (
    PairedDirectionalResult,
    paired_directional_verdict,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict


@claim_bridge(
    source='gamma',
    target='return_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('seed',),
    scope=(
        (pl.col('env_id') == 'CartPole-v1')
        & pl.col('gamma').is_in([0.80, 0.99])
    ),
    predicted_direction='a_gt_b',
)
def higher_gamma_improves_return(
    paired_directional: PairedDirectionalResult,
    *,
    alpha: float = 0.05,
    sesoi_dz: float = 0.5,
    minimum_pairs: int = 3,
) -> tuple[Verdict, RefutationClass | None]:
    """Gamma 0.99 improves final mean return over gamma 0.80."""
    del alpha, sesoi_dz, minimum_pairs
    return paired_directional_verdict(paired_directional)


BRIDGES = (higher_gamma_improves_return,)


__all__ = ['BRIDGES', 'higher_gamma_improves_return']
