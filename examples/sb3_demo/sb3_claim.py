"""Executable claim test for the SB3 runs.

This module knows the estimand and the decision rule: which
assigned parameter, on which outcome, over which scope (the scope
pins the two compared values — scope-as-extent), and what
statistical rule maps the evidence to a verdict.  It knows
nothing about file layouts or how the runs were produced — it
evaluates against any DataFrame that carries the named columns.
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
