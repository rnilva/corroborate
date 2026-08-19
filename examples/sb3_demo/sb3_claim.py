"""Executable claim test for the SB3 runs.

This module knows the estimand and the decision rule: which exact
parameter values define reference and treatment, on which outcome,
over which population, and what statistical rule maps the evidence
to a verdict. It knows nothing about file layouts or how the runs
were produced, and the declaration does not verify external
assignment; it evaluates any compatible DataFrame.
"""
from __future__ import annotations

import polars as pl

from corroborate.analyses.paired.paired_directional import (
    PairedDirectionalResult,
    paired_directional_verdict,
)
from corroborate.bridge import (
    Direction,
    RefutationClass,
    Tier,
    Verdict,
    claim_bridge,
)
from corroborate.core import DoEffect


GAMMA_EFFECT = DoEffect.from_values(
    source='gamma',
    reference=0.80,
    treatment=0.99,
)


@claim_bridge(
    source=GAMMA_EFFECT,
    target='return_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('seed',),
    scope=pl.col('env_id') == 'CartPole-v1',
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


__all__ = ['BRIDGES', 'GAMMA_EFFECT', 'higher_gamma_improves_return']
