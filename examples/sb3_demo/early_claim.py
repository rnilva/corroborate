"""An independently authored question over the same unmodified SB3 records.

This is a second budget-specific claim, not a causal explanation of the
relationship between early and late performance. Both claims use the
same experimental units; they must not be pooled as independent tests.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypeIs

import polars as pl

from corroborate.analyses.paired.paired_directional import (
    PairedDirectionalResult,
    paired_directional_verdict,
)
from corroborate.bridge import Direction, RefutationClass, Tier, Verdict, claim_bridge
from corroborate.measurables import measurable

from examples.sb3_demo.sb3_claim import GAMMA_EFFECT


def _is_list(value: object) -> TypeIs[list[object]]:
    return isinstance(value, list)


@measurable(
    name='eval_reward_at_15k',
    reads=('eval/mean_reward', 'time/total_timesteps'),
)
def eval_reward_at_15k(record: Mapping[str, object]) -> float:
    """The declared evaluation budget, never the last available dump."""
    steps = record.get('time/total_timesteps')
    rewards = record.get('eval/mean_reward')
    if not _is_list(steps) or not _is_list(rewards):
        return math.nan
    values = [
        float(value)
        for step, value in zip(steps, rewards)
        if step == 15_000 and isinstance(value, (int, float))
        and not isinstance(value, bool) and math.isfinite(value)
    ]
    if not values or any(value != values[0] for value in values[1:]):
        return math.nan
    return values[0]


@claim_bridge(
    source=GAMMA_EFFECT,
    target='eval_reward_at_15k',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('config/seed',),
    scope=pl.col('config/env_id') == 'CartPole-v1',
    predicted_direction='a_gt_b',
)
def higher_gamma_improves_early_return(
    paired_directional: PairedDirectionalResult,
    *,
    alpha: float = 0.05,
    sesoi_dz: float = 0.5,
    minimum_pairs: int = 3,
) -> tuple[Verdict, RefutationClass | None]:
    """Gamma 0.99 improves evaluation reward at 15k over gamma 0.80."""
    del alpha, sesoi_dz, minimum_pairs
    return paired_directional_verdict(paired_directional)
