"""Executable claim test for the SB3 runs.

This module knows the estimand and the decision rule: which exact
parameter values define reference and treatment, on which measured
quantity, at which horizon, over which population, and what
statistical rule maps the evidence to a verdict. It knows nothing
about file layouts or how the runs were produced, and the
declaration does not verify external assignment; it evaluates any
compatible DataFrame.

The horizon is part of the claim. ``eval_reward_at_25k`` reads the
producer's own record columns (``eval/mean_reward`` aligned by
``time/total_timesteps``) and answers "the evaluation at step
25 000, NaN if this run was never evaluated there" — per run, no
record-wide anything. A claim at a different budget registers a
different measurable; the record itself never privileges one.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TypeIs

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
from corroborate.measurables import measurable

EVAL_HORIZON = 25_000


def _is_list(value: object) -> TypeIs[list[object]]:
    return isinstance(value, list)


def _is_number(value: object) -> TypeIs[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@measurable(
    name='eval_reward_at_25k',
    reads=('eval/mean_reward', 'time/total_timesteps'),
)
def eval_reward_at_25k(record: Mapping[str, object]) -> float:
    """Mean evaluation reward at the claim's declared horizon —
    the dump where ``time/total_timesteps`` is exactly 25 000.
    NaN for a run not evaluated there: the horizon is pinned, a
    shorter run is missing evidence, never rebased."""
    steps = record.get('time/total_timesteps')
    rewards = record.get('eval/mean_reward')
    if not _is_list(steps) or not _is_list(rewards):
        return float('nan')
    for step, reward in zip(steps, rewards):
        if step == EVAL_HORIZON and _is_number(reward):
            return float(reward)
    return float('nan')


GAMMA_EFFECT = DoEffect.from_values(
    source='config/gamma',
    reference=0.80,
    treatment=0.99,
)


@claim_bridge(
    source=GAMMA_EFFECT,
    target='eval_reward_at_25k',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('config/seed',),
    scope=pl.col('config/env_id') == 'CartPole-v1',
    predicted_direction='a_gt_b',
)
def higher_gamma_improves_return(
    paired_directional: PairedDirectionalResult,
    *,
    alpha: float = 0.05,
    sesoi_dz: float = 0.5,
    minimum_pairs: int = 3,
) -> tuple[Verdict, RefutationClass | None]:
    """Gamma 0.99 improves evaluation reward at 25k steps over
    gamma 0.80."""
    del alpha, sesoi_dz, minimum_pairs
    return paired_directional_verdict(paired_directional)


BRIDGES = (higher_gamma_improves_return,)


__all__ = [
    'BRIDGES',
    'EVAL_HORIZON',
    'GAMMA_EFFECT',
    'eval_reward_at_25k',
    'higher_gamma_improves_return',
]
