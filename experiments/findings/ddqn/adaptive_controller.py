"""CLAIM 2 corroboration — Pearl rung-2 adaptive controller.

The dormancy proxy (per-batch `max_Q − mean_Q ≥ σ_floor_factor ×
σ_Q × √(2 log |A|)`) dispatches between DDQN and vanilla.

- `adaptive_dqn_recovers_ddqn_benefit__fourrooms_factor_0p5`: HELD
  on FR (Hasselt-floor ACTIVE) — adaptive recovers DDQN benefit.
- `adaptive_dqn_fails_to_avoid_attenuation__spaceinvaders_1m`: on
  SI 1M dormancy proxy doesn't fire → controller ≡ DDQN → inherits
  attenuation. Together encode "dormancy necessary, not sufficient."

Both AWAITING DATA: adaptive sweep corpora absent from current
postfix rebuild. Historical: FR g=+0.78 p<0.001; SI g=-0.46 p=0.016."""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.paired_g import PairedGResult
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.core.intervention import DoEffect

from experiments.findings.ddqn._arms import ADAPTIVE_DQN_FACTOR_0P5_SWAP


@claim_bridge(
    source=DoEffect(treatment=(ADAPTIVE_DQN_FACTOR_0P5_SWAP,), baseline=()),
    target='eval_final_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & pl.col('corpus').is_in(
            ('adaptive_dqn_fourrooms_sweep', 'expectile_3way'),
        )
    ),
)
def adaptive_dqn_recovers_ddqn_benefit__fourrooms_factor_0p5(
    paired_g: PairedGResult,
) -> Verdict:
    """Per-batch dormancy proxy `max_Q − mean_Q ≥ 0.5 × σ_Q ×
    √(2 log |A|)` recovers DDQN's benefit on FR. HELD when g ≥
    +0.50 AND p<0.05. Historical: g=+0.78, p<0.001. AWAITING DATA."""
    if paired_g.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.g):
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g >= 0.50 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source=DoEffect(treatment=(ADAPTIVE_DQN_FACTOR_0P5_SWAP,), baseline=()),
    target='eval_final_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'SpaceInvaders-MinAtar')
        & pl.col('corpus').is_in(
            ('adaptive_dqn_spaceinvaders_1m', 'minatar_1M_spaceinvaders'),
        )
    ),
)
def adaptive_dqn_fails_to_avoid_attenuation__spaceinvaders_1m(
    paired_g: PairedGResult,
) -> Verdict:
    """Scope-limitation: on SI 1M, dormancy proxy doesn't fire,
    controller ≡ DDQN, inherits attenuation. HELD when g(adaptive
    vs vanilla) ≤ -0.30, p<0.05. INVARIANT_VIOLATION if adaptive
    unexpectedly helps. Historical: g=-0.46, p=0.016. AWAITING DATA."""
    if paired_g.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.g):
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g <= -0.30 and paired_g.p_value < 0.05:
        return Verdict.HELD
    if paired_g.g >= 0.30 and paired_g.p_value < 0.05:
        return Verdict.INVARIANT_VIOLATION
    return Verdict.NO_EFFECT


BRIDGES = (
    adaptive_dqn_recovers_ddqn_benefit__fourrooms_factor_0p5,
    adaptive_dqn_fails_to_avoid_attenuation__spaceinvaders_1m,
)
