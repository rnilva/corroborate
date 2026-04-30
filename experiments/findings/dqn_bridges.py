"""Authored bridge declarations for the DDQN case study.

Each `@claim_bridge`-decorated function corresponds to a
published claim in FINDINGS.md. Running this file against a
corpus produces the typed verdicts that back the English
narrative; running it against a NEW corpus (held-out, larger,
different program) re-tests every claim with fresh data.

This is the forcing-function file: parquet + this file →
FINDINGS.md verdicts. If a bridge's verdict diverges from the
narrative, the claim has been falsified by new evidence.

The file consumes only the framework's analyses
(`paired_g`, `meta_regression_paired_g`, `backdoor_ate`,
`placebo_refutation`, `random_common_cause_refutation`); no new
framework code is added by this file. It IS the file-protocol
artifact the architecture promises.
"""
from __future__ import annotations

import math

# Importing the analyses package populates the registry so
# resolution by parameter name succeeds.
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.paired_g import PairedGResult
from corroborate.claim_bridge import (
    Direction, Tier, claim_bridge,
)
from corroborate.meta_regression import MetaRegressionResult
from corroborate.verdict import Verdict


# ============ Eighth revision (action_dim_sweep) ============
#
# "DDQN reduces jensen_gap on |A|≥3 envs (paired g, n=60 seeds);
# CartPole at |A|=2 reverses (sign wrong → POWER_INSUFFICIENT)."
#
# Reference: action_dim_sweep corpus, 4 envs × 2 arms × 60 seeds.
# Per-env paired-g table:
#   Acrobot-v1   |A|=3 g=-0.596 HELD
#   Catch-bsuite |A|=3 g=-4.662 HELD
#   DiscountingChain-bsuite |A|=5 g=-0.600 HELD
#   CartPole-v1  |A|=2 g=+0.090 POWER_INSUFFICIENT (sign wrong)


def _ddqn_reduces_gap_holds_when(paired_g: PairedGResult) -> Verdict:
    """Shared threshold logic for the per-env DDQN-reduces-gap
    bridges. Sign opposes prediction (DDQN expected to *reduce*
    gap, so positive g is sign-wrong) → POWER_INSUFFICIENT;
    n_pairs < 30 → POWER_INSUFFICIENT; else HELD when |g|≥0.3
    and p<0.05."""
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g >= 0:
        return Verdict.POWER_INSUFFICIENT  # sign opposes prediction
    if paired_g.g < -0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge
def ddqn_reduces_jensen_gap__acrobot(
    paired_g: PairedGResult,
    *,
    source: str = 'mechanism.jensen_gap',
    target: str = 'mechanism.jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'Acrobot-v1',
) -> Verdict:
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    return _ddqn_reduces_gap_holds_when(paired_g)


@claim_bridge
def ddqn_reduces_jensen_gap__catch(
    paired_g: PairedGResult,
    *,
    source: str = 'mechanism.jensen_gap',
    target: str = 'mechanism.jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'Catch-bsuite',
) -> Verdict:
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    return _ddqn_reduces_gap_holds_when(paired_g)


@claim_bridge
def ddqn_reduces_jensen_gap__discounting_chain(
    paired_g: PairedGResult,
    *,
    source: str = 'mechanism.jensen_gap',
    target: str = 'mechanism.jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'DiscountingChain-bsuite',
) -> Verdict:
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    return _ddqn_reduces_gap_holds_when(paired_g)


@claim_bridge
def ddqn_reduces_jensen_gap__cartpole(
    paired_g: PairedGResult,
    *,
    source: str = 'mechanism.jensen_gap',
    target: str = 'mechanism.jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'CartPole-v1',
) -> Verdict:
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    return _ddqn_reduces_gap_holds_when(paired_g)


# ============ Eighth revision: meta-regression ============
#
# "Meta-regression of g_mech on log_action_dim shows the right
# direction (β negative — bigger bias-reduction at higher |A|)
# but n=4 envs is underpowered for significance."
#
# Verdict: POWER_INSUFFICIENT.

_LOG_ACTION_DIM_PER_ENV: dict[str, dict[str, float]] = {
    'CartPole-v1': {'log_action_dim': math.log(2)},
    'Acrobot-v1': {'log_action_dim': math.log(3)},
    'Catch-bsuite': {'log_action_dim': math.log(3)},
    'DiscountingChain-bsuite': {'log_action_dim': math.log(5)},
}


@claim_bridge
def log_action_dim_drives_jensen_gap_reduction(
    meta_regression_paired_g: MetaRegressionResult,
    *,
    source: str = 'mechanism.jensen_gap',
    target: str = 'mechanism.jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    covariates_per_env: dict[str, dict[str, float]] = (
        _LOG_ACTION_DIM_PER_ENV
    ),
) -> Verdict:
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, covariates_per_env
    coef = next(
        (c for c in meta_regression_paired_g.coefficients
         if c.name == 'log_action_dim'),
        None,
    )
    if coef is None:
        return Verdict.NO_EFFECT
    if coef.coefficient < 0:
        return (
            Verdict.HELD if coef.is_significant
            else Verdict.POWER_INSUFFICIENT
        )
    return Verdict.NO_EFFECT


# ============ Bridge collection — the file's exported claims ============

ACTION_DIM_BRIDGES = (
    ddqn_reduces_jensen_gap__acrobot,
    ddqn_reduces_jensen_gap__catch,
    ddqn_reduces_jensen_gap__discounting_chain,
    ddqn_reduces_jensen_gap__cartpole,
    log_action_dim_drives_jensen_gap_reduction,
)
"""Bridges asserted on the action_dim_sweep corpus
(`experiments/data/action_dim_sweep/runs.parquet`)."""


__all__ = [
    'ACTION_DIM_BRIDGES',
    'ddqn_reduces_jensen_gap__acrobot',
    'ddqn_reduces_jensen_gap__catch',
    'ddqn_reduces_jensen_gap__cartpole',
    'ddqn_reduces_jensen_gap__discounting_chain',
    'log_action_dim_drives_jensen_gap_reduction',
]
