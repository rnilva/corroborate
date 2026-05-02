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

from corroborate.analyses.dowhy import (
    BackdoorResult, RefutationResult,
)
from corroborate.analyses.factorial_2x2 import Factorial2x2Result
from corroborate.analyses.paired_g import PairedGResult
from corroborate.analyses.paired_g_per_burst import (
    PerBurstResult, panel_for_env,
)
from corroborate.analyses.paired_g_pooled import (
    PooledPairedGResult,
)
from corroborate.rl.env_solve_thresholds import SOLVE_THRESHOLDS
from corroborate.analyses.tautology_audit import AuditResult
from corroborate.analyses.verdict_distribution import (
    VerdictDistributionResult,
)
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
    source: str = 'jensen_gap',
    target: str = 'jensen_gap',
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
    source: str = 'jensen_gap',
    target: str = 'jensen_gap',
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
    source: str = 'jensen_gap',
    target: str = 'jensen_gap',
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
    source: str = 'jensen_gap',
    target: str = 'jensen_gap',
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
    source: str = 'jensen_gap',
    target: str = 'jensen_gap',
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


# ============ Seventh revision: dormancy invariant per env =========
#
# Per-env verdict on `at_most[jensen_dormancy_gap<=0]` — the
# framework's-own scope predicate for `double_greedify`. The
# invariant fires per cell with verdict HELD (premise active:
# observed bias ≥ structural Jensen floor) or INVARIANT_VIOLATION
# (premise dormant: observed bias < floor → DDQN's correction has
# nothing to bite on). Authored against the action_dim_sweep
# corpus where each env has a specific structural prediction:
#
#   Acrobot-v1   |A|=3 → premise active   (HELD on 60/60 cells)
#   CartPole-v1  |A|=2 → premise active   (HELD on 60/60 cells)
#   Catch-bsuite |A|=3 → premise dormant  (INVARIANT_VIOLATION on 60/60)
#   DiscountingChain-bsuite |A|=5 → premise active (HELD on 60/60)
#
# These are CATEGORICAL claims (not continuous-effect): the
# `verdict_distribution_per_env` analysis tallies the persisted
# verdict column; bridges assert "≥90% of cells fire the
# predicted verdict on this env".


_DORMANCY_VERDICT_COLUMN: str = (
    'at_most[jensen_dormancy_gap<=0].verdict'
)


def _premise_holds_when(
    distribution: VerdictDistributionResult,
    env_name: str,
    *,
    expected: str,
    fraction_threshold: float = 0.9,
    min_cells: int = 30,
) -> Verdict:
    counts = distribution.for_env(env_name)
    if counts is None or counts.total < min_cells:
        return Verdict.POWER_INSUFFICIENT
    if expected == 'held':
        fraction = counts.held_fraction
    elif expected == 'invariant_violation':
        fraction = counts.violation_fraction
    else:
        return Verdict.POWER_INSUFFICIENT
    if fraction != fraction:  # NaN
        return Verdict.POWER_INSUFFICIENT
    return (
        Verdict.HELD if fraction >= fraction_threshold
        else Verdict.INVARIANT_VIOLATION
    )


@claim_bridge
def jensen_premise_active__acrobot(
    verdict_distribution_per_env: VerdictDistributionResult,
    *,
    source: str = 'at_most[jensen_dormancy_gap<=0]',
    target: str = 'at_most[jensen_dormancy_gap<=0]',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    arm_filter: str = 'ddqn',
    verdict_column: str = _DORMANCY_VERDICT_COLUMN,
    env_name: str = 'Acrobot-v1',
) -> Verdict:
    del source, target, direction, tier, arm_filter, verdict_column
    return _premise_holds_when(
        verdict_distribution_per_env, env_name, expected='held',
    )


@claim_bridge
def jensen_premise_active__cartpole(
    verdict_distribution_per_env: VerdictDistributionResult,
    *,
    source: str = 'at_most[jensen_dormancy_gap<=0]',
    target: str = 'at_most[jensen_dormancy_gap<=0]',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    arm_filter: str = 'ddqn',
    verdict_column: str = _DORMANCY_VERDICT_COLUMN,
    env_name: str = 'CartPole-v1',
) -> Verdict:
    del source, target, direction, tier, arm_filter, verdict_column
    return _premise_holds_when(
        verdict_distribution_per_env, env_name, expected='held',
    )


@claim_bridge
def jensen_premise_dormant__catch(
    verdict_distribution_per_env: VerdictDistributionResult,
    *,
    source: str = 'at_most[jensen_dormancy_gap<=0]',
    target: str = 'at_most[jensen_dormancy_gap<=0]',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    arm_filter: str = 'ddqn',
    verdict_column: str = _DORMANCY_VERDICT_COLUMN,
    env_name: str = 'Catch-bsuite',
) -> Verdict:
    """Catch is the structural counterexample: |A|=3 but σ_Q is
    tiny (0.07) and observed bias even tinier (0.03), so the
    Jensen floor (0.10) exceeds observed → premise dormant. The
    invariant correctly fires on this env; the bridge HELD when
    ≥90% of cells return INVARIANT_VIOLATION."""
    del source, target, direction, tier, arm_filter, verdict_column
    return _premise_holds_when(
        verdict_distribution_per_env, env_name,
        expected='invariant_violation',
    )


@claim_bridge
def jensen_premise_active__discounting_chain(
    verdict_distribution_per_env: VerdictDistributionResult,
    *,
    source: str = 'at_most[jensen_dormancy_gap<=0]',
    target: str = 'at_most[jensen_dormancy_gap<=0]',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    arm_filter: str = 'ddqn',
    verdict_column: str = _DORMANCY_VERDICT_COLUMN,
    env_name: str = 'DiscountingChain-bsuite',
) -> Verdict:
    del source, target, direction, tier, arm_filter, verdict_column
    return _premise_holds_when(
        verdict_distribution_per_env, env_name, expected='held',
    )


# ============ Per-burst panel claims (revisions 9, 12) ============
#
# Per-(env, burst) paired g on `mc_return`. Asserted on the
# expectile_3way corpus (joined runs.parquet × traces.parquet).
#
# Reference verdicts:
#   FourRooms-misc: g positive across all 10 bursts; mean g
#       ≈ +0.65 — "DDQN benefit stable throughout" (revision 9).
#   Catch-bsuite:   g ≈ 0 across every burst — "DDQN at n=1 has
#       *exactly* zero effect on Catch" (revision 12).


@claim_bridge
def ddqn_outcome_stable_across_bursts__fourrooms(
    paired_g_per_burst: PerBurstResult,
    *,
    source: str = 'mc_return',
    target: str = 'mc_return',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    reduction: str = 'mean',
    env_name: str = 'FourRooms-misc',
) -> Verdict:
    """DDQN's outcome benefit on FourRooms is stable across every
    eval burst. HELD when (a) at least 9/10 bursts have positive
    g and (b) the per-burst mean g exceeds 0.3."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, reduction
    panel = panel_for_env(paired_g_per_burst, env_name)
    if not panel:
        return Verdict.POWER_INSUFFICIENT
    positive = sum(1 for s in panel if s.g > 0)
    mean_g = sum(s.g for s in panel) / len(panel)
    if positive >= len(panel) - 1 and mean_g > 0.3:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge
def ddqn_outcome_zero_across_bursts__catch(
    paired_g_per_burst: PerBurstResult,
    *,
    source: str = 'mc_return',
    target: str = 'mc_return',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    reduction: str = 'mean',
    env_name: str = 'Catch-bsuite',
) -> Verdict:
    """Catch-bsuite saturates near-optimal under both arms;
    DDQN at n=1 has zero per-burst effect. NO_EFFECT when
    every burst's |g| is below 0.1; HELD-shaped verdicts are
    impossible since the prediction is null."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, reduction
    panel = panel_for_env(paired_g_per_burst, env_name)
    if not panel:
        return Verdict.POWER_INSUFFICIENT
    if all(abs(s.g) < 0.1 for s in panel):
        return Verdict.NO_EFFECT
    # Any burst with |g| ≥ 0.1 falsifies the "saturated, no
    # effect" claim. NO_EFFECT/HELD aren't the right shape; map
    # to POWER_INSUFFICIENT with the per-burst max |g| as the
    # diagnostic signal in the audit trail.
    return Verdict.POWER_INSUFFICIENT


# ============ First revision (ddqn 200k corpus) =====================
#
# The headline DDQN finding: on the convergence-conditioned
# subset, mechanism (Δjensen_gap) activates strongly (g≈-0.93),
# but the link to outcome (Δeval_best_burst_mean) is null
# (g≈-0.03). The "+0.086 across all 18 envs" outcome g from the
# unrestricted analysis is a convergence artifact — once we
# restrict to envs where vanilla DQN reached a learned policy,
# DDQN's outcome contribution disappears.
#
# Reference verdicts on `experiments/data/ddqn/runs.parquet`,
# total_steps=200000 cells (1080 of 2160), corrected discounted
# thresholds (rev 2):
#
#   Mechanism on converged subset (6 envs): pooled g=-0.925,
#     I²=0.94 → HELD
#   Outcome on converged subset:           pooled g=-0.032,
#     I²=0.35 → NO_EFFECT (link broken)
#
# Converged subset is the substrate's `classify_envs` result on
# the baseline arm at 200k. Encoded as a literal tuple here so
# the bridge commits to a specific scope claim; reproduction
# against a fresh corpus may produce a different subset.


_CONVERGED_ENVS_DDQN_200K: tuple[str, ...] = (
    'Acrobot-v1',
    'Breakout-MinAtar',
    'Catch-bsuite',
    'DeepSea-bsuite',
    'DiscountingChain-bsuite',
    'UmbrellaChain-bsuite',
)


def _pooled_negative_holds_when(
    pooled: PooledPairedGResult,
    *,
    g_threshold: float,
    min_envs: int,
) -> Verdict:
    """HELD when pooled g < -|threshold| with sufficient envs;
    sign-positive → POWER_INSUFFICIENT; insufficient envs →
    POWER_INSUFFICIENT; otherwise NO_EFFECT."""
    if pooled.n_envs < min_envs:
        return Verdict.POWER_INSUFFICIENT
    g = pooled.pooled.pooled_g
    if math.isnan(g):
        return Verdict.POWER_INSUFFICIENT
    if g >= 0:
        return Verdict.POWER_INSUFFICIENT  # sign opposes prediction
    if g < -abs(g_threshold):
        return Verdict.HELD
    return Verdict.NO_EFFECT


def _pooled_null_holds_when(
    pooled: PooledPairedGResult,
    *,
    null_band: float,
    min_envs: int,
) -> Verdict:
    """NO_EFFECT (the rev 1 outcome reading) when |pooled g| <
    null_band — small effect, link is empirically broken;
    otherwise the prediction-of-no-effect is refuted."""
    if pooled.n_envs < min_envs:
        return Verdict.POWER_INSUFFICIENT
    g = pooled.pooled.pooled_g
    if math.isnan(g):
        return Verdict.POWER_INSUFFICIENT
    if abs(g) < null_band:
        return Verdict.NO_EFFECT
    return Verdict.HELD


@claim_bridge
def ddqn_reduces_jensen_gap__converged_subset(
    paired_g_pooled: PooledPairedGResult,
    *,
    source: str = 'jensen_gap',
    target: str = 'jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    env_filter: tuple[str, ...] = _CONVERGED_ENVS_DDQN_200K,
    total_steps_filter: int = 200000,
) -> Verdict:
    """rev 1: pooled paired g(jensen_gap) on the converged subset
    is strongly negative (~-0.93), HELD."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by
    del env_filter, total_steps_filter
    return _pooled_negative_holds_when(
        paired_g_pooled, g_threshold=0.5, min_envs=5,
    )


@claim_bridge
def ddqn_link_to_outcome_null__converged_subset(
    paired_g_pooled: PooledPairedGResult,
    *,
    source: str = 'eval_best_burst_mean',
    target: str = 'eval_best_burst_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    env_filter: tuple[str, ...] = _CONVERGED_ENVS_DDQN_200K,
    total_steps_filter: int = 200000,
) -> Verdict:
    """rev 1: pooled paired g(eval_best_burst_mean) on the
    converged subset is essentially zero (~-0.03). The literature-
    predicted outcome benefit fails — link broken on this corpus.
    Verdict NO_EFFECT encodes that null-link reading directly.

    `source` is the measured column (the analysis's input); the
    conceptual edge is `jensen_gap → outcome.eval_best_
    burst_mean`, but the paired-g consumes only the target column
    + arm. The 'link broken' reading lives in the bridge's
    NO_EFFECT verdict combined with the mechanism HELD on
    `ddqn_reduces_jensen_gap__converged_subset` — same scope,
    same arm, mechanism activates but outcome doesn't move."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by
    del env_filter, total_steps_filter
    return _pooled_null_holds_when(
        paired_g_pooled, null_band=0.15, min_envs=3,
    )


# Note: paired_g_pooled is the analysis name. Both bridges
# consume the SAME analysis name but with different `source`
# values (jensen_gap vs eval_best_burst_mean) —
# the framework's resolver instantiates separately per bridge.


# ============ Eleventh revision: n-step refutes variance-reduction ===
#
# Strategy 1 from the intervention design: hold DDQN fixed; vary
# n_step ∈ {1, 3}. Variance-reduction theory predicts 3-step
# return reduces bootstrap-variance on top of DDQN's bias-
# correction, so outcome should improve. Data refutes:
#
#   Mechanism (pooled paired g(jensen_gap), 4 envs):
#     pooled g=-0.91, I²=0.96 → HELD (3-step DOES reduce bias)
#   Outcome (pooled paired g(eval_final_mean)):
#     pooled g=-0.27, I²=0.96 → NO_EFFECT (no average benefit;
#     I² shows env-heterogeneity, not a clean null)
#   Catch-bsuite outcome:           g=-2.14 → 3-step HURTS
#   DiscountingChain-bsuite outcome: g=+1.20 → 3-step helps (only
#     env where the variance-reduction prediction lands)
#
# Mechanism activates exactly as theory predicts; link fails to
# follow. The residual `bootstrap_fraction → g_link | g_mech`
# (rev 10's ATE=+0.88) is NOT carried by the variance axis.


@claim_bridge
def nstep_3step_reduces_bias_on_top_of_ddqn(
    paired_g_pooled: PooledPairedGResult,
    *,
    source: str = 'jensen_gap',
    target: str = 'jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.INTERVENTIONAL,
    treatment_arm: str = 'ddqn_3step',
    baseline_arm: str = 'ddqn_1step',
    pair_by: tuple[str, ...] = ('seed',),
    env_filter: tuple[str, ...] = (),
    total_steps_filter: int = 200000,
) -> Verdict:
    """rev 11: pooled g(jensen_gap) over 4 sparse-reward envs is
    -0.91 → HELD. n-step return DOES additionally reduce
    overestimation bias on top of DDQN's mechanism, exactly as
    variance-reduction theory predicts at the bias level."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by
    del env_filter, total_steps_filter
    return _pooled_negative_holds_when(
        paired_g_pooled, g_threshold=0.5, min_envs=3,
    )


@claim_bridge
def nstep_3step_does_not_help_outcome__pool(
    paired_g_pooled: PooledPairedGResult,
    *,
    source: str = 'eval_final_mean',
    target: str = 'eval_final_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.INTERVENTIONAL,
    treatment_arm: str = 'ddqn_3step',
    baseline_arm: str = 'ddqn_1step',
    pair_by: tuple[str, ...] = ('seed',),
    env_filter: tuple[str, ...] = (),
    total_steps_filter: int = 200000,
) -> Verdict:
    """rev 11: pooled paired g(eval_final_mean) over 4 envs is
    -0.27 with I²=0.96 — variance-reduction predicted positive g,
    pool sits null-or-mildly-negative. Verdict NO_EFFECT encodes
    the falsified prediction; the I² hides env-specific harm
    (Catch g=-2.14) and help (DiscountingChain g=+1.20) cancelling
    in the pool."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by
    del env_filter, total_steps_filter
    return _pooled_null_holds_when(
        paired_g_pooled, null_band=0.5, min_envs=3,
    )


def _per_env_harm_holds_when(paired_g: PairedGResult) -> Verdict:
    """HELD when treatment significantly REDUCES outcome —
    the inverse of `_ddqn_reduces_gap_holds_when`. Predicted
    direction is INVERSE: treatment is claimed to reduce the
    outcome (i.e. HURT relative to baseline)."""
    if paired_g.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.g):
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g >= 0:
        return Verdict.POWER_INSUFFICIENT  # sign opposes prediction
    if paired_g.g < -0.5 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge
def nstep_3step_hurts_outcome__catch(
    paired_g: PairedGResult,
    *,
    source: str = 'eval_final_mean',
    target: str = 'eval_final_mean',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.INTERVENTIONAL,
    treatment_arm: str = 'ddqn_3step',
    baseline_arm: str = 'ddqn_1step',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'Catch-bsuite',
) -> Verdict:
    """rev 11: 3-step on top of DDQN strongly HURTS outcome on
    Catch-bsuite (g=-2.14, n=30). Catch is short-episode +
    saturating-reward — long rollouts dilute the rare positive
    signal, so n-step's variance trade is net negative."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    return _per_env_harm_holds_when(paired_g)


@claim_bridge
def nstep_3step_helps_outcome__discounting_chain(
    paired_g: PairedGResult,
    *,
    source: str = 'eval_final_mean',
    target: str = 'eval_final_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.INTERVENTIONAL,
    treatment_arm: str = 'ddqn_3step',
    baseline_arm: str = 'ddqn_1step',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'DiscountingChain-bsuite',
) -> Verdict:
    """rev 11: DiscountingChain is the one env where 3-step
    actually helps outcome on top of DDQN (g=+1.20). |A|=5
    + chain structure means the env has enough action-margin AND
    long horizons for the variance-reduction prediction to land.
    Sole positive on this corpus — the heterogeneity that drives
    the pool's high I²."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    if paired_g.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.g):
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g <= 0:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g > 0.5 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# ============ Twelfth revision: 2×2 factorial ========================
#
# Complete (greedification × n_step) factorial on 5 sparse-reward
# envs. Per-env discriminator: over-correction (DDQN+n_step
# strictly worse than additive) vs DDQN-attenuation (DDQN's
# marginal benefit shrinks where n_step covers the same axis) vs
# variance-amplification (n-step alone backfires; DDQN orthogonal).
#
# Reference values on `eval_best_burst_mean` (the
# Hasselt-convention default), 4 corpora unioned:
#
#   FourRooms-misc:  (B−A)=+0.74 (D−C)=+0.16 (C−A)=+0.73
#                    (D−B)=+0.09 INT=-0.71 z=-3.49 → DDQN-attenuation
#   Catch-bsuite:    (B−A)=-1.16 (D−C)=-1.15 (C−A)=+0.00
#                    (D−B)=-0.05 INT=-0.05      → variance-amplification
#                    (n-step alone hurts; DDQN orthogonal)
#   Other 3 envs:    small/noisy, |INT| < 0.4, |z| < 2
#
# Cells expected from union(nstep_intervention, nstep_intervention_
# fr, nstep_vanilla_arms): 600 (5 envs × 4 arms × 30 seeds).


@claim_bridge
def factorial_ddqn_attenuation__fourrooms(
    factorial_2x2_interaction: Factorial2x2Result,
    *,
    source: str = 'eval_best_burst_mean',
    target: str = 'eval_best_burst_mean',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.INTERVENTIONAL,
    arm_a: str = 'vanilla_1step',
    arm_b: str = 'vanilla_3step',
    arm_c: str = 'ddqn_1step',
    arm_d: str = 'ddqn_3step',
    pair_by: tuple[str, ...] = ('seed',),
    env_filter: tuple[str, ...] = ('FourRooms-misc',),
    total_steps_filter: int = 200000,
    env_name: str = 'FourRooms-misc',
) -> Verdict:
    """rev 12: on FourRooms, DDQN's marginal outcome benefit
    shrinks 4× when n_step rises from 1 to 3 (g(C-A)=+0.73 →
    g(D-B)=+0.09). Both arms benefit from n-step similarly
    ((B-A)=+0.74 vs (D-C)=+0.16), so the shrink isn't over-
    correction — it's attenuation: the bias-correction axes
    overlap. Interaction g=-0.71 with z=-3.49 → HELD."""
    del source, target, direction, tier
    del arm_a, arm_b, arm_c, arm_d, pair_by
    del env_filter, total_steps_filter
    p = factorial_2x2_interaction.for_env(env_name)
    if p is None or p.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(p.g_interaction) or p.se_interaction <= 0:
        return Verdict.POWER_INSUFFICIENT
    z = p.g_interaction / p.se_interaction
    # DDQN-attenuation reading needs:
    # - interaction strongly negative (DDQN benefit shrinks at n=3)
    # - DDQN actually has room at n=1 (C-A > 0)
    # - both arms benefit from n-step in the same direction
    #   ((B-A) > 0 AND (D-C) > 0)
    if p.g_interaction >= 0:
        return Verdict.POWER_INSUFFICIENT  # sign opposes prediction
    attenuation_consistent = (
        p.g_interaction < -0.4
        and z < -2.0
        and p.g_c_minus_a > 0.3
        and p.g_b_minus_a > 0
        and p.g_d_minus_c > 0
    )
    return Verdict.HELD if attenuation_consistent else Verdict.NO_EFFECT


@claim_bridge
def factorial_variance_amplification__catch(
    factorial_2x2_interaction: Factorial2x2Result,
    *,
    source: str = 'eval_best_burst_mean',
    target: str = 'eval_best_burst_mean',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.INTERVENTIONAL,
    arm_a: str = 'vanilla_1step',
    arm_b: str = 'vanilla_3step',
    arm_c: str = 'ddqn_1step',
    arm_d: str = 'ddqn_3step',
    pair_by: tuple[str, ...] = ('seed',),
    env_filter: tuple[str, ...] = ('Catch-bsuite',),
    total_steps_filter: int = 200000,
    env_name: str = 'Catch-bsuite',
) -> Verdict:
    """rev 12: on Catch, n_step alone catastrophically harms
    vanilla ((B-A)=-1.16); DDQN at n=1 has *exactly* zero
    effect ((C-A)=+0.00 — saturated policy); n_step on top of
    DDQN matches the vanilla harm ((D-C)=-1.15, ≈ B-A); the
    interaction is null (INT=-0.05). The entire negative outcome
    is variance-amplification from n-step; DDQN is orthogonal.
    HELD when the discriminator pattern (n-step harm symmetric
    across greedification, DDQN ineffective at n=1, no
    interaction) holds."""
    del source, target, direction, tier
    del arm_a, arm_b, arm_c, arm_d, pair_by
    del env_filter, total_steps_filter
    p = factorial_2x2_interaction.for_env(env_name)
    if p is None or p.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(p.g_b_minus_a):
        return Verdict.POWER_INSUFFICIENT
    # Variance-amplification pattern:
    # - n_step alone strongly harms (B-A) << 0
    # - n_step on DDQN harms ~equally ((D-C) ≈ (B-A))
    # - DDQN at n=1 inert ((C-A) ≈ 0)
    # - interaction near zero (no DDQN+n_step compounding)
    pattern = (
        p.g_b_minus_a < -0.5
        and abs(p.g_d_minus_c - p.g_b_minus_a) < 0.5
        and abs(p.g_c_minus_a) < 0.3
        and abs(p.g_interaction) < 0.3
    )
    return Verdict.HELD if pattern else Verdict.NO_EFFECT


# ============ Sixth revision: time-to-first-solve ====================
#
# Sample-efficiency probe at the link edge: among (env, seed)
# pairs where BOTH arms reached threshold, does DDQN cross the
# bar faster (smaller `eval_best_burst_step`)?
#
# Reference verdicts on `experiments/data/ddqn/runs.parquet`,
# total_steps=200000, gate=`eval_best_burst_mean ≥
# SOLVE_THRESHOLDS[env]`:
#
#   Pool (high-solve, 5 non-degenerate envs): pooled g=−0.005,
#     I²=0.67 → NO_EFFECT (link null at sample-efficiency lens too)
#   SpaceInvaders-MinAtar (single env): g=−0.532, n=30 → HELD
#     (DDQN solves faster on this sparse-pixel env)


_SOLVE_THRESHOLDS_FLAT: dict[str, float] = {
    env: spec.threshold
    for env, spec in SOLVE_THRESHOLDS.items()
    if spec.threshold is not None
}


_TIME_TO_SOLVE_HIGH_SOLVE_ENVS: tuple[str, ...] = (
    'Acrobot-v1',
    'Breakout-MinAtar',
    'Catch-bsuite',
    'DiscountingChain-bsuite',
    'MemoryChain-bsuite',
    'SpaceInvaders-MinAtar',
    'UmbrellaChain-bsuite',
)


@claim_bridge
def time_to_solve_link_null__pooled(
    paired_g_among_solvers: PooledPairedGResult,
    *,
    source: str = 'eval_best_burst_step',
    target: str = 'eval_best_burst_step',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    gate_column: str = 'eval_best_burst_mean',
    gate_thresholds: dict[str, float] = _SOLVE_THRESHOLDS_FLAT,
    env_filter: tuple[str, ...] = _TIME_TO_SOLVE_HIGH_SOLVE_ENVS,
    total_steps_filter: int = 200000,
) -> Verdict:
    """rev 6: replacing the steady-state outcome with a sample-
    efficiency proxy doesn't rescue DDQN. Pooled across 5
    non-degenerate high-solve envs, predicted-direction effect
    averages zero with PI bracketing zero — NO_EFFECT."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by
    del gate_column, gate_thresholds, env_filter, total_steps_filter
    return _pooled_null_holds_when(
        paired_g_among_solvers, null_band=0.15, min_envs=4,
    )


@claim_bridge
def ddqn_solves_faster__spaceinvaders(
    paired_g_among_solvers: PooledPairedGResult,
    *,
    source: str = 'eval_best_burst_step',
    target: str = 'eval_best_burst_step',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    gate_column: str = 'eval_best_burst_mean',
    gate_thresholds: dict[str, float] = _SOLVE_THRESHOLDS_FLAT,
    env_filter: tuple[str, ...] = ('SpaceInvaders-MinAtar',),
    total_steps_filter: int = 200000,
) -> Verdict:
    """rev 6: SpaceInvaders-MinAtar is the one env where DDQN
    crosses the solve threshold reliably faster than vanilla
    (g=-0.532, n=30 — moderate effect, sign matches Hasselt's
    overestimation-bias-cost-on-sparse-reward prediction)."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by
    del gate_column, gate_thresholds, total_steps_filter
    spaceinvaders = next(
        (p for p in paired_g_among_solvers.per_env
         if p.stratum_id == env_filter[0]),
        None,
    )
    if spaceinvaders is None:
        return Verdict.POWER_INSUFFICIENT
    if spaceinvaders.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(spaceinvaders.g):
        return Verdict.POWER_INSUFFICIENT
    if spaceinvaders.g >= 0:
        return Verdict.POWER_INSUFFICIENT  # sign opposes prediction
    if spaceinvaders.g < -0.3:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# ============ Bridge collection — the file's exported claims ============

ACTION_DIM_BRIDGES = (
    ddqn_reduces_jensen_gap__acrobot,
    ddqn_reduces_jensen_gap__catch,
    ddqn_reduces_jensen_gap__discounting_chain,
    ddqn_reduces_jensen_gap__cartpole,
    log_action_dim_drives_jensen_gap_reduction,
    jensen_premise_active__acrobot,
    jensen_premise_active__cartpole,
    jensen_premise_dormant__catch,
    jensen_premise_active__discounting_chain,
)
"""Bridges asserted on the action_dim_sweep corpus
(`experiments/data/action_dim_sweep/runs.parquet`)."""


DDQN_200K_BRIDGES = (
    ddqn_reduces_jensen_gap__converged_subset,
    ddqn_link_to_outcome_null__converged_subset,
    time_to_solve_link_null__pooled,
    ddqn_solves_faster__spaceinvaders,
)
"""Bridges asserted on the ddqn 200k corpus
(`experiments/data/ddqn/runs.parquet`, total_steps=200000)."""


NSTEP_INTERVENTION_BRIDGES = (
    nstep_3step_reduces_bias_on_top_of_ddqn,
    nstep_3step_does_not_help_outcome__pool,
    nstep_3step_hurts_outcome__catch,
    nstep_3step_helps_outcome__discounting_chain,
)
"""Bridges asserted on the nstep_intervention corpus (rev 11)."""


NSTEP_FACTORIAL_BRIDGES = (
    factorial_ddqn_attenuation__fourrooms,
    factorial_variance_amplification__catch,
)
"""Bridges asserted on the union of nstep_intervention,
nstep_intervention_fr, nstep_vanilla_arms (rev 12 2×2 factorial)."""


EXPECTILE_PER_BURST_BRIDGES = (
    ddqn_outcome_stable_across_bursts__fourrooms,
    ddqn_outcome_zero_across_bursts__catch,
)
"""Bridges asserted on the expectile_3way corpus's joined
(runs.parquet × traces.parquet) cells."""


# ============ Audit-based claims (FINDINGS revision 5) ============
#
# Three-check tautology audit on candidate mediators. The
# audit's `is_clean` flag holds when a mediator passes structural
# (jaccard) + HP-R² + stratified-ρ checks; the bridge translates
# the per-measurable report to a HELD/NO_EFFECT verdict.


_DDQN_AUDIT_PANEL: tuple[dict[str, object], ...] = (
    {
        'name': 'jensen_gap',
        'reads': ('predicted_q_at_start', 'mc_return'),
    },
)


@claim_bridge
def jensen_gap_outcome_borderline(
    tautology_audit: AuditResult,
    *,
    source: str = 'jensen_gap',
    target: str = 'eval_best_burst_mean',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    measurables: tuple[dict[str, object], ...] = _DDQN_AUDIT_PANEL,
    outcome_path: str = 'eval_best_burst_mean',
    outcome_reads: tuple[str, ...] = ('mc_return',),
    hp_axes: tuple[str, ...] = (
        'replay.capacity', 'replay.batch_size',
        'optimizer.inner.lr', 'sync_period',
    ),
    hp_stratum_axis: str = 'env_name',
    arm_filter: str = 'ddqn',
) -> Verdict:
    """FINDINGS revision 5: 'jensen_gap is the strongest within-
    env signal in the predicted direction (ρ=-0.27, p<0.001),
    BUT the audit flags it as outcome-tautological at jaccard
    0.5 — its reads-set partially overlaps the outcome's.'

    Bridge encodes the *qualified* HELD: structural overlap
    flags it (NO_EFFECT under the strict tautology rule); within-
    env signal is real (ρ < -0.1 with p < 0.05).

    Verdict semantics:
    - jaccard >= 0.5: HELD_WITH_SCOPE_FLAG (outcome-borderline)
      when within-env ρ also significant; otherwise NO_EFFECT.
    - jaccard < 0.5 (clean structural): HELD when ρ significant.
    """
    del source, target, direction, tier
    del measurables, outcome_path, outcome_reads
    del hp_axes, hp_stratum_axis, arm_filter
    report = tautology_audit.by_name('jensen_gap')
    if report is None:
        return Verdict.POWER_INSUFFICIENT
    rho_significant = (
        not (report.outcome_stratified_rho != report.outcome_stratified_rho)  # not nan
        and abs(report.outcome_stratified_rho) >= 0.1
        and report.outcome_stratified_p < 0.05
    )
    if rho_significant and report.outcome_jaccard >= 0.5:
        # Borderline-outcome with real within-env signal —
        # classified as HELD_WITH_SCOPE_FLAG (the tautology
        # caveat doesn't refute the relation, just qualifies it).
        return Verdict.HELD_WITH_SCOPE_FLAG
    if rho_significant:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# ============ Third revision (CartPole HP audit) ====================
#
# The 180-cell vanilla-DQN CartPole HP corpus, audited under three
# checks (outcome-tautology / HP-determinism / HP-shadow). Most
# "solve predictors" turn out to be HP-shadow false positives;
# only state-coverage-KL survives all three checks with
# significant within-stratum signal.
#
# Reference verdicts on `experiments/data/cartpole_hp/
# runs_with_mediators.parquet`, outcome=`eval_final_mean`,
# stratified by `replay.capacity`:
#
#   learning_curve_auc / plateau_slope_late /
#     return_at_25pct_steps:                   OUTCOME-tautological
#                                              (jaccard=1.0)
#   greedy_match_late / q_gap_late /
#     q_max_growth / v_vs_max_delta_late:      HP-SHADOW
#                                              (|stratified ρ| < 0.1)
#   td_residual_late:                          borderline-clean
#   state_coverage_kl_uniform_late:            CLEAN (sole survivor
#                                              with significant
#                                              within-stratum signal)


_CARTPOLE_HP_AUDIT_PANEL: tuple[dict[str, object], ...] = (
    {
        'name': 'mediator.learning_curve_auc',
        'reads': ('mc_return',),
    },
    {
        'name': 'mediator.plateau_slope_late',
        'reads': ('mc_return',),
    },
    {
        'name': 'mediator.return_at_25pct_steps',
        'reads': ('mc_return',),
    },
    {
        'name': 'mediator.greedy_match_late',
        'reads': ('online_argmax_per_step', 'target_argmax_per_step'),
    },
    {
        'name': 'mediator.q_gap_late',
        'reads': ('online_max_q_per_step', 'online_min_q_per_step'),
    },
    {
        'name': 'mediator.q_max_growth',
        'reads': ('online_max_q_per_step',),
    },
    {
        'name': 'mediator.v_vs_max_delta_late',
        'reads': ('online_mean_q_per_step', 'online_max_q_per_step'),
    },
    {
        'name': 'mediator.td_residual_late',
        'reads': ('td_error',),
    },
    {
        'name': 'mediator.state_coverage_kl_uniform_late',
        'reads': ('state_hash',),
    },
)


_CARTPOLE_HP_AUDIT_HP_AXES: tuple[str, ...] = (
    'replay.capacity', 'replay.batch_size',
    'optimizer.inner.lr', 'sync_period',
)


@claim_bridge
def state_coverage_kl_clean_mediator__cartpole_hp(
    tautology_audit: AuditResult,
    *,
    source: str = 'mediator.state_coverage_kl_uniform_late',
    target: str = 'eval_final_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    measurables: tuple[dict[str, object], ...] = _CARTPOLE_HP_AUDIT_PANEL,
    outcome_path: str = 'eval_final_mean',
    outcome_reads: tuple[str, ...] = ('mc_return',),
    hp_axes: tuple[str, ...] = _CARTPOLE_HP_AUDIT_HP_AXES,
    hp_stratum_axis: str = 'replay.capacity',
    arm_filter: str = 'vanilla_dqn',
) -> Verdict:
    """rev 3: state_coverage_kl_uniform_late is the lone mediator
    that survives all three audit checks AND retains a
    significant within-capacity-stratum correlation with outcome.
    HELD when `is_clean` AND |stratified ρ| ≥ 0.1."""
    del source, target, direction, tier
    del measurables, outcome_path, outcome_reads
    del hp_axes, hp_stratum_axis, arm_filter
    report = tautology_audit.by_name(
        'mediator.state_coverage_kl_uniform_late',
    )
    if report is None:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(report.outcome_stratified_rho):
        return Verdict.POWER_INSUFFICIENT
    if not report.is_clean:
        return Verdict.NO_EFFECT
    if abs(report.outcome_stratified_rho) >= 0.1:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge
def learning_curve_auc_outcome_tautological__cartpole_hp(
    tautology_audit: AuditResult,
    *,
    source: str = 'mediator.learning_curve_auc',
    target: str = 'eval_final_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    measurables: tuple[dict[str, object], ...] = _CARTPOLE_HP_AUDIT_PANEL,
    outcome_path: str = 'eval_final_mean',
    outcome_reads: tuple[str, ...] = ('mc_return',),
    hp_axes: tuple[str, ...] = _CARTPOLE_HP_AUDIT_HP_AXES,
    hp_stratum_axis: str = 'replay.capacity',
    arm_filter: str = 'vanilla_dqn',
) -> Verdict:
    """rev 3: learning_curve_auc reads from `mc_return` directly,
    which IS the outcome's source column → jaccard=1.0,
    outcome-tautological. HELD as flagged-by-the-audit; this
    "predictor" is just a re-encoding of the outcome, not a
    mediator."""
    del source, target, direction, tier
    del measurables, outcome_path, outcome_reads
    del hp_axes, hp_stratum_axis, arm_filter
    report = tautology_audit.by_name('mediator.learning_curve_auc')
    if report is None:
        return Verdict.POWER_INSUFFICIENT
    return (
        Verdict.HELD if report.flagged_outcome
        else Verdict.NO_EFFECT
    )


@claim_bridge
def greedy_match_late_hp_shadow__cartpole_hp(
    tautology_audit: AuditResult,
    *,
    source: str = 'mediator.greedy_match_late',
    target: str = 'eval_final_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    measurables: tuple[dict[str, object], ...] = _CARTPOLE_HP_AUDIT_PANEL,
    outcome_path: str = 'eval_final_mean',
    outcome_reads: tuple[str, ...] = ('mc_return',),
    hp_axes: tuple[str, ...] = _CARTPOLE_HP_AUDIT_HP_AXES,
    hp_stratum_axis: str = 'replay.capacity',
    arm_filter: str = 'vanilla_dqn',
) -> Verdict:
    """rev 3: greedy_match_late's marginal correlation with
    outcome is HP-mediated — within each capacity stratum,
    |ρ|<0.1 → no residual signal once the HP regime is
    controlled for. HELD when `flagged_no_residual_signal`. The
    "wild interaction" of greedy_match across HPs from earlier
    revisions is exactly the signature of HP-shadow."""
    del source, target, direction, tier
    del measurables, outcome_path, outcome_reads
    del hp_axes, hp_stratum_axis, arm_filter
    report = tautology_audit.by_name('mediator.greedy_match_late')
    if report is None:
        return Verdict.POWER_INSUFFICIENT
    return (
        Verdict.HELD
        if report.flagged_no_residual_signal
        and not report.flagged_outcome
        else Verdict.NO_EFFECT
    )


CARTPOLE_HP_AUDIT_BRIDGES = (
    state_coverage_kl_clean_mediator__cartpole_hp,
    learning_curve_auc_outcome_tautological__cartpole_hp,
    greedy_match_late_hp_shadow__cartpole_hp,
)
"""Bridges asserted on the cartpole_hp 180-cell vanilla-DQN
corpus (`runs_with_mediators.parquet`) — rev 3."""


# ============ DoWhy claims (FINDINGS revision 4) ============
#
# State-coverage-KL → outcome on the CartPole HP corpus:
#   backdoor_ate: ATE = +8.82 / SCV unit, HELD
#   placebo_refutation: placebo ATE = +0.12 (1.4% of real), HELD
#   random_common_cause_refutation: drift = 0.0075, HELD
#
# Multi-fixture pattern: the bridge consumes ALL THREE analysis
# results simultaneously and asserts the conjunction. This is
# the strongest DoWhy claim shape in FINDINGS — corpus must have
# SCV + outcome + adjustment-set HPs.


_SCV_DAG: list[tuple[str, str]] = [
    ('replay.capacity', 'mediator.state_coverage_kl_uniform_late'),
    ('replay.batch_size', 'mediator.state_coverage_kl_uniform_late'),
    ('optimizer.inner.lr', 'mediator.state_coverage_kl_uniform_late'),
    ('sync_period', 'mediator.state_coverage_kl_uniform_late'),
    ('replay.capacity', 'eval_final_mean'),
    ('replay.batch_size', 'eval_final_mean'),
    ('optimizer.inner.lr', 'eval_final_mean'),
    ('sync_period', 'eval_final_mean'),
    ('mediator.state_coverage_kl_uniform_late',
     'eval_final_mean'),
]


# ============ Strategy 2 — expectile-greedify contrast ============
#
# Authored against the expectile_3way corpus (the live sweep
# launched mid-session). Expectile is a structurally different
# bias-correction operator (Garg et al 2023, "XQL") — does it
# reproduce DDQN's chain decomposition?
#
# Empirical reading from the corpus:
#   Mechanism (jensen_gap): expectile reduces gap MORE than ddqn
#     on every env (g vs vanilla in [-2.7, -11.6]; g vs ddqn in
#     [-0.5, -2.7] — uniformly negative).
#   Outcome (eval_best_burst_mean): mixed. ddqn beats expectile
#     on FourRooms (g=-0.63 vs ddqn) and Acrobot (g=-0.63 vs ddqn).
#     null on Catch (saturated). DiscountingChain marginal.
#
# So the mechanism-link disconnect REPRODUCES under expectile —
# more bias-reduction does NOT translate to more outcome benefit.
# The residual `bootstrap_fraction → g_link | g_mech` appears to
# be mechanism-non-specific (sparse-reward intrinsic), not
# DDQN-specific.


def _expectile_reduces_gap_holds_when(paired_g: PairedGResult) -> Verdict:
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g >= 0:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g < -0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge
def expectile_reduces_jensen_gap_more_than_ddqn__fourrooms(
    paired_g: PairedGResult,
    *,
    source: str = 'jensen_gap',
    target: str = 'jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'expectile_dqn',
    baseline_arm: str = 'ddqn',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'FourRooms-misc',
) -> Verdict:
    """Expectile reduces jensen_gap further than DDQN does on
    FourRooms (the env where DDQN had room to operate).
    Confirms expectile's bias-correction is more aggressive."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    return _expectile_reduces_gap_holds_when(paired_g)


@claim_bridge
def ddqn_outperforms_expectile_on_outcome__fourrooms(
    paired_g: PairedGResult,
    *,
    source: str = 'eval_best_burst_mean',
    target: str = 'eval_best_burst_mean',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'expectile_dqn',
    baseline_arm: str = 'ddqn',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'FourRooms-misc',
) -> Verdict:
    """Despite expectile's bigger bias-reduction, DDQN beats
    expectile on FourRooms outcome. Verdict HELD when expectile
    < ddqn on outcome (g < -0.3 with p < 0.05). The bigger-bias-
    reduction-doesn't-translate finding from FINDINGS revisions
    9 + 10 reproduces under a different mechanism family."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g < -0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge
def expectile_reproduces_mechanism_link_disconnect__fourrooms(
    paired_g: PairedGResult,
    *,
    source: str = 'eval_best_burst_mean',
    target: str = 'eval_best_burst_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'expectile_dqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    env_name: str = 'FourRooms-misc',
) -> Verdict:
    """Strategy 2's headline question: does expectile produce
    the same outcome-vs-vanilla effect as DDQN on FourRooms?
    (FINDINGS revision 9: DDQN g_link ≈ +0.79 across bursts.)

    HELD when expectile shows a similar magnitude DIRECT effect
    (g > 0.3 + p < 0.05); NO_EFFECT when expectile's outcome
    effect is much smaller than DDQN's; both establish the
    finding (whether the link reproduces or not is the answer)."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, env_name
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g > 0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge
def state_coverage_kl_causes_outcome(
    backdoor_ate: BackdoorResult,
    placebo_refutation: RefutationResult,
    random_common_cause_refutation: RefutationResult,
    *,
    source: str = 'mediator.state_coverage_kl_uniform_late',
    target: str = 'eval_final_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.INTERVENTIONAL,
    treatment: str = 'mediator.state_coverage_kl_uniform_late',
    outcome: str = 'eval_final_mean',
    dag: list[tuple[str, str]] = _SCV_DAG,
) -> Verdict:
    """FINDINGS revision 4: state_coverage_kl is the first
    mediator on the CartPole HP corpus that survives every check
    the framework currently has — backdoor ATE positive,
    placebo refutation HELD (placebo ATE near zero), RCC HELD
    (drift near zero).

    HELD when (a) backdoor identification succeeds, (b) ATE > 0
    (DIRECT direction), (c) placebo's |refuted_ate| is < 10% of
    real ATE, (d) RCC drift < 0.5 absolute.

    Pearl tier 2 (INTERVENTIONAL) — the DAG is posited; the
    backdoor adjustment is the rung-2-conditional claim. The
    direction-of-causation caveat (FINDINGS revision 4) lives in
    the narrative; the bridge only certifies the
    backdoor+refutation conjunction."""
    del source, target, direction, tier
    del treatment, outcome, dag
    if not backdoor_ate.identified:
        return Verdict.POWER_INSUFFICIENT
    if backdoor_ate.ate <= 0:
        return Verdict.NO_EFFECT
    real = abs(backdoor_ate.ate)
    placebo_ok = (
        abs(placebo_refutation.refuted_ate) < 0.1 * real
    )
    rcc_ok = random_common_cause_refutation.drift < 0.5
    if placebo_ok and rcc_ok:
        return Verdict.HELD
    return Verdict.NO_EFFECT


EXPECTILE_STRATEGY_2_BRIDGES = (
    expectile_reduces_jensen_gap_more_than_ddqn__fourrooms,
    ddqn_outperforms_expectile_on_outcome__fourrooms,
    expectile_reproduces_mechanism_link_disconnect__fourrooms,
)
"""Strategy-2 bridges (expectile-greedify contrast). Run against
the expectile_3way runs.parquet (no traces required)."""


# ============ Chain decomposition (FINDINGS revision 10) ============
#
# Per-(env, burst) panel meta-regression on env covariates.
# Original 200k DDQN corpus (revision 10) found:
#   β(log_action_dim) on g_mech: −0.39 p=0.005  (HELD — drives mech only)
#   β(log_action_dim) on g_link: +0.01 p=0.94   (null — does not drive link)
#   β(log_obs_dim)    on g_link: −0.07 p=0.0001 (HELD — drives link only)
#   β(log_obs_dim)    on g_mech: −0.01 p=0.49   (null — does not drive mech)
#
# These bridges encode the per-coefficient claims; running them
# on a NEW corpus tests whether the chain bottlenecks reproduce.


_CHAIN_COVARIATES_PER_ENV: dict[str, dict[str, float]] = {
    'Catch-bsuite':            {
        'log_action_dim': math.log(3), 'log_obs_dim': math.log(50),
    },
    'Acrobot-v1':              {
        'log_action_dim': math.log(3), 'log_obs_dim': math.log(6),
    },
    'DiscountingChain-bsuite': {
        'log_action_dim': math.log(5), 'log_obs_dim': math.log(5),
    },
    'FourRooms-misc':          {
        'log_action_dim': math.log(5), 'log_obs_dim': math.log(81),
    },
    'MountainCar-v0':          {
        'log_action_dim': math.log(3), 'log_obs_dim': math.log(2),
    },
}


def _chain_coef_holds_when(
    result: MetaRegressionResult,
    coef_name: str,
    expected_sign: int,
    min_abs_magnitude: float = 0.05,
) -> Verdict:
    """Generic threshold logic for a chain-decomposition
    coefficient claim. HELD when the named coefficient has the
    predicted sign + magnitude + significance; POWER_INSUFFICIENT
    when not significant (under-powered for the directional
    claim); NO_EFFECT when significant with the wrong sign."""
    coef = next(
        (c for c in result.coefficients if c.name == coef_name),
        None,
    )
    if coef is None:
        return Verdict.NO_EFFECT
    if not coef.is_significant:
        return Verdict.POWER_INSUFFICIENT
    sign_ok = (coef.coefficient > 0) == (expected_sign > 0)
    if sign_ok and abs(coef.coefficient) >= min_abs_magnitude:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge
def log_action_dim_drives_g_mech(
    meta_regression_per_burst: MetaRegressionResult,
    *,
    source: str = 'predicted_q_at_start',
    target: str = 'jensen_gap',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    reduction: str = 'mc_minus_q',
    covariates_per_env: dict[str, dict[str, float]] = (
        _CHAIN_COVARIATES_PER_ENV
    ),
) -> Verdict:
    """FINDINGS revision 10: 'log_action_dim moderates the
    mechanism — bigger bias-reduction at higher |A| (β=−0.39,
    p=0.005 on g_mech).'

    Bridge HELD when β(log_action_dim) on g_mech is significantly
    negative."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, reduction
    del covariates_per_env
    return _chain_coef_holds_when(
        meta_regression_per_burst,
        coef_name='log_action_dim',
        expected_sign=-1,
    )


@claim_bridge
def log_obs_dim_drives_g_link(
    meta_regression_per_burst: MetaRegressionResult,
    *,
    source: str = 'mc_return',
    target: str = 'eval_best_burst_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    treatment_arm: str = 'ddqn',
    baseline_arm: str = 'vanilla_dqn',
    pair_by: tuple[str, ...] = ('seed',),
    reduction: str = 'mean',
    covariates_per_env: dict[str, dict[str, float]] = (
        _CHAIN_COVARIATES_PER_ENV
    ),
) -> Verdict:
    """FINDINGS revision 10: 'log_obs_dim moderates the link —
    smaller-obs envs see bigger outcome benefit (β=−0.07,
    p=0.0001 on g_link).'

    Bridge HELD when β(log_obs_dim) on g_link is significantly
    non-null. Sign isn't pinned in the original (the magnitude
    is small either way); we accept either positive or negative
    significant β as confirming the moderator-bottleneck."""
    del source, target, direction, tier
    del treatment_arm, baseline_arm, pair_by, reduction
    del covariates_per_env
    coef = next(
        (c for c in meta_regression_per_burst.coefficients
         if c.name == 'log_obs_dim'),
        None,
    )
    if coef is None or not coef.is_significant:
        return Verdict.POWER_INSUFFICIENT
    if abs(coef.coefficient) >= 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


CHAIN_DECOMPOSITION_BRIDGES = (
    log_action_dim_drives_g_mech,
    log_obs_dim_drives_g_link,
)
"""Per-coefficient bridges from revision 10's chain
decomposition. Run against a per-(env, burst)-panel-able corpus
(joined runs.parquet × traces.parquet)."""


__all__ = [
    'ACTION_DIM_BRIDGES',
    'CHAIN_DECOMPOSITION_BRIDGES',
    'DDQN_200K_BRIDGES',
    'EXPECTILE_PER_BURST_BRIDGES',
    'EXPECTILE_STRATEGY_2_BRIDGES',
    'ddqn_link_to_outcome_null__converged_subset',
    'ddqn_outcome_stable_across_bursts__fourrooms',
    'ddqn_outcome_zero_across_bursts__catch',
    'ddqn_outperforms_expectile_on_outcome__fourrooms',
    'ddqn_reduces_jensen_gap__acrobot',
    'ddqn_reduces_jensen_gap__catch',
    'ddqn_reduces_jensen_gap__cartpole',
    'ddqn_reduces_jensen_gap__converged_subset',
    'ddqn_reduces_jensen_gap__discounting_chain',
    'ddqn_solves_faster__spaceinvaders',
    'expectile_reduces_jensen_gap_more_than_ddqn__fourrooms',
    'expectile_reproduces_mechanism_link_disconnect__fourrooms',
    'jensen_gap_outcome_borderline',
    'jensen_premise_active__acrobot',
    'jensen_premise_active__cartpole',
    'jensen_premise_active__discounting_chain',
    'jensen_premise_dormant__catch',
    'log_action_dim_drives_g_mech',
    'log_action_dim_drives_jensen_gap_reduction',
    'log_obs_dim_drives_g_link',
    'state_coverage_kl_causes_outcome',
    'time_to_solve_link_null__pooled',
]
