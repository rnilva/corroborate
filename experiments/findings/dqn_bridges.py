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
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Literal

import polars as pl

# Importing the analyses package populates the registry so
# resolution by parameter name succeeds.
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

# Substrate measurables (jensen_gap, jensen_dormancy_*, eval_*,
# etc.) are registered by importing the rl.dqn module — without
# this, the runner's measurable-signature manifest is empty
# because `get_registered(name)` returns None for every required
# name.
import corroborate_rl.dqn.measurables  # noqa: F401  # pyright: ignore[reportUnusedImport]

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
from corroborate_rl.env_catalogue import SOLVE_THRESHOLDS
from corroborate.analyses.tautology_audit import AuditResult
from corroborate.analyses.verdict_distribution import (
    VerdictDistributionResult,
)
from corroborate.analyses.panel import per_stratum_panel
from corroborate.bridge.analysis import analysis
from corroborate.bridge.bridge import (
    Direction, Tier, claim_bridge,
)
from corroborate.bridge.predicates import (
    finite_lt, partition_aggregate,
)
from corroborate.core.intervention import ArmRole, DoEffect, Intervention
from corroborate.corpus.schema import StratumG
from corroborate.measurables import Measurable
from corroborate.stats import MetaRegressionResult
from corroborate.stats.meta_regression import Pool, meta_regress_panel
from corroborate.bridge.verdict import Verdict
from corroborate_rl.dqn.claims.bootstrap import (
    bootstrap, double_greedify, expectile_greedify,
)
from corroborate_rl.dqn.measurables import (
    jensen_bias_per_burst_mean,
    mc_return_per_burst_mean,
)

import numpy as np
import numpy.typing as npt
from collections.abc import Mapping as _Mapping


# Typed per-burst reductions consumed by the bridges below that
# pass `source` to `paired_g_per_burst` / `meta_regression_per_burst`.
# `_MC_RETURN_PER_BURST_MEAN` is the link-side projection (per-eps
# mean of the actual sampled return); `_JENSEN_BIAS_PER_BURST_MEAN`
# is the mech-side (per-eps mean of `mc_return − predicted_q_at_start`,
# the structural Jensen-bias signal). Canonical instances live in
# the substrate (`corroborate_rl.dqn.measurables`); local aliases
# preserve the existing call-site names.
_MC_RETURN_PER_BURST_MEAN = mc_return_per_burst_mean
_JENSEN_BIAS_PER_BURST_MEAN = jensen_bias_per_burst_mean


# Typed structural deltas used across bridges in this zoo. Each
# `Intervention` is a single-slot replacement on the claim graph;
# `DoEffect` composes them into treatment / baseline arms.
DDQN_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(bootstrap, greedification=double_greedify),
)
EXPECTILE_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(
        bootstrap,
        greedification=partial(expectile_greedify, tau=0.7),
    ),
)


# File-level intervention: most bridges in this zoo test the
# DDQN-vs-vanilla contrast (`do(bootstrap = ddqn-style) →
# effect`). Bridges that test a different mechanism contrast
# (expectile-vs-DDQN, expectile-vs-vanilla) override via the
# per-decorator `source = DoEffect(...)` kwarg.
INTERVENTION = DoEffect(treatment=(DDQN_SWAP,), baseline=())


# ============ Canonical training-regime envelopes ============
#
# Substrate corpora explore different HP regimes for different
# scientific questions; a bridge scoped only by env pools cells
# from EVERY corpus that has that env — including non-canonical-
# HP cells the bridge author didn't intend. The principled
# scope-axis is endogenous (cf. ANALYSIS_RECIPE.md §0); until
# those are authored, these HP envelopes are the minimal-disambiguating
# subset that recovers the bridge's intended pool.
#
# Empirically minimal:
#   - `_FOURROOMS_REGIME`: just `lr==1e-4`. The contaminating
#     corpora at FourRooms (`ddqn`, `cartpole_hp`-style) all use
#     lr=1e-3; this single filter excludes them.
#   - `_ACTION_DIM_SWEEP_REGIME`: `lr==1e-3 & capacity==50000`.
#     `lr==1e-3` excludes lr=1e-4 corpora; `capacity==50000`
#     excludes the `ddqn`-200k corpus (cap=10000) which shares
#     lr=1e-3 but reaches Q-explosion regime on Acrobot
#     (mean_jensen=1232).

_FOURROOMS_REGIME: pl.Expr = pl.col('optimizer.inner.lr') == 0.0001
# ^ Transient debt: lr is HP setup with no clean endogenous
# correlate in the cache (would need `gradient_norm_late` or
# `td_error_late` measurables to migrate). Until then, scoping on
# lr is the most honest pin to the FourRooms HPO regime.

# Endogenous reading of the original capacity HP filter: the
# documented intent was "exclude cells in the Q-explosion regime."
# `partition_aggregate` computes a NaN-safe per-stratum mean of
# `q_divergence_score`; `< 5` selects strata whose Q stays
# bounded relative to the Bellman fixed-point bound. The lr part
# remains an HP filter (no endogenous correlate yet — see above).
#
# IMPORTANT: lr is included in the partition keys (not just env +
# capacity). Polars's `.over(key)` aggregates over the input
# dataframe BEFORE filter predicates in the same expression
# resolve, so a per-`(env, cap)` partition would mix lr values
# (e.g., CartPole cap=50k mean across all lr = 20.6, but at
# lr=0.001 alone = 1.25). Including lr in the partition keeps
# the aggregate cohort-specific.
#
# Empirical (lr=0.001, env, capacity) mean q_div within partition:
#   Acrobot-v1 cap=10k: 6.24 (excluded), cap=50k: 0.06 (included)
#   CartPole-v1 cap=10k: 291.78 (excluded), cap=50k: 1.25 (included)
#   Catch-bsuite cap={10k, 50k}: 0 (both included)
#   DiscountingChain-bsuite cap={10k, 50k}: ~0 (both included)
# vs prior `capacity == 50000`:
#   Acrobot, CartPole identical cell sets; verdicts identical.
#   Catch +120 cells (cap=10k now included), g shifts -5.00 → -4.58
#     (both HELD, threshold |g| < null_band).
#   DC +120 cells, g shifts -0.60 → -0.91 (both HELD).
_ACTION_DIM_SWEEP_REGIME: pl.Expr = (
    (pl.col('optimizer.inner.lr') == 0.001)
    & finite_lt(
        partition_aggregate(
            'q_divergence_score',
            by=[
                'optimizer.inner.lr',
                'env_name',
                'replay.capacity',
            ],
            op='mean',
        ),
        5.0,
    )
)


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


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_ACTION_DIM_SWEEP_REGIME & (pl.col('env_name') == 'Acrobot-v1'),
)
def ddqn_reduces_jensen_gap__acrobot(
    paired_g: PairedGResult,
) -> Verdict:
    return _ddqn_reduces_gap_holds_when(paired_g)


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_ACTION_DIM_SWEEP_REGIME & (pl.col('env_name') == 'Catch-bsuite'),
)
def ddqn_reduces_jensen_gap__catch(
    paired_g: PairedGResult,
) -> Verdict:
    return _ddqn_reduces_gap_holds_when(paired_g)


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_ACTION_DIM_SWEEP_REGIME & (pl.col('env_name') == 'DiscountingChain-bsuite'),
)
def ddqn_reduces_jensen_gap__discounting_chain(
    paired_g: PairedGResult,
) -> Verdict:
    return _ddqn_reduces_gap_holds_when(paired_g)


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_ACTION_DIM_SWEEP_REGIME & (pl.col('env_name') == 'CartPole-v1'),
)
def ddqn_reduces_jensen_gap__cartpole(
    paired_g: PairedGResult,
) -> Verdict:
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


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
)
def log_action_dim_drives_jensen_gap_reduction(
    meta_regression_paired_g: MetaRegressionResult,
    *,
    covariates_per_env: dict[str, dict[str, float]] = (
        _LOG_ACTION_DIM_PER_ENV
    ),
) -> Verdict:
    del covariates_per_env
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
    'jensen_dormancy_premise_active'
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


@claim_bridge(
    source='jensen_dormancy_premise_active',
    target='jensen_dormancy_premise_active',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(pl.col('env_name') == 'Acrobot-v1'),
)
def jensen_premise_active__acrobot(
    verdict_distribution_per_env: VerdictDistributionResult,
    *,
    arm_filter: ArmRole = ArmRole.TREATMENT,
    verdict_column: str = _DORMANCY_VERDICT_COLUMN,
) -> Verdict:
    del arm_filter, verdict_column
    return _premise_holds_when(
        verdict_distribution_per_env, 'Acrobot-v1', expected='held',
    )


@claim_bridge(
    source='jensen_dormancy_premise_active',
    target='jensen_dormancy_premise_active',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(pl.col('env_name') == 'CartPole-v1'),
)
def jensen_premise_active__cartpole(
    verdict_distribution_per_env: VerdictDistributionResult,
    *,
    arm_filter: ArmRole = ArmRole.TREATMENT,
    verdict_column: str = _DORMANCY_VERDICT_COLUMN,
) -> Verdict:
    del arm_filter, verdict_column
    return _premise_holds_when(
        verdict_distribution_per_env, 'CartPole-v1', expected='held',
    )


@claim_bridge(
    source='jensen_dormancy_premise_active',
    target='jensen_dormancy_premise_active',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(pl.col('env_name') == 'Catch-bsuite'),
)
def jensen_premise_dormant__catch(
    verdict_distribution_per_env: VerdictDistributionResult,
    *,
    arm_filter: ArmRole = ArmRole.TREATMENT,
    verdict_column: str = _DORMANCY_VERDICT_COLUMN,
) -> Verdict:
    """Catch is the structural counterexample: |A|=3 but σ_Q is
    tiny (0.07) and observed bias even tinier (0.03), so the
    Jensen floor (0.10) exceeds observed → premise dormant. The
    invariant correctly fires on this env; the bridge HELD when
    ≥90% of cells return INVARIANT_VIOLATION."""
    del arm_filter, verdict_column
    return _premise_holds_when(
        verdict_distribution_per_env, 'Catch-bsuite',
        expected='invariant_violation',
    )


@claim_bridge(
    source='jensen_dormancy_premise_active',
    target='jensen_dormancy_premise_active',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(pl.col('env_name') == 'DiscountingChain-bsuite'),
)
def jensen_premise_active__discounting_chain(
    verdict_distribution_per_env: VerdictDistributionResult,
    *,
    arm_filter: ArmRole = ArmRole.TREATMENT,
    verdict_column: str = _DORMANCY_VERDICT_COLUMN,
) -> Verdict:
    del arm_filter, verdict_column
    return _premise_holds_when(
        verdict_distribution_per_env, 'DiscountingChain-bsuite',
        expected='held',
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


@claim_bridge(
    source=INTERVENTION,
    target='mc_return',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    # Pin n_step to the default DDQN-1step regime — n=3/5/10 cells
    # from `nstep_*` corpora at FourRooms otherwise pool into the
    # same (env, seed) bucket as different intervention regimes.
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('n_step').is_null() | (pl.col('n_step') == 1))
    ),
)
def ddqn_outcome_stable_across_bursts__fourrooms(
    paired_g_per_burst: PerBurstResult,
    *,
    source: Measurable[
        _Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
) -> Verdict:
    """DDQN's outcome benefit on FourRooms is stable across every
    eval burst. HELD when (a) at least 9/10 bursts have positive
    g and (b) the per-burst mean g exceeds 0.3."""
    del source  # forwarded to paired_g_per_burst
    panel = panel_for_env(paired_g_per_burst, 'FourRooms-misc')
    if not panel:
        return Verdict.POWER_INSUFFICIENT
    positive = sum(1 for s in panel if s.g > 0)
    mean_g = sum(s.g for s in panel) / len(panel)
    if positive >= len(panel) - 1 and mean_g > 0.3:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source=INTERVENTION,
    target='mc_return',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(pl.col('env_name') == 'Catch-bsuite'),
)
def ddqn_outcome_zero_across_bursts__catch(
    paired_g_per_burst: PerBurstResult,
    *,
    source: Measurable[
        _Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
) -> Verdict:
    """Catch-bsuite saturates near-optimal under both arms;
    DDQN at n=1 has zero per-burst effect. NO_EFFECT when
    every burst's |g| is below 0.1; HELD-shaped verdicts are
    impossible since the prediction is null."""
    del source  # forwarded to paired_g_per_burst
    panel = panel_for_env(paired_g_per_burst, 'Catch-bsuite')
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


def _pooled_null_prediction_holds_when(
    pooled: PooledPairedGResult,
    *,
    null_band: float,
    min_envs: int,
) -> Verdict:
    """For bridges with `predicted_direction='null'` (xfail-style):
    HELD when |pooled g| < null_band — the no-effect prediction
    is confirmed (small effect, link/outcome empirically null);
    NO_EFFECT when |pooled g| >= null_band — the no-effect
    prediction is REFUTED (an effect was observed when none was
    predicted; the unexpected-pass / xpass analog).

    Verdict semantics are uniform across all four
    `predicted_direction` values: HELD = prediction confirmed.
    Pair this body with `predicted_direction='null'` on the
    decorator so the (verdict, predicted_direction) tuple at the
    report layer reads unambiguously."""
    if pooled.n_envs < min_envs:
        return Verdict.POWER_INSUFFICIENT
    g = pooled.pooled.pooled_g
    if math.isnan(g):
        return Verdict.POWER_INSUFFICIENT
    if abs(g) < null_band:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=pl.col('env_name').is_in(list(_CONVERGED_ENVS_DDQN_200K)),
)
def ddqn_reduces_jensen_gap__converged_subset(
    paired_g_pooled: PooledPairedGResult,
    *,
    total_steps_filter: int = 200000,
) -> Verdict:
    """rev 1: pooled paired g(jensen_gap) on the converged subset
    is strongly negative (~-0.93), HELD."""
    del total_steps_filter
    return _pooled_negative_holds_when(
        paired_g_pooled, g_threshold=0.5, min_envs=5,
    )


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    predicted_direction='null',
    scope=pl.col('env_name').is_in(list(_CONVERGED_ENVS_DDQN_200K)),
)
def ddqn_link_to_outcome_null__converged_subset(
    paired_g_pooled: PooledPairedGResult,
    *,
    total_steps_filter: int = 200000,
) -> Verdict:
    """rev 1: pooled paired g(eval_best_burst_mean) on the
    converged subset is essentially zero (~-0.03). The
    literature-predicted outcome benefit fails — link broken on
    this corpus. Authored with `predicted_direction='null'`
    (xfail-style); HELD encodes "the null prediction was
    confirmed" — the link is empirically broken at the
    converged-subset scope.

    `source` is the measured column (the analysis's input); the
    conceptual edge is `jensen_gap → outcome.eval_best_
    burst_mean`, but the paired-g consumes only the target column
    + arm. The 'link broken' reading lives in the bridge's HELD
    verdict (under predicted_direction='null') combined with the
    mechanism HELD on `ddqn_reduces_jensen_gap__converged_subset`
    — same scope, same arm, mechanism activates but outcome
    doesn't move."""
    del total_steps_filter
    return _pooled_null_prediction_holds_when(
        paired_g_pooled, null_band=0.15, min_envs=3,
    )


# Note: paired_g_pooled is the analysis name. Both bridges
# consume the SAME analysis name but with different `source`
# values (jensen_gap vs eval_best_burst_mean) —
# the framework's resolver instantiates separately per bridge.


# ============ Eleventh revision: n-step attenuates DDQN ============
#
# Bias-compounding theory predicts: as `n_step` grows, the
# bootstrap-target's bias contribution shrinks (because the MC
# component dominates), so DDQN — which exists to cut the
# overestimation in the bootstrap target — has less to fix.
# Equivalently: DDQN-vs-vanilla's |g_jensen| should be *larger*
# at n=1 than at n=3.
#
# The original rev-11 bridges authored a within-DDQN HP-cleavage
# (`ddqn_3step` vs `ddqn_1step`); under the Phase-6 typed-
# Intervention contract, both arms collapse to the same canonical
# arm_key (both apply `DDQN_SWAP`) and the contrast is no longer
# expressible as a DoEffect. The principled cross-arm form is
# DDQN-vs-vanilla scoped to each `n_step` value — the four
# bridges below capture the same scientific story.
#
# Verdict convention (HELD = prediction confirmed):
#   - n=1 mech: HELD when DDQN reduces bias (strong negative g)
#   - n=3 mech: HELD when attenuated (|g| < null_band) — bias-
#     compounding theory predicts smallness, so HELD-as-null
#   - n=1 outcome: HELD when DDQN helps (positive g) — per
#     `findings_nstep_falsification.md`, Δ=+0.087, p=0.0003
#   - n=3 outcome: HELD when attenuated (|g| < null_band) — DDQN
#     advantage collapses to Δ=+0.002, ns
#
# Together they assert the slope: |g_jensen(n=1)| > |g_jensen(n=3)|
# AND DDQN's outcome advantage attenuates with n. Per memory
# `findings_nstep_falsification.md`, the corpus collapses
# monotonically Δ=+0.087 (n=1) → +0.002 (n=3) on FourRooms;
# the four bridges encode the endpoints of that slope.
#
# Target column: `eval_best_burst_mean` (Hasselt convention per
# CLAUDE.md and the file's other DDQN bridges; the original rev-11
# bridges used `eval_final_mean` but the per-burst best-mean is
# the published comparable). Corpus: `nstep_lambda_fourrooms`
# (single env: FourRooms-misc, 5 n_step values × 2 arms × 30 seeds).


def _attenuated_holds_when(
    paired_g: PairedGResult, *, null_band: float,
) -> Verdict:
    """HELD when |g| < null_band — the attenuation reading. The
    theorem predicts smallness; HELD encodes prediction confirmed.
    HELD-strong-positive or HELD-strong-negative would refute
    attenuation, but bridges that want to encode that should
    declare a separate DIRECT/INVERSE bridge with their own
    threshold. n_pairs < 30 → POWER_INSUFFICIENT."""
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.g):
        return Verdict.POWER_INSUFFICIENT
    if abs(paired_g.g) < null_band:
        return Verdict.HELD
    return Verdict.NO_EFFECT


def _ddqn_helps_outcome_holds_when(
    paired_g: PairedGResult, *, g_threshold: float,
) -> Verdict:
    """HELD when DDQN improves outcome (g > threshold AND p<0.05).
    Sign opposes prediction (negative g is sign-wrong) →
    POWER_INSUFFICIENT; n_pairs < 30 → POWER_INSUFFICIENT."""
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.g):
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g <= 0:
        return Verdict.POWER_INSUFFICIENT  # sign opposes prediction
    if paired_g.g > g_threshold and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _FOURROOMS_REGIME
        & (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('n_step') == 1)
    ),
)
def ddqn_reduces_jensen_gap__fourrooms_n1(
    paired_g: PairedGResult,
) -> Verdict:
    """At full bootstrap (n=1) on FourRooms, DDQN-vs-vanilla
    g(jensen_gap) is strongly negative — DDQN cuts the
    bootstrap-bias just as theory predicts. HELD when g < -0.3
    with p<0.05 (uses the shared per-env helper)."""
    return _ddqn_reduces_gap_holds_when(paired_g)


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    predicted_direction='null',
    scope=(
        _FOURROOMS_REGIME
        & (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('n_step') == 3)
    ),
)
def ddqn_attenuates_jensen_gap__fourrooms_n3(
    paired_g: PairedGResult,
) -> Verdict:
    """At n=3 on FourRooms, the MC component pre-empts most of
    the bootstrap-bias so DDQN has less to fix. The attenuation
    prediction: |g(jensen_gap)| should land in the null band.
    HELD when |g| < 0.3 (HELD-as-null convention; the theorem
    predicts smallness here)."""
    return _attenuated_holds_when(paired_g, null_band=0.3)


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _FOURROOMS_REGIME
        & (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('n_step') == 1)
    ),
)
def ddqn_helps_outcome__fourrooms_n1(
    paired_g: PairedGResult,
) -> Verdict:
    """At n=1 on FourRooms, DDQN improves outcome — per
    `findings_nstep_falsification.md`, Δ=+0.087, p=0.0003. HELD
    when g > 0.3 with p<0.05."""
    return _ddqn_helps_outcome_holds_when(paired_g, g_threshold=0.3)


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    predicted_direction='null',
    scope=(
        _FOURROOMS_REGIME
        & (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('n_step') == 3)
    ),
)
def ddqn_outcome_attenuates__fourrooms_n3(
    paired_g: PairedGResult,
) -> Verdict:
    """At n=3 on FourRooms, DDQN's outcome advantage collapses
    to Δ=+0.002 (ns) — variance-reduction theory's prediction
    that n-step rescues the link is refuted. HELD when
    |g(outcome)| < 0.3 (the attenuated reading)."""
    return _attenuated_holds_when(paired_g, null_band=0.3)


# ============ N-step slope: meta-regression over n_step ============
#
# Form (C) per ANALYSIS_RECIPE.md §2: the slope-form companion to
# the (n=1, n=3) endpoint bridges above. Stratifies cells by
# `n_step ∈ {1, 2, 3, 5, 10}`, computes per-stratum paired g of
# DDQN-vs-vanilla on the chosen target, then meta-regresses the
# stratum panel on `log(n_step)`.
#
# Tier.ASSOCIATIONAL is the honest acknowledgement that `n_step`
# is currently a hyperparameter scalar, not a typed Intervention.
# A negative `log_n_step` slope on `eval_best_burst_mean` means
# DDQN's outcome benefit attenuates as the bootstrap target shifts
# toward MC — the bias-compounding theory's slope-form prediction.

@analysis
def meta_regression_paired_g_by_nstep(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...],
    source: str,
    arm_field: str = 'arm_key',
    pool: Pool = 'random',
) -> MetaRegressionResult:
    """Per-`n_step` paired-g panel + meta-regression on
    `log(n_step)`. Substrate-specific stratifier (n_step is an
    RL-substrate hyperparameter); the framework's
    `meta_regression_paired_g` hardcodes `env_name`-stratification,
    which doesn't fit a single-env n-step sweep.

    For each `n_step` value present in `cells`, runs paired_g on
    the (treatment, baseline) cells; packs the per-stratum results
    into `StratumG[int]`; calls `meta_regress_panel` with
    `{n_step: {'log_n_step': log(n_step)}}` covariates. The slope
    on `log_n_step` is the moderator-direction estimate."""
    cells_list = list(cells)

    def _stratify(cell: Mapping[str, object]) -> int | None:
        n = cell.get('n_step')
        return n if isinstance(n, int) else None

    def _analyze(subset: Sequence[Mapping[str, object]]) -> PairedGResult:
        from corroborate.analyses.paired_g import paired_g
        return paired_g.fn(
            subset,
            treatment_arm=treatment_arm,
            baseline_arm=baseline_arm,
            source=source,
            pair_by=pair_by,
            arm_field=arm_field,
        )

    panel_raw = per_stratum_panel(
        cells_list, stratify_by=_stratify, analysis=_analyze,
        min_cells_per_stratum=2,
    )
    panel = tuple(
        StratumG[int](
            stratum_id=n,
            g=r.g,
            se=r.se,
            n_pairs=r.n_pairs,
        )
        for n, r in panel_raw
    )
    covariates: dict[int, Mapping[str, float]] = {
        n: {'log_n_step': math.log(n)} for n, _ in panel_raw
    }
    return meta_regress_panel(
        panel,
        covariates_per_stratum=covariates,
        pool=pool,
    )


def _slope_holds_when(
    meta: MetaRegressionResult, *,
    covariate: str = 'log_n_step',
    sign: Literal['negative', 'positive'] = 'negative',
    min_strata: int = 3,
) -> Verdict:
    """HELD when the meta-regression coefficient on `covariate`
    has the predicted sign AND its CI excludes zero. Underpowered
    panels (fewer strata than `min_strata`) return
    POWER_INSUFFICIENT."""
    if meta.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    coef = next(
        (c for c in meta.coefficients if c.name == covariate), None,
    )
    if coef is None:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(coef.coefficient):
        return Verdict.POWER_INSUFFICIENT
    if sign == 'negative':
        if coef.coefficient >= 0:
            return Verdict.POWER_INSUFFICIENT  # sign opposes prediction
        if coef.ci_hi < 0:  # CI strictly below zero
            return Verdict.HELD
    else:
        if coef.coefficient <= 0:
            return Verdict.POWER_INSUFFICIENT
        if coef.ci_lo > 0:
            return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_FOURROOMS_REGIME & (pl.col('env_name') == 'FourRooms-misc'),
    pair_by=('seed',),
)
def ddqn_outcome_slope_attenuates_with_log_nstep__fourrooms(
    meta_regression_paired_g_by_nstep: MetaRegressionResult,
) -> Verdict:
    """The slope form of the bias-compounding prediction on
    FourRooms. As `n_step` grows, the bootstrap-target shifts
    toward MC and DDQN has less bias to fix; the per-stratum g
    of DDQN-vs-vanilla on `eval_best_burst_mean` should attenuate
    monotonically. HELD when the `log_n_step` coefficient is
    significantly negative (CI strictly below zero) across at
    least 3 of the 5 strata (n ∈ {1, 2, 3, 5, 10})."""
    return _slope_holds_when(
        meta_regression_paired_g_by_nstep,
        covariate='log_n_step',
        sign='negative',
        min_strata=3,
    )


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_FOURROOMS_REGIME & (pl.col('env_name') == 'FourRooms-misc'),
    pair_by=('seed',),
)
def ddqn_jensen_slope_attenuates_with_log_nstep__fourrooms(
    meta_regression_paired_g_by_nstep: MetaRegressionResult,
) -> Verdict:
    """The slope form on the mechanism (jensen_gap). Theory:
    `|g_jensen|` should shrink as `n_step` grows because the
    bootstrap-bias compounds less under MC-leaning targets. HELD
    when the `log_n_step` slope is significantly positive
    (g(jensen) is negative; growing toward zero is a positive
    slope)."""
    return _slope_holds_when(
        meta_regression_paired_g_by_nstep,
        covariate='log_n_step',
        sign='positive',
        min_strata=3,
    )


@claim_bridge(
    source=INTERVENTION,
    target='eval_final_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_FOURROOMS_REGIME & (pl.col('env_name') == 'FourRooms-misc'),
    pair_by=('seed',),
)
def ddqn_final_outcome_slope_attenuates_with_log_nstep__fourrooms(
    meta_regression_paired_g_by_nstep: MetaRegressionResult,
) -> Verdict:
    """Sibling of the `eval_best_burst_mean` slope on the
    `eval_final_mean` (steady-state) target. Best-burst is a peak
    metric that compresses long-run differences once both arms hit
    their per-arm ceiling; final-mean is sensitive to long-run
    learning quality and exposes the bias-compounding attenuation
    more cleanly. On `nstep_lambda_fourrooms`: β≈−0.27 per
    log(n_step), CI strictly below zero, p≈0.025 → HELD. n=10
    even crosses (vanilla beats DDQN on final mean), consistent
    with bias-compounding theory's prediction that DDQN's edge
    inverts when bootstrap-bias is no longer the dominant
    failure mode.

    Documents the eval_best_burst_mean / eval_final_mean
    dissociation: peak = no slope-significance,
    steady-state = clear attenuation. Per memory
    `findings_l2_acrobot_goldilocks`, scalar best-burst can hide
    effects the time-resolved metrics expose."""
    return _slope_holds_when(
        meta_regression_paired_g_by_nstep,
        covariate='log_n_step',
        sign='negative',
        min_strata=3,
    )


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


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(pl.col('env_name') == 'FourRooms-misc'),
)
def factorial_ddqn_attenuation__fourrooms(
    factorial_2x2_interaction: Factorial2x2Result,
    *,
    arm_a: str = 'vanilla_1step',
    arm_b: str = 'vanilla_3step',
    arm_c: str = 'ddqn_1step',
    arm_d: str = 'ddqn_3step',
    env_filter: tuple[str, ...] = ('FourRooms-misc',),
    total_steps_filter: int = 200000,
) -> Verdict:
    """rev 12: on FourRooms, DDQN's marginal outcome benefit
    shrinks 4× when n_step rises from 1 to 3 (g(C-A)=+0.73 →
    g(D-B)=+0.09). Both arms benefit from n-step similarly
    ((B-A)=+0.74 vs (D-C)=+0.16), so the shrink isn't over-
    correction — it's attenuation: the bias-correction axes
    overlap. Interaction g=-0.71 with z=-3.49 → HELD."""
    del arm_a, arm_b, arm_c, arm_d
    del env_filter, total_steps_filter
    p = factorial_2x2_interaction.for_env('FourRooms-misc')
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


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(pl.col('env_name') == 'Catch-bsuite'),
)
def factorial_variance_amplification__catch(
    factorial_2x2_interaction: Factorial2x2Result,
    *,
    arm_a: str = 'vanilla_1step',
    arm_b: str = 'vanilla_3step',
    arm_c: str = 'ddqn_1step',
    arm_d: str = 'ddqn_3step',
    env_filter: tuple[str, ...] = ('Catch-bsuite',),
    total_steps_filter: int = 200000,
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
    del arm_a, arm_b, arm_c, arm_d
    del env_filter, total_steps_filter
    p = factorial_2x2_interaction.for_env('Catch-bsuite')
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


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_step',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    predicted_direction='null',
)
def time_to_solve_link_null__pooled(
    paired_g_among_solvers: PooledPairedGResult,
    *,
    gate_column: str = 'eval_best_burst_mean',
    gate_thresholds: dict[str, float] = _SOLVE_THRESHOLDS_FLAT,
    env_filter: tuple[str, ...] = _TIME_TO_SOLVE_HIGH_SOLVE_ENVS,
    total_steps_filter: int = 200000,
) -> Verdict:
    """rev 6: replacing the steady-state outcome with a sample-
    efficiency proxy doesn't rescue DDQN. Pooled across 5
    non-degenerate high-solve envs, predicted-direction effect
    averages zero with PI bracketing zero. Authored with
    `predicted_direction='null'`; HELD encodes "the null
    prediction was confirmed — sample-efficiency-as-outcome
    doesn't break the rev-1 link-null."""
    del gate_column, gate_thresholds, env_filter, total_steps_filter
    return _pooled_null_prediction_holds_when(
        paired_g_among_solvers, null_band=0.15, min_envs=4,
    )


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_step',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
)
def ddqn_solves_faster__spaceinvaders(
    paired_g_among_solvers: PooledPairedGResult,
    *,
    gate_column: str = 'eval_best_burst_mean',
    gate_thresholds: dict[str, float] = _SOLVE_THRESHOLDS_FLAT,
    env_filter: tuple[str, ...] = ('SpaceInvaders-MinAtar',),
    total_steps_filter: int = 200000,
) -> Verdict:
    """rev 6: SpaceInvaders-MinAtar is the one env where DDQN
    crosses the solve threshold reliably faster than vanilla
    (g=-0.532, n=30 — moderate effect, sign matches Hasselt's
    overestimation-bias-cost-on-sparse-reward prediction)."""
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
    ddqn_reduces_jensen_gap__fourrooms_n1,
    ddqn_attenuates_jensen_gap__fourrooms_n3,
    ddqn_helps_outcome__fourrooms_n1,
    ddqn_outcome_attenuates__fourrooms_n3,
    ddqn_outcome_slope_attenuates_with_log_nstep__fourrooms,
    ddqn_jensen_slope_attenuates_with_log_nstep__fourrooms,
    ddqn_final_outcome_slope_attenuates_with_log_nstep__fourrooms,
)
"""Bridges asserted on the `nstep_lambda_fourrooms` corpus
(FourRooms-misc, n_step ∈ {1, 2, 3, 5, 10} × {vanilla, ddqn} × 30
seeds). The (n=1, n=3) endpoint bridges + slope-form
meta-regression bridges together encode the bias-compounding
theory's attenuation prediction. The slope form (last two) is
the higher-power test using all 5 n_step strata."""


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


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
)
def jensen_gap_outcome_borderline(
    tautology_audit: AuditResult,
    *,
    measurables: tuple[dict[str, object], ...] = _DDQN_AUDIT_PANEL,
    outcome_path: str = 'eval_best_burst_mean',
    outcome_reads: tuple[str, ...] = ('mc_return',),
    hp_axes: tuple[str, ...] = (
        'replay.capacity', 'replay.batch_size',
        'optimizer.inner.lr', 'sync_period',
    ),
    hp_stratum_axis: str = 'env_name',
    arm_filter: ArmRole = ArmRole.TREATMENT,
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


@claim_bridge(
    source='mediator.state_coverage_kl_uniform_late',
    target='eval_final_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
)
def state_coverage_kl_clean_mediator__cartpole_hp(
    tautology_audit: AuditResult,
    *,
    measurables: tuple[dict[str, object], ...] = _CARTPOLE_HP_AUDIT_PANEL,
    outcome_path: str = 'eval_final_mean',
    outcome_reads: tuple[str, ...] = ('mc_return',),
    hp_axes: tuple[str, ...] = _CARTPOLE_HP_AUDIT_HP_AXES,
    hp_stratum_axis: str = 'replay.capacity',
    arm_filter: ArmRole = ArmRole.BASELINE,
) -> Verdict:
    """rev 3: state_coverage_kl_uniform_late is the lone mediator
    that survives all three audit checks AND retains a
    significant within-capacity-stratum correlation with outcome.
    HELD when `is_clean` AND |stratified ρ| ≥ 0.1."""
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


@claim_bridge(
    source='mediator.learning_curve_auc',
    target='eval_final_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
)
def learning_curve_auc_outcome_tautological__cartpole_hp(
    tautology_audit: AuditResult,
    *,
    measurables: tuple[dict[str, object], ...] = _CARTPOLE_HP_AUDIT_PANEL,
    outcome_path: str = 'eval_final_mean',
    outcome_reads: tuple[str, ...] = ('mc_return',),
    hp_axes: tuple[str, ...] = _CARTPOLE_HP_AUDIT_HP_AXES,
    hp_stratum_axis: str = 'replay.capacity',
    arm_filter: ArmRole = ArmRole.BASELINE,
) -> Verdict:
    """rev 3: learning_curve_auc reads from `mc_return` directly,
    which IS the outcome's source column → jaccard=1.0,
    outcome-tautological. HELD as flagged-by-the-audit; this
    "predictor" is just a re-encoding of the outcome, not a
    mediator."""
    del measurables, outcome_path, outcome_reads
    del hp_axes, hp_stratum_axis, arm_filter
    report = tautology_audit.by_name('mediator.learning_curve_auc')
    if report is None:
        return Verdict.POWER_INSUFFICIENT
    return (
        Verdict.HELD if report.flagged_outcome
        else Verdict.NO_EFFECT
    )


@claim_bridge(
    source='mediator.greedy_match_late',
    target='eval_final_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
)
def greedy_match_late_hp_shadow__cartpole_hp(
    tautology_audit: AuditResult,
    *,
    measurables: tuple[dict[str, object], ...] = _CARTPOLE_HP_AUDIT_PANEL,
    outcome_path: str = 'eval_final_mean',
    outcome_reads: tuple[str, ...] = ('mc_return',),
    hp_axes: tuple[str, ...] = _CARTPOLE_HP_AUDIT_HP_AXES,
    hp_stratum_axis: str = 'replay.capacity',
    arm_filter: ArmRole = ArmRole.BASELINE,
) -> Verdict:
    """rev 3: greedy_match_late's marginal correlation with
    outcome is HP-mediated — within each capacity stratum,
    |ρ|<0.1 → no residual signal once the HP regime is
    controlled for. HELD when `flagged_no_residual_signal`. The
    "wild interaction" of greedy_match across HPs from earlier
    revisions is exactly the signature of HP-shadow."""
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


@claim_bridge(
    source=DoEffect(treatment=(EXPECTILE_SWAP,), baseline=(DDQN_SWAP,)),
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(pl.col('env_name') == 'FourRooms-misc'),
)
def expectile_reduces_jensen_gap_more_than_ddqn__fourrooms(
    paired_g: PairedGResult,
) -> Verdict:
    """Expectile reduces jensen_gap further than DDQN does on
    FourRooms (the env where DDQN had room to operate).
    Confirms expectile's bias-correction is more aggressive."""
    return _expectile_reduces_gap_holds_when(paired_g)


@claim_bridge(
    source=DoEffect(treatment=(EXPECTILE_SWAP,), baseline=(DDQN_SWAP,)),
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(pl.col('env_name') == 'FourRooms-misc'),
)
def ddqn_outperforms_expectile_on_outcome__fourrooms(
    paired_g: PairedGResult,
) -> Verdict:
    """Despite expectile's bigger bias-reduction, DDQN beats
    expectile on FourRooms outcome. Verdict HELD when expectile
    < ddqn on outcome (g < -0.3 with p < 0.05). The bigger-bias-
    reduction-doesn't-translate finding from FINDINGS revisions
    9 + 10 reproduces under a different mechanism family."""
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g < -0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source=DoEffect(treatment=(EXPECTILE_SWAP,), baseline=()),
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(pl.col('env_name') == 'FourRooms-misc'),
)
def expectile_reproduces_mechanism_link_disconnect__fourrooms(
    paired_g: PairedGResult,
) -> Verdict:
    """Strategy 2's headline question: does expectile produce
    the same outcome-vs-vanilla effect as DDQN on FourRooms?
    (FINDINGS revision 9: DDQN g_link ≈ +0.79 across bursts.)

    HELD when expectile shows a similar magnitude DIRECT effect
    (g > 0.3 + p < 0.05); NO_EFFECT when expectile's outcome
    effect is much smaller than DDQN's; both establish the
    finding (whether the link reproduces or not is the answer)."""
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g > 0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source='mediator.state_coverage_kl_uniform_late',
    target='eval_final_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
)
def state_coverage_kl_causes_outcome(
    backdoor_ate: BackdoorResult,
    placebo_refutation: RefutationResult,
    random_common_cause_refutation: RefutationResult,
    *,
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


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    # FINDINGS rev-10 panel was the original 200k DDQN cohort.
    # Pin the HP fingerprint so cells from intervention sweeps
    # at different (sync, lr, capacity, n_step, gamma) regimes
    # don't pool into the same (env, seed) bucket.
    scope=(
        (pl.col('total_steps') == 200_000)
        & (pl.col('eval_every') == 20_000)
        & (pl.col('gamma') == 0.99)
        & (pl.col('sync_period') == 100)
        & (pl.col('replay.capacity') == 10_000)
        & (pl.col('replay.batch_size') == 32)
        & (pl.col('n_step').is_null() | (pl.col('n_step') == 1))
    ),
)
def log_action_dim_drives_g_mech(
    meta_regression_per_burst: MetaRegressionResult,
    *,
    source: Measurable[
        _Mapping[str, object], npt.NDArray[np.floating],
    ] = _JENSEN_BIAS_PER_BURST_MEAN,
    covariates_per_env: dict[str, dict[str, float]] = (
        _CHAIN_COVARIATES_PER_ENV
    ),
) -> Verdict:
    """FINDINGS revision 10: 'log_action_dim moderates the
    mechanism — bigger bias-reduction at higher |A| (β=−0.39,
    p=0.005 on g_mech).'

    Bridge HELD when β(log_action_dim) on g_mech is significantly
    negative."""
    del source, covariates_per_env  # source forwarded
    return _chain_coef_holds_when(
        meta_regression_per_burst,
        coef_name='log_action_dim',
        expected_sign=-1,
    )


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    # Same FINDINGS rev-10 cohort pinning as the g_mech sibling.
    scope=(
        (pl.col('n_step').is_null() | (pl.col('n_step') == 1))
        & (pl.col('replay.capacity') == 10_000)
    ),
)
def log_obs_dim_drives_g_link(
    meta_regression_per_burst: MetaRegressionResult,
    *,
    source: Measurable[
        _Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
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
    del source, covariates_per_env  # source forwarded
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


# Canonical name the runner imports — union of every bridge in
# this file. Per-corpus tuples (`ACTION_DIM_BRIDGES`,
# `DDQN_200K_BRIDGES`, etc.) stay for legacy `run_dqn_bridges.py`
# call sites that route specific corpora to specific subsets.
BRIDGES = (
    *ACTION_DIM_BRIDGES,
    *DDQN_200K_BRIDGES,
    *NSTEP_INTERVENTION_BRIDGES,
    *NSTEP_FACTORIAL_BRIDGES,
    *EXPECTILE_PER_BURST_BRIDGES,
    *CARTPOLE_HP_AUDIT_BRIDGES,
    *EXPECTILE_STRATEGY_2_BRIDGES,
    *CHAIN_DECOMPOSITION_BRIDGES,
)


__all__ = [
    'ACTION_DIM_BRIDGES',
    'BRIDGES',
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
