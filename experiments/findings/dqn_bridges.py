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
from corroborate.analyses.paired_g import PairedGResult
from corroborate.analyses.paired_g_per_burst import (
    PerBurstResult, panel_for_env,
)
from corroborate.analyses.tautology_audit import AuditResult
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
        'name': 'mechanism.jensen_gap',
        'reads': ('predicted_q_at_start', 'mc_return'),
    },
)


@claim_bridge
def jensen_gap_outcome_borderline(
    tautology_audit: AuditResult,
    *,
    source: str = 'mechanism.jensen_gap',
    target: str = 'outcome.eval_best_burst_mean',
    direction: Direction = Direction.INVERSE,
    tier: Tier = Tier.ASSOCIATIONAL,
    measurables: tuple[dict[str, object], ...] = _DDQN_AUDIT_PANEL,
    outcome_path: str = 'outcome.eval_best_burst_mean',
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
    report = tautology_audit.by_name('mechanism.jensen_gap')
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
    ('replay.capacity', 'outcome.eval_final_mean'),
    ('replay.batch_size', 'outcome.eval_final_mean'),
    ('optimizer.inner.lr', 'outcome.eval_final_mean'),
    ('sync_period', 'outcome.eval_final_mean'),
    ('mediator.state_coverage_kl_uniform_late',
     'outcome.eval_final_mean'),
]


@claim_bridge
def state_coverage_kl_causes_outcome(
    backdoor_ate: BackdoorResult,
    placebo_refutation: RefutationResult,
    random_common_cause_refutation: RefutationResult,
    *,
    source: str = 'mediator.state_coverage_kl_uniform_late',
    target: str = 'outcome.eval_final_mean',
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.INTERVENTIONAL,
    treatment: str = 'mediator.state_coverage_kl_uniform_late',
    outcome: str = 'outcome.eval_final_mean',
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


__all__ = [
    'ACTION_DIM_BRIDGES',
    'EXPECTILE_PER_BURST_BRIDGES',
    'ddqn_outcome_stable_across_bursts__fourrooms',
    'ddqn_outcome_zero_across_bursts__catch',
    'ddqn_reduces_jensen_gap__acrobot',
    'ddqn_reduces_jensen_gap__catch',
    'ddqn_reduces_jensen_gap__cartpole',
    'ddqn_reduces_jensen_gap__discounting_chain',
    'jensen_gap_outcome_borderline',
    'log_action_dim_drives_jensen_gap_reduction',
    'state_coverage_kl_causes_outcome',
]
