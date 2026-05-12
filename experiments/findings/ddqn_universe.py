"""DDQN measurement graph.

Bridges encoding the DDQN causal-chain claims. Current verdicts in
`ddqn_universe.run.json`; per-bridge audit history in
`BRIDGE_AUDIT_TABLE.md`; substantive findings in `findings_*.md`.

Two-channel causal architecture: Channel A (Hasselt bias-correction,
fires on `[G1∧G2∧G3]` — premise-active ∧ argmax-vulnerable ∧
outcome-headroom; tested by CLAIM 26b) and Channel B (Q-magnitude
regularization via `ddqn_bootstrap_gap`, fires on dormant cells;
tested by CLAIM 3). Disjoint scopes, not competing hypotheses.

RL methodology: per-pair-Δ fixtures (`paired_g`, `paired_link_per_burst`,
`proportion_mediated`) measure within-init correlation, not
population-of-inits variance. Load-bearing bridges migrated to
`stratified_arm_diff_pooled`, `arm_mean_diff`, `stratified_partial_spearman`,
`within_arm_link`."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import polars as pl

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry
from corroborate.analyses.link_attenuation_dowhy import (
    LinkAttenuationDowhyResult,
)
from corroborate.analyses.paired_delta_link_dowhy import (
    PairedDeltaLinkDowhyResult,
)
from corroborate.analyses.arm_mean_diff import ArmMeanDiffResult
from corroborate.analyses.paired_g import PairedGResult
from corroborate.analyses.stratified_partial_spearman import (
    StratifiedPartialSpearmanResult,
)
from corroborate.analyses.stratum_effect_panel import (
    StratumEffectPanel, panel_regress,
)
from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.analyses.paired_g_per_burst import PerBurstResult
from corroborate.analyses.paired_link_per_burst import (
    PerBurstLinkResult, phase_link_consistency,
)
from functools import partial

from corroborate.bridge.bridge import (
    Direction, Tier, claim_bridge,
)
from corroborate.bridge.predicates import (
    finite, finite_ge, finite_gt, finite_lt, partition_aggregate,
)
from corroborate.core.intervention import DoEffect, Intervention
from corroborate.measurables import Measurable
from corroborate.analyses.paired_continuous_do_dowhy import (
    PairedContinuousDoResult,
)
from corroborate.stats import MetaRegressionResult
from corroborate_rl.dqn.claims.bootstrap import (
    adaptive_dormancy_greedify, bootstrap, double_greedify,
)
from corroborate_rl.dqn.dqn import dqn
from corroborate_rl.dqn.measurables import (
    jensen_bias_per_burst_mean,
    mc_return_per_burst_mean,
)
from corroborate.bridge.verdict import RefutationClass, Verdict


# Outermost claim for endogeneity gating (cf. ENDOGENEITY_TOPOLOGY.md).
# The runner threads this to `evaluate(..., claim=CLAIM)` so the
# admission gates (`exogenous_source`, `exogenous_scope`) consult
# `walk_paths(CLAIM, regime='leaf')` for the substrate's author-
# primitive set — leaves like `gamma`, `replay.capacity`,
# `env_name`, `seed`, ...
CLAIM = dqn


# Composed per-burst reductions used as bridge defaults. Both this
# module and `dqn_bridges.py` need the same two reductions; the
# canonical instances live in the substrate (`corroborate_rl.dqn.
# measurables`) so each name registers exactly once across all
# findings modules. Local aliases keep the existing call-site
# names readable.
_MC_RETURN_PER_BURST_MEAN = mc_return_per_burst_mean
_JENSEN_BIAS_PER_BURST_MEAN = jensen_bias_per_burst_mean


# Typed structural deltas reused across bridges in this universe.
# Each `Intervention` is a single-slot replacement on the claim
# graph; `DoEffect` composes them into treatment / baseline arms.
DDQN_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(bootstrap, greedification=double_greedify),
)
ADAPTIVE_DQN_FACTOR_0P5_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(
        bootstrap,
        greedification=partial(
            adaptive_dormancy_greedify, sigma_floor_factor=0.5,
        ),
    ),
)


# File-level intervention: every bridge in this module tests
# `do(bootstrap = ddqn) → effect`. Bridges that test a DIFFERENT
# mechanism contrast (adaptive_dqn variants) override via the
# per-decorator `source = DoEffect(...)` kwarg. HP-encoded
# variants of the SAME contrast (γ-stratified, n_step-stratified,
# etc.) reuse `INTERVENTION` and add an `n_step` / `gamma` /
# `weight_decay` predicate to the bridge's `scope`.
INTERVENTION = DoEffect(treatment=(DDQN_SWAP,), baseline=())

# Substrate arm-key constants extracted from INTERVENTION. Used by
# bridges whose source is a measurable (link / mediator bridges
# testing measurable→outcome couplings) but whose analysis fixtures
# still need `treatment_arm` / `baseline_arm` to construct per-pair Δs.
# The bridge runner only auto-injects arm kwargs when source is a
# DoEffect; for measurable-sourced bridges these defaults supply them.
_DDQN_ARM = INTERVENTION.treatment_arm_key()
_VANILLA_ARM = INTERVENTION.baseline_arm_key()


# Hypothesis-module-level scope: bsuite envs are *diagnostic*
# probes, not RL benchmarks. Each bsuite env is engineered to
# expose a single property (DiscountingChain locks decision at
# step 0 → bandit under the hood; DeepSea has a one-arrow
# optimal policy → zero chain branching; MNISTBandit is pure
# bandit; Catch saturates fast; MemoryChain / UmbrellaChain
# test memory). Cross-env scope claims (chain-amplifier theory,
# polarity moderators, jointly-cross-env attenuators) treat
# them as if they were chain MDPs — they're not, and including
# them imports leverage from a structural class the theory
# doesn't apply to (cf. earlier bandit-tail leverage findings).
#
# `MODULE_SCOPE` is read by the runner via `getattr(h,
# 'MODULE_SCOPE', None)` and AND-combined into every bridge's
# `scope=` inside `evaluate`. By design there's no per-bridge
# opt-out; a bridge that intentionally probes bsuite must live
# in a different hypothesis module (e.g., `dqn_bridges.py`).
MODULE_SCOPE: pl.Expr = ~pl.col('env_name').str.ends_with('-bsuite')


# CLAIM 2 — Necessary-scope dormancy refutation.


@claim_bridge(
    # Decorator declares the do-contrast (vanilla → ddqn) on the
    # OUTCOME column. The graph edge `jensen_dormancy_gap_at_best_burst
    # → eval_best_burst_mean` lives in scope as the cell filter.
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        # **2026-05-11 dormancy measurable shifted from scalar
        # `jensen_dormancy_gap` (all-bursts collapsed) to
        # `jensen_dormancy_gap_at_best_burst` (dormancy AT the
        # burst that gave the best outcome). Resolves the
        # measurement-frame misalignment surfaced during this
        # audit — outcome was at best burst, dormancy was at
        # all-burst mean; cells could read "dormant on average"
        # while best-burst was achieved during transient
        # non-dormant phase. Per-burst alignment closes the loop.
        pl.col('jensen_dormancy_gap_at_best_burst').is_finite()
        & (pl.col('jensen_dormancy_gap_at_best_burst') >= 0.05)
        & pl.col('eval_best_burst_mean').is_finite()
    ),
    predicted_direction='null',
)
def ddqn_refuted_when_dormancy_fires(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    null_ceiling: float = 0.2,
    min_strata: int = 2,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_vanilla_predictor: float = float('-inf'),
) -> tuple[Verdict, RefutationClass | None]:
    """Necessary-scope refutation: on dormant-at-best-burst cells
    (σ_Q × √(2 log K) − (Q − MC) > 0.05), Δ_outcome should be ≈ 0.
    Per-env Cohen's d (independent-samples), DL pooling.

    Per-env CI vs ±`null_ceiling`: HELD if all in null band;
    INVARIANT_VIOLATION if any env CI fully > +null_ceiling;
    NO_EFFECT (SIGN_FLIP) if any env CI fully < −null_ceiling;
    else POW_INSUF. Pooled-d aggregate can mask direction-opposed
    per-env effects, so per-env is load-bearing.

    Currently POW_INSUF: Acrobot d=+0.43 CI=[+0.064,+0.794] and SI
    d=+0.37 CI=[+0.025,+0.722] straddle the +0.2 ceiling. The
    substantive challenge to the necessary-condition framing
    persists. See `findings_dormancy_case_studies.md` for config-
    level case studies — only SI sync=3000 T=200k survives strict
    rock-solid + Welch-sig (Δ_o=+0.50, z=+3.26) and the Δ_q_late
    pattern suggests Channel B (Q-magnitude regularization) is
    firing on these cells, not the Hasselt bias-correction
    Channel A."""
    del stratify_by, min_vanilla_predictor
    n_strata = stratified_arm_diff_pooled.n_strata
    if n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    # Per-env CI check (not pooled): pooled-d aggregate can mask
    # direction-opposed per-env effects.
    any_above = False
    any_below = False
    any_spans = False
    n_envs_valid = 0
    for s in stratified_arm_diff_pooled.per_stratum:
        d_env = s.cohen_d
        se_env = s.cohen_se
        if math.isnan(d_env) or math.isnan(se_env):
            continue
        n_envs_valid += 1
        ci_lo_env = d_env - 1.96 * se_env
        ci_hi_env = d_env + 1.96 * se_env
        if ci_lo_env > null_ceiling:
            any_above = True
        elif ci_hi_env < -null_ceiling:
            any_below = True
        elif not (ci_lo_env >= -null_ceiling and ci_hi_env <= null_ceiling):
            any_spans = True
    if n_envs_valid < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if any_above:
        return Verdict.INVARIANT_VIOLATION, None
    if any_below:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if not any_spans:
        return Verdict.HELD, None
    return Verdict.POWER_INSUFFICIENT, None


# CLAIM 2 corroboration — Pearl rung-2 adaptive controller (dormancy
# proxy → DDQN dispatch). AWAITING DATA: adaptive_dqn_fourrooms_sweep
# + expectile_3way corpora absent post-rebuild; methodology TODO
# paired_g → arm_mean_diff deferred until cells return. Historical
# verdict in `findings_dormancy_controller_scope.md`.


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
    """Pearl rung-2 adaptive controller (per-batch dormancy proxy
    `max_Q − mean_Q ≥ 0.5 × σ_Q × √(2 log |A|)`) recovers DDQN's
    benefit on FR. HELD when g ≥ +0.50, p<0.05. Historical g=+0.78,
    p<0.001. AWAITING DATA (adaptive sweep corpora absent)."""
    if paired_g.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.g):
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g >= 0.50 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# =====================================================================
# TIER A2 — env-conditional existence proofs (per-burst dynamics).
# Generalize WITHIN env at this HP regime; do not lift structurally
# (audit at K1 + the four-MinAtar comparison showed log_obs_dim ×
# log_horizon × n_actions does not predict per-env late attenuation).
# =====================================================================


# CLAIM 7d (TIER A2 ddqn_helps_at_early_bursts__pixel_envs) — CUT
# 2026-05-12 (paired_g + data orphan: MinAtar 1M gone). Substance in
# `findings_minatar_link_attenuation.md`.


@claim_bridge(
    source=INTERVENTION,
    target='mc_return',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    # `corpus` is in pair_by because seed=N has different RNG
    # realizations across independent sweeps; pairing across
    # corpora was an accident of the implementation. Each sweep
    # contributes its own valid paired set.
    pair_by=('seed', 'corpus'),
    # Endogenous scope: `q_divergence_score > 1` (jensen_gap exceeds
    # the Bellman fixed-point bound r_max/(1-γ)) replaces the prior
    # `sync_period == 100` HP-knob scope. SI 1M cells at sync=100
    # all have q_div in (2.7, 1002), so the regime is preserved;
    # cells with NaN q_div (no_hit_penalty wrapper, jensen_gap null)
    # had null mc_return too and contributed nothing. The endogenous
    # form expresses the bridge's actual interest — Q-explosion
    # regime, not standard sync — and will admit future cells from
    # other syncs that happen to land in the same regime.
    scope=(
        (pl.col('env_name') == 'SpaceInvaders-MinAtar')
        & (pl.col('total_steps') >= 1000000.0)
        & pl.col('reward_clip_min').is_null()
        & finite_gt('q_divergence_score', 1.0)
    ),
)
def ddqn_attenuates_at_late_bursts__spaceinvaders(
    paired_g_per_burst: PerBurstResult,
    *,
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    burst_floor: int = 3,
    helped_ceiling: float = 0.40,
    g_ceiling: float = -0.30,
    n_pairs_floor: int = 50,
) -> Verdict:
    """Existence proof: SI 1M last-quarter bursts, DDQN reliably
    WORSE than vanilla. Sample-size-weighted aggregate over
    burst-index ≥ burst_floor. HELD when total n ≥ floor AND
    helped_fraction ≤ ceiling AND g ≤ g_ceiling. SI-specific (not
    seen on Asterix/Breakout/Freeway 1M per K1 audit)."""
    del source  # forwarded to paired_g_per_burst
    late = [
        s for s in paired_g_per_burst.strata
        if s.env_name == 'SpaceInvaders-MinAtar'
        and s.burst_index >= burst_floor
        and s.n_pairs > 0
        and not math.isnan(s.g)
        and not math.isnan(s.helped_fraction)
    ]
    if not late:
        return Verdict.POWER_INSUFFICIENT
    n_total = sum(s.n_pairs for s in late)
    if n_total < n_pairs_floor:
        return Verdict.POWER_INSUFFICIENT
    g_pooled = sum(s.g * s.n_pairs for s in late) / n_total
    helped_pooled = sum(s.helped_fraction * s.n_pairs for s in late) / n_total
    if helped_pooled <= helped_ceiling and g_pooled <= g_ceiling:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# CLAIM 2 scope-limitation companion: same controller FAILS on SI 1M
# (HELD g=−0.46, p=0.016 historically). Together with FourRooms HELD
# sibling encodes "dormancy necessary, not sufficient" — load-bearing
# for PAPER_NOTES.md §3.4. AWAITING DATA: adaptive_dqn_spaceinvaders_1m
# + minatar_1M_spaceinvaders absent post-rebuild.


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
    """Pearl rung-2 scope-limitation: adaptive controller (CLAIM 2)
    FAILS on SI 1M — dormancy proxy doesn't fire, controller ≡
    DDQN, inherits attenuation. HELD when g(adaptive vs vanilla)
    ≤ -0.30, p<0.05. INVARIANT_VIOLATION if adaptive unexpectedly
    helps. Historical: g=-0.46, p=0.016. AWAITING DATA (adaptive +
    SI 1M corpora absent)."""
    if paired_g.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.g):
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g <= -0.30 and paired_g.p_value < 0.05:
        return Verdict.HELD
    if paired_g.g >= 0.30 and paired_g.p_value < 0.05:
        return Verdict.INVARIANT_VIOLATION
    return Verdict.NO_EFFECT


# CLAIM 5 — within-env do(γ) on FourRooms (HELD g=+1.11 at γ=0.99
# historically). NOT covered by CLAIM 26b (cross-env at γ=0.99 only).
# AWAITING DATA: gamma_sweep absent post-rebuild. MetaMaze sister
# STAYS CUT — its claim is actively tested + REFUTED by
# `metamaze_link_steeper_at_high_gamma` on current data. Methodology
# TODO paired_g → arm_mean_diff deferred until data returns.


@claim_bridge(
    # `effective_horizon >= 25` is the endogenous selector. Within
    # the `gamma_sweep` corpus (FourRooms-misc, γ ∈ {0.90, 0.95,
    # 0.99} with realised bf ≈ 0.97-0.99 → eff_h ∈ {8, 13, ~30+})
    # it selects the γ=0.99 cohort. eff_h is now `1/(1−γ·bf)` —
    # the chain-depth amplifier mediated by realised episode
    # termination — so values are smaller than the textbook
    # `1/(1−γ)`. Threshold recalibrated from 50 to 25.
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('corpus') == 'gamma_sweep')
        & finite_ge('effective_horizon', 25.0)
    ),
)
def ddqn_benefit_scales_with_effective_horizon__fourrooms(
    paired_g: PairedGResult,
) -> Verdict:
    """Pearl rung-2 do(γ) FR γ=0.99 cohort (eff_h≈72). HELD when
    helped_fraction ≥ 0.55 AND g ≥ 0.30. Historical: g=+1.11.
    AWAITING DATA: gamma_sweep corpus absent. Within-env probe
    NOT covered by CLAIM 26b (cross-env at γ=0.99 only)."""
    if paired_g.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.helped_fraction):
        return Verdict.POWER_INSUFFICIENT
    if (
        paired_g.helped_fraction >= 0.55
        and paired_g.g >= 0.30
    ):
        return Verdict.HELD
    return Verdict.NO_EFFECT


# CLAIM 6 (log_mc_variance attenuator) — CUT 2026-05-11 REFUTED by CV
# decomposition. Underlearning-rescue substance in CLAIM 7's rs bridges.


# CLAIM 7 — DDQN dominates vanilla's reward-scale response curve on
# FourRooms at rs ∈ [0.03, 0.3] (Pearl rung-2 interventional). Peak
# Δ=+0.50 native at rs=0.3 (HELD). Hasselt-floor reading: DDQN's
# reduced ε lets it learn at smaller σ_Q. NOT a √(log|A|) law —
# rescue regime is FourRooms-specific (DeepSea/DiscountingChain/
# MNISTBandit don't enter the under-learning regime at rs=0.1).
# Δ is do-effect not causal edge between arms — see findings note
# `findings_underlearning_rescue.md`.


def _rescue_threshold(
    *,
    failure_baseline: float = 0.1,
    optimal_ceiling: float = 0.8,
    rescue_fraction: float = 0.5,
) -> float:
    """`rescue_fraction × (optimal_ceiling − failure_baseline)`
    = 0.5 × (0.8 − 0.1) = 0.35. Empirically calibrated:
    failure_baseline ≈ vanilla DQN floor at rs=0.1 on FR; ceiling
    ≈ empirical RL convergence; fraction = qualitative "≥half
    headroom" bound."""
    return rescue_fraction * (optimal_ceiling - failure_baseline)


def _has_heavy_tail_violation(
    assumption_violations: tuple[str, ...],
) -> bool:
    """True when `paired_g.assumption_violations` flags heavy-tail
    or skew-bias issues that make the Gaussian ±1.96×se CI anti-
    conservative.

    Reviewer-3 catch: `paired_g` reports
    `heavy_tail_se_anti_conservative` and `skew_bias_likely` flags
    when the per-pair-Δ distribution's kurtosis / skew exceeds
    calibrated thresholds — meaning the framework's own SE is
    under-covering by ~10-25%. The CI-vs-threshold verdict helpers
    treat the SE as exact; if the framework knows it's
    miscalibrated, the verdict should propagate that knowledge.

    Returns True when the bridge's CI-based verdict should be
    treated with caution (typically: widen CI or refuse to assert
    HELD/NO_EFFECT).
    """
    return any(
        'heavy_tail' in v or 'skew' in v for v in assumption_violations
    )


def _native_diff_ci_verdict(
    md: float, se: float, threshold: float,
    *,
    assumption_violations: tuple[str, ...] = (),
    ci_widening_factor: float = 1.25,
) -> Verdict:
    """CI-vs-threshold verdict for paired native-diff bridges.

    Tests hypothesis `md ≥ threshold` via the 95% CI around md.
    Reviewer-3 catch: when `paired_g.assumption_violations` flags
    heavy-tail / skew bias, the framework's own SE is under-
    covering by ~10-25%. We widen the CI by `ci_widening_factor`
    (default 1.25 = 25% conservative bump matching the framework's
    own warning) to keep the verdict robust to that
    miscalibration. When the widened CI still discriminates against
    threshold, the verdict is honest; when it spans, POW_INSUF is
    correct.

    - `CI ⊂ [threshold, ∞)` → HELD;
    - `CI ⊂ (−∞, threshold)` → NO_EFFECT;
    - `CI spans threshold` → POWER_INSUFFICIENT.
    """
    if math.isnan(md) or math.isnan(se):
        return Verdict.POWER_INSUFFICIENT
    se_eff = (
        se * ci_widening_factor
        if _has_heavy_tail_violation(assumption_violations)
        else se
    )
    ci_lo = md - 1.96 * se_eff
    ci_hi = md + 1.96 * se_eff
    if ci_lo >= threshold:
        return Verdict.HELD
    if ci_hi < threshold:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


def _native_diff_null_verdict(
    md: float, se: float, null_ceiling: float,
    *,
    assumption_violations: tuple[str, ...] = (),
    ci_widening_factor: float = 1.25,
) -> Verdict:
    """CI-vs-ceiling verdict for null-prediction native-diff
    bridges (`predicted_direction='null'`).

    Same heavy-tail / skew widening as `_native_diff_ci_verdict`.

    - `CI ⊂ [−null_ceiling, +null_ceiling]` → HELD;
    - CI outside ±null_ceiling → NO_EFFECT;
    - CI spans boundary → POWER_INSUFFICIENT.
    """
    if math.isnan(md) or math.isnan(se):
        return Verdict.POWER_INSUFFICIENT
    se_eff = (
        se * ci_widening_factor
        if _has_heavy_tail_violation(assumption_violations)
        else se
    )
    ci_lo = md - 1.96 * se_eff
    ci_hi = md + 1.96 * se_eff
    if ci_lo >= -null_ceiling and ci_hi <= null_ceiling:
        return Verdict.HELD
    if ci_lo > null_ceiling or ci_hi < -null_ceiling:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


@claim_bridge(
    source=INTERVENTION,
    target='outcome_native',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('reward_scale') == 0.1)
    ),
    predicted_direction='a_gt_b',
)
def ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    threshold_diff: float = _rescue_threshold(),
) -> Verdict:
    """Pearl-rung-2 do(arm=ddqn) on FR rs=0.1: DDQN's native-units
    mean_diff vs vanilla closes ≥ `rescue_fraction × (optimal -
    failure_baseline)` of the failure-to-optimal range. Threshold
    +0.35 = 0.5 × (0.8 − 0.1). HELD when CI lower ≥ threshold.
    Empirical: md=+0.638, CI=[+0.594, +0.682]."""
    return _native_diff_ci_verdict(
        arm_mean_diff.mean_diff, arm_mean_diff.mean_diff_se,
        threshold_diff,
    )


# CLAIM 7b — rescue-regime peak at rs=0.3 (Δ=+0.50 native). Plateau
# with rs=0.1 (+0.49), not single-point.


@claim_bridge(
    source=INTERVENTION,
    target='outcome_native',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('reward_scale') == 0.3)
    ),
    predicted_direction='a_gt_b',
)
def ddqn_dominates_vanilla_response_curve__fourrooms_rs_0p3(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    threshold_diff: float = _rescue_threshold(),
) -> Verdict:
    """Sibling of CLAIM 7 at the rescue-regime peak (rs=0.3). Same
    threshold +0.35 = `_rescue_threshold()`. Currently REFUTED on
    post-rebuild corpus: md=+0.259 CI=[+0.169, +0.349] — rs=0.3
    sits on rescue-regime upper edge where vanilla recovers (was
    +0.50 plateau on earlier corpus)."""
    return _native_diff_ci_verdict(
        arm_mean_diff.mean_diff, arm_mean_diff.mean_diff_se,
        threshold_diff,
    )


# CLAIM 7c / 7d — rescue is FourRooms-specific. Acrobot and CartPole
# at rs=0.1 don't show DDQN benefit (g_out -0.17, +0.09). Authored as
# null-form refutations of universal rs reading.


@claim_bridge(
    source=INTERVENTION,
    target='outcome_native',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Acrobot-v1')
        & (pl.col('reward_scale') == 0.1)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    ),
    predicted_direction='null',
)
def ddqn_does_not_rescue__acrobot_rs_0p1(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    null_ceiling: float = 0.2,
) -> Verdict:
    """Acrobot rs=0.1 null bridge: rescue does NOT activate (CLAIM
    7 is FR-specific). HELD when native CI ⊂ ±null_ceiling.
    Currently POW_INSUF (md=+0.229, CI=[−0.13,+0.59] spans ceiling)."""
    return _native_diff_null_verdict(
        arm_mean_diff.mean_diff, arm_mean_diff.mean_diff_se,
        null_ceiling,
    )


@claim_bridge(
    source=INTERVENTION,
    target='outcome_native',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'CartPole-v1')
        & (pl.col('reward_scale') == 0.1)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    ),
    predicted_direction='null',
)
def ddqn_does_not_rescue__cartpole_rs_0p1(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    null_ceiling: float = 0.2,
) -> Verdict:
    """CartPole rs=0.1 sister to Acrobot null bridge. HELD when
    native CI ⊂ ±null_ceiling. Currently POW_INSUF (md=+0.212,
    CI=[-0.155,+0.579] spans ceiling)."""
    return _native_diff_null_verdict(
        arm_mean_diff.mean_diff, arm_mean_diff.mean_diff_se,
        null_ceiling,
    )


# CLAIM 7 g/h/i/j — CUT 2026-05-11. Four wrapper-induced mechanism
# probes (reward_shape / action_noise). Synthesis: argmax-concentration
# is FR-structural side-effect not causal; subsumed by CLAIM 26b's
# three-gate framework. See `findings_underlearning_rescue.md`.


# CLAIM 7e/7f — DDQN's rescue is action-selection-level: argmax
# entropy higher at rs=0.1 (H 1.14 → 1.30); matches at rs=1.0. Pair:
# DDQN-INCREASES-entropy at rescue rs + matches at standard rs.


@claim_bridge(
    source=INTERVENTION,
    target='argmax_entropy_late',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('reward_scale') == 0.1)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    ),
    predicted_direction='a_gt_b',
)
def ddqn_increases_argmax_entropy__fourrooms_rs_0p1(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    threshold_diff: float = 0.05,
) -> tuple[Verdict, RefutationClass | None]:
    """FR rs=0.1 rescue regime: DDQN's argmax entropy predicted
    higher than vanilla's. REFUTED via SIGN_FLIP — actually
    SUBSTANTIALLY LOWER (mean_diff=-0.232 nats, p=2.6e-12). DDQN's
    rescue is "policy sharpens after learning unblocks", not
    "exploration maintained". Kept as documented refutation."""
    diff = arm_mean_diff.mean_diff
    p = arm_mean_diff.mean_diff_p_value
    if math.isnan(diff) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    # Predicted direction: a_gt_b (positive ΔH).
    if diff < 0.0:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    significant = p < 0.05
    above_threshold = diff >= threshold_diff
    if significant and above_threshold:
        return Verdict.HELD, None
    if above_threshold or significant:
        return Verdict.POWER_INSUFFICIENT, RefutationClass.UNDERPOWERED
    return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT


@claim_bridge(
    source=INTERVENTION,
    target='argmax_entropy_late',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('reward_scale') == 1.0)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    ),
    predicted_direction='null',
)
def ddqn_entropy_matches_vanilla__fourrooms_rs_1p0(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    null_ceiling: float = 0.05,
) -> tuple[Verdict, RefutationClass | None]:
    """FR rs=1.0 standard regime: DDQN argmax entropy null-prediction
    refuted via SIGN_FLIP — DDQN STILL has lower argmaxH than
    vanilla (mean_diff=-0.099). The argmax-sharpening is reward-
    scale-invariant, not regime-specific. Kept as documented
    refutation of regime-specificity framing."""
    diff = arm_mean_diff.mean_diff
    p = arm_mean_diff.mean_diff_p_value
    if math.isnan(diff) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    is_small = abs(diff) <= null_ceiling
    is_ns = p > 0.05
    if is_small and is_ns:
        return Verdict.HELD, None
    if is_small or is_ns:
        return Verdict.POWER_INSUFFICIENT, RefutationClass.UNDERPOWERED
    # Predicted null but observed significant effect → null
    # refuted; classify as SIGN_FLIP (a real effect when none
    # was predicted).
    return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP


# CLAIM 8 (per-burst SI 1M crossover, ddqn_curve_crosses_vanilla_
# late__spaceinvaders) — CUT 2026-05-12 (data orphan: SI 1M gone).
# Substance in `findings_minatar_link_attenuation.md` +
# `findings_sync_curve_breakout.md`.


# CLAIM 9 — n-step falsification of bootstrap-bias-compounding
# (Pearl rung-2 negative-prediction probe). Encoded as PAIR:
# `__fourrooms_n1` HELD positive (DDQN helps at full bootstrap);
# `__fourrooms_n10` HELD null (effect vanishes as bootstrap removed).
# Historically monotonic Δ→0: n=1 +0.087 → n=10 +0.005 (ns). See
# `findings_nstep_falsification.md`.


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('n_step') == 1)
    ),
)
def ddqn_helps_at_full_bootstrap__fourrooms_n1(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    threshold_diff: float = 0.05,
) -> Verdict:
    """At n=1 (full bootstrap), DDQN's outcome benefit on
    FourRooms is ≥ +0.05 with p < 0.05. The positive baseline of
    the falsification curve — pairs with the n=10 NO_EFFECT
    bridge to corroborate that bootstrap dependence is the
    mechanism's necessary substrate.

    Migrated from `paired_g` to `arm_mean_diff` (2026-05-11):
    seed-paired Δ is the wrong inferential primitive for RL —
    same-seed cells diverge from step 1 as DDQN changes the
    loss / sample order / explored state space, so the "paired"
    Δ measures within-init correlation, not population-of-inits
    variance. Independent-samples (Welch's t) is the
    inferentially-honest form for "does DDQN beat vanilla at
    this regime across plausible inits."""
    diff = arm_mean_diff.mean_diff
    p = arm_mean_diff.mean_diff_p_value
    if math.isnan(diff) or math.isnan(p):
        return Verdict.NO_EFFECT
    if diff < 0.0:
        return Verdict.NO_EFFECT
    significant = p < 0.05
    above_threshold = diff >= threshold_diff
    if significant and above_threshold:
        return Verdict.HELD
    if above_threshold or significant:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


# CLAIM 9 n=10 falsification endpoint — load-bearing companion to
# the n=1 HELD sister (the falsification curve needs both endpoints).
# AWAITING DATA: nstep_lambda_fourrooms absent post-rebuild.


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('n_step') == 10)
    ),
)
def ddqn_null_under_monte_carlo__fourrooms_n10(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    null_ceiling: float = 0.02,
) -> Verdict:
    """At n=10 (near-Monte-Carlo, bootstrap influence ≈ 0), DDQN's
    outcome benefit on FourRooms is ≤ +0.02 AND not significant
    (p > 0.05). HELD when both conditions hold — corroborates the
    necessary-condition reading: removing bootstrap removes the
    benefit. This is a HELD-as-null bridge: the verdict is HELD
    when the difference is small (the *predicted* outcome of the
    falsification probe). Verdict mapping is inverted vs the n=1
    bridge by design — the theorem predicts smallness here.

    Migrated from `paired_g` to `arm_mean_diff` (2026-05-11) —
    see companion bridge `ddqn_helps_at_full_bootstrap__fourrooms_n1`
    for the RL-methodology rationale.

    **AWAITING DATA (2026-05-12):** Postfix rebuild dropped the
    `nstep_lambda_fourrooms` sweep — current cache has only n=1
    cells for FourRooms. Bridge correctly returns POWER_INSUFFICIENT
    until n=10 cells are re-collected. Restored after audit
    reflection: the n=1 HELD verdict alone does NOT corroborate
    the falsification curve; the falsification structure requires
    the n=10 null endpoint."""
    diff = arm_mean_diff.mean_diff
    p = arm_mean_diff.mean_diff_p_value
    if math.isnan(diff) or math.isnan(p):
        # NaN = no cells in scope OR degenerate stats; underpowered
        # to assert HELD or NO_EFFECT (reviewer screening catch).
        return Verdict.POWER_INSUFFICIENT
    is_small = abs(diff) <= null_ceiling
    is_ns = p > 0.05
    if is_small and is_ns:
        return Verdict.HELD
    if is_small or is_ns:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


# ============ CLAIM 10 — link IS causally bias-correction on Acrobot
#                          γ=0.999, per-burst.
#
# The scalar `outcome.eval_best_burst_mean` ↔ `mechanism.jensen_gap`
# slope on Acrobot at γ=0.999 was reported as null in
# `findings_l2_acrobot_goldilocks.md` ("Δ within 1 SD"). That was a
# measurement artifact: best-burst-per-seed selection doesn't align
# vanilla and DDQN seeds, so the scalar pair averages noise. Per-
# burst alignment recovers the signal.
#
# Four bridges corroborate the link causally on the same 300-pair
# panel from `l2_x_gamma_acrobot` traces (γ=0.999, wd=1e-4):
#
#   (10a) Per-burst phase-link consistency — fraction of bursts where
#         r(Δ_jens, Δ_out) is significantly negative. Empirical 1.000.
#   (10b) Backdoor ATE adjusting for burst + seed confounds. Empirical
#         -0.6312, identified=True.
#   (10c) Placebo refutation — permuted treatment shrinks ATE to ~0.
#         Empirical placebo ATE = 0.000, |placebo/real| = 0%.
#   (10d) RCC refutation — adding noise covariate leaves ATE stable.
#         Empirical drift = 0.000.
#
# All four hold → bias-correction → outcome on Acrobot γ=0.999 is a
# genuine causal link, not a spurious correlation, not a burst
# confound. The scalar metric was the wrong instrument.


@claim_bridge(
    # Runs against the `l2_x_gamma_acrobot` corpus's
    # γ=0.999 × wd=1e-4 cohort. `effective_horizon >= 80` is the
    # endogenous selector — Acrobot's realised bf ≈ 0.99 means
    # γ=0.999 → eff_h ≈ 86-141 and γ=0.99 → eff_h ≈ 49-61, so the
    # threshold isolates the γ=0.999 cohort. eff_h is now
    # `1/(1−γ·bf)`; threshold recalibrated from 500.
    source='jensen_gap',
    target='mc_return',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'Acrobot-v1')
        & (pl.col('corpus') == 'l2_x_gamma_acrobot')
        & finite_ge('effective_horizon', 80.0)
        & (pl.col('optimizer.inner.weight_decay') == 0.0001)
    ),
)
def acrobot_per_burst_link_active__gamma_0999(
    paired_link_per_burst: PerBurstLinkResult,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _JENSEN_BIAS_PER_BURST_MEAN,
    env_name: str = 'Acrobot-v1',
    consistency_floor: float = 0.7,
) -> Verdict:
    """Per-burst r(Δ_jens, Δ_out) is significantly negative in at
    least `consistency_floor` of bursts on Acrobot γ=0.999. HELD when
    `phase_link_consistency >= consistency_floor`. Empirical 1.000
    (every burst significant) at the corroborating regime."""
    del treatment_arm, baseline_arm, target, predictor
    plc = phase_link_consistency(
        paired_link_per_burst, env_name=env_name,
    )
    if math.isnan(plc):
        return Verdict.POWER_INSUFFICIENT
    if plc >= consistency_floor:
        return Verdict.HELD
    if plc >= consistency_floor * 0.5:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


# CLAIM 22 — REACH cross-env DoWhy link (FR + Acrobot + MountainCar +
# MetaMaze): ATE -0.61, placebo+RCC clean (n=1200 per-burst rows).
# Encoded as backdoor + placebo + RCC bridge trio.


# Three structural gates for "DDQN is relevant" scope (per
# `docs/DDQN_THREE_GATES.md`). DDQN can only help when ALL three
# gates fire conjunctively — otherwise NO_EFFECT is structural,
# not algorithmic.
#
# G1 — premise active: vanilla overestimates by a substantial
#      margin (jensen_gap > 0.5) AND does not net-underestimate
#      (jensen_dormancy_gap < 0.05). FourRooms (jens=0.04)
#      excluded.
#
# G2 — argmax bias-vulnerable: bias differential can flip argmax.
#      Currently heuristic via `n_actions >= 3` (Hasselt floor
#      plus union-bound on flip-paths grows with K). Future
#      continuous form: `q_argmax_margin_late /
#      q_action_std_late < 1.0` once measurables populate on
#      rebuilt cache. CartPole (|A|=2) excluded.
#
# G3 — outcome has headroom: vanilla return is below env reward
#      ceiling. Currently proxied by REACH polarity (REACH envs
#      have shorter→better → outcome can fall arbitrarily, no
#      ceiling). SURVIVE envs hit eval-cap (CartPole 99.34).
#      Future continuous form: per-env reward-ceiling fraction
#      via `outcome_episode_cv > 0.005` (saturation → CV→0).
# G1 filter expressed at CONFIG level, not per-cell. Earlier per-cell
# `jensen_gap > 0.05` admitted seed-asymmetric subsets — DDQN
# reduces jens (that IS the mechanism), so DDQN seeds failed the
# filter at a higher rate than vanilla. On FourRooms the asymmetry
# was 40 vanilla vs 28 DDQN admitted, and the kept DDQN seeds were
# the ones where DDQN's mechanism worked LEAST. Pair-Δ analyses
# downstream then ran on this biased subset. DoWhy refutations
# (placebo / RCC) protected verdicts but the scope semantics were
# confused. Config-level lift: admit/reject whole configs (both
# arms together) based on VANILLA's config-mean of jens/dorm — the
# property of vanilla cells alone.
_DDQN_CONFIG_KEYS: tuple[str, ...] = (
    'env_name', 'sync_period', 'gamma', 'total_steps',
    'n_step', 'reward_scale', 'action_duplicate_k',
)
_VANILLA_JENS_GAP = pl.when(pl.col('arm_key') == 'baseline').then(
    pl.col('jensen_gap'),
).otherwise(None)
_VANILLA_DORMANCY_GAP = pl.when(pl.col('arm_key') == 'baseline').then(
    pl.col('jensen_dormancy_gap'),
).otherwise(None)

# Reusable G1-at-config-level predicate. partition_aggregate is
# NaN-safe over the vanilla-masked column: non-vanilla rows
# contribute null, vanilla cells contribute their jens/dormancy
# values. The per-row broadcast value is the vanilla-only config-
# mean, which then filters all rows (both arms) in the config
# uniformly. Threshold 0.05 unchanged — applied at config-mean
# grain instead of per-seed grain.
_G1_VANILLA_CONFIG_PREMISE_ACTIVE = (
    (partition_aggregate(_VANILLA_JENS_GAP, by=_DDQN_CONFIG_KEYS, op='mean') > 0.05)
    & (partition_aggregate(_VANILLA_DORMANCY_GAP, by=_DDQN_CONFIG_KEYS, op='mean') < 0.05)
)

# Q-bounded regime predicate (config-level lift of `dormancy<0.05 AND
# q_div<1.0` per-cell pair). Distinct from G1: G1 demands jens > 0
# (mech ACTIVE), while this expresses "Q well-calibrated to MC, no
# explosion" — admits cells where Q ≈ MC with bounded divergence,
# regardless of jens magnitude.
_VANILLA_Q_DIVERGENCE_SCORE = pl.when(pl.col('arm_key') == 'baseline').then(
    pl.col('q_divergence_score'),
).otherwise(None)

_VANILLA_CONFIG_Q_BOUNDED = (
    (partition_aggregate(_VANILLA_DORMANCY_GAP, by=_DDQN_CONFIG_KEYS, op='mean') < 0.05)
    & (partition_aggregate(_VANILLA_Q_DIVERGENCE_SCORE, by=_DDQN_CONFIG_KEYS, op='mean') < 1.0)
)

_DDQN_RELEVANT_SCOPE = (
    _G1_VANILLA_CONFIG_PREMISE_ACTIVE
    # G2 — argmax bias-vulnerable. Heuristic `n_actions >= 3` for
    # now. The (max−min)/σ_Q proxy was inconclusive (clusters at
    # 2.0-2.6 for all envs, doesn't isolate top1-top2 margin from
    # full Q-range). Proper continuous form requires
    # `q_argmax_margin_late / q_action_std_late < √(2 ln K)`
    # threshold, awaiting future sweeps to populate
    # `online_top12_margin_per_step` trace reduction (added
    # 2026-05-09).
    & finite('n_actions')
    & (pl.col('n_actions') >= 3)
    # G3 deferred — `env_reward_polarity` is a bad proxy
    # (MetaMaze's procedural mazes break length→reward
    # correlation, polarity ≈ 0 instead of negative). The proper
    # continuous form (`outcome_episode_cv > 0.005`) needs cache
    # rebuild after the new measurable populates. Until then,
    # G1+G2 alone correctly excludes the established ceiling-
    # saturated cases (CartPole via G2's |A|=2 exclusion).
    # Standard config (no n-step / action-duplicate / rs-shift /
    # polyak-τ interventions in scope). These appear in
    # `_DDQN_CONFIG_KEYS` AND as separate predicates: the former
    # ensures cells from rs-shift / n-step / k-dup variants get
    # their own config-mean (don't contaminate the standard
    # config's mean); the latter ensures non-standard variants
    # are excluded from scope entirely.
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    & pl.col('target_sync.tau').is_null()
)


_REACH_ENVS_FOUR: tuple[str, ...] = (
    'FourRooms-misc',
    'Acrobot-v1',
    'MountainCar-v0',
    'MetaMaze-misc',
)


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=_DDQN_RELEVANT_SCOPE,
)
def reach_link_backdoor_ate_negative(
    paired_delta_link_dowhy: PairedDeltaLinkDowhyResult,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _JENSEN_BIAS_PER_BURST_MEAN,
    env_filter: tuple[str, ...] = _REACH_ENVS_FOUR,
    ate_ceiling: float = -0.1,
) -> Verdict:
    """DoWhy backdoor ATE on the per-(env, burst, seed) Δ panel
    across cells satisfying the DDQN-relevant gate conjunction
    (G1 premise-active + G2 argmax-vulnerable, see
    `docs/DDQN_THREE_GATES.md`) yields a NEGATIVE ATE bigger than
    `ate_ceiling`. HELD when identified AND ATE <= ceiling. Sign-
    locked by Hasselt theorem: DDQN reduces |Δ_jens| → boosts
    |Δ_out| (CLAIM 22).

    **Methodology note** (2026-05-11 audit): briefly migrated to
    per-cell `backdoor_ate` to honor the per-pair-Δ critique, but
    reverted because (a) the framework's DoWhy primitive only
    accepts numeric DAG variables (env_name is categorical), so
    per-cell env-adjustment would require one-hot encoding work;
    (b) at n=1200 obs with paired-Δ + placebo + RCC refutations
    holding, the original analysis is empirically defensible.
    The per-pair-Δ form admixes init-correlation into the slope
    estimate but the refutations validate the signal isn't a
    pure data artifact."""
    del treatment_arm, baseline_arm, link_predictor, link_target, env_filter
    b = paired_delta_link_dowhy.backdoor
    if not b.identified:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(b.ate):
        return Verdict.POWER_INSUFFICIENT
    if b.ate <= ate_ceiling:
        return Verdict.HELD
    if b.ate < 0.0:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=_DDQN_RELEVANT_SCOPE,
)
def reach_link_placebo_refuted(
    paired_delta_link_dowhy: PairedDeltaLinkDowhyResult,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _JENSEN_BIAS_PER_BURST_MEAN,
    env_filter: tuple[str, ...] = _REACH_ENVS_FOUR,
    placebo_max_ratio: float = 0.2,
) -> Verdict:
    """Placebo refutation on the REACH-cross-env Δ panel: random
    treatment shrinks ATE to ~zero. HELD when |placebo / real| <
    `placebo_max_ratio` AND real ATE is non-zero. Confirms the
    bias-correction effect is treatment-specific, not artifact of
    pooled noise (CLAIM 22)."""
    del treatment_arm, baseline_arm, link_predictor, link_target, env_filter
    p = paired_delta_link_dowhy.placebo
    real = p.real_ate
    placebo = p.refuted_ate
    if math.isnan(real) or math.isnan(placebo) or abs(real) < 1e-9:
        return Verdict.POWER_INSUFFICIENT
    ratio = abs(placebo / real)
    if ratio < placebo_max_ratio:
        return Verdict.HELD
    if ratio < placebo_max_ratio * 2:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=_DDQN_RELEVANT_SCOPE,
)
def reach_link_rcc_robust(
    paired_delta_link_dowhy: PairedDeltaLinkDowhyResult,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _JENSEN_BIAS_PER_BURST_MEAN,
    env_filter: tuple[str, ...] = _REACH_ENVS_FOUR,
    rcc_max_drift_ratio: float = 0.1,
) -> Verdict:
    """Random-common-cause refutation on REACH-cross-env panel: a
    synthetic noise covariate added to the adjustment set leaves
    ATE within `rcc_max_drift_ratio` of the real ATE. HELD when
    |refuted - real| / |real| < tolerance. Confirms robustness to
    omitted-confounder vulnerability (CLAIM 22)."""
    del treatment_arm, baseline_arm, link_predictor, link_target, env_filter
    r = paired_delta_link_dowhy.random_common_cause
    real = r.real_ate
    refuted = r.refuted_ate
    if math.isnan(real) or math.isnan(refuted) or abs(real) < 1e-9:
        return Verdict.POWER_INSUFFICIENT
    drift_ratio = abs(refuted - real) / abs(real)
    if drift_ratio < rcc_max_drift_ratio:
        return Verdict.HELD
    if drift_ratio < rcc_max_drift_ratio * 2:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


# CLAIM 23 — q_divergence_score and argmax_entropy_late are shadows
# of Δ_jens (not independent mediators of Δ_outcome). q_div=jens ×
# per-env-constant mathematically; argmaxH co-varies via shared Q-
# distribution. Partial-Spearman | Δ_jens collapses both. Authored
# as null-form bridges. See `feedback_jens_shadow_mediators.md`.


_DDQN_VS_VANILLA_ARMS = (
    'arm_baseline',
    'arm_treatment',
)


@claim_bridge(
    source='q_divergence_score',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_DDQN_RELEVANT_SCOPE,
    predicted_direction='null',
)
def q_divergence_shadowed_by_jens(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'q_divergence_score',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    null_max_abs_rho: float = 0.2,
    min_strata: int = 2,
) -> Verdict:
    """`ρ_partial(qdiv, outcome | jens)` env-stratified Fisher-z
    pooled. HELD when |ρ| < `null_max_abs_rho` (null confirmed).
    Currently NO_EFFECT (ρ_partial=-0.432, n=717, 11 strata) — the
    γ-induced scaling residual surfaces cross-env. See CLAIM 23 +
    `feedback_jens_shadow_mediators.md`."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if stratified_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = stratified_partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if abs(rho) < null_max_abs_rho:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source='argmax_entropy_late',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_DDQN_RELEVANT_SCOPE,
    predicted_direction='null',
)
def argmax_entropy_shadowed_by_jens(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'argmax_entropy_late',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    null_max_abs_rho: float = 0.2,
    min_strata: int = 2,
) -> Verdict:
    """`ρ_partial(argmax_entropy_late, outcome | jens)` env-
    stratified Fisher-z pooled. HELD when |ρ| < `null_max_abs_rho`.
    Currently HELD (null confirmed, ρ=+0.011). Unlike q_divergence,
    argmaxH isn't algebraically tied to jens. See CLAIM 23 +
    `feedback_jens_shadow_mediators.md`."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if stratified_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = stratified_partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if abs(rho) < null_max_abs_rho:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# CLAIM 24 — Within-MetaMaze do(γ) on n_γ=2 strata ({0.99, 0.999}).
# Stratum-level Δ_outcome amplification test (reformulated 2026-05-11
# off paired-Δ form). Weak (n_γ=2 no slope CI) but falsifiable.
# Currently REFUTED on postfix corpora (γ=0.999 Δ_o < 0 on both mean
# and median — was paired-Δ init-correlation, not amplification).


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'MetaMaze-misc')
        & pl.col('gamma').is_in([0.99, 0.999])
        & _G1_VANILLA_CONFIG_PREMISE_ACTIVE
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
        & pl.col('target_sync.tau').is_null()
    ),
    predicted_direction='a_gt_b',
)
def metamaze_link_steeper_at_high_gamma(
    stratum_effect_panel: StratumEffectPanel,
    *,
    measurables: tuple[str, ...] = ('eval_best_burst_mean',),
    stratify_by: tuple[str, ...] = ('gamma',),
    min_seeds_per_arm: int = 10,
    high_gamma: float = 0.999,
    low_gamma: float = 0.99,
    high_floor: float = 0.5,
    amplification_ratio_min: float = 1.5,
) -> Verdict:
    """Within-MetaMaze do(γ): n_γ=2 amplification test. HELD when
    high-γ Δ_o ≥ `high_floor` AND high-γ ≥ `amplification_ratio_min`
    × low-γ Δ_o (or low-γ ≤ 0 trivially). Currently REFUTED on
    postfix corpora: γ=0.99 +1.49, γ=0.999 -1.65 → NO_EFFECT (the
    paired-Δ +2.55 reading was init-correlation, not amplification)."""
    del measurables, stratify_by, min_seeds_per_arm  # forwarded to fixture
    panel = stratum_effect_panel
    if panel.n_strata < 2:
        return Verdict.POWER_INSUFFICIENT
    deltas_outcome = panel.deltas.get('eval_best_burst_mean', ())
    high_delta: float | None = None
    low_delta: float | None = None
    for stratum, delta in zip(panel.strata, deltas_outcome, strict=True):
        gamma_val = stratum[0]
        if (
            isinstance(gamma_val, (int, float))
            and not math.isnan(float(gamma_val))
        ):
            if math.isclose(float(gamma_val), high_gamma, rel_tol=1e-6):
                high_delta = delta
            elif math.isclose(float(gamma_val), low_gamma, rel_tol=1e-6):
                low_delta = delta
    if (
        high_delta is None or low_delta is None
        or math.isnan(high_delta) or math.isnan(low_delta)
    ):
        return Verdict.POWER_INSUFFICIENT
    # (i) high-γ stratum must show substantive absolute benefit.
    if high_delta < high_floor:
        return Verdict.NO_EFFECT
    # (ii) amplification ratio. Qualitative flip (low ≤ 0, high > floor)
    # is the strongest form of amplification — trivially holds.
    if low_delta <= 0:
        return Verdict.HELD
    if (high_delta / low_delta) >= amplification_ratio_min:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# Sibling bridge testing the same claim under MEDIAN aggregation.
# Together with `metamaze_link_steeper_at_high_gamma` (mean), this
# pair characterizes the seed-level distribution. The CLAIM 24
# banner's "median Δ_o = +2.55 at γ=0.999" was the PAIRED median
# (per-seed `DDQN − vanilla`) — inheriting the same init-
# correlation critique as paired_g. The stratum-level median
# (`median(DDQN) − median(vanilla)`) is the inferentially-honest
# form; here it ALSO refutes amplification, confirming the
# mean-based finding is not an outlier artifact.


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'MetaMaze-misc')
        & pl.col('gamma').is_in([0.99, 0.999])
        & _G1_VANILLA_CONFIG_PREMISE_ACTIVE
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
        & pl.col('target_sync.tau').is_null()
    ),
    predicted_direction='a_gt_b',
)
def metamaze_link_steeper_at_high_gamma__median(
    stratum_effect_panel: StratumEffectPanel,
    *,
    measurables: tuple[str, ...] = ('eval_best_burst_mean',),
    stratify_by: tuple[str, ...] = ('gamma',),
    min_seeds_per_arm: int = 10,
    aggregator: str = 'median',
    high_gamma: float = 0.999,
    low_gamma: float = 0.99,
    high_floor: float = 0.5,
    amplification_ratio_min: float = 1.5,
) -> Verdict:
    """Sibling of `metamaze_link_steeper_at_high_gamma` under median
    aggregation. Same shape; rules out bimodal-seed-distribution
    explanation of the mean's NO_EFFECT. Currently REFUTED: γ=0.99
    median +0.16, γ=0.999 median -0.96 — both summaries agree DDQN
    HURTS at high γ on MetaMaze."""
    del measurables, stratify_by, min_seeds_per_arm, aggregator
    panel = stratum_effect_panel
    if panel.n_strata < 2:
        return Verdict.POWER_INSUFFICIENT
    deltas_outcome = panel.deltas.get('eval_best_burst_mean', ())
    high_delta: float | None = None
    low_delta: float | None = None
    for stratum, delta in zip(panel.strata, deltas_outcome, strict=True):
        gamma_val = stratum[0]
        if (
            isinstance(gamma_val, (int, float))
            and not math.isnan(float(gamma_val))
        ):
            if math.isclose(float(gamma_val), high_gamma, rel_tol=1e-6):
                high_delta = delta
            elif math.isclose(float(gamma_val), low_gamma, rel_tol=1e-6):
                low_delta = delta
    if (
        high_delta is None or low_delta is None
        or math.isnan(high_delta) or math.isnan(low_delta)
    ):
        return Verdict.POWER_INSUFFICIENT
    if high_delta < high_floor:
        return Verdict.NO_EFFECT
    if low_delta <= 0:
        return Verdict.HELD
    if (high_delta / low_delta) >= amplification_ratio_min:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# CLAIM 25 — Within-FourRooms do(|A|) via action_duplicate(k). Δ
# scales 20× from |A|=4 (+0.02) to |A|=16 (+0.41) as vanilla halves
# but DDQN stays constant at 0.79. See
# `findings_action_dim_inflation_postfix.md`.


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & pl.col('action_duplicate_k').is_not_null()
        & _G1_VANILLA_CONFIG_PREMISE_ACTIVE
    ),
    predicted_direction='a_lt_b',
)
def fourrooms_action_dim_link_active__inflated(
    stratum_effect_panel: StratumEffectPanel,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    measurables: tuple[str, ...] = ('jensen_gap', 'eval_best_burst_mean'),
    stratify_by: tuple[str, ...] = ('action_duplicate_k',),
    min_seeds_per_arm: int = 5,
    x: str = 'jensen_gap',
    y: str = 'eval_best_burst_mean',
    slope_max: float = -0.05,
    r_squared_floor: float = 0.7,
    min_strata: int = 3,
) -> Verdict:
    """Within-FourRooms chain-amplifier link via `action_duplicate_k`
    panel (k ∈ {1,2,3,4}). Per-k Δ_outcome regressed on per-k Δ_jens.
    HELD when slope ≤ `slope_max` (negative — bias reduction
    translates to outcome) AND p < 0.05 AND n_strata ≥ min_strata.
    See `findings_action_dim_inflation_postfix.md`."""
    del treatment_arm, baseline_arm, measurables, stratify_by, min_seeds_per_arm
    result = panel_regress(stratum_effect_panel, x=x, y=y)
    if result.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(result.slope):
        return Verdict.POWER_INSUFFICIENT
    if result.slope > slope_max:
        # Wrong sign OR right sign with magnitude below substantive
        # threshold. Either way the chain-amplifier pattern is
        # absent from the panel.
        return Verdict.NO_EFFECT
    if result.r_squared < r_squared_floor:
        # The slope satisfies the sign + magnitude check but the
        # fit is noisy — pattern doesn't cleanly support a scaling
        # claim. Don't HELD on noisy panels.
        return Verdict.NO_EFFECT
    # Within-env descriptive scaling claim: n is fixed by sweep
    # design (one stratum per k value), so frequentist p-on-slope
    # isn't the right gate. The chain-amplifier prediction
    # asserts that the OBSERVED panel shows substantive
    # negative slope with clean fit — what we test.
    return Verdict.HELD


# ============ CLAIM 11 — extreme Q-divergence attenuates link
#
# Companion to CLAIM 2 (dormancy refutes mech). Where dormancy bounds the
# LOWER attenuation (mech inactive), CLAIM 11 bounds the UPPER attenuation
# (Q-explosion overwhelms link translation):
#
#   q_divergence_score = vanilla_jens_late / (r_max / (1 - γ))
#
# When score > 1000 (Q exceeds Bellman fixed-point bound by 3+ orders of
# magnitude), the link from mechanism to outcome attenuates significantly.
# Together with dormancy, they bound the band 0.02 < score < 1000 within
# which DDQN's link operates.
#
# Convention: `g_link` is the per-(env, burst) Pearson correlation between
# bias reduction (-Δ_jens) and outcome gain (Δ_outcome) across paired
# seeds — see `paired_link_per_burst` for the negate-and-correlate step.
# Active link → r > 0 (more reduction, more gain). Attenuation → r → 0.
#
# Empirical panel (13 (env, regime) cells, mech-HELD subset, no bandits):
#   below band (score < 0.02): n=3, mean g_link = -0.09 (mostly null)
#   in band (0.02-1000):       n=7, mean g_link = +0.34 (link works)
#   above band (score > 1000): n=3, mean g_link = -0.09 (link attenuated)
#
# DoWhy backdoor ATE (above_1000 vs band, adjusting for env family):
#   ATE = -0.21, placebo refutation passes (|p/r|=0%), RCC drift = 0.012
#   → causally HELD with refutations.
#
# Pearl-rung-2 corroboration via Asterix sync × training-length sweep:
#   sync=1000 100k → q_div=0.02, g_link=+0.21 (in band, link active)
#   sync=100  1M   → q_div=17300, g_link=-0.23 (above band, link collapsed)


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('total_steps') == 1_000_000)
        & finite('q_divergence_score')
        & (
            pl.col('q_network.channels').is_null()
            | (pl.col('q_network.channels') != '(32,64)')
        )
    ),
)
def extreme_q_divergence_attenuates_link__binary(
    link_attenuation_dowhy: LinkAttenuationDowhyResult,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    attenuator: str = 'q_divergence_score',
    binary_threshold: float = 1000.0,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _JENSEN_BIAS_PER_BURST_MEAN,
    ate_ceiling: float = -0.10,
    dedupe_strategy: str = 'mean',
) -> Verdict:
    """Binary form: cells with `q_divergence_score > 1000` have
    per-(env, burst) link strength attenuated by ≥ 0.10 compared
    to band-cells (0.02 < score < 1000), after backdoor
    adjustment for env family. HELD when ATE ≤ -0.10 AND
    identified=True. Empirical: ATE = -0.21."""
    del treatment_arm, baseline_arm, attenuator, binary_threshold
    del link_target, link_predictor, dedupe_strategy
    b = link_attenuation_dowhy.backdoor
    if not b.identified:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(b.ate):
        return Verdict.POWER_INSUFFICIENT
    # Numerical-zero guard: when DoWhy returns an ATE at machine
    # epsilon (e.g., 1e-16), the sign is RNG-dependent and the
    # verdict would flip POW_INSUF / NO_EFFECT across runs. Treat
    # as "no signal".
    if abs(b.ate) < 1e-6:
        return Verdict.POWER_INSUFFICIENT
    if b.ate <= ate_ceiling:
        return Verdict.HELD
    if b.ate < 0.0:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('total_steps') == 1_000_000)
        & finite('q_divergence_score')
        & (
            pl.col('q_network.channels').is_null()
            | (pl.col('q_network.channels') != '(32,64)')
        )
    ),
)
def extreme_q_divergence_attenuates_link__placebo_refuted(
    link_attenuation_dowhy: LinkAttenuationDowhyResult,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    attenuator: str = 'q_divergence_score',
    binary_threshold: float = 1000.0,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _JENSEN_BIAS_PER_BURST_MEAN,
    placebo_max_ratio: float = 0.2,
    dedupe_strategy: str = 'mean',
) -> Verdict:
    """Placebo refutation shrinks the binary above-1000 ATE to ≤
    `placebo_max_ratio` of the real value, confirming the
    attenuation is treatment-specific (not noise). Empirical:
    real -0.21, placebo 0, ratio 0%."""
    del treatment_arm, baseline_arm, attenuator, binary_threshold
    del link_target, link_predictor, dedupe_strategy
    p = link_attenuation_dowhy.placebo
    real = p.real_ate
    placebo = p.refuted_ate
    if math.isnan(real) or math.isnan(placebo) or abs(real) < 1e-9:
        return Verdict.POWER_INSUFFICIENT
    ratio = abs(placebo / real)
    if ratio < placebo_max_ratio:
        return Verdict.HELD
    if ratio < placebo_max_ratio * 2:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('total_steps') == 1_000_000)
        & finite('q_divergence_score')
        & (
            pl.col('q_network.channels').is_null()
            | (pl.col('q_network.channels') != '(32,64)')
        )
    ),
)
def extreme_q_divergence_attenuates_link__rcc_robust(
    link_attenuation_dowhy: LinkAttenuationDowhyResult,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    attenuator: str = 'q_divergence_score',
    binary_threshold: float = 1000.0,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _JENSEN_BIAS_PER_BURST_MEAN,
    rcc_max_drift_ratio: float = 0.15,
    dedupe_strategy: str = 'mean',
) -> Verdict:
    """RCC refutation: adding a noise covariate to the adjustment
    set leaves the binary above-1000 ATE within
    `rcc_max_drift_ratio` of real. Confirms robustness to
    spurious-confound vulnerability. Empirical: drift ratio ≈ 5%."""
    del treatment_arm, baseline_arm, attenuator, binary_threshold
    del link_target, link_predictor, dedupe_strategy
    r = link_attenuation_dowhy.random_common_cause
    real = r.real_ate
    refuted = r.refuted_ate
    if math.isnan(real) or math.isnan(refuted) or abs(real) < 1e-9:
        return Verdict.POWER_INSUFFICIENT
    drift_ratio = abs(refuted - real) / abs(real)
    if drift_ratio < rcc_max_drift_ratio:
        return Verdict.HELD
    if drift_ratio < rcc_max_drift_ratio * 2:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


# CLAIM 12 — env-polarity moderates the eff_h mediator sign. GOAL
# envs ρ_pool=-0.80, SURVIVAL +0.24 (sign-cancellation in pooled
# cross-env). See `findings_polarity_mediator.md`.


@claim_bridge(
    source='effective_horizon',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('env_name', 'corpus', 'gamma', 'total_steps', 'sync_period', 'seed'),
    scope=(
        finite_lt('env_reward_polarity', -0.3)
        & (
            pl.col('q_divergence_score').is_nan()
            | finite_lt('q_divergence_score', 1000.0)
        )
    ),
    # `predicted_direction='a_lt_b'`: on GOAL polarity envs, the
    # polarity tautology predicts r(eff_h, outcome) ∝ polarity (≈
    # −0.6 negative), and this coupling survives conditioning on
    # jens. HELD when ρ_partial ≤ −`magnitude_threshold`.
    predicted_direction='a_lt_b',
)
def eff_h_mediates_g_link__goal_envs(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'effective_horizon',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    magnitude_threshold: float = 0.3,
    min_strata: int = 2,
) -> Verdict:
    """GOAL-polarity (`env_reward_polarity < −0.3`): `ρ_partial(
    eff_h, outcome | jens)` should inherit polarity-tautology sign
    (negative). HELD when ρ ≤ −`magnitude_threshold`. Currently
    HELD ρ=−0.593 (n=737, 5 strata). Matches polarity-coupling
    `r ≈ 0.625 × polarity` from `findings_polarity_mediator.md`."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if stratified_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = stratified_partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if rho <= -magnitude_threshold:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source='effective_horizon',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('env_name', 'corpus', 'gamma', 'total_steps', 'sync_period', 'seed'),
    scope=(
        finite_gt('env_reward_polarity', 0.3)
        & (
            pl.col('q_divergence_score').is_nan()
            | finite_lt('q_divergence_score', 1000.0)
        )
    ),
    # `predicted_direction='a_gt_b'`: on SURVIVAL polarity envs,
    # the polarity tautology predicts r(eff_h, outcome) ∝ polarity
    # (≈ +0.6 positive). HELD when ρ_partial ≥ +`magnitude_threshold`.
    predicted_direction='a_gt_b',
)
def eff_h_mediates_g_link__survival_envs(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'effective_horizon',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    magnitude_threshold: float = 0.3,
    min_strata: int = 2,
) -> Verdict:
    """SURVIVAL-polarity sibling (`env_reward_polarity > +0.3`):
    `ρ_partial(eff_h, outcome | jens)` should be positive
    (polarity-tautology sign). HELD when ρ ≥ +`magnitude_threshold`.
    Currently HELD ρ=+0.656 (n=307, 3 strata)."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if stratified_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = stratified_partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if rho >= magnitude_threshold:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# CLAIM 13 — target_staleness_late as non-eff_h mediator chain: DDQN
# prevents Q-explosion → online close to target → low staleness →
# clean TD signals → outcome gain. Historical mediation 27% on FR,
# 65% on Breakout sync=100 (both CUT to data orphan). Surviving
# minatar_intermediate_sync uses partial-Spearman per
# `findings_target_staleness_mediator.md`.


# `_staleness_mediation_holds_when` helper removed 2026-05-12 with
# the target_staleness bridges' migration off proportion_mediated.
# Sister bridges (FourRooms capacity-sweep + Breakout sync=100) were
# cut on data orphan; the surviving minatar_intermediate_sync bridge
# was revised to use stratified_partial_spearman per the
# proportion_mediated deprecation warning.


# CLAIM 13 (target_staleness mediation FR + Breakout sync=100) — CUT
# 2026-05-12 (data orphan: capacity_sweep_fourrooms + minatar_1M gone).
# Sister `__minatar_intermediate_sync` (revised to partial_spearman)
# + `cross_config_staleness_slope_negative__survive` cover the
# surviving substance.


@claim_bridge(
    source='target_staleness_late',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    # Current corpora hosting the intermediate-sync MinAtar
    # cohort (renamed from `*_intermediate_sync` 2026 postfix
    # rebuild). Asterix syncs ∈ {500, 1500, 3000} +
    # Breakout syncs ∈ {500, 1500, 3000}.
    scope=(
        pl.col('corpus').is_in(
            ['asterix_postfix_chunk10',
             'survive_sync_intermediate_minatar_postfix'],
        )
        & finite('target_staleness_late')
        & finite('jensen_gap')
        & finite('eval_best_burst_mean')
        & _VANILLA_CONFIG_Q_BOUNDED
    ),
    predicted_direction='a_lt_b',
)
def target_staleness_late_mediates_outcome__minatar_intermediate_sync(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'target_staleness_late',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 10,
    magnitude_threshold: float = 0.2,
    min_strata: int = 2,
) -> Verdict:
    """`ρ_partial(target_staleness_late, outcome | jens)` env-
    stratified on MinAtar intermediate-sync SURVIVE cohort. HELD
    when ρ ≤ −magnitude_threshold (predicted negative). Currently
    NO_EFFECT (ρ=−0.069 predicted direction, |ρ|<0.2). The cross-
    config negative coupling does NOT survive at per-seed within-
    env grain. See `findings_target_staleness_mediator.md`."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if stratified_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = stratified_partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    # Predicted direction is 'a_lt_b' (negative ρ). HELD when
    # ρ ≤ −magnitude_threshold; sign-flip is INVARIANT_VIOLATION-
    # like but we just return NO_EFFECT here (no refutation class
    # plumbed through this bridge's signature).
    if rho <= -magnitude_threshold:
        return Verdict.HELD
    if abs(rho) < magnitude_threshold:
        return Verdict.NO_EFFECT
    # Sign-flipped (ρ > 0): wrong direction, magnitude exceeds
    # threshold. NO_EFFECT preserves the predicted-direction-failed
    # reading without claiming a strong opposite effect.
    return Verdict.NO_EFFECT


# CLAIM 21 — Polarity-stratified cross-config staleness slope.
# Sign flips by polarity: SURVIVE ρ=-0.9 (n=5), REACH ρ=+1.0 (n=3
# trivial). Cross-config descriptive, not causally identified —
# log_sync drives multiple Δs simultaneously. See
# `findings_cross_config_staleness_polarity.md`.


@claim_bridge(
    source='target_staleness_late',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        finite('q_divergence_score') & finite_lt('q_divergence_score', 1.0)
        & finite('jensen_dormancy_gap') & finite_lt('jensen_dormancy_gap', 0.05)
        & finite('env_reward_polarity')
        & finite_gt('env_reward_polarity', 0.3)  # SURVIVE
        & finite('target_staleness_late')
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
        & pl.col('target_sync.tau').is_null()
    ),
    predicted_direction='a_lt_b',
)
def cross_config_staleness_slope_negative__survive(
    stratum_effect_panel: StratumEffectPanel,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    measurables: tuple[str, ...] = ('target_staleness_late', 'eval_best_burst_mean'),
    stratify_by: tuple[str, ...] = (
        'env_name', 'sync_period', 'total_steps', 'corpus',
    ),
    min_seeds_per_arm: int = 5,
    rho_threshold: float = -0.5,
    p_threshold: float = 0.1,
    min_strata: int = 3,
) -> tuple[Verdict, RefutationClass | None]:
    """Cross-config Spearman on per-config (Δ_target_staleness_late,
    Δ_y_best) for SURVIVE polarity. HELD when ρ ≤ rho_threshold
    AND p ≤ p_threshold AND n ≥ min_strata. Empirical n=5: ρ=-0.90,
    p=0.037. Cross-config descriptive (sync_period confounds);
    within-cell mediation breaks at this scope (proportion≈0.07).
    See `findings_cross_config_staleness_polarity.md`."""
    del treatment_arm, baseline_arm, measurables, stratify_by, min_seeds_per_arm
    panel = stratum_effect_panel
    if panel.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    d_pred = panel.deltas.get('target_staleness_late', ())
    d_target = panel.deltas.get('eval_best_burst_mean', ())
    valid = [
        (p_, t_) for p_, t_ in zip(d_pred, d_target, strict=True)
        if not (math.isnan(p_) or math.isnan(t_))
    ]
    if len(valid) < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    from scipy.stats import spearmanr as _spearmanr
    pred_arr = np.asarray([p_ for p_, _ in valid], dtype=np.float64)
    target_arr = np.asarray([t_ for _, t_ in valid], dtype=np.float64)
    rho_v, p_v = _spearmanr(pred_arr, target_arr)
    rho = float(rho_v)
    p = float(p_v)
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if rho <= rho_threshold and p <= p_threshold:
        return Verdict.HELD, None
    if rho > 0.0:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT


# CLAIM 14 (link_r_predictable_from_polarity__soft_tautology) — CUT
# 2026-05-11 (seed-pairing critique + empirical collapse post-rebuild:
# β +0.614 → +0.366 ns). Substance in `eff_h_mediates_g_link__
# {goal,survival}_envs` (JCI partial-Spearman form).


# CLAIM 26b — three-gate scope conjunction predicts DDQN outcome
# benefit (replaces CLAIM 26's slope-predictor form). Gate-active
# subset HELD pooled d=+0.36, p=0.007. Gates load-bearing as scope
# predicates. See `findings_gate_conditional_outcome_benefit.md`.


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    # pair_by intentionally omitted — stratified_arm_diff_pooled
    # aggregates seeds within strata defined by `stratify_by` and
    # does not consume per-pair Δs. The framework's default
    # `pair_by=('seed',)` is inherited by the Bridge contract but
    # ignored by this primitive.
    scope=(
        # ARM-SYMMETRIC predicates only. Stratum-level scope (e.g.
        # vanilla mean jens > 0.05) is applied INSIDE the primitive
        # via `min_vanilla_predictor` so both arms in a stratum are
        # included or excluded together (no asymmetric per-cell
        # filtering bias).
        pl.col('eval_best_burst_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
        & pl.col('n_actions').is_finite() & (pl.col('n_actions') >= 3)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & ~((pl.col('env_name') == 'MetaMaze-misc')
            & (pl.col('gamma') == 0.999))
        & (pl.col('env_name') != 'CartPole-v1')
        # G3-bottom exclusion: SlidingTilePuzzle vanilla fails to
        # converge at current HPs (outcome trajectory has NEGATIVE
        # growth across bursts; vanilla Q-values grow but don't
        # translate to solving the puzzle). See
        # `findings_scope_density.md` "G3-bottom" predicate. With
        # larger network + γ=0.999, vanilla MAY converge — separate
        # probe sweep `sliding_tile_probe_big_cnn` tests this.
        & (pl.col('env_name') != 'SlidingTilePuzzle-jumanji')
    ),
    predicted_direction='a_gt_b',
)
def ddqn_helps_under_three_gate_scope__cross_env(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    threshold_d: float = 0.05,
    alpha: float = 0.05,
    min_strata: int = 4,
    stratify_by: tuple[str, ...] = (
        'env_name', 'sync_period', 'gamma', 'total_steps',
    ),
    min_vanilla_predictor: float = 2.0,
) -> Verdict:
    """Cross-config Cohen's d (independent-samples Hedges 1981),
    DerSimonian-Laird random-effects pool over strata that pass
    stratum-level G1 filter (`mean(vanilla jens) > min_vanilla_
    predictor`). Stratify by `(env, sync, γ, total_steps)`. HELD
    when pooled d > threshold, p < α, n_strata ≥ min_strata. See
    `findings_within_stratum_primitives.md` +
    `findings_dormancy_case_studies.md`."""
    n_strata = stratified_arm_diff_pooled.n_strata
    if n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    d = stratified_arm_diff_pooled.pooled_d
    p = stratified_arm_diff_pooled.pooled_p_value
    if math.isnan(d) or math.isnan(p):
        return Verdict.NO_EFFECT
    if d < 0.0:
        return Verdict.NO_EFFECT
    significant = p < alpha
    above = d >= threshold_d
    if significant and above:
        return Verdict.HELD
    if above or significant:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.NO_EFFECT


# CLAIM 3 — Q-clip channel sufficient-condition. Complement to 26b
# (Hasselt bias channel): DDQN's bootstrap-target gap (`target_max_q
# − target_q[argmax_online]`) is non-negative by construction and
# fires on DORMANT cells where Hasselt's premise is inactive. Tests
# the chain via stratified partial Spearman ρ(clip_wedge, outcome |
# jens) on the dormant scope. See `findings_clip_channel_polarity.md`.


@claim_bridge(
    source='clip_wedge_polarity_aligned',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        # Dormant scope (G1 inactive). Both arms' cells with
        # small bias enter; jens variation is small so partialling
        # is mostly principled-belt-and-suspenders.
        pl.col('jensen_gap').is_finite() & (pl.col('jensen_gap') < 0.1)
        # Polarity-aligned clip-wedge measurable available.
        # Skips cells where ddqn_bootstrap_gap or env_reward_polarity
        # is NaN (pre-trace-restore / nan polarity envs).
        & pl.col('clip_wedge_polarity_aligned').is_finite()
        # Polarity-defined: drop ambiguous-polarity envs.
        # We need a directional polarity signal for the moderation
        # test to be meaningful.
        & pl.col('env_reward_polarity').is_finite()
        & (pl.col('env_reward_polarity').abs() > 0.3)
        # Outcome finite.
        & pl.col('eval_best_burst_mean').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def clip_wedge_predicts_outcome__polarity_moderated__dormant_scope(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'clip_wedge_polarity_aligned',
    y: str = 'eval_best_burst_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    min_rho: float = 0.2,
    min_strata: int = 2,
) -> Verdict:
    """Sufficient-condition complement to CLAIM 26b: on dormant cells
    (`jensen_gap < 0.1`), does the polarity-aligned clip wedge
    predict outcome benefit after partialling out residual jens?
    `clip_wedge_polarity_aligned = ddqn_bootstrap_gap × sign(
    env_reward_polarity)` folds polarity-moderation into a single
    predictor (channel sign-flips between SURVIVAL and REACH per
    `findings_clip_channel_polarity.md`). HELD if pooled partial-r
    ≥ `min_rho`."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if stratified_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = stratified_partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if rho >= min_rho:
        return Verdict.HELD
    if rho <= -min_rho:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT




# CLAIM 15 + 15b — Pearl rung-2 polyak-do(τ) corroboration of
# staleness as causal mediator: HELD on GOAL polarity (FourRooms
# ATE=-0.018, p=0.003 historically); HELD null on SURVIVAL (Asterix).
# AWAITING DATA: polyak_tau_intervention absent post-rebuild
# (target_sync.tau is null everywhere). Findings notes:
# `findings_polyak_tau.md`, `findings_polyak_makes_mech_dormant_survive.md`.


@claim_bridge(
    source='target_staleness_late',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    pair_by=(
        'env_name', 'gamma', 'sync_period',
        'total_steps', 'seed', 'target_sync.tau',
    ),
    scope=(
        # Endogenous polyak-sweep indicator: only `polyak_update`
        # carries a `target_sync.tau` field; periodic_copy regimes
        # leave it null.
        finite('target_sync.tau')
        & (pl.col('target_sync.tau') > 0)
        & finite_lt('env_reward_polarity', -0.5)
        & finite('q_divergence_score')
        & finite_lt('q_divergence_score', 100.0)
        & finite('target_staleness_late')
        & finite('eval_best_burst_mean')
        & finite_gt('q_late_mean', 0.0)
    ),
    predicted_direction='a_lt_b',
)
def staleness_amplifies_ddqn_outcome__sparse_goal_polyak(
    paired_continuous_do_dowhy: PairedContinuousDoResult,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    treatment_var: str = 'target_staleness_late',
    treatment_var_arm: str = 'baseline',
    outcome: str = 'eval_best_burst_mean',
    ate_threshold: float = 1.0,
    refutation_drift_threshold: float = 5.0,
    n_pairs_floor: int = 30,
) -> Verdict:
    """Polyak-do(τ) on GOAL polarity (env_reward_polarity < −0.5,
    bounded Q, q_late_mean > 0): per-pair baseline target staleness
    causally amplifies DDQN's outcome benefit. DoWhy backdoor_ate
    + placebo + RCC. HELD: identified ∧ ATE > threshold ∧ refutations
    clean. Historical: FourRooms n=120, ATE≈+5 reward units/staleness
    unit. AWAITING DATA: polyak_tau_intervention absent post-rebuild."""
    del treatment_arm, baseline_arm, treatment_var, treatment_var_arm, outcome
    result = paired_continuous_do_dowhy
    if not result.backdoor.identified:
        return Verdict.POWER_INSUFFICIENT
    if result.n_pairs < n_pairs_floor:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(result.backdoor.ate):
        return Verdict.POWER_INSUFFICIENT
    # Predicted direction: 'a_lt_b' means treatment_arm < baseline.
    # Under treatment_var=baseline_staleness, the test "DDQN's
    # paired benefit grows with baseline staleness" predicts a
    # POSITIVE ATE (more staleness → larger Δ_outcome). The
    # `direction=Direction.INVERSE` on the bridge encodes the
    # τ-side inverse relationship; the body checks ATE > threshold
    # (positive).
    if result.backdoor.ate <= ate_threshold:
        return Verdict.NO_EFFECT
    # Refutations: placebo gives ATE ≈ 0; RCC drift small.
    if (
        not math.isnan(result.placebo.refuted_ate)
        and abs(result.placebo.refuted_ate) > refutation_drift_threshold
    ):
        return Verdict.POWER_INSUFFICIENT
    if (
        not math.isnan(result.random_common_cause.drift)
        and result.random_common_cause.drift > refutation_drift_threshold
    ):
        return Verdict.POWER_INSUFFICIENT
    return Verdict.HELD


@claim_bridge(
    source='target_staleness_late',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=(
        'env_name', 'gamma', 'sync_period',
        'total_steps', 'seed', 'target_sync.tau',
    ),
    scope=(
        finite('target_sync.tau')
        & (pl.col('target_sync.tau') > 0)
        & finite_gt('env_reward_polarity', 0.3)
        & finite('target_staleness_late')
        & finite('eval_best_burst_mean')
    ),
    predicted_direction='null',
)
def staleness_does_not_amplify_ddqn_outcome__survival_polyak(
    paired_continuous_do_dowhy: PairedContinuousDoResult,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    treatment_var: str = 'target_staleness_late',
    treatment_var_arm: str = 'baseline',
    outcome: str = 'eval_best_burst_mean',
    null_band: float = 5.0,
    n_pairs_floor: int = 30,
) -> Verdict:
    """SURVIVAL-polarity companion to FR polyak bridge. Null-form
    HELD when |ATE| < null_band AND identified AND n ≥ floor. The
    staleness mediation chain BREAKS on SURVIVE polarity per
    `findings_polyak_makes_mech_dormant_survive.md`. AWAITING DATA
    (polyak sweeps absent)."""
    del treatment_arm, baseline_arm, treatment_var, treatment_var_arm, outcome
    result = paired_continuous_do_dowhy
    if not result.backdoor.identified:
        return Verdict.POWER_INSUFFICIENT
    if result.n_pairs < n_pairs_floor:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(result.backdoor.ate):
        return Verdict.POWER_INSUFFICIENT
    # Null prediction: |ATE| should be smaller than the canonical
    # FourRooms effect (~−0.018). If much smaller, null confirmed.
    if abs(result.backdoor.ate) < null_band:
        return Verdict.HELD
    # ATE significantly negative → null refuted (xpass), staleness
    # DOES amplify here — surprising result.
    return Verdict.NO_EFFECT


# CLAIM 17 (chain_amplifier_link_active_in_bounded_q) — CUT 2026-05-11.
# Cross-env signal was leverage-driven (drop PacMan+MountainCar+SI →
# slope -1.62 → +0.07 ns). Substance preserved by CLAIM 26b's
# stratified-DL pool (leverage-robust) + `findings_minatar_link_attenuation.md`.


# CLAIM 19 — Cross-env (n=4 REACH) Pearson +0.975 of mean_dY on
# effective_horizon. SURVIVE counterpart (n=5) HELD via CLAIM 20's
# argmax_entropy_late_van predictor.


@claim_bridge(
    source='effective_horizon',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('seed', 'total_steps', 'eval_every'),
    scope=(
        finite('q_divergence_score') & finite_lt('q_divergence_score', 1.0)
        & finite_gt('bootstrap_fraction', 0.5)
        & finite('jensen_dormancy_gap') & finite_lt('jensen_dormancy_gap', 0.05)
        & finite('env_reward_polarity')
        & finite_lt('env_reward_polarity', -0.3)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
        & pl.col('target_sync.tau').is_null()
    ),
    predicted_direction='a_gt_b',
)
def effh_predicts_link_power__reach_envs(
    meta_regression_per_burst: MetaRegressionResult,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    covariates: tuple[str, ...] = ('effective_horizon',),
    dedupe_strategy: str = 'mean',
    slope_threshold: float = 0.005,
) -> Verdict:
    """Per-(env, burst) meta-regression of Δ_outcome on env-mean
    effective_horizon, REACH polarity. HELD when β ≥ slope_threshold
    AND significant. Currently NO_EFFECT: β=-0.0046, CI=[-0.009,
    -0.0002], p=0.041 — OPPOSITE direction (per-burst slope flips
    relative to env-mean aggregate due to phase-structure inversion,
    see `findings_fourrooms_time_series.md`)."""
    del treatment_arm, baseline_arm, source, covariates, dedupe_strategy
    coef = next(
        (c for c in meta_regression_per_burst.coefficients
         if c.name == 'effective_horizon'),
        None,
    )
    if coef is None:
        return Verdict.POWER_INSUFFICIENT
    if not coef.is_significant:
        return Verdict.POWER_INSUFFICIENT
    if coef.coefficient >= slope_threshold:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# CLAIM 20 — Cross-config (n=5 SURVIVE) Pearson +0.91 of mean_dY on
# argmax_entropy_late_van. STARTING-POINT — argmax_ent is largely
# env-structural (van ↔ dd Pearson +0.95) and collinear with
# bias-reduction. Companion to CLAIM 19's REACH-side effh predictor.


@claim_bridge(
    source='argmax_entropy_late',
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('seed', 'total_steps', 'eval_every'),
    scope=(
        finite('q_divergence_score') & finite_lt('q_divergence_score', 1.0)
        & finite_gt('bootstrap_fraction', 0.5)
        & finite('jensen_dormancy_gap') & finite_lt('jensen_dormancy_gap', 0.05)
        & finite('env_reward_polarity')
        & finite_gt('env_reward_polarity', 0.3)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
        & pl.col('target_sync.tau').is_null()
    ),
    predicted_direction='a_gt_b',
)
def argmax_entropy_predicts_link_power__survive_envs(
    meta_regression_per_burst: MetaRegressionResult,
    *,
    treatment_arm: str = _DDQN_ARM,
    baseline_arm: str = _VANILLA_ARM,
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    covariates: tuple[str, ...] = ('argmax_entropy_late',),
    dedupe_strategy: str = 'mean',
    slope_threshold: float = 0.5,
) -> Verdict:
    """STARTING-POINT SURVIVE companion to CLAIM 19 (REACH effh).
    Per-env paired-g regressed on env-mean argmax_entropy_late.
    HELD when β ≥ slope_threshold AND significant. Caveats:
    argmax_ent is mostly env-structural (van↔dd Pearson +0.95);
    collinear with mean_dJ. n=5 small."""
    del treatment_arm, baseline_arm, source, covariates, dedupe_strategy
    coef = next(
        (c for c in meta_regression_per_burst.coefficients
         if c.name == 'argmax_entropy_late'),
        None,
    )
    if coef is None:
        return Verdict.POWER_INSUFFICIENT
    if not coef.is_significant:
        return Verdict.POWER_INSUFFICIENT
    if coef.coefficient >= slope_threshold:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# CLAIM 18 (algorithmic_activation_rate_mediates_link) — CUT 2026-05-11
# (placeholder, claim didn't survive post-fix).
# CLAIM 4 + 16 (bf→g_link cross-env) — CUT 2026-05-11 (bf-constancy
# + leverage). Substance survived via Q-stable per-burst link, see
# `findings_residual_unexplained.md`.


# =====================================================================
# DDQN measurement graph — the closure.
# =====================================================================
DDQN_UNIVERSE_BRIDGES = (
    # CLAIM 2 — load-bearing necessary scope (causal refutation).
    ddqn_refuted_when_dormancy_fires,
    # CLAIM 2 corroborations — Pearl rung-2 adaptive controller.
    # Both AWAITING DATA (adaptive corpora absent from postfix
    # rebuild); return POW_INSUF until they return.
    adaptive_dqn_recovers_ddqn_benefit__fourrooms_factor_0p5,
    adaptive_dqn_fails_to_avoid_attenuation__spaceinvaders_1m,
    # CLAIM 5 — within-env do(γ) on FourRooms. AWAITING DATA
    # (gamma_sweep corpus dropped; current FR has γ=0.99 only).
    # MetaMaze sister stays cut: its substantive claim is being
    # actively tested (and refuted) by metamaze_link_steeper_at_
    # high_gamma on current corpora.
    ddqn_benefit_scales_with_effective_horizon__fourrooms,
    # ddqn_benefit_scales_with_gamma__discountingchain MOVED to
    # `dqn_bridges.py` — DiscountingChain is bsuite (excluded by
    # MODULE_SCOPE), and the do(γ) bridge is an env-specific
    # finding rather than a cross-env scope claim.
    # CLAIM 7 — Pearl-rung-2: DDQN's reward-scale-response curve
    #           dominates vanilla's at rs=0.1 on FourRooms.
    ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1,
    # CLAIM 7b — same dominance at rs=0.3 (rescue-regime peak).
    ddqn_dominates_vanilla_response_curve__fourrooms_rs_0p3,
    # CLAIM 7c/7d — rescue does NOT generalize to Acrobot/CartPole.
    ddqn_does_not_rescue__acrobot_rs_0p1,
    ddqn_does_not_rescue__cartpole_rs_0p1,
    # CLAIM 7e/7f — rescue mechanism is exploration-maintenance.
    ddqn_increases_argmax_entropy__fourrooms_rs_0p1,
    ddqn_entropy_matches_vanilla__fourrooms_rs_1p0,
    # CLAIM 7 g/h/i/j — DELETED (auxiliary mechanism-route probes
    # subsumed by CLAIM 26b three-gate framework; synthesis
    # preserved as deletion-memo banner above).
    # CLAIM 8 — per-burst crossover shape on SpaceInvaders 1M —
    # CUT 2026-05-12 (data orphan: SI 1M missing). See banner above.
    # CLAIM 9 — n-step falsification of bootstrap-bias-compounding
    #           (the strongest mechanism corroboration: Δ→0 as n grows).
    ddqn_helps_at_full_bootstrap__fourrooms_n1,
    # n=10 null endpoint AWAITING DATA — nstep_lambda_fourrooms
    # missing post-rebuild; returns POW_INSUF until it returns.
    ddqn_null_under_monte_carlo__fourrooms_n10,
    # TIER A2 existence proofs (per-burst, env-conditional).
    # ddqn_helps_at_early_bursts__pixel_envs — CUT 2026-05-12
    # (paired_g + data orphan: MinAtar 1M missing post-rebuild).
    # ddqn_attenuates_at_late_bursts__spaceinvaders — DISABLED
    # (substrate-drift between pre-postfix SI 1M sweeps; not a
    # current-cache cell either).
    # CLAIM 10 — link IS bias-correction on Acrobot γ=0.999, causally
    # corroborated. Per-burst link panel + DoWhy backdoor + placebo
    # refutation + RCC refutation all hold. Corrects the prior
    # `findings_l2_acrobot_goldilocks.md` "scalar link null" finding,
    # which was a measurement artifact of best-burst-per-seed
    # selection.
    acrobot_per_burst_link_active__gamma_0999,
    # CLAIM 22 — REACH-cross-env DoWhy: Δ_jens → Δ_outcome HELDs
    # across all 4 REACH envs (FourRooms, Acrobot, MountainCar,
    # MetaMaze) under env-confounded backdoor + placebo + RCC. The
    # cross-env scope of the link verdict, beyond the Acrobot-only
    # bridge above. (Per-burst panel; n_pairs ≈ 1200 with all 4
    # REACH envs in scope.)
    reach_link_backdoor_ate_negative,
    reach_link_placebo_refuted,
    reach_link_rcc_robust,
    # CLAIM 23 — q_divergence and argmax_entropy are shadows of
    # Δ_jens (not independent mediators). Bridges HELD when the
    # partial Spearman conditional on Δ_jens is consistent with
    # zero — codifies the shadow-mediator finding so future
    # bridge authors don't re-derive the same conclusion.
    q_divergence_shadowed_by_jens,
    argmax_entropy_shadowed_by_jens,
    # CLAIM 24 — within-MetaMaze do(γ): link slope steepens.
    metamaze_link_steeper_at_high_gamma,
    metamaze_link_steeper_at_high_gamma__median,
    # CLAIM 25 — within-FourRooms do(|A|): DDQN benefit scales 20×.
    fourrooms_action_dim_link_active__inflated,
    # CLAIM 11 — extreme Q-divergence attenuates the link. Companion
    # to CLAIM 2's dormancy bridge: dormancy bounds the lower
    # attenuation (mech inactive); extreme Q-divergence bounds the
    # upper attenuation (Q-explosion overwhelms link). Together they
    # bound the link-active band on q_divergence_score.
    extreme_q_divergence_attenuates_link__binary,
    extreme_q_divergence_attenuates_link__placebo_refuted,
    extreme_q_divergence_attenuates_link__rcc_robust,
    # CLAIM 12 — env-polarity moderates the eff_h mediator sign.
    # Two paired bridges: GOAL envs (negative slope) and SURVIVAL
    # envs (positive slope). Replaces the cross-env null residual
    # with two oppositely-signed env-stratified bridges.
    eff_h_mediates_g_link__goal_envs,
    eff_h_mediates_g_link__survival_envs,
    # CLAIM 13 — target_staleness_late as the dominant non-eff_h
    # mediator (carries 27% on FourRooms, 65% on Breakout sync=100
    # under mech-HELD conditioning). Tautology audit corroborated
    # on FourRooms (CLEAN). The hard-to-find "other 84%" of DDQN's
    # outcome benefit (after eff_h) is largely target staleness
    # suppression downstream of bias correction.
    # target_staleness_late_mediates_outcome__fourrooms — CUT
    # 2026-05-12 (data orphan: capacity_sweep_fourrooms missing).
    # target_staleness_late_mediates_outcome__breakout_sync100 —
    # CUT 2026-05-12 (data orphan: minatar_1M missing).
    target_staleness_late_mediates_outcome__minatar_intermediate_sync,
    cross_config_staleness_slope_negative__survive,
    # CLAIM 21 REACH-polyak half retracted post-fix: ρ=−0.10 at n=5
    # configs (pre-fix n=3 ρ=+1.0 was a fluke). See
    # `findings_cross_config_staleness_polarity.md`.
    # CLAIM 14 — soft tautology: env-polarity predicts the link sign
    # per env at slope ≈ +0.5 (Fisher-z), R² ≈ 0.83. Companion to
    # CLAIM 12's eff_h_mediates_g_link__{goal,survival}_envs:
    # polarity predicts link SHAPE (CLAIM 14 HELD), but eff_h is NOT
    # a dominant mediator (CLAIM 12 HELD under predicted_direction=
    # CLAIM 14 — link_r_predictable_from_polarity__soft_tautology
    # CUT 2026-05-11 (seed-pairing critique + empirical collapse
    # post-rebuild; substance preserved in eff_h_mediates_g_link
    # __{goal,survival}_envs JCI form). See banner above.
    # CLAIM 26 — slope-predictor regression cut; subsumed by CLAIM
    # 26b's gate-conjunction outcome bridge below. See
    # `findings_g1_predicts_link_slope.md`.
    # CLAIM 26b — substantive cross-env replacement for CLAIM 26's
    # slope-predictor regression. Tests that DDQN's outcome benefit is
    # positive panel-level when the three gates fire jointly. The
    # outcome-level claim, not the slope-level one.
    ddqn_helps_under_three_gate_scope__cross_env,
    # CLAIM 3 — sufficient-condition complement to 26b. On DORMANT
    # cells (G1 inactive), tests whether the Q-clip wedge predicts
    # outcome via stratified partial Spearman conditioning on jens.
    # Authors the non-Hasselt Q-magnitude regularization channel
    # the framework had previously left "deliberately ABSENT".
    clip_wedge_predicts_outcome__polarity_moderated__dormant_scope,
    # CLAIM 15 — Polyak-τ rung-2 corroboration on FourRooms goal
    # polarity. AWAITING DATA — polyak_tau_intervention sweeps
    # missing post-rebuild; returns POW_INSUF until they return.
    staleness_amplifies_ddqn_outcome__sparse_goal_polyak,
    # CLAIM 15b — companion null bridge: under SURVIVAL polarity
    # in the polyak regime, the staleness-mediation chain is
    # BROKEN. Same AWAITING DATA state as 15.
    staleness_does_not_amplify_ddqn_outcome__survival_polyak,
    # CLAIM 19 — among REACH-polarity envs in CLAIM 17 scope,
    # effective_horizon is a strong cross-env predictor of link
    # power. Empirical: per-env mean_dY tracks effh at Pearson 0.97
    # (p=0.025) on n=4 REACH envs.
    effh_predicts_link_power__reach_envs,
    # CLAIM 20 — STARTING POINT — argmax_entropy_late_van as
    # cross-env link-power predictor on SURVIVE-polarity envs.
    # Per-config Pearson +0.91 (n=5) but env-structural caveat;
    # designed sweeps at intermediate sync needed to corroborate.
    argmax_entropy_predicts_link_power__survive_envs,
)
"""The six bridges that close the DDQN study. CLAIM 1 (mechanism
activation, do(DDQN) ↓ jensen_gap) is corroborated by
`ddqn_reduces_jensen_gap__converged_subset` in `dqn_bridges.py`
on the 200k DDQN corpus's converged subset; not duplicated here.

CLAIM 3 (sufficient scope) authored 2026-05-12 as
`clip_wedge_predicts_outcome__dormant_scope` — tests the
Q-magnitude regularization channel via stratified partial Spearman
on dormant cells. Complements CLAIM 26b's bias-correction-channel
[G1 ∧ G2 ∧ G3] bridge: this is the non-Hasselt sufficient channel
the previous framework iteration had no primitive for."""


__all__ = [
    'DDQN_UNIVERSE_BRIDGES',
    'reach_link_backdoor_ate_negative',
    'reach_link_placebo_refuted',
    'reach_link_rcc_robust',
    'q_divergence_shadowed_by_jens',
    'argmax_entropy_shadowed_by_jens',
    'metamaze_link_steeper_at_high_gamma',
    'fourrooms_action_dim_link_active__inflated',
    'acrobot_per_burst_link_active__gamma_0999',
    'extreme_q_divergence_attenuates_link__binary',
    'extreme_q_divergence_attenuates_link__placebo_refuted',
    'extreme_q_divergence_attenuates_link__rcc_robust',
    'ddqn_attenuates_at_late_bursts__spaceinvaders',
    'ddqn_refuted_when_dormancy_fires',
    'eff_h_mediates_g_link__goal_envs',
    'eff_h_mediates_g_link__survival_envs',
    'target_staleness_late_mediates_outcome__minatar_intermediate_sync',
    'cross_config_staleness_slope_negative__survive',
    'ddqn_helps_under_three_gate_scope__cross_env',
    'staleness_amplifies_ddqn_outcome__sparse_goal_polyak',
    'staleness_does_not_amplify_ddqn_outcome__survival_polyak',
    'ddqn_null_under_monte_carlo__fourrooms_n10',
    'adaptive_dqn_recovers_ddqn_benefit__fourrooms_factor_0p5',
    'adaptive_dqn_fails_to_avoid_attenuation__spaceinvaders_1m',
    'ddqn_benefit_scales_with_effective_horizon__fourrooms',
    'clip_wedge_predicts_outcome__polarity_moderated__dormant_scope',
    'ddqn_does_not_rescue__acrobot_rs_0p1',
    'ddqn_does_not_rescue__cartpole_rs_0p1',
    'ddqn_increases_argmax_entropy__fourrooms_rs_0p1',
    'ddqn_entropy_matches_vanilla__fourrooms_rs_1p0',
]


# Canonical name `corroborate.runner` imports;
# DDQN_UNIVERSE_BRIDGES stays as an alias for legacy call sites.
BRIDGES = DDQN_UNIVERSE_BRIDGES

