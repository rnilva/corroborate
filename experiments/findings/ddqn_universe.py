"""DDQN measurement graph — final closure.

Three load-bearing causal claims, audited for predicate
endogeneity. Authored against the universal paired-delta
datasets (cell-mean and per-burst), the existing 200k DDQN
corpus's converged subset, and a Pearl-rung-2 designed
intervention sweep on FourRooms.

Tier framework (refinement of the framework's existing
ASSOCIATIONAL/INTERVENTIONAL):

  TIER A1 — universal exogenous predicates (env-feature / time /
            HP); claims generalize to envs we haven't measured.
  TIER A2 — sampled exogenous predicates (env_name); existence
            proofs over our specific benchmark sample.
  TIER INT — Pearl-rung-2: claim backed by a designed
            intervention (the adaptive controller sweep).
  TIER B  — control-trajectory-endogenous predicates; descriptive
            only, not actionable. NOT EXPORTED HERE; live in
            `dqn_bridges.py` zoo.

# 1. MECHANISM (universal causal):
#    do(arm=ddqn) ↓ jensen_gap on premise-active envs.
#    Bridge: `ddqn_reduces_jensen_gap__converged_subset` — lives
#    in `dqn_bridges.py`. Multi-env paired g pooled across the
#    convergence-conditioned 6-env subset; HELD with g=−0.93.
#    Universal across our benchmark (TIER A1).

# 2. NECESSARY SCOPE (causal refutation, load-bearing claim):
#    dormancy_gap > 0 ⇒ DDQN does NOT help on outcome.
#    Bridge: `ddqn_refuted_when_dormancy_fires`.
#    σ_Q × √(2 log |A|) is the Hasselt-2010 structural Jensen
#    floor — when observed bias falls below it, DDQN's
#    correction has nothing to bite on. Helped fraction drops
#    from baseline 31% to 7.1% across 394 dormant cells.
#    Pearl-rung-2 corroboration: `adaptive_dqn_recovers_ddqn_
#    benefit__fourrooms_factor_0p5` — a designed intervention
#    sweep where DDQN's per-batch greedification dispatches via
#    the dormancy heuristic — recovers DDQN's outcome benefit on
#    FourRooms (g=+0.78 vs vanilla, p<0.001, n=30).

# 3. NO SUFFICIENT SCOPE (negative result, framework-honest):
#    No exogenous predicate gets helped% above ~57%. The link
#    from mechanism to outcome is irreducibly env+time
#    conditional. Documented as a non-claim — included here to
#    fix the closure shape but NOT authored as a bridge.

# 4. INDEPENDENT LINK-SIDE SCOPE (residual unexplained-by-dormancy):
#    Bridge: `bootstrap_fraction_drives_g_link__net_of_dormancy`.
#    Panel-level meta-regression of g_link on
#    {log_action_dim, log_obs_dim, log_horizon, bootstrap_fraction,
#    dormancy_env_mean} on the DDQN 200k corpus shows
#    β(bootstrap_fraction) = +2.716 (z=+3.27, p=0.0012) — a
#    robust positive moderator of DDQN's outcome benefit on the
#    LINK edge, INDEPENDENT of dormancy on the mechanism edge.
#    Theoretically: high bootstrap_fraction = long bootstrap
#    chains = bigger per-step bias amplification = larger DDQN
#    outcome payoff. CLAIM 2 (dormancy on mechanism) and CLAIM 4
#    (bootstrap_fraction on link) operate on DIFFERENT edges of
#    the chain — neither subsumes the other.

The two env-conditional helper bridges below are TIER A2
existence proofs: per-env temporal-window claims that
generalize WITHIN their env at this HP regime but not across
envs structurally.
"""
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
from corroborate.analyses.mundlak_decomposition import MundlakResult
from corroborate.analyses.paired_delta_link_dowhy import (
    PairedDeltaLinkDowhyResult,
)
from corroborate.analyses.paired_g import PairedGResult
from corroborate.analyses.paired_g_per_burst import PerBurstResult
from corroborate.analyses.paired_link_per_burst import (
    PerBurstLinkResult, phase_link_consistency,
)
from corroborate.bridge.bridge import (
    Direction, Tier, claim_bridge,
)
from corroborate.core.intervention import DoEffect
from corroborate.measurables import Measurable
from corroborate.stats import MetaRegressionResult
from corroborate.measurables.reductions import from_key, reduce_axis
from corroborate_rl.dqn.measurables import jensen_bias_per_eps
from corroborate.bridge.verdict import Verdict


# Composed per-burst reductions consumed by bridges below. Module-
# level so each bridge that wants the canonical "per-burst-mean"
# shape can reference the same Measurable instance (auto-registered
# at @claim_bridge decode time, so the runner's transitive_reads
# walks pick up `mc_return` / `predicted_q_at_start` for free).
_MC_RETURN_PER_BURST_MEAN: Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
] = reduce_axis(from_key('mc_return'), axis=-1, op='mean')
_JENSEN_BIAS_PER_BURST_MEAN: Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
] = reduce_axis(jensen_bias_per_eps, axis=-1, op='mean')


# File-level intervention: every bridge in this module tests
# `do(arm=ddqn) → effect`. Bridges that test a DIFFERENT
# intervention override via decorator `source = DoEffect(...)`.
INTERVENTION = DoEffect(treatment_arm='ddqn', baseline_arm='vanilla_dqn')


# =====================================================================
# Per-env covariate table for the DDQN 200k corpus's 18 envs.
#
# Structural covariates (log_action_dim, log_obs_dim, log_horizon)
# come straight from the env spec — these are theory-known
# pre-treatment features.
#
# `bootstrap_fraction` (1 − mean(done) over vanilla_dqn cells of
# that env, averaged across 30 seeds) and `dormancy_env_mean`
# (per-env mean of the universal-dataset `dormancy_gap_avg`) are
# corpus-empirical. Both are computed once on the 200k DDQN runs
# and frozen here so the bridge is reproducible without re-
# touching traces.
# =====================================================================


# Static covariates per env are no longer baked inline. The
# bridges that use env-level features (bootstrap_fraction,
# log_action_dim, log_obs_dim, log_horizon, jensen_dormancy_gap)
# now reference them as column NAMES; the meta-regression
# analysis groups by env and computes per-env means from the
# materialised cache columns at evaluation time. See the
# `bootstrap_fraction_drives_g_link__net_of_dormancy` bridge.


# =====================================================================
# CLAIM 2 — Necessary scope (load-bearing dormancy refutation).
# =====================================================================


@claim_bridge(
    # Decorator declares the do-contrast (vanilla → ddqn) on the
    # OUTCOME column. The graph edge `jensen_dormancy_gap →
    # eval_best_burst_mean` (mech-state predicate of refutation
    # → outcome) lives in the docstring/scope rather than the
    # source field — the source field now carries the contrast
    # exclusively. paired_g computes Δ on `target=eval_best_burst_mean`
    # under the file's INTERVENTION, which IS the actual refutation
    # predicate; the dormancy-gap is consumed by `scope` as a cell
    # filter, not as paired_g's measurement axis.
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('seed', 'env_name'),
    # `is_finite()` excludes both null and NaN — required because
    # polars admits NaN through `>=` comparisons (NaN >= 1e-9 → True
    # in polars's filter context, unlike IEEE-754 semantics). Cells
    # with NaN `jensen_dormancy_gap` are "couldn't evaluate" and
    # should drop, not pass.
    scope=(
        pl.col('jensen_dormancy_gap').is_finite()
        & (pl.col('jensen_dormancy_gap') >= 1e-9)
    ),
)
def ddqn_refuted_when_dormancy_fires(
    paired_g: PairedGResult,
    *,
    dedupe_strategy: str = 'mean',
) -> Verdict:
    """Necessary-condition claim. The framework's-own Jensen
    dormancy invariant operationalizes the Hasselt-2010 structural
    floor `σ_Q × √(2 log |A|)` against observed bias. When the gap
    fires (gap > 0, premise dormant), DDQN's bias-correction
    mechanism has nothing to operate on.

    HELD when helped_fraction ≤ 0.15 AND |g| ≤ 0.20 — the
    refutation prediction is corroborated (DDQN does NOT help
    outcome on dormant cells). INVARIANT_VIOLATION when DDQN
    unexpectedly helps despite dormancy (helped > 0.40).

    The Pearl-rung-2 corroboration via `adaptive_dqn_recovers_
    ddqn_benefit__fourrooms_factor_0p5` validates this as
    actionable: a runtime controller using a per-batch dormancy
    proxy (max_Q − mean_Q vs σ_Q × √(2 log |A|)) recovers DDQN's
    outcome benefit on FourRooms (g=+0.78 vs vanilla, p<0.001)."""
    del dedupe_strategy  # forwarded to paired_g
    if paired_g.n_pairs < 50:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.helped_fraction):
        return Verdict.POWER_INSUFFICIENT
    if (
        paired_g.helped_fraction <= 0.15
        and abs(paired_g.g) <= 0.20
    ):
        return Verdict.HELD
    if paired_g.helped_fraction > 0.40:
        return Verdict.INVARIANT_VIOLATION
    return Verdict.NO_EFFECT


# =====================================================================
# CLAIM 2 — Pearl-rung-2 corroboration of the necessary-scope claim.
# Designed-intervention sweep `adaptive_dqn_fourrooms_sweep` runs the
# adaptive controller (`adaptive_dormancy_greedify` with
# `sigma_floor_factor=0.5`) on FourRooms-misc, paired against the
# existing `expectile_3way` FourRooms vanilla_dqn + ddqn cells.
# =====================================================================


@claim_bridge(
    source=DoEffect(treatment_arm='adaptive_dqn_factor_0p5', baseline_arm='vanilla_dqn'),
    target='eval_final_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    # Pearl-rung-2 pairing: adaptive_dqn cells from the
    # designed-intervention sweep (`adaptive_dqn_fourrooms_sweep`,
    # which carries no vanilla_dqn arm) against the existing
    # `expectile_3way` FourRooms vanilla_dqn cohort (rs≈0.1, 200k
    # steps, comparable eval scale). Without this corpus filter,
    # `paired_g` pools across 7 other vanilla_dqn FourRooms-misc
    # cohorts at unrelated reward scales (e.g. `reward_scale_sweep`
    # at eval_mean=2.74 vs. expectile_3way at 0.56), inflating
    # `mean_diff` by an order of magnitude and silently
    # misclassifying a refutation as no_effect.
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
    """Pearl-rung-2 designed-intervention bridge. The adaptive
    controller (`adaptive_dormancy_greedify` with
    sigma_floor_factor=0.5) dispatches per-batch between vanilla
    `max_greedify` and DDQN `double_greedify` based on the in-
    batch dormancy proxy `max_Q − mean_Q ≥ 0.5 × σ_Q ×
    √(2 log |A|)`.

    The claim: a runtime controller built FROM the framework's
    own dormancy invariant recovers DDQN's outcome benefit on
    FourRooms — corroborates the dormancy-as-necessary-condition
    claim at Pearl rung-2.

    HELD when g(adaptive − vanilla) ≥ +0.50 AND p < 0.05.
    Empirically: g=+0.78 (final), p<0.001, n=30 paired seeds.
    The mean_jensen on adaptive (0.074) tracks DDQN (0.084), not
    vanilla (0.362), confirming the controller engages DDQN at
    the right batches.

    Auxiliary observation (NOT load-bearing): adaptive trends
    slightly ABOVE pure DDQN (g(adaptive − ddqn) = +0.21 on
    final mean), but the test underpowers at n=30 (p=0.25).
    The trend is consistent with "occasional vanilla fallback
    on dormant batches strictly helps", but a tighter claim
    needs a wider seed budget. Tracked as a non-claim here."""
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


@claim_bridge(
    source=INTERVENTION,
    target='mc_return_first_quarter',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('seed', 'env_name'),
    scope=(
        (pl.col('log_obs_dim') >= 5.0)
        & (pl.col('total_steps') >= 1000000.0)
        & pl.col('reward_clip_min').is_null()
        & (pl.col('sync_period') == 100)
    ),
)
def ddqn_helps_at_early_bursts__pixel_envs(
    paired_g: PairedGResult,
    *,
    dedupe_strategy: str = 'mean',
) -> Verdict:
    """TIER A2 existence proof: at the first eval-burst quarter
    on long-horizon high-obs-dim envs (MinAtar 1M), DDQN's
    outcome delta is positive in the majority of cells with
    substantial pooled effect. Per-cell helped=56.7%, g=+0.30,
    n=120 in the original analysis.

    Generalizes within the MinAtar 1M sample, not across all
    high-obs-dim envs (log_obs_dim alone is not predictive —
    see K1 LOO + four-MinAtar comparison)."""
    del dedupe_strategy  # forwarded to paired_g
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.helped_fraction):
        return Verdict.POWER_INSUFFICIENT
    if (
        paired_g.helped_fraction >= 0.55
        and paired_g.g >= 0.20
    ):
        return Verdict.HELD
    return Verdict.NO_EFFECT


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
    scope=(
        (pl.col('env_name') == 'SpaceInvaders-MinAtar')
        & (pl.col('total_steps') >= 1000000.0)
        & pl.col('reward_clip_min').is_null()
        & (pl.col('sync_period') == 100)
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
    """TIER A2 existence proof: on SpaceInvaders-MinAtar at 1M
    training steps, in the last quarter of training bursts,
    DDQN's outcome is reliably WORSE than vanilla.

    Consumes `paired_g_per_burst` — a per-(env, burst) panel where
    each stratum holds (g, n_pairs, helped_fraction) computed from
    seed-paired Δ in that burst. With `pair_by=('seed', 'corpus')`,
    independent sweeps probing the same nominal regime
    (SpaceInvaders 1M, sync=100, no reward clip) contribute
    distinct paired observations rather than being silently
    averaged: each sweep's seed=N is its own RNG realization.

    Filters strata to `env_name == 'SpaceInvaders-MinAtar'` AND
    `burst_index >= burst_floor`, then aggregates: `n_pairs_total
    = Σ n_pairs` and sample-size-weighted means for
    `helped_fraction` and `g`.

    HELD when (a) cell-burst total ≥ `n_pairs_floor` (=50), AND
    (b) pooled helped_fraction ≤ `helped_ceiling` (=0.40), AND
    (c) pooled g ≤ `g_ceiling` (=−0.30).

    Env-specific: other long-horizon high-obs-dim envs (Asterix,
    Breakout, Freeway at 1M) do NOT show the same late
    attenuation (K1 audit). SpaceInvaders is the canonical
    example, not an instance of a structural law. The dormancy
    invariant captures this regime more cleanly via CLAIM 2
    (necessary scope) — when Q-network explodes at 1M,
    observed_bias rises faster than σ_Q × √(2 log |A|), so
    dormancy doesn't fire on this proxy and DDQN keeps engaging
    counterproductively. Pearl-rung-2 corroboration of the
    dormancy-blind-spot reading: bridge
    `adaptive_dqn_fails_to_avoid_attenuation__spaceinvaders_1m`
    runs the dormancy controller on this regime; it tracks DDQN
    (g≈0) and inherits the attenuation (g=−0.46 vs vanilla)."""
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


# =====================================================================
# CLAIM 2 — Scope limitation of the dormancy-aware controller.
# Designed-intervention sweep `adaptive_dqn_spaceinvaders_1m` runs
# the same `adaptive_dormancy_greedify(sigma_floor_factor=0.5)`
# controller on SpaceInvaders-MinAtar at 1M training steps.
# Paired against the existing `minatar_1M` SpaceInvaders cells.
# =====================================================================


@claim_bridge(
    source=DoEffect(treatment_arm='adaptive_dqn', baseline_arm='vanilla_dqn'),
    target='eval_final_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    # Pearl-rung-2 pairing: adaptive_dqn from the 1M-step
    # designed-intervention sweep (`adaptive_dqn_spaceinvaders_1m`)
    # against the existing minatar_1M SpaceInvaders vanilla_dqn
    # cells. Other corpora carry vanilla_dqn at SpaceInvaders-MinAtar
    # at different `total_steps` regimes (50k in `ddqn`, 200k in
    # `ddqn_effective_cohort`) which `pair_by=('seed',)` would
    # silently pool with the 1M cells.
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
    """Pearl-rung-2 scope-limitation bridge. The same dormancy-
    aware controller that recovers DDQN's benefit on FourRooms
    FAILS on SpaceInvaders-MinAtar at 1M training steps — the
    per-batch dormancy proxy at sigma_floor_factor=0.5 doesn't
    fire on this regime, so the controller is indistinguishable
    from pure DDQN (g ≈ 0) and inherits DDQN's late-burst
    attenuation.

    Auxiliary observation (empirical, also paired n=30):
      g(adaptive vs ddqn) = −0.06, p=0.74  → controller ≡ DDQN
      g(ddqn vs vanilla)  = −0.44, p=0.022 → DDQN itself hurts
    The shape "adaptive_dqn ≡ ddqn ∧ ddqn < vanilla" implies
    "adaptive_dqn < vanilla" by transitivity. This bridge tests
    that downstream chain.

    HELD when g(adaptive vs vanilla) ≤ −0.30 AND p < 0.05 — the
    scope-limitation prediction is corroborated. Empirically:
    g=−0.46, p=0.016, n=30. INVARIANT_VIOLATION when adaptive
    unexpectedly RECOVERS the benefit (g ≥ +0.30, p<0.05) — which
    would refute the dormancy-blind-spot reading.

    Causal reading: this bridge corroborates `ddqn_attenuates_at_
    late_bursts__spaceinvaders` from the OTHER side. Not only does
    DDQN hurt on this regime — the dormancy invariant cannot tell
    you to stop. The structural Hasselt floor σ_Q × √(2 log |A|)
    is a NECESSARY condition for predicting where DDQN helps
    (CLAIM 2) but not a SUFFICIENT one — long-horizon pixel envs
    have late-burst failure modes the floor doesn't capture.

    Together with CLAIM 2's FourRooms recovery: the controller's
    actionable scope is "envs where dormancy fires and DDQN's
    bias-correction has bias to bite on", not "all DDQN-hurt
    envs"."""
    if paired_g.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.g):
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g <= -0.30 and paired_g.p_value < 0.05:
        return Verdict.HELD
    if paired_g.g >= 0.30 and paired_g.p_value < 0.05:
        return Verdict.INVARIANT_VIOLATION
    return Verdict.NO_EFFECT


# =====================================================================
# CLAIM 5 — Effective-horizon scope (Pearl-rung-2 designed γ sweep).
#
# Within FourRooms (held bf ≈ const), varying γ ∈ {0.99, 0.95, 0.90}
# directly varies effective_horizon = 1/(1−γ·bf). DDQN's outcome
# benefit collapses ~25× as effective horizon shrinks 7×:
#
#   γ=0.99 (eff_h=72): mean Δ_outcome_best = +0.20, g=+1.11
#   γ=0.95 (eff_h=19): mean Δ_outcome_best = +0.02, g=+0.56
#   γ=0.90 (eff_h=10): mean Δ_outcome_best ≈ 0,    g=+0.27 (ns)
#
# Effective horizon is the within-env scope predicate the env-level
# bootstrap_fraction signal hides. Source: gamma_sweep designed
# intervention (3 envs × 3 γ × 2 arms × 30 seeds, do(γ)).
# =====================================================================


@claim_bridge(
    # gamma_sweep stamps γ into the arm name (ddqn_g090 / _g095 /
    # _g099); the high-effective-horizon scope picks γ=0.99, so the
    # contrast is the γ=0.99 pair specifically. File-level
    # INTERVENTION's plain `ddqn`/`vanilla_dqn` doesn't match any
    # cell here.
    source=DoEffect(treatment_arm='ddqn_g099', baseline_arm='vanilla_dqn_g099'),
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('corpus') == 'gamma_sweep')
        & (pl.col('effective_horizon') >= 50.0)
    ),
)
def ddqn_benefit_scales_with_effective_horizon__fourrooms(
    paired_g: PairedGResult,
) -> Verdict:
    """Pearl-rung-2 designed-γ-intervention bridge. Filters
    gamma_sweep's FourRooms cells to the high-effective-horizon
    cohort (γ=0.99, eff_h≈72) and asserts DDQN's benefit is
    substantially positive there. Falsification companion is
    implicit in the scope: at low effective_horizon (γ=0.90,
    eff_h≈10) the same comparison is null, demonstrating the
    scope predicate at the within-env level.

    The within-env γ-axis variation rules out env-confounders
    that bf-residual-on-g_link couldn't: bf ≈ 0.996 on every cell
    here, so any monotone relationship with effective_horizon
    must be γ-driven, hence intervention-causal under do(γ).

    Causal decomposition (do(γ) on FourRooms, n=30 per γ):
      γ=0.99: g_mech=−2.12, g_link=+1.11
      γ=0.95: g_mech=−1.71, g_link=+0.56
      γ=0.90: g_mech=−3.41, g_link=+0.27
    The MECHANISM (DDQN→↓jensen_gap) is large + invariant in γ;
    only the LINK collapses. So effective_horizon moderates the
    amplification path from per-step bias-reduction to integrated
    outcome — NOT the mechanism arrow itself.

    Theoretical reading: V(s) = Σₖ (γ·bf)ᵏ · per_step_diff, so a
    per-step DDQN correction ε integrates to ε/(1−γ·bf) =
    ε · effective_horizon. The integrated link strength is
    linearly proportional to effective_horizon. CLAIM 4
    (bf-on-link) and CLAIM 5 (γ-on-link) thus unify as the same
    structural moderator.

    Falsification + extension probes (2026-05-01, gamma_sweep_
    fourrooms_low + gamma_sweep_metamaze_high):
      FourRooms γ=0.80 → eff_h=5: g_link=0.00 (ns) ✓ falsifies
      FourRooms γ=0.50 → eff_h=2: g_link=+0.22 (ns) ✓ falsifies
      MetaMaze γ=0.999 → eff_h=20: g_link=+0.40, p=.034 ✓ activates
    The chain-depth amplifier IS portable across MLP-friendly
    envs once eff_h ≥ ~20; FourRooms-specificity caveat from the
    earlier MetaMaze-at-γ=0.99 null is retracted.

    Within-env Spearman (whole per-burst dataset):
    ρ(delta_bias, delta_mc | env) = −0.503,
    partial ρ | mc_progress = −0.440 — mechanism→link causal
    edge is robust to saturation control across most envs.

    HELD when helped_fraction ≥ 0.55 AND g ≥ 0.30 on the
    high-eff_h subset. NO_EFFECT or INVARIANT_VIOLATION otherwise."""
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


@claim_bridge(
    # gamma_sweep_metamaze_high stamps γ into the arm name
    # (ddqn_g0995 / ddqn_g0999); the asserted activation is at
    # γ=0.999 (eff_h≈1000 in this corpus's effective_horizon
    # encoding), so contrast the γ=0.999 pair specifically.
    source=DoEffect(
        treatment_arm='ddqn_g0999', baseline_arm='vanilla_dqn_g0999',
    ),
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'MetaMaze-misc')
        & (pl.col('corpus') == 'gamma_sweep_metamaze_high')
        & (pl.col('effective_horizon') >= 18.0)
    ),
)
def ddqn_benefit_scales_with_effective_horizon__metamaze_high_gamma(
    paired_g: PairedGResult,
) -> Verdict:
    """Portability probe — chain-depth-amplifier activates on
    MetaMaze when γ pushed to 0.999 (eff_h≈20). The earlier
    MetaMaze-at-γ=0.99 null was just eff_h-too-low; pushing γ
    higher activates the same signature, refuting the
    FourRooms-specificity caveat.

    Empirical (gamma_sweep_metamaze_high, n=30):
      γ=0.999, eff_h≈20: g_link=+0.40, p=0.034 ✓
      γ=0.995, eff_h≈18: g_link=+0.10 (ns)

    Combined with the FourRooms low-γ truncation probe
    (g_link=0 at eff_h<10), this brackets the operating range
    of the chain-depth-amplifier: ~10-20 effective steps minimum.

    HELD when helped_fraction ≥ 0.45 AND g ≥ 0.20."""
    if paired_g.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.helped_fraction):
        return Verdict.POWER_INSUFFICIENT
    if (
        paired_g.helped_fraction >= 0.45
        and paired_g.g >= 0.20
    ):
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    # gamma_sweep_more stamps γ into the arm name (ddqn_g090 /
    # _g095 / _g099); the high-γ scope picks γ=0.99, so contrast
    # the γ=0.99 pair specifically.
    source=DoEffect(treatment_arm='ddqn_g099', baseline_arm='vanilla_dqn_g099'),
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'DiscountingChain-bsuite')
        & (pl.col('corpus') == 'gamma_sweep_more')
        & (pl.col('gamma') >= 0.985)
    ),
)
def ddqn_benefit_scales_with_gamma__discountingchain(
    paired_g: PairedGResult,
) -> Verdict:
    """Pearl-rung-2 do(γ) bridge on DiscountingChain. Filters
    gamma_sweep_more's DiscountingChain cells to γ=0.99 only and
    asserts DDQN's benefit is positive there. The companion γ<0.99
    cells show null effect.

    Empirical (n=30 per γ on DiscountingChain):
      γ=0.99: g_link=+0.41 (p=.03), g_mech=−1.03 (p<.0001)
      γ=0.95: g_link=0,  g_mech=−0.19 (p=.30)
      γ=0.90: g_link=0,  g_mech=−0.08 (p=.67)

    Different from FourRooms: BOTH the mechanism AND the link
    weaken with γ here. Consistent with the published
    "γ shrinks bias itself" story (Lan et al. 2022; Amit et al.
    2020): low γ truncates the chain at the *per-step* level so
    the bias has no opportunity to compound, and DDQN's
    correction has nothing to bite on. FourRooms's pattern is
    different (mechanism stays strong, link weakens) — the
    chain-depth-as-amplifier reading.

    HELD when g ≥ 0.30 AND helped_fraction ≥ 0.20 on the
    high-γ subset. helped threshold is lower than other bridges
    because DC is sparse-reward bimodal: many seeds score 0
    (never find goal), so helped_fraction undershoots even when
    the mean effect is significant."""
    if paired_g.n_pairs < 20:
        return Verdict.POWER_INSUFFICIENT
    if math.isnan(paired_g.helped_fraction):
        return Verdict.POWER_INSUFFICIENT
    if (
        paired_g.helped_fraction >= 0.20
        and paired_g.g >= 0.30
    ):
        return Verdict.HELD
    return Verdict.NO_EFFECT


# =====================================================================
# CLAIM 4 — Independent link-side scope predicate (bootstrap_fraction).
#
# At the (env, burst) panel level on the DDQN 200k corpus,
# bootstrap_fraction (≈ 1 − P(terminate per step), an env-level
# structural feature of how often training updates bootstrap from
# a non-terminal target) predicts g_link (the outcome side of
# DDQN's chain) WITH dormancy_env_mean already in the model.
#
# Empirically (n_strata=149, 15 envs):
#   β(bootstrap_fraction → g_link) = +2.716, z=+3.27, p=0.0012
#   β(dormancy_env_mean → g_link) ≈ 0     (dormancy is on g_mech)
#
# This rules out the "dormancy absorbs bootstrap_fraction"
# reading: bootstrap_fraction is a SEPARATE scope predicate on
# the LINK edge, not a coarse proxy for dormancy.
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('seed', 'total_steps', 'eval_every'),
)
def bootstrap_fraction_drives_g_link__net_of_dormancy(
    meta_regression_per_burst: MetaRegressionResult,
    *,
    # `source` pins the per-burst measurable that the underlying
    # `paired_g_per_burst` projects each cell to. The decorator's
    # `source=INTERVENTION` carries the do-contrast (treatment /
    # baseline arms); the bridge's `target='eval_best_burst_mean'`
    # would otherwise be auto-injected as the analysis source
    # (a scalar, no per-burst structure). The body default below
    # routes the panel computation back onto `mc_return` per the
    # claim's g_link reading.
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    # Column-name covariates: each is materialised per-cell by the
    # @measurable cache, then averaged to env-level inside the
    # analysis. Replaces the inline `_DDQN_UNIVERSE_COVARIATES_PER_ENV`
    # static dict (frozen values from the 200k corpus, applied to
    # whatever corpus runs now). Per-env values now come from the
    # corpus itself: `log_action_dim`/`log_obs_dim`/`log_horizon`
    # are env-catalogue lookups, `bootstrap_fraction` is per-cell
    # `1 - mean(done)`, `invariant.jensen_dormancy_gap` is the raw
    # invariant column from runs.parquet.
    covariates: tuple[str, ...] = (
        'log_action_dim',
        'log_obs_dim',
        'log_horizon',
        'bootstrap_fraction',
        'jensen_dormancy_gap',
    ),
    dedupe_strategy: str = 'mean',
) -> Verdict:
    """Independent link-side scope predicate. The (env, burst)
    panel meta-regression of g_link on the 5-covariate set
    {log_action_dim, log_obs_dim, log_horizon, bootstrap_fraction,
    dormancy_env_mean} produces a robust positive coefficient on
    bootstrap_fraction even with dormancy_env_mean controlled.

    Theoretical reading: in envs where most updates bootstrap
    (long episodes / sparse termination), the chain of
    bootstrapped Q-values amplifies the per-step max-bias, so
    DDQN's bias correction yields a larger downstream outcome
    benefit. Where episodes terminate often (low bootstrap
    fraction), the truncation cuts off the bias amplification,
    and DDQN's correction carries less of an outcome signal.

    HELD when β(bootstrap_fraction) ≥ +1.0 AND p < 0.05 (a
    medium-large coefficient with the expected positive sign).
    POWER_INSUFFICIENT when not significant. NO_EFFECT when
    significant with β below the magnitude floor.

    The independence-of-dormancy claim is encoded in the
    covariate set itself: dormancy_env_mean is in the model, so
    a surviving β(bootstrap_fraction) is a partial coefficient,
    not a marginal one."""
    del source, covariates, dedupe_strategy
    coef = next(
        (c for c in meta_regression_per_burst.coefficients
         if c.name == 'bootstrap_fraction'),
        None,
    )
    # `coef is None` means the covariate was dropped by the
    # regression — typically because it was all-NaN at the cells
    # available (e.g. `bootstrap_fraction` reads the per-step
    # `done` trace which isn't joined into the runner's cache).
    # That's a data-availability gap, not "no signal", so call
    # POWER_INSUFFICIENT rather than silently NO_EFFECT.
    if coef is None:
        return Verdict.POWER_INSUFFICIENT
    if not coef.is_significant:
        return Verdict.POWER_INSUFFICIENT
    if coef.coefficient >= 1.0:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# =====================================================================
# CLAIM 6 — log_mc_variance attenuates DDQN benefit (between-env).
# =====================================================================
# Reframed (2026-05-02): the original observational finding was
# log_mc_variance has a between-env attenuating effect on g_link
# (univariate β=−0.019, p=0.018 OLS-style; CR1 β=−0.061, p=0.064;
# Mundlak between β=−0.061, p=0.064 CR1 — borderline). The bridge
# verdicts POWER_INSUFFICIENT under cluster-robust SEs (11 effective
# clusters too few).
#
# **Important reframing from `reward_scale_sweep` (CLAIM 7 below)**:
# the mc_variance reading was a SHADOW of the under-learning
# rescue mechanism. In native-outcome units, DDQN's largest
# benefit appears at LOW reward scale (rs=0.1 on FourRooms,
# native diff +0.49 ★★★) where vanilla CATASTROPHICALLY UNDER-
# LEARNS, not at high mc_variance. Standardized Hedges' g hid
# this because pooled SD scales with reward; native units don't.
# The mc_variance attenuator description was capturing "envs
# where vanilla doesn't fail" via the wrong proxy. Keep the
# bridge as POWER_INSUFFICIENT observational claim but treat the
# under-learning-rescue framing (CLAIM 7) as the load-bearing
# causal story.
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('seed', 'total_steps', 'eval_every'),
)
def mc_variance_attenuates_g_link__between_env(
    mundlak_paired_g_per_burst: MundlakResult,
    *,
    # See `bootstrap_fraction_drives_g_link__net_of_dormancy`
    # for the rationale: pin the per-burst measurable that
    # `paired_g_per_burst` (called inside Mundlak) projects each
    # cell onto. The decorator's `source=INTERVENTION` carries
    # the do-contrast; the body default below routes the panel
    # computation back onto `mc_return` per the g_link reading.
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    predictor_name: str = 'log_mc_variance_per_burst',
    predictor_arm_filter: str = 'vanilla_dqn',
    dedupe_strategy: str = 'mean',
) -> Verdict:
    """Single-level (between-env) attenuator. The Mundlak
    decomposition of `log_mc_variance` over the (env, burst)
    g_link panel produces a between-env coefficient that, when
    significant and negative, indicates that envs with higher
    return-variance see smaller DDQN link benefit.

    HELD when between coefficient < 0 AND between p < 0.05.
    POWER_INSUFFICIENT when |between coefficient| ≥ |0.01| but
    p ≥ 0.05 (signal in expected direction but underpowered).
    NO_EFFECT otherwise.

    Within-env coefficient is reported but NOT asserted on —
    the prior "within enabler" framing was a methodology
    artifact. Mundlak guards future readers against repeating it.

    Pearl-rung-2 corroboration comes from `reward_scale_sweep`
    (causal probe via reward × k intervention)."""
    del source, predictor_name, predictor_arm_filter, dedupe_strategy
    coef = mundlak_paired_g_per_burst.between
    if not coef.p_value < 0.05:
        if coef.coefficient < -0.01:
            return Verdict.POWER_INSUFFICIENT
        return Verdict.NO_EFFECT
    if coef.coefficient < 0.0:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# =====================================================================
# CLAIM 7 — DDQN's reward-scale-response curve dominates vanilla's
#           on FourRooms at low reward scale (Pearl rung-2,
#           interventional contrast).
# =====================================================================
# `reward_scale_sweep` + `reward_scale_low_fourrooms` mapped the
# native-outcome curves of both arms across rs ∈ [0.01, 10] on
# FourRooms:
#
#                rs       vanilla_native     ddqn_native    Δ (interventional)
#                0.01     0.08               0.17           +0.087
#                0.03     0.05               0.30           +0.245
#                0.10     0.06               0.54           +0.488 ★★★
#                0.30     0.24               0.74           +0.497 ★★★
#                1.00     0.70               0.79           +0.075
#                10.00    0.80               0.80           ~0.000  (ceiling)
#
# The interventional contrast `do(arm=ddqn) − do(arm=baseline)`
# at fixed (FourRooms, rs, seed) is large at rs ∈ [0.03, 0.3] —
# peak +0.50 native at rs=0.3.
#
# **Defensive framing (load-bearing, applies to all
# treatment-vs-baseline claims)**: Δ is NOT an observational
# edge between two outcome nodes. Vanilla and DDQN runs are
# independent training trajectories under different algorithms;
# they share log_rs, env_name, seed, but have no causal arrow
# between their outputs. JCI on (vanilla_native, ddqn_native,
# log_rs) correctly drops the vanilla↔ddqn edge at depth 1
# (partial r ≈ −0.04 conditional on log_rs). The two arms have
# two INDEPENDENT response curves to log_rs; Δ is the
# interventional contrast, NOT a causal mechanism between them.
#
# The narrative "DDQN rescues vanilla's under-learning" is
# misleading — there's no causal arrow from vanilla's
# under-learning TO DDQN's success. Both arms have independent
# learning dynamics under log_rs; DDQN's curve simply
# saturates at smaller rs than vanilla's. The Hasselt-floor
# theory predicts this directly: ε = σ_Q · √(2 log|A|) sets the
# minimum reward magnitude where each arm's gradient signal
# overcomes its compounded Jensen bias; DDQN's reduced ε lets
# it learn at smaller σ_Q, hence smaller reward scales. The
# interventional gap is the visible signature of two arms with
# different ε.
#
# Pearl-rung-2: paired (env, rs, seed) cells with randomized
# arm assignment. The mean_diff IS the do-effect; cluster-count
# irrelevant.
#
# Predicted: mean_diff(do(ddqn), do(vanilla) | FourRooms, rs=0.1) ≥ +0.4
# Observed: +0.486, p=2.2e-16 — HELD.
#
# **Scope of the rescue regime is FourRooms-specific, NOT a
# general √(log|A|) phenomenon.** The Hasselt-floor reading above
# predicts that at rs=0.1 the rescue gap should scale across envs
# as √(log|A|): DeepSea (|A|=2) at 71% of FourRooms's gap,
# DiscountingChain (|A|=5) at 107%, MNISTBandit (|A|=10) at 140%.
# `action_dim_at_low_rs` (4 envs × 2 rs × 30 seeds) tested this
# corollary directly:
#
#                env                |A|    obs Δ at rs=0.1   pred Δ
#                DeepSea-bsuite      2     +0.06 (ns)        +0.35
#                FourRooms-misc      4     +0.49 (HELD)      +0.49 (anchor)
#                DiscountingChain    5     +0.02 (ns)        +0.53
#                MNISTBandit-bsuite  10    +0.00 (degen)     +0.63
#
# Pearson r(predicted, observed) = −0.16 → REFUTED. The other
# three envs do not enter the rescue regime at rs=0.1: DeepSea's
# vanilla solves at native 0.80 (no failure to rescue);
# MNISTBandit is a bandit, both arms cap at the supervised-MNIST
# floor regardless of bias; DiscountingChain solves quickly. The
# Hasselt-floor formula is necessary at the mechanism layer (DDQN
# does reduce Jensen bias, and the floor is real) but it doesn't
# *predict gap magnitude across envs* because vanilla doesn't
# fail at rs=0.1 outside FourRooms's specific under-learning
# window. The bridge below holds at FourRooms; we don't author
# sibling bridges at other |A| envs because the premise (vanilla
# fails) doesn't hold there.
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='outcome_native',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('reward_scale') == 0.1)
    ),
)
def ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1(
    paired_g: PairedGResult,
    *,
    threshold_diff: float = 0.4,
    dedupe_strategy: str = 'mean',
) -> Verdict:
    """Pearl-rung-2 interventional contrast: do(arm=ddqn) on
    FourRooms at reward_scale=0.1 produces native-outcome
    ≥ +0.4 above the do(arm=vanilla_dqn) baseline.

    Generic primitive shape: consumes `paired_g` with
    `target='outcome_native'` (the registered measurable
    `eval_best_burst_mean / reward_scale`) under
    `source=INTERVENTION` (do(ddqn) − do(vanilla_dqn) contrast)
    and `scope=(env_name == 'FourRooms-misc') & (reward_scale
    == 0.1)` to filter the corpus. No bespoke analysis — the bridge
    supplies the measurable name + scope, the framework runs
    `paired_g` and injects the result.

    HELD when `paired_g.mean_diff ≥ threshold_diff (=+0.4)` AND
    `paired_g.mean_diff_p_value < 0.05`. POWER_INSUFFICIENT
    when diff in expected direction but underpowered.
    NO_EFFECT otherwise.

    Asserts on `paired_g.mean_diff` (the interventional contrast
    in native units), NOT `paired_g.g` (standardized Hedges' g
    pools SD that scales with reward, hiding the interventional
    effect under apparent sweet-spotting at baseline).

    **Defensive note**: `mean_diff` is the do-effect of arm
    assignment at fixed (env, rs, seed); it is NOT an
    observational edge between vanilla and ddqn outcome nodes.
    The two arms run as independent training trajectories under
    different algorithms — JCI confirms vanilla_native ⊥
    ddqn_native | log_rs (partial r ≈ −0.04). The bridge tests
    a CONTRAST between two independent reward-scale-response
    curves, not a causal arrow between cell outputs."""
    del dedupe_strategy  # forwarded to paired_g; not used in body
    diff = paired_g.mean_diff
    p = paired_g.mean_diff_p_value
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


# =====================================================================
# CLAIM 7b — same interventional contrast at rs=0.3 (rescue-regime
#            peak in the reward_scale_low_fourrooms sweep map).
# =====================================================================
# At rs=0.3 the do-effect peaks at +0.50 native (vanilla 0.24,
# ddqn 0.74); at rs=0.1 the do-effect is +0.49 (vanilla 0.06,
# ddqn 0.54). Both clear the +0.4 threshold. The ratio
# 0.30 → 0.50 vs 0.10 → 0.49 confirms the rescue-regime is a
# plateau, not a single-point effect.
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='outcome_native',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('reward_scale') == 0.3)
    ),
)
def ddqn_dominates_vanilla_response_curve__fourrooms_rs_0p3(
    paired_g: PairedGResult,
    *,
    threshold_diff: float = 0.4,
) -> Verdict:
    """Sibling of CLAIM 7 at the rescue-regime peak (rs=0.3).
    Same primitive shape (`paired_g` with measurable source +
    extra_filters); just a different scale point. Together with
    CLAIM 7 (rs=0.1) and the underpowered-but-direction-correct
    rs=0.03 / rs=0.01 datapoints, establishes the rescue regime
    as a plateau, not a knife-edge point. Generalizes the
    reward-scale-response-curve-dominance claim across the
    interior of the inverted-U.

    Same defensive framing as CLAIM 7: `mean_diff` is the
    interventional contrast, not an observational edge between
    arm outputs."""
    diff = paired_g.mean_diff
    p = paired_g.mean_diff_p_value
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


# =====================================================================
# CLAIM 8 — Per-burst learning-curve crossover on SpaceInvaders.
# =====================================================================
# Per-burst analysis on minatar_1M (paired across 30 seeds, env=
# SpaceInvaders-MinAtar, total_steps=1M) shows a sharp sign-flip
# in the interventional contrast `do(arm=ddqn) − do(arm=vanilla)`:
#
#   burst 0:    diff +1.97 (DDQN ahead, significant)
#   bursts 1-2: ~tied
#   bursts 3-5: DDQN sliding (-0.5 to -1.0, ns)
#   bursts 6+:  diff -1.4 to -2.0, all significant negative
#
# Other 3 MinAtar envs (Asterix, Breakout, Freeway) do NOT show
# this crossover under the same paired_g_per_burst test.
# SpaceInvaders is the only MinAtar env with stochastic NEGATIVE
# reward (-1 per hit, random enemy fire); the others are
# positive-only or zero-stochasticity.
#
# **Burst-level JCI on (mc, bias, burst_index) per arm**
# refines the mechanism reading:
#   - BOTH arms decline late on SpaceInvaders. vanilla goes
#     9.2→7.5 native; ddqn 11.1→6.1. The "late crossover" isn't
#     "DDQN fails while vanilla succeeds" — it's "both arms
#     decay, DDQN decays faster from a higher start."
#   - DDQN has STRONGER mc↔bias coupling (ρ=−0.70 vs vanilla's
#     −0.58). Counter-intuitive: DDQN's bias-correction was
#     supposed to break the bias→mc dependence, but it
#     TIGHTENS it. Plausible reading: vanilla's overestimation
#     noise decouples observed mc from instantaneous bias;
#     DDQN removes the noise → mc faithfully reflects the
#     residual bias trajectory → tighter negative ρ.
#   - bias↔burst_index in both arms (ρ=+0.95-0.97) — Q-magnitude
#     inflates monotonically over training. The underlying
#     issue is shared late-training Q-explosion; DDQN doesn't
#     cause it, but its noise-reduction makes the consequences
#     visible.
#
# Two competing mechanism hypotheses, distinguished by the
# queued `spaceinvaders_no_hit_penalty` Pearl-rung-2 sweep:
#   1. Pessimism: vanilla's overestimation acts as
#      optimism-under-hit-uncertainty regularization that helps
#      the agent stay aggressive. Stripping the −1 hit removes
#      the differential pessimism source → crossover disappears.
#   2. Q-explosion: late training, Q-values diverge in both
#      arms (bias↔burst ρ=+0.95). DDQN's bias correction makes
#      mc track the explosion more faithfully, so the curve
#      drops faster. Stripping the hit doesn't fix divergence
#      → crossover persists.
#
# Bridge below detects the SHAPE only — the crossover-and-late-
# negative pattern. The mechanism claim awaits the no-hit-
# penalty sweep result.
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='mc_return[per_burst]',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'SpaceInvaders-MinAtar')
        & (pl.col('total_steps') == 1_000_000)
        & pl.col('reward_clip_min').is_null()
        & (pl.col('sync_period') == 100)
    ),
)
def ddqn_curve_crosses_vanilla_late__spaceinvaders(
    paired_g_per_burst: PerBurstResult,
    *,
    # Pin the per-burst measurable: the decorator's target
    # `mc_return[per_burst]` is the human-readable name of what
    # the analysis projects, but `paired_g_per_burst` indexes
    # cells by the raw per-burst column `mc_return`. The body
    # default routes the per-burst projection back onto the
    # underlying 2-D `mc_return` array.
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    env_name: str = 'SpaceInvaders-MinAtar',
    crossover_burst_min: int = 3,
    crossover_burst_max: int = 10,
    late_negative_floor: float = -0.3,
    dedupe_strategy: str = 'mean',
) -> Verdict:
    """Per-burst crossover detection on SpaceInvaders 1M.

    Walks `paired_g_per_burst.strata` filtered to `env_name`, in
    burst-index order. HELD when:
      (1) The first stratum where g < 0 lies in
          [crossover_burst_min, crossover_burst_max] inclusive
          (the curves cross within the expected window).
      (2) The mean g across late bursts (index ≥ crossover) is
          ≤ `late_negative_floor` (the late-training side of the
          curve has a substantial negative drift, not just a
          one-burst fluctuation).

    POWER_INSUFFICIENT when (1) holds but (2) fails (crossover
    detected, but late drift is too small to assert magnitude).
    NO_EFFECT when no crossover or curve never goes negative.

    Reads `paired_g_per_burst.strata[i].g` (standardized Hedges'
    g) — within env, the SD-scaling is ~constant so sign
    detection is invariant. The bridge is a SHAPE claim about
    the per-burst-index curve, not a single aggregate effect."""
    del source, dedupe_strategy
    env_strata = sorted(
        (s for s in paired_g_per_burst.strata if s.env_name == env_name),
        key=lambda s: s.burst_index,
    )
    if len(env_strata) < crossover_burst_max + 1:
        return Verdict.NO_EFFECT

    # Find the crossover: first burst where g < 0 (after
    # potentially seeing positive g earlier — the bridge isn't
    # strict about an early-positive prefix; the late-floor check
    # in step (2) carries the magnitude signal).
    crossover_idx: int | None = None
    for s in env_strata:
        if math.isnan(s.g):
            continue
        if s.g < 0.0:
            crossover_idx = s.burst_index
            break
    if crossover_idx is None:
        return Verdict.NO_EFFECT
    if not (crossover_burst_min <= crossover_idx <= crossover_burst_max):
        return Verdict.NO_EFFECT

    # Late-side magnitude floor.
    late_gs = [
        s.g for s in env_strata
        if s.burst_index >= crossover_idx and not math.isnan(s.g)
    ]
    if not late_gs:
        return Verdict.NO_EFFECT
    late_mean = float(np.mean(late_gs))
    if late_mean > late_negative_floor:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.HELD


# =====================================================================
# CLAIM 9 — n-step falsification of bootstrap-bias-compounding.
# =====================================================================
# Pearl-rung-2 negative-prediction probe of the Hasselt mechanism.
# `nstep_lambda_fourrooms` (n ∈ {1, 2, 3, 5, 10} × ddqn vs vanilla
# × 30 seeds × FourRooms γ=0.99) interventionally varies bootstrap
# dependence: n=1 is full bootstrap, n→∞ is Monte Carlo with no
# bootstrap. The Hasselt theorem says DDQN's benefit comes from
# correcting bias amplified along the bootstrap chain — so the
# do-effect of (do(arm=ddqn) − do(arm=vanilla)) on outcome should
# DECLINE monotonically with n and reach zero when bootstrap
# influence is gone.
#
# Observed:
#   n=1   Δ = +0.087   p = 0.0003   ★ (full benefit)
#   n=2   Δ = +0.009   p = 0.03     (10× shrinkage)
#   n=3   Δ = +0.002   p = 0.62     (NULL)
#   n=5   Δ = +0.007   p = 0.23     (NULL)
#   n=10  Δ = +0.005   p = 0.48     (NULL)
#
# Encoded as TWO bridges that together corroborate the
# falsification:
#   - `ddqn_helps_at_full_bootstrap__fourrooms_n1` HELD at n=1
#   - `ddqn_null_under_monte_carlo__fourrooms_n10` HELD-as-null
#     at n=10 (negative prediction; DDQN should be ineffective
#     when bootstrap is removed)
#
# Confounds tend to add positive correlations, not subtract them;
# the monotonic Δ→0 with reduced bootstrap dependence directly
# confirms the bootstrap-bias-compounding mechanism. Alternative
# stories (regularization, exploration, optimization stability)
# would persist at high n.
# =====================================================================


@claim_bridge(
    source=DoEffect(treatment_arm='ddqn_n1', baseline_arm='vanilla_n1'),
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(pl.col('env_name') == 'FourRooms-misc'),
)
def ddqn_helps_at_full_bootstrap__fourrooms_n1(
    paired_g: PairedGResult,
    *,
    threshold_diff: float = 0.05,
) -> Verdict:
    """At n=1 (full bootstrap), DDQN's outcome benefit on
    FourRooms is ≥ +0.05 with p < 0.05. The positive baseline of
    the falsification curve — pairs with the n=10 NO_EFFECT
    bridge to corroborate that bootstrap dependence is the
    mechanism's necessary substrate."""
    diff = paired_g.mean_diff
    p = paired_g.mean_diff_p_value
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


@claim_bridge(
    source=DoEffect(treatment_arm='ddqn_n10', baseline_arm='vanilla_n10'),
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(pl.col('env_name') == 'FourRooms-misc'),
)
def ddqn_null_under_monte_carlo__fourrooms_n10(
    paired_g: PairedGResult,
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
    bridge by design — the theorem predicts smallness here."""
    diff = paired_g.mean_diff
    p = paired_g.mean_diff_p_value
    if math.isnan(diff) or math.isnan(p):
        return Verdict.NO_EFFECT
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
    # Override: this bridge runs on the `l2_x_gamma_acrobot` traces
    # which authored γ + weight-decay into the arm name strings
    # (substrate-side smuggle). File-level INTERVENTION's plain
    # `ddqn` / `vanilla_dqn` won't match; consume the corpus's
    # actual arm shape via a per-bridge override.
    source=DoEffect(
        treatment_arm='ddqn_g0999_wd1em4',
        baseline_arm='vanilla_g0999_wd1em4',
    ),
    target='mc_return',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=(pl.col('env_name') == 'Acrobot-v1'),
)
def acrobot_per_burst_link_active__gamma_0999(
    paired_link_per_burst: PerBurstLinkResult,
    *,
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
    del target, predictor
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


@claim_bridge(
    source=DoEffect(
        treatment_arm='ddqn_g0999', baseline_arm='vanilla_dqn_g0999',
    ),
    target='outcome.eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Acrobot-v1')
        & (pl.col('gamma') == 0.999)
    ),
)
def acrobot_link_backdoor_ate_negative__gamma_0999(
    paired_delta_link_dowhy: PairedDeltaLinkDowhyResult,
    *,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _JENSEN_BIAS_PER_BURST_MEAN,
    env_filter: tuple[str, ...] = ('Acrobot-v1',),
    ate_ceiling: float = -0.1,
) -> Verdict:
    """DoWhy backdoor adjustment over the per-(env, burst, seed)
    paired-Δ panel (treatment=Δ_jens, outcome=Δ_out, adjusters=
    burst dummies) on Acrobot γ=0.999 yields a NEGATIVE ATE
    bigger than `ate_ceiling` (i.e. ATE <= -0.1). HELD when
    identified AND ATE <= ceiling. Empirical -0.6312."""
    del link_predictor, link_target, env_filter
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
    source=DoEffect(
        treatment_arm='ddqn_g0999', baseline_arm='vanilla_dqn_g0999',
    ),
    target='outcome.eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Acrobot-v1')
        & (pl.col('gamma') == 0.999)
    ),
)
def acrobot_link_placebo_refuted__gamma_0999(
    paired_delta_link_dowhy: PairedDeltaLinkDowhyResult,
    *,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _JENSEN_BIAS_PER_BURST_MEAN,
    env_filter: tuple[str, ...] = ('Acrobot-v1',),
    placebo_max_ratio: float = 0.2,
) -> Verdict:
    """Placebo refutation shrinks ATE to ≤ `placebo_max_ratio` of
    the real ATE on Acrobot γ=0.999. HELD when |placebo / real|
    < placebo_max_ratio AND real ATE is non-zero. Confirms the
    bias-correction effect is treatment-specific (not noise).
    Empirical: real -0.6312, placebo 0.0000, ratio 0%."""
    del link_predictor, link_target, env_filter
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
    source=DoEffect(
        treatment_arm='ddqn_g0999', baseline_arm='vanilla_dqn_g0999',
    ),
    target='outcome.eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'Acrobot-v1')
        & (pl.col('gamma') == 0.999)
    ),
)
def acrobot_link_rcc_robust__gamma_0999(
    paired_delta_link_dowhy: PairedDeltaLinkDowhyResult,
    *,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _JENSEN_BIAS_PER_BURST_MEAN,
    env_filter: tuple[str, ...] = ('Acrobot-v1',),
    rcc_max_drift_ratio: float = 0.1,
) -> Verdict:
    """Random-common-cause refutation: adding a noise covariate
    to the adjustment set leaves ATE within `rcc_max_drift_ratio`
    of the real ATE on Acrobot γ=0.999. HELD when |refuted -
    real| / |real| < rcc_max_drift_ratio. Confirms robustness to
    spurious-confound vulnerability. Empirical drift = 0.000."""
    del link_predictor, link_target, env_filter
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
    source=INTERVENTION,
    target='outcome.eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(pl.col('total_steps') == 1_000_000),
)
def extreme_q_divergence_attenuates_link__binary(
    link_attenuation_dowhy: LinkAttenuationDowhyResult,
    *,
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
    del attenuator, binary_threshold, link_target, link_predictor
    del dedupe_strategy
    b = link_attenuation_dowhy.backdoor
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
    source=INTERVENTION,
    target='outcome.eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(pl.col('total_steps') == 1_000_000),
)
def extreme_q_divergence_attenuates_link__placebo_refuted(
    link_attenuation_dowhy: LinkAttenuationDowhyResult,
    *,
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
    del attenuator, binary_threshold, link_target, link_predictor
    del dedupe_strategy
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
    source=INTERVENTION,
    target='outcome.eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(pl.col('total_steps') == 1_000_000),
)
def extreme_q_divergence_attenuates_link__rcc_robust(
    link_attenuation_dowhy: LinkAttenuationDowhyResult,
    *,
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
    del attenuator, binary_threshold, link_target, link_predictor
    del dedupe_strategy
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


# =====================================================================
# DDQN measurement graph — the closure.
# =====================================================================
DDQN_UNIVERSE_BRIDGES = (
    # CLAIM 2 — load-bearing necessary scope (causal refutation).
    ddqn_refuted_when_dormancy_fires,
    # CLAIM 2 corroborations — Pearl rung-2 designed interventions.
    adaptive_dqn_recovers_ddqn_benefit__fourrooms_factor_0p5,
    adaptive_dqn_fails_to_avoid_attenuation__spaceinvaders_1m,
    # CLAIM 4 — independent link-side scope (residual after dormancy).
    bootstrap_fraction_drives_g_link__net_of_dormancy,
    # CLAIM 5 — effective-horizon scope (Pearl rung-2 do(γ) sweep).
    ddqn_benefit_scales_with_effective_horizon__fourrooms,
    ddqn_benefit_scales_with_effective_horizon__metamaze_high_gamma,
    ddqn_benefit_scales_with_gamma__discountingchain,
    # CLAIM 6 — between-env mc_variance attenuates g_link
    #           (POWER_INSUFFICIENT under CR1; SHADOW of CLAIM 7).
    mc_variance_attenuates_g_link__between_env,
    # CLAIM 7 — Pearl-rung-2: DDQN's reward-scale-response curve
    #           dominates vanilla's at rs=0.1 on FourRooms.
    ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1,
    # CLAIM 7b — same dominance at rs=0.3 (rescue-regime peak).
    ddqn_dominates_vanilla_response_curve__fourrooms_rs_0p3,
    # CLAIM 8 — per-burst crossover shape on SpaceInvaders 1M.
    ddqn_curve_crosses_vanilla_late__spaceinvaders,
    # CLAIM 9 — n-step falsification of bootstrap-bias-compounding
    #           (the strongest mechanism corroboration: Δ→0 as n grows).
    ddqn_helps_at_full_bootstrap__fourrooms_n1,
    ddqn_null_under_monte_carlo__fourrooms_n10,
    # TIER A2 existence proofs (per-burst, env-conditional).
    ddqn_helps_at_early_bursts__pixel_envs,
    # ddqn_attenuates_at_late_bursts__spaceinvaders — DISABLED.
    # The two SpaceInvaders 1M sweeps that match this bridge's scope
    # (`minatar_1M_spaceinvaders`, Apr 30; `spaceinvaders_no_hit_penalty`
    # default arm, May 1) trained on substantially different substrate
    # commits — env-wrapper plumbing was refactored, RewardClippedEnv
    # introduced, and 9 other commits landed between the two trainings.
    # Same nominal HPs, same seeds (0..29), but seed=0 trajectories
    # diverge from burst 0 — the cells are observations of different
    # code, not replications of the same experiment. Cross-corpus
    # pooling shifts pooled g from -0.42 (Apr 30 single-corpus) to
    # -0.07 (May 1 single-corpus), or -0.28 averaged. The shape
    # sister bridge (`ddqn_curve_crosses_vanilla_late__spaceinvaders`,
    # HELD) carries the late-attenuation claim more robustly without
    # committing to a magnitude threshold under substrate drift.
    # Re-enable after re-running spaceinvaders_no_hit_penalty under
    # the current substrate (or pinning substrate version per cell).
    # CLAIM 10 — link IS bias-correction on Acrobot γ=0.999, causally
    # corroborated. Per-burst link panel + DoWhy backdoor + placebo
    # refutation + RCC refutation all hold. Corrects the prior
    # `findings_l2_acrobot_goldilocks.md` "scalar link null" finding,
    # which was a measurement artifact of best-burst-per-seed
    # selection.
    acrobot_per_burst_link_active__gamma_0999,
    acrobot_link_backdoor_ate_negative__gamma_0999,
    acrobot_link_placebo_refuted__gamma_0999,
    acrobot_link_rcc_robust__gamma_0999,
    # CLAIM 11 — extreme Q-divergence attenuates the link. Companion
    # to CLAIM 2's dormancy bridge: dormancy bounds the lower
    # attenuation (mech inactive); extreme Q-divergence bounds the
    # upper attenuation (Q-explosion overwhelms link). Together they
    # bound the link-active band on q_divergence_score.
    extreme_q_divergence_attenuates_link__binary,
    extreme_q_divergence_attenuates_link__placebo_refuted,
    extreme_q_divergence_attenuates_link__rcc_robust,
)
"""The six bridges that close the DDQN study. CLAIM 1 (mechanism
activation, do(DDQN) ↓ jensen_gap) is corroborated by
`ddqn_reduces_jensen_gap__converged_subset` in `dqn_bridges.py`
on the 200k DDQN corpus's converged subset; not duplicated here.

CLAIM 3 (sufficient scope) is deliberately ABSENT — no exogenous
predicate corroborates a sufficient condition for DDQN's outcome
benefit, and we don't author null bridges."""


__all__ = [
    'DDQN_UNIVERSE_BRIDGES',
    'acrobot_link_backdoor_ate_negative__gamma_0999',
    'acrobot_link_placebo_refuted__gamma_0999',
    'acrobot_link_rcc_robust__gamma_0999',
    'acrobot_per_burst_link_active__gamma_0999',
    'extreme_q_divergence_attenuates_link__binary',
    'extreme_q_divergence_attenuates_link__placebo_refuted',
    'extreme_q_divergence_attenuates_link__rcc_robust',
    'adaptive_dqn_fails_to_avoid_attenuation__spaceinvaders_1m',
    'adaptive_dqn_recovers_ddqn_benefit__fourrooms_factor_0p5',
    'bootstrap_fraction_drives_g_link__net_of_dormancy',
    'ddqn_attenuates_at_late_bursts__spaceinvaders',
    'ddqn_benefit_scales_with_effective_horizon__fourrooms',
    'ddqn_benefit_scales_with_effective_horizon__metamaze_high_gamma',
    'ddqn_benefit_scales_with_gamma__discountingchain',
    'ddqn_helps_at_early_bursts__pixel_envs',
    'ddqn_refuted_when_dormancy_fires',
]


# Canonical name `corroborate.runner` imports;
# DDQN_UNIVERSE_BRIDGES stays as an alias for legacy call sites.
BRIDGES = DDQN_UNIVERSE_BRIDGES

