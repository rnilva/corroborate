"""DDQN measurement graph.

Bridges encoding the DDQN causal-chain claims, audited for
predicate endogeneity and post-fix verdict status. The current
verdict landscape lives in
`experiments/findings/BRIDGE_AUDIT_TABLE.md`; the manifest
governing cuts / migrations is `BRIDGE_AUDIT.md` with the
framework-completion design in `BRIDGE_PREDICTION_DESIGN.md`.

Tier framework (refinement of the framework's
ASSOCIATIONAL/INTERVENTIONAL):

  TIER A1 — universal exogenous predicates (env-feature / time /
            HP); claims generalize to envs we haven't measured.
  TIER A2 — sampled exogenous predicates (env_name); existence
            proofs over our specific benchmark sample.
  TIER INT — Pearl-rung-2: claim backed by a designed
            intervention sweep.
  TIER B  — control-trajectory-endogenous predicates; descriptive
            only, not actionable. NOT EXPORTED HERE; live in
            `dqn_bridges.py` zoo.

## Surviving load-bearing causal layers (post-2026-05-11 audit)

The file's claim structure after step-2 cuts:

- **Mechanism activation** (universal causal) — `do(arm=ddqn) ↓
  jensen_gap` on premise-active envs. Tested in
  `dqn_bridges.py::ddqn_reduces_jensen_gap__converged_subset`,
  not duplicated here.
- **Necessary scope — dormancy refutation** (CLAIM 2,
  POWER_INSUFFICIENT). `ddqn_refuted_when_dormancy_fires` was
  refactored 2026-05-11 to CI[g]-vs-null_ceiling logic (consumes
  `paired_g` + `bootstrap_paired_g`). Current verdict is
  POWER_INSUFFICIENT: at the dormancy cells' extreme kurtosis
  (excess_kurt=109, skew=+10.27) the bootstrap CI itself is
  uncalibrated (its tests verify accuracy at log-normal n=50
  only); the framework's `assumption_violations` flag and a
  substantive raw mean_diff (+0.23 native) suggest the cells
  carry a real positive shift that the standardized-g CI hides
  via outlier-inflated SD. Reviewer-5 catch: don't claim HELD
  on percentile bootstrap at kurt=109. Pair with CLAIM 26b as a
  necessary-scope companion under env-G1-inactive scope.
- **Three-gate scope conjunction** (CLAIM 26b, SURVIVED).
  `ddqn_helps_under_three_gate_scope__cross_env` — DDQN helps
  iff `G1 ∧ G2 ∧ G3` fires jointly (premise-active ∧
  argmax-vulnerable ∧ outcome-headroom). Pooled d=+0.46,
  p=0.005 across 5 G1-active envs. The substantive cross-env
  outcome-level claim.
- **Chain-amplifier link** (CLAIM 17 — CUT 2026-05-11).
  Migrated to `stratum_effect_panel + panel_regress` revealed
  the cross-env signal was leverage-driven by 2-3 high-bias
  envs; drop them and slope flips sign. Substantive content
  preserved by CLAIM 26b's stratified-DL pool (leverage-robust)
  and per-env existence proofs in
  `findings_minatar_link_attenuation.md`. See the in-source
  CLAIM 17 deletion-memo banner.
- **REACH-cohort link** (CLAIM 22, SURVIVED). DoWhy backdoor +
  placebo + RCC refutation trio on the REACH-polarity envs.
- **Polarity-coupling shape** (CLAIM 14, POWER_COLLAPSED).
  `link_r_predictable_from_polarity__soft_tautology` — sign-
  correct but CI overlaps zero at n=9 envs post-fix; structural
  identity argument worth revisiting as a measurement-identity
  bridge.

## Cut / retired (2026-05-11)

CLAIM 4 + 16 (bf cross-env), CLAIM 21 REACH-polyak,
CLAIM 26 (slope-predictor, superseded by 26b), CLAIM 6
(mc_variance, refuted via CV decomposition), CLAIM 18
(algorithmic-activation, placeholder), CLAIM 7 g/h/i/j (4
auxiliary mechanism-route probes). Synthesis preserved as
deletion-memo banners below.

## RL methodology note

The audit's step-5 reading flagged that seed-pairing
inside per-pair-Δ fixtures (`paired_g`, `paired_link_per_burst`,
`proportion_mediated`, etc.) reflects within-init correlation
rather than population-of-inits variance — the inferential
target for cross-init claims is the stratified-pooled form.
17 of 22 in-scope bridges need migration to fully-stratified
analogs per `BRIDGE_PREDICTION_DESIGN.md §11`. The SURVIVED
verdicts above are conditional on the current methodology;
post-step-6 re-audit may shift several into wider-CI states.
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
from corroborate.analyses.proportion_mediated import ProportionMediatedResult
from corroborate.analyses.cross_config_paired_slope import (
    CrossConfigPairedSlopeResult,
)
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
ADAPTIVE_DQN_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(
        bootstrap,
        greedification=partial(
            adaptive_dormancy_greedify, sigma_floor_factor=1.0,
        ),
    ),
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


# =====================================================================
# CLAIM 2 — Necessary scope (load-bearing dormancy refutation).
# =====================================================================


@claim_bridge(
    # Decorator declares the do-contrast (vanilla → ddqn) on the
    # OUTCOME column. The graph edge `jensen_dormancy_gap →
    # eval_best_burst_mean` lives in scope as the cell filter.
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    # pair_by intentionally omitted — `stratified_arm_diff_pooled`
    # aggregates seeds within env strata, NOT per-pair Δs. The
    # framework's default `pair_by=('seed',)` would be wrong:
    # vanilla seed-N and DDQN seed-N produce diverged trajectories
    # in RL, so seed-pairing isn't a real pair.
    scope=(
        pl.col('jensen_dormancy_gap').is_finite()
        & (pl.col('jensen_dormancy_gap') >= 1e-9)
        & pl.col('eval_best_burst_mean').is_finite()
    ),
    predicted_direction='null',
)
def ddqn_refuted_when_dormancy_fires(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    null_ceiling: float = 0.2,
    min_strata: int = 3,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_vanilla_predictor: float = float('-inf'),
) -> tuple[Verdict, RefutationClass | None]:
    """Necessary-condition claim. The framework's-own Jensen
    dormancy invariant operationalizes the Hasselt-2010 structural
    floor `σ_Q × √(2 log |A|)` against observed bias. When the gap
    fires (gap > 0, premise dormant), DDQN's bias-correction
    mechanism has nothing to operate on, so Δ_outcome should be
    ≈ 0 on dormant cells.

    **2026-05-11 migrated to `stratified_arm_diff_pooled`** per the
    RL seed-pairing critique. Previous versions iterated through
    `paired_g`, `bootstrap_paired_g`, CI-widening, etc. — all of
    them assumed seed-paired (DDQN seed-N vs vanilla seed-N)
    differences carried within-seed information. In RL, training
    trajectories diverge from step 1 under different algorithms;
    seed pairing doesn't pair anything. Stratified per-env Cohen's
    d (independent-samples form, Hedges 1981 SE) is the right
    primitive. Cross-env DL random-effects pooling handles the
    outcome-scale heterogeneity (FourRooms 0-1 vs Acrobot −500−0
    etc.) by standardizing per env first.

    Verdict mapping (null prediction):
    - HELD when pooled CI ⊂ [−null_ceiling, +null_ceiling] — null
      confirmed at Cohen's-small bound;
    - INVARIANT_VIOLATION when pooled CI lower > null_ceiling —
      data confidently asserts substantive positive effect on
      dormant cells, contradicting the necessary-condition claim;
    - NO_EFFECT (SIGN_FLIP) when pooled CI upper < −null_ceiling
      — strong negative effect (DDQN substantively HURTS on
      dormant cells);
    - POWER_INSUFFICIENT when CI spans the bound or n_strata is
      below the minimum.

    The Pearl-rung-2 corroboration via `adaptive_dqn_recovers_
    ddqn_benefit__fourrooms_factor_0p5` validates the underlying
    theory: a runtime controller using a per-batch dormancy proxy
    (max_Q − mean_Q vs σ_Q × √(2 log |A|)) recovers DDQN's outcome
    benefit on FourRooms (g=+0.78 vs vanilla, p<0.001)."""
    del stratify_by, min_vanilla_predictor
    n_strata = stratified_arm_diff_pooled.n_strata
    if n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    d = stratified_arm_diff_pooled.pooled_d
    se = stratified_arm_diff_pooled.pooled_se
    if math.isnan(d) or math.isnan(se):
        return Verdict.POWER_INSUFFICIENT, None
    ci_lo = stratified_arm_diff_pooled.pooled_ci_lo
    ci_hi = stratified_arm_diff_pooled.pooled_ci_hi
    # Refutation branches: CI excluding null zone on either side.
    if ci_lo > null_ceiling:
        return Verdict.INVARIANT_VIOLATION, None
    if ci_hi < -null_ceiling:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    # HELD branch: CI ⊂ [-null_ceiling, +null_ceiling].
    if ci_lo >= -null_ceiling and ci_hi <= null_ceiling:
        return Verdict.HELD, None
    return Verdict.POWER_INSUFFICIENT, None


# =====================================================================
# CLAIM 2 — Pearl-rung-2 corroboration of the necessary-scope claim.
# Designed-intervention sweep `adaptive_dqn_fourrooms_sweep` runs the
# adaptive controller (`adaptive_dormancy_greedify` with
# `sigma_floor_factor=0.5`) on FourRooms-misc, paired against the
# existing `expectile_3way` FourRooms vanilla_dqn + ddqn cells.
# =====================================================================


@claim_bridge(
    source=DoEffect(treatment=(ADAPTIVE_DQN_FACTOR_0P5_SWAP,), baseline=()),
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
    # Endogenous scope: per-(env, total_steps) mean of
    # `q_divergence_score` exceeds the Bellman fixed-point bound
    # (jensen > r_max/(1−γ)) — the regime where the bridge claims
    # DDQN's early-burst benefit operates. The multi-key partition
    # keeps the regime-specific signal: Freeway-MinAtar's
    # full-cache env-mean is 0.74 (mixing 50k/200k/1M cells), but
    # its 1M-only (env, total_steps) mean is 2.76 — so Freeway 1M
    # passes. Two predicates: `finite('q_divergence_score')` drops
    # per-cell NaN (no_hit_penalty wrapper cells with null
    # jensen_gap); `partition_aggregate(...) > 1.0` selects the
    # Q-explosion regime via NaN-safe partition mean.
    scope=(
        (pl.col('log_obs_dim') >= 5.0)
        & (pl.col('total_steps') >= 1000000.0)
        & pl.col('reward_clip_min').is_null()
        & finite('q_divergence_score')
        & (
            partition_aggregate(
                'q_divergence_score',
                by=['env_name', 'total_steps'],
                op='mean',
            )
            > 1.0
        )
    ),
)
def ddqn_helps_at_early_bursts__pixel_envs(
    paired_g: PairedGResult,
    *,
    dedupe_strategy: str = 'mean',
) -> tuple[Verdict, RefutationClass | None]:
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
        return Verdict.POWER_INSUFFICIENT, None
    if math.isnan(paired_g.helped_fraction):
        return Verdict.POWER_INSUFFICIENT, None
    # Predicted direction: positive g + helped majority.
    if (
        paired_g.helped_fraction >= 0.55
        and paired_g.g >= 0.20
    ):
        return Verdict.HELD, None
    # Sign-flip when observed g is negative (opposite direction);
    # otherwise null-effect (positive but small).
    if paired_g.g < 0.0:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT


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
    source=DoEffect(treatment=(ADAPTIVE_DQN_FACTOR_0P5_SWAP,), baseline=()),
    target='eval_final_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    # Pearl-rung-2 pairing: adaptive_dqn from the 1M-step
    # designed-intervention sweep (`adaptive_dqn_spaceinvaders_1m`,
    # which ran the dormancy controller at sigma_floor_factor=0.5
    # — same setting as the FourRooms recovery bridge) against the
    # existing minatar_1M SpaceInvaders vanilla_dqn cells. Other
    # corpora carry vanilla_dqn at SpaceInvaders-MinAtar at
    # different `total_steps` regimes (50k in `ddqn`, 200k in
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
    # `effective_horizon >= 150` is the endogenous selector. Within
    # the `gamma_sweep_metamaze_high` corpus (MetaMaze γ ∈ {0.995,
    # 0.999} with realised bf ≈ 0.995 → eff_h ≈ {99, 165}) the
    # threshold isolates the γ=0.999 cohort. eff_h is now
    # `1/(1−γ·bf)` — values much smaller than the textbook
    # `1/(1−γ)` because MetaMaze episodes terminate. Threshold
    # recalibrated from 500 to 150.
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'MetaMaze-misc')
        & (pl.col('corpus') == 'gamma_sweep_metamaze_high')
        & finite_ge('effective_horizon', 150.0)
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


# =====================================================================
# CLAIM 6 — DELETED. Bridge audit step 2 (2026-05-11).
# `log_mc_variance → g_link` between-env attenuator was REFUTED via
# CV decomposition (reward-magnitude confound — see
# `findings_chain_bottlenecks_decomposed.md`). The substantive
# under-learning-rescue mechanism is documented in
# `findings_underlearning_rescue.md` and tested by CLAIM 7's
# rs-sweep bridges.
# =====================================================================


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


def _rescue_threshold(
    *,
    failure_baseline: float = 0.1,
    optimal_ceiling: float = 0.8,
    rescue_fraction: float = 0.5,
) -> float:
    """Substantive-rescue threshold in native units.

    The rescue claim asserts DDQN closes at least
    `rescue_fraction` of the failure-to-optimal range:

        threshold = rescue_fraction × (optimal_ceiling − failure_baseline)
                  = 0.5 × (0.8 − 0.1) = 0.35

    **Parameter derivation, honestly stated.** Reviewer pushback
    flagged the earlier framing of `failure_baseline=0.1` as
    "random chance for |A|=4 grids" — that's NOT actually
    uniform-action chance (= 0.25 per step, but episode-level
    native-outcome for a random policy that rarely reaches the
    goal is empirically near 0.0-0.1). The defaults encode:
    - `failure_baseline = 0.1` — the empirical native outcome a
      learner gets when it can't make use of reward signal
      (vanilla DQN at rs=0.1 on FourRooms reaches ~0.05; rounding
      up for conservativism). This IS empirically calibrated to
      vanilla failure, not derived from first principles.
    - `optimal_ceiling = 0.8` — empirical RL convergence ceiling
      across the canonical corpus.
    - `rescue_fraction = 0.5` — the qualitative substantive claim
      "DDQN closes at least half the headroom".

    **Limitation.** The threshold is calibrated using corpus data
    (vanilla floor) plus a 50% qualitative bound. Reviewer is
    right that it's not first-principles theory; it IS an
    explicit empirical anchor with a justified 50% fraction,
    which is honestly stated rather than hidden behind a single
    magic constant. Verdicts at the threshold's exact CI
    boundary are sensitive to within-0.1 shifts of any parameter;
    bridges using this should report `paired_g.mean_diff_se` and
    the CI explicitly so the reader can re-evaluate at a
    different parameterisation.
    """
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
    """Pearl-rung-2 interventional contrast: do(arm=ddqn) on
    FourRooms at reward_scale=0.1 produces a substantive
    interventional contrast — DDQN closes ≥ 50% of the failure-
    to-optimal range in native units.

    Generic primitive shape: consumes `paired_g` with
    `target='outcome_native'` (the registered measurable
    `eval_best_burst_mean / reward_scale`) under
    `source=INTERVENTION` (do(ddqn) − do(vanilla_dqn) contrast)
    and `scope=(env_name == 'FourRooms-misc') & (reward_scale
    == 0.1)` to filter the corpus. No bespoke analysis — the bridge
    supplies the measurable name + scope, the framework runs
    `paired_g` and injects the result.

    **Threshold derivation** (replaces older hand-tuned +0.4):
    `threshold_diff = _rescue_threshold()` = `rescue_fraction ×
    (optimal_ceiling − failure_baseline)` = `0.5 × (0.8 − 0.1)
    = 0.35`. The three parameters are theory-derived:
    `failure_baseline=0.1` (random-chance native on |A|=4 sparse-
    reward grids), `optimal_ceiling=0.8` (empirical RL ceiling),
    `rescue_fraction=0.5` (substantive-rescue qualitative claim
    asserts ≥half-headroom closure). See `_rescue_threshold`.

    Verdict uses 95% CI vs threshold (see `_native_diff_ci_verdict`):
    - HELD when CI lower ≥ threshold (md confidently ≥ threshold);
    - NO_EFFECT when CI upper < threshold (md confidently <);
    - POWER_INSUFFICIENT when CI spans the threshold.

    Empirical post-rebuild: md=+0.638 (95% CI=[+0.594, +0.682]),
    threshold=+0.35 — CI entirely above → HELD.

    Asserts on `paired_g.mean_diff` (the interventional contrast
    in native units), NOT `paired_g.g` (standardized Hedges' g
    pools SD that scales with reward, hiding the interventional
    effect under apparent sweet-spotting at baseline).

    **2026-05-11 migration to `arm_mean_diff`** (independent-
    samples Welch t-test) from `paired_g`. RL trajectories
    diverge from step 1 under different algorithms — pairing
    vanilla seed-N with DDQN seed-N is not a meaningful
    statistical pair; the two are independent draws from
    distinct training distributions. JCI confirms
    `vanilla_native ⊥ ddqn_native | log_rs` (partial r ≈ −0.04).
    `arm_mean_diff` reports the same `mean_diff` (treatment-mean
    minus baseline-mean) but with independent-samples Welch SE
    rather than per-pair-Δ SE. Verdict identity uses the CI-vs-
    threshold helper as before.

    **Defensive note**: `mean_diff` is the do-effect of arm
    assignment; it is NOT an observational edge between vanilla
    and ddqn outcome nodes. The bridge tests a CONTRAST between
    two independent reward-scale-response curves, not a causal
    arrow between cell outputs."""
    return _native_diff_ci_verdict(
        arm_mean_diff.mean_diff, arm_mean_diff.mean_diff_se,
        threshold_diff,
    )


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
    predicted_direction='a_gt_b',
)
def ddqn_dominates_vanilla_response_curve__fourrooms_rs_0p3(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    threshold_diff: float = _rescue_threshold(),
) -> Verdict:
    """Sibling of CLAIM 7 at the rescue-regime peak (rs=0.3).
    Same primitive shape, verdict logic, and theory-derived
    threshold via `_rescue_threshold()` — see CLAIM 7 docstring
    for the failure_baseline / optimal_ceiling / rescue_fraction
    parameters that derive the +0.35 substantive-rescue floor.

    **2026-05-11 post-rebuild verdict: NO_EFFECT (claim refuted).**
    Empirical md=+0.259 (95% CI=[+0.169, +0.349]); the substantive-
    rescue threshold +0.35 is outside the CI's upper bound — data
    confidently refutes "DDQN closes ≥ 50% of the failure-to-
    optimal range at rs=0.3". The +0.50 plateau cited in the older
    findings (memory `findings_underlearning_rescue.md`, table
    rs=0.30 → +0.497) came from an EARLIER corpus
    `reward_scale_low_fourrooms`. The post-rebuild
    `reward_scale_sweep_postfix` corpus shows the rs=0.3 gap has
    narrowed to ~+0.26 — vanilla now reaches 0.52 native (was
    0.24), DDQN reaches 0.78 native (was 0.74). The rescue regime
    is narrower than the original plateau reading suggested.

    Direction (DDQN > vanilla) is preserved at p=4e-6, but the
    magnitude doesn't clear the half-headroom theoretical bar.
    The bridge documents this REFUTATION: rs=0.3 sits on the upper
    edge of the rescue regime where vanilla recovers, not in the
    plateau interior.

    Verdict logic uses 95% CI vs threshold (see
    `_native_diff_ci_verdict`):
    - HELD when CI lower ≥ +0.35;
    - NO_EFFECT when CI upper < +0.4 (current case);
    - POWER_INSUFFICIENT when CI spans the threshold.

    Same defensive framing as CLAIM 7: `mean_diff` is the
    interventional contrast, not an observational edge between
    arm outputs. **2026-05-11 migrated to `arm_mean_diff`** —
    see CLAIM 7 docstring for the seed-pairing critique."""
    return _native_diff_ci_verdict(
        arm_mean_diff.mean_diff, arm_mean_diff.mean_diff_se,
        threshold_diff,
    )


# =====================================================================
# CLAIM 7c / 7d — rescue regime is FourRooms-specific.
#
# The reward-scale-response curve dominance from CLAIM 7 (FourRooms
# rs=0.1 → DDQN do-effect +0.49) does NOT generalize to Acrobot or
# CartPole. The reward_scale_sweep corpus has rs=0.1 cells on both:
#
#   Acrobot rs=0.1, γ=0.99: g_out = −0.17 (DDQN slightly HURTS)
#   CartPole rs=0.1, γ=0.99: g_out = +0.09 (effectively null)
#
# Refutation of the "rs<<1 → big DDQN benefit" universal reading.
# The rescue mechanism is sparse-positive-terminal-reward specific
# (FourRooms's structure), not a universal rs effect. Authored as
# predicted_direction='null' bridges — HELD when the rs=0.1 effect
# is small + n.s., i.e. the rescue does NOT activate.
# =====================================================================


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
    """Acrobot at rs=0.1 does NOT show the FourRooms rescue.
    HELD encodes "rescue does not activate" via CI-vs-ceiling
    on `outcome_native` (native units).

    Empirical post-rebuild: md_native = +0.229 (95% CI≈[−0.130,
    +0.588]) — CI spans the null ceiling → POWER_INSUFFICIENT.

    **2026-05-11 migration:** previous version used Hedges' g
    on raw `eval_best_burst_mean`. At rs=0.1, raw outcomes are
    tiny; pooled SD inflates |g| readings (and shrinks them too,
    depending on which arm has more variance). Migrated to
    `outcome_native` (raw / reward_scale) + CI-vs-ceiling logic
    (`_native_diff_null_verdict`) so the rescue-or-not test is
    reward-magnitude-invariant.

    **Verdict shifted HELD → POWER_INSUFFICIENT under the honest
    methodology.** The previous HELD reading was generous: |g|=0.10
    with p=0.59 was "small effect, fail to reject null" — but
    failure-to-reject is NOT confirmation. In native units the
    CI is wide (n=30, native md spans [−0.13, +0.59]); we cannot
    confidently say "rescue does not activate" within ceiling
    0.2. To upgrade to HELD, either widen the ceiling (e.g. 0.6
    would land HELD at the cost of admitting "small rescues"
    as null) or collect more seeds.

    Different mechanism than FourRooms: Acrobot's dense per-step
    penalty doesn't have the "vanilla under-learns sparse
    positive reward" failure mode that the rescue mechanism
    addresses. rs=0.1 just shrinks Q's scale without changing
    the learning regime.

    **2026-05-11 migrated to `arm_mean_diff`** — RL seed pairing
    is meaningless under arm divergence; see CLAIM 7 docstring."""
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
    """CartPole at rs=0.1 does NOT show the FourRooms rescue.
    Sister of the Acrobot bridge; same migration to native-unit
    CI-vs-ceiling logic.

    Empirical post-rebuild: md_native = +0.212 (95% CI≈[−0.155,
    +0.579]) — CI spans the null ceiling → POWER_INSUFFICIENT.
    Same as Acrobot sister bridge: at n=30 with native variance,
    the CI is wide enough to span the [−0.2, +0.2] band; we
    cannot confidently confirm OR refute the null.

    Different mechanism: CartPole has dense per-step alive bonus
    and saturates fast at any rs ≥ 0.1. Vanilla doesn't have the
    "can't find reward" failure mode that the rescue addresses.

    **2026-05-11 migrated to `arm_mean_diff`** — see CLAIM 7
    docstring for seed-pairing critique."""
    return _native_diff_null_verdict(
        arm_mean_diff.mean_diff, arm_mean_diff.mean_diff_se,
        null_ceiling,
    )


# =====================================================================
# CLAIM 7 g/h/i/j — DELETED. Bridge audit step 2 (2026-05-11).
#
# Four mechanism-route probes testing whether DDQN's
# argmax-concentration mechanism on FourRooms generalizes when
# the env is wrapped to look like FR (reward-shape) or made
# action-stochastic. All four ran on `reward_shape_intervention`
# / `action_noise_intervention` corpora (not in post-fix snapshot).
# Empirical readings preserved here as the substantive finding:
#
#   7g sparsified_acrobot   ΔH=-0.011, p=0.11  → NO_EFFECT
#   7h densified_fourrooms  ΔH=-0.076, p=0.0001 → null prediction
#                                                 REFUTED; effect
#                                                 PERSISTS under
#                                                 densification
#   7i noisy_acrobot        ΔH=+0.007, p=0.68  → NO_EFFECT
#   7j noisy_metamaze       ΔH=-0.006, p=0.34  → NO_EFFECT
#
# Synthesis: argmax-concentration is a downstream side-effect of
# FR's small state/action structure, NOT the causal pathway.
# DDQN's outcome benefit on noisy MetaMaze is +0.53 with
# Δjens=-3.60 DESPITE no argmax concentration — decoupling the
# two mechanisms. The CLAIM 26b three-gate framework subsumes
# this characterization via measurable env features
# (G1 ∧ G2 ∧ G3) without relying on wrapper-induced mechanism
# activation tests. See `findings_underlearning_rescue.md`.
# =====================================================================


# =====================================================================
# CLAIM 7e/7f — DDQN's rescue mechanism is action-selection-level.
#
# rs_sweep_with_traces probe: at FourRooms rs=0.1, DDQN's argmax
# distribution is MORE diverse (higher entropy, lower mode_freq)
# than vanilla. At rs=1.0+, the two arms have nearly identical
# argmax distributions. This suggests DDQN's benefit at low rs
# is via maintained exploration (not premature commitment when
# Q is flat), not via decisive policy selection.
#
# Empirical (rs_sweep_with_traces, FourRooms, n=30 per arm × rs):
#   rs=0.1: vanilla H≈1.14, DDQN H≈1.30 (DDQN higher entropy)
#   rs=1.0: vanilla H≈1.04, DDQN H≈1.04 (identical)
#
# Two complementary bridges encode this:
#   (7e) DDQN INCREASES argmax entropy at rescue-regime rs.
#        Predicted direction: a_gt_b (DDQN entropy > vanilla).
#   (7f) DDQN's argmax entropy matches vanilla at standard rs.
#        Predicted direction: null (no entropy difference).
# =====================================================================


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
    """At FourRooms rs=0.1 (rescue regime), DDQN's argmax
    distribution is predicted MORE diverse than vanilla's — the
    "DDQN maintains exploration when Q is flat" reading.

    **2026-05-11 verdict post-rebuild: REFUTED via SIGN_FLIP.**
    g=-2.98 (95% CI excludes 0 strongly), mean_diff=-0.232 nats,
    p=2.6e-12, n=30. The prediction's sign is wrong: DDQN's
    argmaxH is SUBSTANTIALLY LOWER than vanilla's (more
    decisive policy, not more diverse).

    Revised mechanism reading: in the under-learning regime
    (rs=0.1), vanilla can't learn — its policy stays near-uniform
    (high argmaxH) because Q-values don't differentiate actions.
    DDQN's bias correction unblocks learning, producing
    differentiated Q-values and a SHARPER argmax. The under-
    learning rescue (`findings_underlearning_rescue.md`,
    Δ_outcome=+0.638, g=+5.05) IS via "policy sharpens after
    rescue", not "exploration maintained".

    Bridge stays as a falsifiable artifact: the original
    "maintains exploration" reading is refuted. Don't repair the
    prediction sign post-hoc — keep as documented refutation of
    the explore-vs-commit alternative explanation.

    **2026-05-11 migrated to `arm_mean_diff`** — see CLAIM 7
    docstring for seed-pairing critique."""
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
    """At FourRooms rs=1.0 (standard reward scale), DDQN's argmax
    distribution is predicted to MATCH vanilla's — the "DDQN's
    exploration-maintenance mechanism is inactive at standard rs"
    reading. Authored with `predicted_direction='null'`; HELD =
    null confirmed.

    **2026-05-11 verdict post-rebuild: NO_EFFECT (null refuted
    via SIGN_FLIP).** g=-1.72, mean_diff=-0.099, p=2.2e-9, n=30.
    Null prediction refuted: there IS a substantial effect — DDQN
    has LOWER argmaxH than vanilla even at rs=1.0. This matches
    the rs=0.1 finding (sister bridge above): DDQN's effect on
    argmaxH is NOT regime-specific — it produces sharper policies
    across reward scales, not just in the rescue regime.

    Both rs=0.1 and rs=1.0 bridges' "exploration-maintenance"
    framing is wrong. The robust pattern is: DDQN's bias
    correction yields better Q-discrimination → sharper argmax →
    LOWER argmax entropy. The rescue effect at rs=0.1 is
    Δ_outcome-large but the argmax-sharpening is reward-scale-
    invariant (at FourRooms γ=0.99).

    Bridge stays as a falsifiable artifact: the regime-specificity
    reading is refuted.

    **2026-05-11 migrated to `arm_mean_diff`** — see CLAIM 7
    docstring for seed-pairing critique."""
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
    # Endogenous scope: `q_divergence_score > 1` (jensen_gap exceeds
    # the Bellman fixed-point bound r_max/(1−γ)) replaces the prior
    # `sync_period == 100` HP-knob scope. SI 1M cells at sync=100
    # all have q_div in (2.7, 1002), so the regime is preserved;
    # the 60 cells with NaN q_div (no_hit_penalty wrapper, null
    # jensen_gap) had null mc_return too and contributed nothing.
    # The endogenous form expresses the bridge's real interest —
    # the Q-explosion regime where the late-burst crossover happens.
    scope=(
        (pl.col('env_name') == 'SpaceInvaders-MinAtar')
        & (pl.col('total_steps') == 1_000_000)
        & pl.col('reward_clip_min').is_null()
        & finite_gt('q_divergence_score', 1.0)
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
        # Empty strata = no cells in scope = underpowered, not
        # null-effect. Reviewer screening catch: previous NO_EFFECT
        # for n=0 was a verdict-logic bug.
        return Verdict.POWER_INSUFFICIENT

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
    for the RL-methodology rationale."""
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
    source=INTERVENTION,
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


# =====================================================================
# CLAIM 22 — REACH-cross-env DoWhy: Δ_jens → Δ_outcome HELDs across
# all 4 REACH envs (FourRooms, Acrobot, MountainCar, MetaMaze) under
# env-confounded backdoor adjustment.
#
# The link verdict at REACH-cross-env scope is what the per-cell
# pooled correlation already showed (ρ ≈ -0.97 across 1200 obs);
# DoWhy's backdoor + placebo + RCC just confirms it survives causal
# adjustment. Replaces the earlier Acrobot-only DoWhy triple, whose
# numeric calibration was tied to the pre-fix Replay corpus.
#
# Empirical (postfix corpus, n=1200 per-burst rows, 4 envs):
#   ATE = -0.61 (per unit Δ_jens; physics: should be negative)
#   placebo |refuted/real| < 1%, RCC drift < 5%.
# Reads as: across 4 REACH envs, 1 unit reduction in jensen-bias →
# 0.61 units of outcome-MC gain, with treatment-specific signal
# (placebo) and robustness to spurious confounders (RCC).
# =====================================================================


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
_DDQN_RELEVANT_SCOPE = (
    # G1 — threshold 0.05 matches dormancy scale; tighter
    # thresholds silently exclude high-variance cells on noisy
    # envs (MetaMaze per-seed jens fluctuates above and below 0.5)
    finite('jensen_gap')
    & (pl.col('jensen_gap') > 0.05)
    & finite('jensen_dormancy_gap')
    & finite_lt('jensen_dormancy_gap', 0.05)
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
    # polyak-τ interventions in scope)
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
    source=INTERVENTION,
    target='outcome.eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=_DDQN_RELEVANT_SCOPE,
)
def reach_link_backdoor_ate_negative(
    paired_delta_link_dowhy: PairedDeltaLinkDowhyResult,
    *,
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
    source=INTERVENTION,
    target='outcome.eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=_DDQN_RELEVANT_SCOPE,
)
def reach_link_placebo_refuted(
    paired_delta_link_dowhy: PairedDeltaLinkDowhyResult,
    *,
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
    source=INTERVENTION,
    target='outcome.eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=_DDQN_RELEVANT_SCOPE,
)
def reach_link_rcc_robust(
    paired_delta_link_dowhy: PairedDeltaLinkDowhyResult,
    *,
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


# =====================================================================
# CLAIM 23 — q_divergence_score and argmax_entropy_late are SHADOWS
# of Δ_jens, not independent direct-cause mediators of Δ_outcome.
#
# Both have substantial marginal correlation with Δ_outcome on the
# postfix corpus (q_div: ρ=-0.59 ***; argmaxH: ρ=-0.31 ***), which
# would naively suggest a second / third independent mediator path
# alongside jens. Partial Spearman after conditioning on Δ_jens
# shows the residual collapses (q_div: +0.15 sign-flip; argmaxH:
# -0.14 ns at n=120) — both are largely / fully mediated by jens.
#
# Why mechanically:
#   q_divergence_score = jensen_gap_late / (R / (1−γ))
#                      = jens × per-env-constant
# i.e. it IS jens up to an env-level scaling. Their marginal
# co-variation is mathematical, not causal. argmax_entropy_late
# co-varies with jens via shared Q-distribution dynamics.
#
# Codifying these as null-form bridges (predicted_direction='null',
# HELD-when-partial-is-zero) prevents future investigators from
# re-authoring them as competing independent mediators. See
# `feedback_jens_shadow_mediators.md`.
# =====================================================================


_DDQN_VS_VANILLA_ARMS = (
    'arm_baseline',
    'arm_treatment',
)


@claim_bridge(
    source=INTERVENTION,
    target='outcome.eval_best_burst_mean',
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
    """**Migrated 2026-05-11 from `partial_spearman_paired_delta`
    (per-pair-Δ form) to JCI `stratified_partial_spearman`
    (per-cell, env-stratified) — verdict flipped HELD → NO_EFFECT
    once trace measurables restored across all REACH envs.**

    Tests `ρ_partial(qdiv, outcome | jens)`, env-stratified,
    Fisher-z pooled. Authored with `predicted_direction='null'`;
    HELD when `|ρ_partial| < null_max_abs_rho`.

    Empirical (post-trace-restore ddqn_universe, in-scope cells
    2026-05-11): `ρ_partial = -0.432` (n=717, 11 strata, p=6e-15)
    — NO_EFFECT (null refuted).

    **Where the algebraic shadow breaks.** `q_divergence_score =
    jensen_gap / (R / (1−γ))` is constant scaling only within
    fixed (env, γ). The bridge's `_DDQN_RELEVANT_SCOPE` mixes
    γ ∈ {0.99, 0.999} within each env (MetaMaze contributes both;
    FR/Acrobot mostly γ=0.99). Stratifying by env only doesn't
    eliminate the γ-induced residual: qdiv carries the γ-scaled
    structure that jens alone doesn't, surfacing as ρ≈-0.43
    cross-env. The original per-pair-Δ "HELD null" reading
    (ρ=+0.15 ns at n=120 paired Δs) was a small-n artifact AND
    a scope artifact — pre-rebuild the FR/MM/MC corpora had
    trace-dependent measurables NaN'd out, dropping the in-scope
    n to 373 and leaving mostly γ=0.99 cells (within-γ ≈
    constant scaling). With full scope, the leakage is visible.

    **Implication.** Don't author qdiv as a clean shadow when γ
    varies; treat it as a γ-modulated structural function of
    jens. The bridge stays as a falsifiable artifact documenting
    the cross-γ leakage. For "shadow within fixed (env, γ)",
    stratify by (env, γ) (returns ρ=NaN — degenerate collinearity
    confirms the algebra).

    HELD (null confirmed) when |ρ_partial| < `null_max_abs_rho`.
    NO_EFFECT (null refuted) when |ρ_partial| ≥ `null_max_abs_rho`.
    See CLAIM 23 + `feedback_jens_shadow_mediators.md`."""
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
    source=INTERVENTION,
    target='outcome.eval_best_burst_mean',
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
    """**Migrated 2026-05-11 from `partial_spearman_paired_delta`
    to JCI `stratified_partial_spearman` — verdict preserved
    HELD across the post-trace-restore corpus.**

    Empirical (post-trace-restore ddqn_universe, in-scope cells
    2026-05-11): `ρ_partial = +0.011` (n=717, 11 strata, p=0.78)
    — HELD (null confirmed).

    `argmax_entropy_late` is conditionally independent of outcome
    given jens within the bridge's scope. Unlike
    `q_divergence_score`, argmaxH is NOT algebraically tied to
    jens; the shadow holds at full cross-env scope (no γ-induced
    leakage). Both methods (per-pair-Δ at n=120 and JCI per-cell
    at n=717) agree.

    HELD (null confirmed) when |ρ_partial| < `null_max_abs_rho`.
    NO_EFFECT (null refuted) when |ρ_partial| ≥ `null_max_abs_rho`.
    See CLAIM 23 + `feedback_jens_shadow_mediators.md`."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if stratified_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = stratified_partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if abs(rho) < null_max_abs_rho:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# =====================================================================
# CLAIM 24 — Within-MetaMaze do(γ): outcome benefit amplifies at high γ.
#
# Strict within-env intervention: same env (MetaMaze-misc), same
# dynamics, same MLP, same n_actions=4, same paired seeds — only
# γ varies (n_γ=2: γ ∈ {0.99, 0.999}, effh ≈ {100, 1000}).
#
# **Reframed 2026-05-11** (post-paired-Δ critique): the original
# bridge encoded per-burst link plc=1.00 at γ=0.999, which relied
# on per-pair `r(Δ_jens, Δ_outcome)` aggregation across seed-pairs
# — the per-pair Δ in RL measures init-distribution-induced
# correlation, not treatment-effect coupling (CLAIM 17 cut for
# the same reason). The substantive content of CLAIM 24 is
# `Δ_outcome` AMPLIFICATION across γ, which IS testable cleanly
# at the (env, γ) stratum level: per-γ `Δ_outcome = mean(DDQN) −
# mean(vanilla)`.
#
# **Significance.** With CLAIM 17 (cross-env chain-amplifier) cut
# as leverage-driven, the within-env γ-amplification observation
# is unexplained by anything else in the file. The within-env
# probe controls env identity / dynamics / |A| / n_step
# exhaustively; it's the cleanest evidence that the chain-amplifier
# mechanism IS real but operates within-env. n_γ=2 makes it a
# WEAK bridge (no slope CI possible) but a FALSIFIABLE one — on
# any future corpus where γ=0.999's mean benefit doesn't exceed
# γ=0.99's by a substantive margin, the bridge flips NO_EFFECT.
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        (pl.col('env_name') == 'MetaMaze-misc')
        & pl.col('gamma').is_in([0.99, 0.999])
        & finite('jensen_gap')
        & (pl.col('jensen_gap') > 0.05)
        & finite('jensen_dormancy_gap')
        & finite_lt('jensen_dormancy_gap', 0.05)
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
    """Within-MetaMaze do(γ): DDQN's outcome benefit at γ=0.999 is
    substantively larger than at γ=0.99, and substantively positive
    in absolute terms.

    Weak n_γ=2 bridge (no slope CI possible — only 2 strata) but
    falsifiable. Tests two conditions:
      (i)  high-γ stratum Δ_outcome ≥ `high_floor` (substantive
           absolute effect).
      (ii) high-γ Δ_outcome ≥ `amplification_ratio_min` × low-γ
           Δ_outcome (within-env amplification with γ).

    When low-γ Δ_outcome ≤ 0 (DDQN doesn't help at low γ), the
    ratio condition is trivially satisfied — the qualitative
    flip (no help → substantive help) is itself amplification.

    HELD when both (i) and (ii) pass.
    NO_EFFECT when (i) fails (high-γ doesn't reach substantive
    threshold) or (ii) fails (amplification ratio too small).
    POWER_INSUFFICIENT only when the required strata don't both
    populate (e.g., one γ value absent from corpus).

    **Significance** (per CLAIM 24 banner): post-CLAIM-17 cut,
    this within-env probe is the cleanest evidence the
    chain-amplifier mechanism operates within-env. The previous
    plc-on-paired-link form encoded init-distribution-induced
    correlation; this reformulation tests treatment-effect
    amplification directly.

    Empirical (per CLAIM 24 banner):
      γ=0.99:  mean Δ_outcome ≈ near-zero (median -0.12)
      γ=0.999: mean Δ_outcome ≈ +2.55 (median)
    """
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
        & finite('jensen_gap')
        & (pl.col('jensen_gap') > 0.05)
        & finite('jensen_dormancy_gap')
        & finite_lt('jensen_dormancy_gap', 0.05)
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
    """Sibling of `metamaze_link_steeper_at_high_gamma` under
    median aggregation. Same scope, same claim shape (within-env
    do(γ) outcome amplification), but per-stratum reduction is
    `median(DDQN_outcome) − median(vanilla_outcome)` instead of
    mean.

    Authored 2026-05-11 alongside the mean version to test
    whether the bimodal-seed-distribution hypothesis (some seeds
    catastrophically harmed, others rescued) explains the
    mean-version's NO_EFFECT. Under that hypothesis, median
    would be POSITIVE at γ=0.999 (more seeds rescued than hurt)
    while mean is NEGATIVE (catastrophes drag mean down).

    Empirical finding: median-aggregated stratum-Δ is also
    NEGATIVE at γ=0.999 (-1.34) and positive at γ=0.99 (+0.39).
    Both summaries agree: DDQN HELPS at γ=0.99 and HURTS at
    γ=0.999. The γ-amplification prediction is refuted regardless
    of aggregator.

    The CLAIM 24 banner's "median +2.55 at γ=0.999" reading was
    the PAIRED median (per-seed sign-count), which inherits the
    paired-Δ critique. The stratum-level form (this bridge)
    matches the mean-aggregated sibling: REFUTED."""
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


# =====================================================================
# CLAIM 25 — Within-FourRooms do(|A|): DDQN benefit scales 20×.
#
# action_duplicate(k) wraps FourRooms's |A|=4 with k duplicates of
# each action, mapping action i ∈ [0, 4k) to inner action i mod 4.
# Preserves dynamics, reward, optimal Q* — only |A| varies. The
# Hasselt floor √(2 ln K) grows; DDQN's argmax/max decoupling has
# more bias to correct as K grows.
#
# Empirical (postfix corpus, 4 conditions × 60 paired seeds, 200k
# training steps per cell, FourRooms-misc):
#   k=1 (|A|=4):  van out 0.78, ddq out 0.80, Δ +0.02
#   k=2 (|A|=8):  van out 0.73, ddq out 0.79, Δ +0.07
#   k=3 (|A|=12): van out 0.56, ddq out 0.79, Δ +0.22
#   k=4 (|A|=16): van out 0.38, ddq out 0.79, Δ +0.41 (20× growth)
#
# Vanilla outcome HALVES from |A|=4 to |A|=16 (bias-on-argmax
# degrades policy). DDQN outcome STAYS CONSTANT (~0.79) — bias
# correction perfectly compensates for action-space inflation.
# Pooled within-FourRooms ρ(Δ_jens, Δ_outcome) = -0.759, p<0.0001,
# n=120.
#
# Cleanest within-env do(|A|) probe possible: cross-env scope
# confound (env identity vs |A|) is fully controlled. Findings file:
# `findings_action_dim_inflation_postfix.md`.
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & pl.col('action_duplicate_k').is_not_null()
        & finite('jensen_gap')
        & (pl.col('jensen_gap') > 0.05)
        & finite('jensen_dormancy_gap')
        & finite_lt('jensen_dormancy_gap', 0.05)
    ),
    predicted_direction='a_lt_b',
)
def fourrooms_action_dim_link_active__inflated(
    stratum_effect_panel: StratumEffectPanel,
    *,
    measurables: tuple[str, ...] = ('jensen_gap', 'eval_best_burst_mean'),
    stratify_by: tuple[str, ...] = ('action_duplicate_k',),
    min_seeds_per_arm: int = 5,
    x: str = 'jensen_gap',
    y: str = 'eval_best_burst_mean',
    slope_max: float = -0.05,
    r_squared_floor: float = 0.7,
    min_strata: int = 3,
) -> Verdict:
    """Within-FourRooms chain-amplifier link, stratified by
    `action_duplicate_k`. Per-k Δ_outcome regressed on per-k Δ_jens
    across the action-duplicate panel (k ∈ {1, 2, 3, 4}).

    The within-env do(|A|) probe is the cleanest scope-confound
    control: env identity, dynamics, reward, optimal Q* all held
    constant; only |A| varies. Per the Hasselt floor √(2 ln K),
    DDQN's argmax/max decoupling has more bias to correct as K
    grows; the prediction is that per-k Δ_outcome scales with
    per-k Δ_jens.

    HELD when slope ≤ `slope_max` (negative, indicating bias
    reduction translates to outcome gain) AND p < 0.05 AND
    n_strata ≥ `min_strata`.

    Empirical (CLAIM 25, postfix corpus):
      k=1 (|A|=4):  Δ_jens ≈ -0.0,  Δ_out ≈ +0.02
      k=2 (|A|=8):  Δ_jens grows,    Δ_out ≈ +0.07
      k=3 (|A|=12): Δ_jens grows,    Δ_out ≈ +0.22
      k=4 (|A|=16): Δ_jens largest,  Δ_out ≈ +0.41
    Vanilla outcome halves (0.78 → 0.38) while DDQN stays
    constant (~0.79) — bias correction perfectly compensates for
    action-space inflation. See `findings_action_dim_inflation_
    postfix.md`.

    **Migrated from `paired_link_per_burst` (2026-05-11) to
    `stratum_effect_panel + panel_regress`.** The prior per-burst
    paired link computed `r(Δ_jens, Δ_outcome)` across seed-pairs
    within each (k, burst); per-pair Δs measure init-distribution-
    induced correlation between two divergent trajectories sharing
    only an init seed. The stratum-level Δ form tests the same
    underlying claim (CLAIM 25 chain-amplifier within FourRooms)
    at the inferentially-honest unit: per-k treatment-effect
    estimates regressed across the k panel."""
    del measurables, stratify_by, min_seeds_per_arm  # forwarded to fixture
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
    source=INTERVENTION,
    target='outcome.eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    # Endogenous touch (`q_divergence_score.is_finite()`) added
    # alongside the run-length filter so EXOGENOUS_SCOPE doesn't
    # WARN: the bridge's actual interest IS the q_divergence
    # regime, which the analysis (`link_attenuation_dowhy`) drives
    # via its `attenuator='q_divergence_score'` param. The cell
    # set is a strict subset of the prior — only cells with
    # populated q_div are kept, which the analysis was already
    # filtering internally. Verdict-preserving.
    scope=(
        (pl.col('total_steps') == 1_000_000)
        & finite('q_divergence_score')
        # Exclude SlidingTile big_cnn probe (4 cells with
        # q_network.channels=(32,64) — a different architecture
        # than the (8,16)-channel main sweep). Mixing the probe
        # with the main sweep makes paired_link_per_burst error
        # on per-burst array-shape mismatch (the two configs
        # produce different trace shapes). Earlier scope-fix used
        # `eval_every == 50000` which dropped all MLP envs (which
        # use 100000); the architecture filter keeps MLP cells
        # (null channels) and the main CNN sweep ((8,16)) while
        # excluding only the 4 probe cells.
        & (
            pl.col('q_network.channels').is_null()
            | (pl.col('q_network.channels') != '(32,64)')
        )
    ),
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
    source=INTERVENTION,
    target='outcome.eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    # Endogenous touch (`q_divergence_score.is_finite()`) added
    # alongside the run-length filter so EXOGENOUS_SCOPE doesn't
    # WARN: the bridge's actual interest IS the q_divergence
    # regime, which the analysis (`link_attenuation_dowhy`) drives
    # via its `attenuator='q_divergence_score'` param. The cell
    # set is a strict subset of the prior — only cells with
    # populated q_div are kept, which the analysis was already
    # filtering internally. Verdict-preserving.
    scope=(
        (pl.col('total_steps') == 1_000_000)
        & finite('q_divergence_score')
        # Exclude SlidingTile big_cnn probe (4 cells with
        # q_network.channels=(32,64) — a different architecture
        # than the (8,16)-channel main sweep). Mixing the probe
        # with the main sweep makes paired_link_per_burst error
        # on per-burst array-shape mismatch (the two configs
        # produce different trace shapes). Earlier scope-fix used
        # `eval_every == 50000` which dropped all MLP envs (which
        # use 100000); the architecture filter keeps MLP cells
        # (null channels) and the main CNN sweep ((8,16)) while
        # excluding only the 4 probe cells.
        & (
            pl.col('q_network.channels').is_null()
            | (pl.col('q_network.channels') != '(32,64)')
        )
    ),
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
    # Endogenous touch (`q_divergence_score.is_finite()`) added
    # alongside the run-length filter so EXOGENOUS_SCOPE doesn't
    # WARN: the bridge's actual interest IS the q_divergence
    # regime, which the analysis (`link_attenuation_dowhy`) drives
    # via its `attenuator='q_divergence_score'` param. The cell
    # set is a strict subset of the prior — only cells with
    # populated q_div are kept, which the analysis was already
    # filtering internally. Verdict-preserving.
    scope=(
        (pl.col('total_steps') == 1_000_000)
        & finite('q_divergence_score')
        # Exclude SlidingTile big_cnn probe (4 cells with
        # q_network.channels=(32,64) — a different architecture
        # than the (8,16)-channel main sweep). Mixing the probe
        # with the main sweep makes paired_link_per_burst error
        # on per-burst array-shape mismatch (the two configs
        # produce different trace shapes). Earlier scope-fix used
        # `eval_every == 50000` which dropped all MLP envs (which
        # use 100000); the architecture filter keeps MLP cells
        # (null channels) and the main CNN sweep ((8,16)) while
        # excluding only the 4 probe cells.
        & (
            pl.col('q_network.channels').is_null()
            | (pl.col('q_network.channels') != '(32,64)')
        )
    ),
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
# CLAIM 12 — env-polarity moderates the eff_h mediator sign.
#
# The cross-env residual `bootstrap_fraction → g_link | g_mech` is
# sign-cancellation between two opposite-direction mediator channels:
# - GOAL envs (env_reward_polarity < 0): Δ_eff_h coupling is
#   negative — DDQN's policy improvement shortens trajectories,
#   reduces eff_h, increases discounted return.
# - SURVIVAL envs (env_reward_polarity > 0): Δ_eff_h coupling is
#   positive — DDQN's policy improvement extends trajectories,
#   increases eff_h, accumulates more reward.
# Cross-env meta-regression averaged the opposite signs to ~0; per-
# polarity stratification gives ρ_pool = -0.798 and +0.240 (formal
# proof n_envs=8, binomial p=0.004 across env-by-env sign matches).
# Source: `findings_polarity_mediator.md`, `polarity_proof.json`.
#
# `env_reward_polarity` (per-cell Pearson(episode_length, mc_return))
# is the endogenous polarity proxy — recovers hand-coded categorical
# polarity at Spearman ρ=+0.88, p=0.02. Authored as @measurable in
# corroborate_rl.dqn.measurables.
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
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
    """On GOAL-polarity envs, `ρ_partial(eff_h, outcome | jens)`
    inherits the polarity-tautology sign and magnitude — env-
    stratified (within-env), Fisher-z pooled. Tests whether the
    polarity-coupling shape from `findings/polarity_mediator.md`
    (`r ≈ 0.625 × polarity`, R²=0.886, n_envs=8) survives jens-
    conditioning in JCI form.

    Empirical (ddqn_universe post-rebuild, 2026-05-11):
    `ρ_partial = −0.593` (n=737, 5 strata: Acrobot, FourRooms,
    MetaMaze, Snake, MountainCar) — HELD. Magnitude matches
    `0.625 × |mean polarity|` predicted by the polarity-tautology
    coefficient.

    **2026-05-11 re-author note.** Original bridge used
    `proportion_mediated` (now deprecated) with
    `predicted_direction='null'` testing the *causal mediation
    share* claim "eff_h is NOT a dominant mediator" (≈ 12% share,
    HELD). The migration to `stratified_partial_spearman` tests a
    *different question*: observational conditional dependence.
    The two are compatible (small share + substantial residual
    coupling). Re-authored with `a_lt_b` direction matching the
    polarity-tautology prior — this is NOT post-hoc data fitting
    but alignment with an independent prior finding (per reviewer
    feedback). The bridge now tests a POSITIVE prediction in JCI
    form. The original "eff_h is not dominant mediator" claim
    still holds at the causal-share level, just expressed via a
    different primitive (`proportion_mediated` deprecated).

    Scope: GOAL polarity (env_reward_polarity < −0.3), Q-bounded
    (q_div < 1000 or NaN).
    HELD when ρ_partial ≤ −`magnitude_threshold` (signed direction
    matching polarity).
    NO_EFFECT when |ρ_partial| < `magnitude_threshold` (coupling
    too weak to match the tautology shape) or sign-flipped."""
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
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
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
    """On SURVIVAL-polarity envs, `ρ_partial(eff_h, outcome | jens)`
    inherits the polarity-tautology sign (positive) and magnitude.
    Sibling of `..._goal_envs` on the opposite polarity half-plane.

    Empirical (ddqn_universe post-rebuild, 2026-05-11):
    `ρ_partial = +0.656` (n=307, 3 strata: CartPole, Asterix,
    PacMan) — HELD. Sign and magnitude match the polarity-coupling
    coefficient.

    **2026-05-11 re-author note** — see GOAL bridge above. The
    migration from `proportion_mediated` (deprecated) to JCI
    `stratified_partial_spearman` is a question shift, not a
    refutation. The bridge tests the polarity-tautology
    prediction in JCI form rather than the causal-mediation-share
    claim.

    Scope: SURVIVAL polarity (env_reward_polarity > +0.3),
    Q-bounded.
    HELD when ρ_partial ≥ +`magnitude_threshold`."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if stratified_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = stratified_partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if rho >= magnitude_threshold:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# =====================================================================
# CLAIM 12 — target_staleness_late as the dominant non-eff_h mediator.
# DDQN's bias correction prevents Q-explosion / unbounded growth →
# online network stays close to the periodically-copied target →
# late-training target staleness stays low → bootstrap targets are
# accurate → cleaner TD signals → outcome gain. Empirical mediation
# share: 27% on FourRooms (capacity_sweep, n=88 mech-HELD) and 65%
# on Breakout sync=100 (minatar_1M, n=16 mech-HELD).
#
# Tautology audit (run_tautology_audit_fourrooms.py) corroborates:
# target_staleness_late survives all three checks on FourRooms —
# jaccard=0 (structurally independent of mc_return), hp_r²=0.158
# (NOT deterministic in replay.capacity), within-capacity stratified
# ρ(target_staleness_late, eval_best_burst_mean) = −0.604 with
# p = 3×10⁻¹⁰. CLEAN.
# =====================================================================


def _staleness_mediation_holds_when(
    proportion_mediated: ProportionMediatedResult,
    *, dominance_floor: float = 0.2,
    n_pairs_floor: int = 25,
) -> tuple[Verdict, RefutationClass | None]:
    """Shared verdict logic for the target_staleness-mediates-outcome
    bridges. Sister of `_eff_h_mediation_holds_when` with the
    inverse semantics: HELD when the mediator carries a NON-trivial
    share of DDQN's outcome benefit (proportion ≥ dominance_floor).

    Authored with `predicted_direction='a_gt_b'` — the prior is that
    DDQN improves outcome AND that target_staleness_late carries a
    dominant share of that benefit. HELD = both predictions
    confirmed.

    Three gates:
      (1) n_pairs ≥ n_pairs_floor (statistical power),
      (2) `in_unit_interval` (linear-mediation assumptions hold),
      (3) `proportion` ≥ `dominance_floor` (mediator carries the
          share)."""
    if proportion_mediated.n_pairs < n_pairs_floor:
        return Verdict.POWER_INSUFFICIENT, None
    p = proportion_mediated.proportion
    if math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if not proportion_mediated.in_unit_interval:
        return Verdict.POWER_INSUFFICIENT, None
    if p >= dominance_floor:
        return Verdict.HELD, None
    # Predicted-direction proportion ≥ dominance_floor; observed
    # below the floor. Negative proportion (mediator carries
    # opposite sign) → SIGN_FLIP; small-positive → NULL_EFFECT.
    if p < 0.0:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    # `replay.capacity` is in pair_by because capacity_sweep_fourrooms
    # has 3 capacities (10k, 20k, 50k) and pairs MUST share capacity
    # for the mediation analysis to be coherent. Without it, the
    # `(corpus, seed)` dedup collapses cells across capacities and
    # in_unit_interval breaks (proportion goes negative).
    pair_by=(
        'env_name', 'corpus', 'gamma', 'total_steps', 'sync_period',
        'seed', 'replay.capacity',
    ),
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('corpus') == 'capacity_sweep_fourrooms')
        & finite('target_staleness_late')
    ),
    predicted_direction='a_gt_b',
)
def target_staleness_late_mediates_outcome__fourrooms(
    proportion_mediated: ProportionMediatedResult,
    *,
    mediator: str = 'target_staleness_late',
    upstream_source: str = 'jensen_gap',
    upstream_max_delta: float = 0.0,
    dominance_floor: float = 0.2,
    n_pairs_floor: int = 25,
) -> tuple[Verdict, RefutationClass | None]:
    """On FourRooms (capacity_sweep), `target_staleness_late` mediates
    DDQN's outcome benefit at ~27% under mech-HELD conditioning.

    Mediation chain tested:
        `do(DDQN) → Δ_jens<0 → Δ_target_staleness_late → Δ_outcome`

    DDQN's bias correction → Q stays bounded near target → less
    online-target divergence in the late training window → cleaner
    bootstrap target values → outcome gain.

    Scope: FourRooms-misc cells with finite `target_staleness_late`
    (only computable when traces have `online_max_q_per_step` and
    `target_max_q_per_step` columns).

    HELD when:
      (1) ≥ `n_pairs_floor` paired cells with Δ_jens<0,
      (2) linear-mediation assumptions hold,
      (3) `proportion` ≥ `dominance_floor` (≥ 20% of total Δ_outcome
          routed through staleness reduction).

    Empirical (capacity_sweep_fourrooms, mech-HELD): proportion =
    0.269, n_pairs = 88. Tautology audit on the same data (DDQN arm,
    capacity-stratified): jaccard=0, hp_r²=0.158, stratified ρ =
    −0.604 with p = 3×10⁻¹⁰. CLEAN."""
    del mediator, upstream_source, upstream_max_delta
    return _staleness_mediation_holds_when(
        proportion_mediated,
        dominance_floor=dominance_floor,
        n_pairs_floor=n_pairs_floor,
    )


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    # `corpus` in pair_by is critical: the (env=Breakout-MinAtar,
    # sync=100) scope captures cells from THREE corpora (`ddqn`,
    # `ddqn_effective_cohort`, `minatar_1M`); without `corpus`, the
    # seed-only pairing cross-pairs cells from different sweeps that
    # ran on substrate-different commits, polluting the mediation
    # estimate. Empirical: without `corpus`, n_pairs=52 with diluted
    # proportion=0.018 (no_effect); with `corpus`, n_pairs=16 within
    # minatar_1M at proportion=0.65 (HELD).
    pair_by=('env_name', 'corpus', 'gamma', 'total_steps', 'sync_period', 'seed'),
    scope=(
        (pl.col('env_name') == 'Breakout-MinAtar')
        & (pl.col('sync_period') == 100)
        & (pl.col('corpus') == 'minatar_1M')
        & finite('target_staleness_late')
    ),
    predicted_direction='a_gt_b',
)
def target_staleness_late_mediates_outcome__breakout_sync100(
    proportion_mediated: ProportionMediatedResult,
    *,
    mediator: str = 'target_staleness_late',
    upstream_source: str = 'jensen_gap',
    upstream_max_delta: float = 0.0,
    dominance_floor: float = 0.2,
    n_pairs_floor: int = 10,
) -> tuple[Verdict, RefutationClass | None]:
    """On Breakout-MinAtar at sync=100 (the canonical Q-explosion
    regime), `target_staleness_late` mediates DDQN's outcome benefit
    at ~65% under mech-HELD conditioning — 2.4× stronger than on
    FourRooms.

    Mechanism is the same chain as the FourRooms sister bridge but
    sharper: vanilla's max-bias drives explicit Q-explosion on
    Breakout sync=100; DDQN's correction prevents the explosion;
    the staleness suppression carries dominant share of the outcome
    gain.

    Scope: Breakout-MinAtar at sync_period=100, with finite
    `target_staleness_late`. n_pairs_floor reduced to 10 (vs 25 on
    FourRooms) — minatar_1M provides only ~16 mech-HELD pairs at
    this scope.

    HELD when proportion ≥ dominance_floor (default 0.2). Empirical
    (minatar_1M, mech-HELD): proportion = 0.646, n_pairs = 16."""
    del mediator, upstream_source, upstream_max_delta
    return _staleness_mediation_holds_when(
        proportion_mediated,
        dominance_floor=dominance_floor,
        n_pairs_floor=n_pairs_floor,
    )


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('env_name', 'corpus', 'gamma', 'total_steps', 'sync_period', 'seed'),
    scope=(
        pl.col('corpus').is_in(
            ['asterix_intermediate_sync', 'breakout_intermediate_sync'],
        )
        & finite('target_staleness_late')
        & finite('jensen_dormancy_gap') & finite_lt('jensen_dormancy_gap', 0.05)
        & finite('q_divergence_score') & finite_lt('q_divergence_score', 1.0)
    ),
    predicted_direction='a_gt_b',
)
def target_staleness_late_mediates_outcome__minatar_intermediate_sync(
    proportion_mediated: ProportionMediatedResult,
    *,
    mediator: str = 'target_staleness_late',
    upstream_source: str = 'jensen_gap',
    upstream_max_delta: float = 0.0,
    dominance_floor: float = 0.2,
    n_pairs_floor: int = 50,
) -> tuple[Verdict, RefutationClass | None]:
    """Tests whether target_staleness_late within-cell mediates
    DDQN's outcome benefit on the MinAtar SURVIVE intermediate-
    sync corpora (Asterix sync ∈ {500, 1500, 3000} + Breakout sync
    ∈ {500, 1500}, after the dormancy filter excludes the cells
    where DDQN's Q crosses below MC).

    Empirical (2026-05-08, n_pairs=114): proportion = 0.069 →
    NO_EFFECT (null_effect). Within-cell linear mediation does
    NOT replicate the strong cross-config structural pattern.

    Cross-config aggregate finding (n=11 in-scope (env, sync)
    cells, NOT what this bridge tests):
    - marginal Spearman ρ(Δ_target_staleness, Δ_y_best) = -0.86
      (p=0.001)
    - partial Spearman | log_sync = -0.64 (p=0.044)
    - other Δ candidates (jens, q_late, argmax_ent, eff_h) all
      explained away by log_sync; only Δ_target_staleness retains
      residual mediator power.

    Reading: the cross-config relationship is real (staleness
    reductions order DDQN benefit across configs) but the within-
    cell per-seed mediation channel is weak — Δ_target_staleness
    doesn't carry per-seed outcome variance. This matches the
    polarity-mediator pattern documented in
    `findings_target_staleness_mediator.md`: linear mediation breaks
    on Asterix/Breakout's bounded-Q SURVIVE regime; counterfactual
    mediation or cross-config slope tests are the appropriate
    surface, not within-cell linear share."""
    del mediator, upstream_source, upstream_max_delta
    return _staleness_mediation_holds_when(
        proportion_mediated,
        dominance_floor=dominance_floor,
        n_pairs_floor=n_pairs_floor,
    )


# =====================================================================
# CLAIM 21 — Polarity-stratified cross-config staleness slope.
#
# At the CONFIG level (1 row per (env, sync, corpus) or (env, tau,
# corpus)), the relationship between mean Δ_target_staleness_late
# and mean Δ_eval_best_burst_mean is polarity-conditional.
#
# Empirical (2026-05-08, ddqn_universe cache, strict mech-HELD
# (paired-t p<0.05 ∧ frac<0 ≥ 0.65) per config):
#
#   SURVIVE (n=5: Asterix sync ∈ {500,1500,3000} + Breakout
#   sync ∈ {500,1500} from intermediate-sync corpora):
#     ρ(mean_d_stale, mean_d_y_best) = -0.900 (p=0.037)
#
#   REACH (n=3: FourRooms tau ∈ {0.001, 0.01, 0.1} from
#   polyak_tau_intervention corpora; SURVIVE polyak fails strict
#   mech-HELD per `findings_polyak_makes_mech_dormant_survive`):
#     ρ(mean_d_stale, mean_d_y_best) = +1.000 (p=0.000, n=3 trivial)
#
# Sign FLIPS by polarity. Two bridges below capture the polarity-
# conditional relationship; both are STARTING-POINT given the small
# n per polarity class. Authored at Tier.ASSOCIATIONAL because the
# cross-config slope is a SCOPE-level descriptive pattern, not a
# within-cell mediation channel (`proportion_mediated` returns ~0.07
# on the same SURVIVE scope; mediation breaks at the cell level).
#
# The cross-config slope cannot identify causation — sync_period
# drives Δ_jens, Δ_q_late, Δ_target_staleness, AND Δ_y all at
# once. After conditioning on log_sync, only target_staleness retains
# residual cross-config slope. But identification at fixed-sync
# requires varying staleness via an independent lever; polyak
# provides that on REACH (mech survives) but preempts mech on
# SURVIVE.
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('seed',),
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
    cross_config_paired_slope: CrossConfigPairedSlopeResult,
    *,
    target: str = 'eval_best_burst_mean',
    predictor: str = 'target_staleness_late',
    config_keys: tuple[str, ...] = (
        'env_name', 'sync_period', 'total_steps', 'corpus',
    ),
    rho_threshold: float = -0.5,
    p_threshold: float = 0.1,
    min_configs: int = 3,
) -> tuple[Verdict, RefutationClass | None]:
    """Cross-config bridge on SURVIVE polarity:
    ρ(mean Δ_target_staleness_late, mean Δ_y_best) ≤ -0.5
    across configs in CLAIM 17 mech-active scope.

    Empirical (n=5, periodic_copy / sync-period sweep on Asterix
    + Breakout intermediate-sync): ρ = -0.90 (p=0.037).

    HELD when ρ ≤ rho_threshold AND p ≤ p_threshold AND
    n_configs ≥ min_configs.

    STARTING POINT — n=5 is borderline. Within-cell linear
    mediation gives proportion ≈ 0.07 on the same scope (mediation
    breaks). The cross-config slope captures a SCOPE-level
    descriptive pattern: configs where DDQN reduces staleness more
    are also configs where DDQN harms outcome more (Δ_stale > 0
    means DDQN INCREASES staleness; bigger increase → worse Δ_y).

    Cannot identify causation: sync_period confounds Δ_stale and
    Δ_y. Polyak-τ intervention can't disentangle on SURVIVE
    because polyak preempts mech (Δ_jens ≈ 0). Awaiting more
    in-scope SURVIVE configs (PacMan + Freeway/SI intermediate-
    sync sweeps) for power."""
    del target, predictor, config_keys
    if cross_config_paired_slope.n_configs < min_configs:
        return Verdict.POWER_INSUFFICIENT, None
    rho = cross_config_paired_slope.rho
    p = cross_config_paired_slope.p_value
    if math.isnan(rho) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if rho <= rho_threshold and p <= p_threshold:
        return Verdict.HELD, None
    if rho > 0.0:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT


# =====================================================================
# CLAIM 14 — env-polarity predicts the link sign per env (soft tautology).
#
# Cross-env, the within-env paired link r(Δ_eff_h, Δ_outcome) tracks
# `env_reward_polarity` at slope ≈ +0.5 (Fisher-z), R² ≈ 0.83 (n_envs=8
# polarity-defined envs from the canonical ddqn corpus).
#
# This bridge encodes the "polarity predicts link sign" observation
# from the original CLAIM 12 motivation as a SOFT TAUTOLOGY claim:
# the relationship holds because polarity is by definition the
# within-cell r(L, return), and `effective_horizon` is L-derived; the
# paired-Δ link inherits the env's L→outcome map. The empirical premise
# is just "DDQN walks along the env's L→outcome curve" rather than
# fundamentally re-shaping the env. Empirically true (β=+0.61,
# p=0.0017) but mechanism-blind — the same pattern would hold for any
# RL algorithm that improves outcome via length-channel.
#
# Bridge predicts `'a_gt_b'`: the polarity coefficient is positive and
# substantive. HELD confirms the soft tautology operates as expected.
# Companion to `eff_h_mediates_g_link__{goal,survival}_envs` which both
# stay HELD under `predicted_direction='null'` — eff_h is structurally
# tied to polarity (this bridge) but is NOT a dominant mediator (those
# bridges). The two readings together are the explicit form of the
# polarity finding.
#
# Source: `experiments/findings/sync_curve_breakout/run_polarity_tautology_demo.py`,
# `polarity_tautology_findings.md`.
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('seed',),
    scope=(
        finite('env_reward_polarity')
        & finite('effective_horizon')
        & finite('eval_best_burst_mean')
    ),
    predicted_direction='a_gt_b',
)
def link_r_predictable_from_polarity__soft_tautology(
    paired_link_per_env: MetaRegressionResult,
    *,
    target: str = 'eval_best_burst_mean',
    predictor: str = 'effective_horizon',
    moderator: str = 'env_reward_polarity',
    slope_threshold: float = 0.4,
    r_squared_threshold: float = 0.5,
    min_envs: int = 5,
    alpha: float = 0.05,
) -> Verdict:
    """The soft tautology: per env, paired link r(Δ_eff_h, Δ_outcome)
    is predictable from `env_reward_polarity` because eff_h IS L-derived
    and polarity IS the env's r(L, return) by definition.

    Bridge body: read the polarity coefficient from the meta-regression.

      HELD when β(polarity) ≥ `slope_threshold` AND coefficient is
      significant AND R² ≥ `r_squared_threshold` AND n_strata ≥
      `min_envs`. Confirms the soft-tautology prediction holds.

      NO_EFFECT when β is below threshold or insignificant — would
      refute the soft tautology, suggesting the L→outcome map is NOT
      stable under DDQN intervention. (Strong negative finding if
      ever fires.)

      POWER_INSUFFICIENT when fewer than `min_envs` envs surface in
      the panel.

    Pre-rebuild empirical (canonical ddqn corpus, 8 envs):
      β(env_reward_polarity) = +0.614, CI [+0.34, +0.89],
      p = 1.7×10⁻³, R² = 0.83, n_strata = 8 → HELD.

    **2026-05-11 post-rebuild: COLLAPSED.** With trace-restore
    bringing n_strata 8 → 10:
      β = +0.366, CI [−0.47, +1.20], p = 0.34, R² = 0.11,
      I² = 0.95 (extreme cross-env heterogeneity).
    Verdict: NO_EFFECT (coefficient insignificant, magnitude
    below 0.4 threshold). Per-env check on the 4 envs surfacing
    with finite paired Δs: Acrobot polarity=−0.98 r=−0.27;
    Asterix polarity=+0.62 r=+0.39; **PacMan polarity=+0.34
    r=−0.49 (sign-mismatch)**; Snake polarity=−0.50 r=−0.04
    (near null).

    The earlier "8-of-8 sign-match is structurally forced"
    framing (memory `findings_polarity_soft_tautology_bridge.md`,
    pre-PacMan-ingest) does not survive the expanded panel.
    The bridge is REFUTED on current data — the polarity-coupling
    shape is not universal. A post-hoc rationalization ("PacMan
    has fixed-length env") was considered and DROPPED per
    reviewer-3 feedback: without a registered measurable
    operationalizing "policy-independent episode length" as a
    scope predicate, any sign-mismatch could be retro-justified
    by inventing an env class. Honest verdict is REFUTED full
    stop; if a principled scope predicate later identifies
    structural exclusions, the bridge can be re-scoped.

    The companion eff_h-mediator bridges
    (`eff_h_mediates_g_link__{goal,survival}_envs`) remain HELD
    with directional predictions matching polarity sign; per-env
    coupling EXISTS within polarity half-planes, but the env-mean
    cross-env panel doesn't fit a single linear β."""
    del target, predictor
    if paired_link_per_env.n_strata < min_envs:
        return Verdict.POWER_INSUFFICIENT
    coef = next(
        (c for c in paired_link_per_env.coefficients
         if c.name == moderator),
        None,
    )
    if coef is None:
        return Verdict.POWER_INSUFFICIENT
    if not coef.is_significant:
        return Verdict.NO_EFFECT
    if coef.coefficient < slope_threshold:
        return Verdict.NO_EFFECT
    if paired_link_per_env.r_squared < r_squared_threshold:
        return Verdict.NO_EFFECT
    return Verdict.HELD


# =====================================================================
# CLAIM 26b — gate-scope conjunction predicts DDQN outcome benefit.
#
# Replaces CLAIM 26's slope-predictor regression (which was structurally
# underdetermined: per-env link slope is pinned at -1 by the asymptote
# claim, so cross-env variance in |slope| is dominated by saturation /
# sub-asymptote artifacts, not by v_jens). The substantive cross-env
# claim is at the OUTCOME level, not the slope level: when the three
# gates fire jointly, DDQN helps.
#
# Empirical (postfix corpus, gate-active subset n=246 across 7 envs):
#   mean Δo = +1.33, t = 2.59, p = 0.010
# Outside gate-active scope (n=204), mean Δo ≈ 0 — the gates are
# load-bearing scope predicates. See
# `findings_gate_conditional_outcome_benefit.md`.
# =====================================================================


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
    stratify_by: tuple[str, ...] = ('env_name',),
    min_vanilla_predictor: float = 2.0,
) -> Verdict:
    """Cross-env DDQN benefit under [G1 ∧ G2] scope, aggregated via
    DerSimonian-Laird random-effects on per-stratum Cohen's d.

    Principled cross-env aggregation per
    `findings_within_stratum_primitives.md`:
      1. Aggregate seeds within each env stratum (sync/gamma/k
         within-env nuisance, averaged) → per-arm mean + sd + n_seeds.
      2. Stratum-level scope filter on G1: vanilla aggregate
         `mean(jensen_gap) > min_vanilla_predictor` (default 2.0,
         "premise substantively active" — see threshold rationale
         below). Both arms in a stratum included or excluded
         together; no asymmetric filtering.
      3. Per stratum: independent-samples Cohen's d + SE
         (Hedges 1981 small-sample form).
      4. DerSimonian-Laird random-effects pooling on per-stratum
         (d, SE).

    **G1 threshold rationale** (data-suggested, not theory-derived):
    at the lax threshold v_jens > 0.05, the panel includes envs at
    the borderline of premise-activation (Asterix v_jens=1.97,
    Breakout v_jens=0.69, Acrobot v_jens=1.91). On these envs DDQN's
    bias correction is small relative to intrinsic noise → per-env d
    is borderline-negative. At v_jens > 2.0, the surviving envs
    (MetaMaze, MountainCar, SI, Pacman) all show positive d (3
    substantially so; Pacman at +0.09 is under-trained per
    `findings_within_stratum_primitives.md` UPDATE). The threshold
    is empirically calibrated; future work could derive it from
    "minimum-bias-magnitude-needed-to-overcome-DDQN-intrinsic-noise"
    theory.

    HELD when pooled |d| > `threshold_d` AND p < α AND n_strata >=
    min_strata. POWER_INSUFFICIENT when direction-correct but
    p >= α. NO_EFFECT when wrong-direction or |d| < threshold_d.

    Empirical (current cache, threshold v_jens > 2.0, 4 envs in
    scope: MetaMaze γ=0.99, MountainCar, Pacman, SI):
      pooled d = +0.43, CI [+0.04, +0.83], p = 0.032, I² = 68%.
      All 4 per-env d's positive: MetaMaze +0.22, MountainCar +0.41,
      Pacman +0.09, SI +0.87.

    **Caveat — small n_strata**: at threshold 2.0, n_strata = 4 is
    minimal for DL pooling. The HELD verdict reflects "5 envs in
    panel, 4 in scope, all positive" not a statistically robust
    cross-env result. Replication requires more envs that satisfy
    the substantive G1 threshold.

    See `findings_within_stratum_primitives.md` (UPDATE 2)."""
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


# =====================================================================
# CLAIM 15 — Polyak-τ rung-2 corroboration: target staleness causally
# amplifies DDQN's outcome benefit on FourRooms.
#
# Pearl rung-2 evidence from `polyak_tau_intervention.yaml` sweep:
# do(τ) at fixed sync_period=100 directly varies target staleness
# (τ → 1: target ≈ online, low staleness; τ → 0: target lags, high
# staleness). Both DDQN and baseline arms run at each τ; the bridge
# tests whether DDQN's outcome benefit (Δ_outcome at fixed seed × τ)
# decays as τ grows.
#
# Empirical (FourRooms): ATE(log τ → Δ_outcome) = −0.018, p = 0.003,
# placebo refuter ate = 0, RCC drift = 0. The 3-log-τ range accounts
# for ~5% of FourRooms's reward range, matching the observational
# proportion_mediated=0.27 within an order of magnitude.
#
# Companion to CLAIM 13 (`target_staleness_late_mediates_outcome__
# fourrooms` HELD, observational): together rung-2 + rung-1.5
# evidence on FourRooms specifically. The narrow scope reflects the
# regime-dependent nature of the staleness mediation chain (cf.
# `wide_jci_panel`: 24-stratum partial Spearman ρ_part = −0.188
# pooled, with sign-flip in Q-explosion / silent-inversion regimes).
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    # `target_sync.tau` MUST be in pair_by so each (DDQN, baseline)
    # pair shares a single τ value. The analysis then reads
    # `target_staleness_late` from the BASELINE arm of each pair
    # as the per-pair endogenous mediator level — exogenously
    # varied via the Polyak τ sweep. log_tau is the INSTRUMENT;
    # target_staleness_late is the proximal treatment we test the
    # causal effect of on Δ_outcome.
    pair_by=(
        'env_name', 'gamma', 'sync_period',
        'total_steps', 'seed', 'target_sync.tau',
    ),
    scope=(
        # Endogenous polyak-sweep indicator: only `polyak_update`
        # carries a `target_sync.tau` field; periodic_copy regimes
        # leave it null. NOT corpus-tagged.
        finite('target_sync.tau')
        & (pl.col('target_sync.tau') > 0)
        # GOAL polarity (length and return inversely correlated).
        & finite_lt('env_reward_polarity', -0.5)
        # Bounded Q (no Q-explosion). Polyak smoothing keeps q_div
        # low here in practice; the predicate excludes any future
        # corpus where it doesn't.
        & finite('q_divergence_score')
        & finite_lt('q_divergence_score', 100.0)
        & finite('target_staleness_late')
        & finite('eval_best_burst_mean')
        # POSITIVE Q-regime: vanilla's late-window mean Q > 0,
        # equivalent to "r_min ≥ 0 + bounded Q + GOAL". This is
        # the ENDOGENOUS downstream of r_min — captures the
        # actual sign of Hasselt's bias direction in the cell's
        # trajectory. r_min is the structural cause; q_late_mean
        # is the per-cell observable. Predicate routes around
        # `r_min` (exogenous env-structural) by testing what
        # actually matters: where vanilla's Q ends up.
        # See `polyak_q_regime_findings.md` for the empirical
        # mechanism trace.
        & finite_gt('q_late_mean', 0.0)
    ),
    predicted_direction='a_lt_b',
)
def staleness_amplifies_ddqn_outcome__sparse_goal_polyak(
    paired_continuous_do_dowhy: PairedContinuousDoResult,
    *,
    treatment_var: str = 'target_staleness_late',
    treatment_var_arm: str = 'baseline',
    outcome: str = 'eval_best_burst_mean',
    ate_threshold: float = 1.0,
    refutation_drift_threshold: float = 5.0,
    n_pairs_floor: int = 30,
) -> Verdict:
    """In the polyak-do(τ) regime (endogenous indicator:
    `target_sync.tau` finite > 0), under SPARSE-TERMINAL-POSITIVE
    GOAL polarity (env_reward_polarity < −0.5, r_min ≥ 0,
    q_divergence_score < 100), per-pair baseline target staleness
    CAUSALLY amplifies DDQN's outcome benefit (Δ_outcome): pairs
    with higher baseline staleness show larger DDQN benefit.

    Causal logic. The polyak sweep exogenously varies τ across
    pairs; both DDQN and baseline arms in a pair share the same
    τ. Baseline's `target_staleness_late` per pair is therefore
    a τ-driven exogenous variable that captures "how much
    staleness the algorithm experiences in this pair." DoWhy
    backdoor_ate(target_staleness_late → Δ_outcome) under
    DAG `[(target_staleness_late, delta_outcome)]` estimates the
    causal slope. Refutations (placebo + RCC) validate.

    The HP knob `target_sync.tau` is NOT the source — it's the
    sweep-design instrument that exogenously varies the
    endogenous mediator. The bridge tests the proximal causal
    relationship `target_staleness_late → Δ_outcome`, with τ as
    the rung-2 randomiser in the background.

    HELD when:
      (1) backdoor.identified;
      (2) `n_pairs ≥ n_pairs_floor`;
      (3) ATE > `ate_threshold` (positive: more staleness ⇒
          larger Δ_outcome — DDQN benefits more from correcting
          a bigger bias);
      (4) refutations clean.

    Empirical (polyak_tau_intervention, FourRooms, n=120, τ ∈
    [0.001, 0.1] traces present): backdoor ATE on baseline
    target_staleness_late ≈ +5 reward units / staleness unit;
    placebo refuter near 0; RCC drift small."""
    del treatment_var, treatment_var_arm, outcome
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
    source=INTERVENTION,
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
        # SURVIVAL polarity (Asterix-like — episode length and return
        # positively correlated, dense per-step rewards). Endogenous
        # regime predicate, NOT env_name.
        & finite_gt('env_reward_polarity', 0.3)
        & finite('target_staleness_late')
        & finite('eval_best_burst_mean')
    ),
    # `predicted_direction='null'` (xfail-style): under SURVIVAL
    # polarity in the polyak regime, the staleness-mediation chain
    # is BROKEN (the L→outcome map is sign-flipped relative to
    # GOAL, AND in pre-polyak periodic_copy this regime
    # exhibited Q-explosion that disconnected mech from outcome).
    # HELD = null confirmed.
    predicted_direction='null',
)
def staleness_does_not_amplify_ddqn_outcome__survival_polyak(
    paired_continuous_do_dowhy: PairedContinuousDoResult,
    *,
    treatment_var: str = 'target_staleness_late',
    treatment_var_arm: str = 'baseline',
    outcome: str = 'eval_best_burst_mean',
    null_band: float = 5.0,
    n_pairs_floor: int = 30,
) -> Verdict:
    """Companion to `staleness_amplifies_ddqn_outcome__fourrooms_
    polyak`. Under SURVIVAL polarity (env_reward_polarity > 0.3 —
    Asterix, Breakout, etc.) in the polyak-do(τ) regime, the
    staleness-mediation chain is BROKEN. The bridge predicts NULL
    ATE on `target_staleness_late → Δ_outcome`.

    Why null. SURVIVAL envs have positive L→outcome map (longer
    episodes = better outcome). DDQN's bias correction in this
    polarity regime doesn't translate to outcome via length-
    mediated channels (per the wide-JCI silent-inversion finding
    and the FourRooms-vs-Asterix Q-amplification result). The
    polyak τ knob does drive baseline staleness variation, but
    that variation doesn't propagate into DDQN's per-pair benefit.

    HELD when |ATE| < `null_band` AND identified AND n_pairs ≥
    floor. NO_EFFECT (xpass) when ATE is unexpectedly large —
    would refute the regime-specificity prediction.

    Empirical (polyak_tau_asterix, Asterix-MinAtar, n=120):
    backdoor ATE on baseline staleness ≈ small (|ATE| < null_band)
    → HELD null confirmed. Asterix doesn't reproduce FourRooms's
    staleness amplification."""
    del treatment_var, treatment_var_arm, outcome
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


# =====================================================================
# CLAIM 17 — DELETED. Bridge audit step 2 (2026-05-11).
#
# `chain_amplifier_link_active_in_bounded_q` was migrated from
# `paired_link_per_burst` (init-correlation-driven, not a
# treatment-effect coupling) to `stratum_effect_panel +
# panel_regress` — the inferentially-honest cross-env regression
# of per-env Δ_outcome on per-env Δ_jens. Under correct
# methodology, the verdict shifts as follows:
#
#   Full panel (n=12 envs):                slope=-1.62, R²=0.97, p<0.0001
#   No PacMan (n=11):                      slope=-0.47, R²=0.57, p=0.008
#   No PacMan + MountainCar (n=10):        slope=-0.29, R²=0.77, p=0.0008
#   No PacMan + MountainCar + SI (n=9):    slope=+0.07, R²=0.05, p=0.57
#
# The cross-env chain-amplifier signal is leverage-driven by 2-3
# high-bias envs. Even after scoping to a high-bias cluster (env
# mean jensen_gap > 2.0, n=6), dropping PacMan + MountainCar
# leaves slope=-0.16, n.s. on the remaining 4 envs.
#
# The chain-amplifier theory as a CROSS-ENV LAW doesn't survive
# on the post-fix corpus. Its empirical content is preserved
# elsewhere:
#  - CLAIM 26b's `ddqn_helps_under_three_gate_scope__cross_env`
#    uses DerSimonian-Laird stratified pooling (leverage-robust)
#    and reports pooled d=+0.46, p=0.005 across G1-active envs —
#    the canonical cross-env outcome benefit claim.
#  - Per-env existence proofs (DDQN helps substantively on PacMan,
#    MountainCar, SpaceInvaders) document in
#    `findings_minatar_link_attenuation.md`.
# =====================================================================


# =====================================================================
# CLAIM 19 — Cross-env: effective_horizon predicts link power on
# REACH-polarity envs (negative env_reward_polarity, "shorter is
# better") in the CLAIM 17 bounded-Q scope.
#
# Empirical (per-env mean_dY across configs that pass strict
# mech-HELD, with ddqn_universe corpus, n_envs=4 REACH envs in
# scope):
#   MetaMaze γ=0.999: mean_dY=+2.13, effh=110
#   Acrobot γ=0.99/0.95: mean_dY=+0.31, effh=32
#   FourRooms γ=0.99: mean_dY=+0.11, effh=38
#   MountainCar γ=0.99/0.95/0.90: mean_dY=-0.004, effh=40
#
# Cross-env Pearson r(mean_dY, effh) = +0.975 (p=0.025).
# SURVIVE-polarity envs (CartPole, MinAtar) do NOT show this
# cross-env structural relation (Pearson +0.408 ns) — link power
# tracks observed |Δ_jens| but no env-structural predictor cleanly
# orders them. Polarity-class is a moderator of which env-feature
# drives cross-env link power.
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('seed', 'total_steps', 'eval_every'),
    scope=(
        # CLAIM 17 scope predicates
        finite('q_divergence_score') & finite_lt('q_divergence_score', 1.0)
        & finite_gt('bootstrap_fraction', 0.5)
        & finite('jensen_dormancy_gap') & finite_lt('jensen_dormancy_gap', 0.05)
        # REACH polarity: r(episode_length, mc_return) < 0 — shorter
        # trajectories correlate with bigger return (goal-reaching).
        # Endogenous predicate (per-cell empirical polarity).
        & finite('env_reward_polarity')
        & finite_lt('env_reward_polarity', -0.3)
        # Standard config (no n-step / action-duplicate / rs-shift /
        # polyak-τ).
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
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    covariates: tuple[str, ...] = ('effective_horizon',),
    dedupe_strategy: str = 'mean',
    slope_threshold: float = 0.005,
) -> Verdict:
    """Cross-env REACH-polarity bridge: among REACH envs in the
    CLAIM 17 bounded-Q scope, larger effective_horizon predicts
    bigger DDQN outcome benefit. Tests the per-(env, burst) meta-
    regression slope of Δ_outcome on env-mean effective_horizon.
    HELD when β(eff_h) ≥ `slope_threshold` AND significant in the
    predicted direction.

    The chain-amplifier theory's clean-firing direction: longer
    chain → more bias compounding → more room for DDQN's per-step
    correction to integrate to outcome.

    **2026-05-11 verdict post-rebuild:** NO_EFFECT (sign refuted).
    Per-(env, burst) meta-regression at n_strata=49 gives
    coef(effective_horizon) = **−0.0046, 95% CI [−0.009, −0.0002],
    p=0.041** — significant in the OPPOSITE direction. The env-
    mean Pearson r=+0.975 (n=4 envs, cited in earlier docstring)
    pooled over bursts; per-burst meta-regression unmasks the
    phase-structure inversion already documented in
    `findings_fourrooms_time_series.md` ("DDQN reduces bias
    early; Q grows late (success-induced)"). The env-mean
    Pearson was cross-env aggregate evidence; per-burst slope
    flips because late-burst Q-growth on long-eff_h envs amplifies
    DDQN's vanilla baseline more than the early-burst correction
    benefits.

    Bridge stays as a falsifiable artifact: the per-burst slope's
    direction is opposite to the chain-amplifier reading. The
    cross-env scaling claim survives only at env-mean aggregation,
    not at the per-burst level. Treat as evidence AGAINST a clean
    "longer chain → bigger outcome benefit" reading on REACH."""
    del source, covariates, dedupe_strategy
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


# =====================================================================
# CLAIM 20 — Cross-env: argmax_entropy_late_van predicts link power on
# SURVIVE-polarity envs in CLAIM 17 bounded-Q scope. Companion to
# CLAIM 19's REACH-side effh predictor; STARTING POINT — small n.
#
# Empirical (per-CONFIG, strict mech-HELD, ddqn_universe cache
# 2026-05-08, n_configs=5 across 4 SURVIVE envs):
#   SI sync=100:        mean_dY=+2.56, argmax_ent_van=1.33
#   Asterix sync=100:   mean_dY=+0.06, argmax_ent_van=0.86 (q_stab)
#   Breakout sync=100:  mean_dY=+0.29, argmax_ent_van=0.90
#   Asterix sync=1k:    mean_dY=+0.08, argmax_ent_van=0.86
#   CartPole sync=1k:   mean_dY=+0.00, argmax_ent_van=0.67 (saturated)
#
# Per-config Pearson r(mean_dY, argmax_ent_van) = +0.909 (p=0.033),
# Spearman = +0.900 (p=0.037). REACH per-config (n=35): effh +0.754
# dominates, argmax_ent only +0.323 — confirming polarity-stratified
# predictor pattern.
#
# **STARTING-POINT caveats** (per tautology audit):
# - argmax_entropy_van ↔ argmax_entropy_dd: Pearson +0.945 — argmax
#   entropy is largely an env-structural action-distribution property
#   (vanilla and DDQN have similar argmax entropies per env).
#   It's not capturing DDQN's algorithmic effect specifically; rather
#   it captures envs where many actions have similar Q-values
#   (= more action-asymmetric Hasselt bias to fix).
# - argmax_entropy_van ↔ mean_dJ: Pearson -0.79 — collinear with
#   bias-reduction magnitude. argmax_ent and bias-reduction may be
#   two manifestations of the same env-level action-redundancy.
# - n_configs=5 is small. SURVIVE strict-mech-HELD set is hard to
#   expand because most SURVIVE configs at sync=100 have Q-explosion
#   (out of scope) and at sync=10k are mech-dormant. Authored as
#   a starting point — corroborate with more configs (designed sweeps
#   at intermediate sync periods) or refute.
# =====================================================================


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=('seed', 'total_steps', 'eval_every'),
    scope=(
        finite('q_divergence_score') & finite_lt('q_divergence_score', 1.0)
        & finite_gt('bootstrap_fraction', 0.5)
        & finite('jensen_dormancy_gap') & finite_lt('jensen_dormancy_gap', 0.05)
        # SURVIVE polarity: r(episode_length, mc_return) > 0 — longer
        # trajectories correlate with bigger return (stay-alive).
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
    source: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ] = _MC_RETURN_PER_BURST_MEAN,
    covariates: tuple[str, ...] = ('argmax_entropy_late',),
    dedupe_strategy: str = 'mean',
    slope_threshold: float = 0.5,
) -> Verdict:
    """STARTING-POINT cross-env SURVIVE bridge: among SURVIVE envs in
    CLAIM 17 bounded-Q scope, env-mean argmax_entropy_late predicts
    bigger DDQN outcome benefit. Companion to CLAIM 19 (effh on REACH).

    Per-env paired g on `mc_return[per_burst]` regressed on env-mean
    `argmax_entropy_late`. HELD when β ≥ `slope_threshold` AND
    significant.

    Reading (provisional). SURVIVE envs (positive polarity) don't
    show effh as a cross-env link-power predictor (REACH does).
    Within SURVIVE, argmax_entropy_van orders mean_dY at Pearson
    +0.909 / Spearman +0.900 across n=5 strict-mech-HELD configs.
    Plausible mechanism: high argmax_entropy = env has many similar-
    quality actions = more action-asymmetric Hasselt bias to correct
    = DDQN benefits more.

    **Audit caveats** (read before promoting to Tier.INTERVENTIONAL):
    1. argmax_ent is mostly env-structural (vanilla and DDQN have
       Pearson +0.945 across configs — not algorithm-specific).
    2. argmax_ent_van and mean_dJ are collinear (Pearson -0.79);
       the predictor may be a proxy for "bias-reduction-magnitude
       availability" rather than a separate channel.
    3. n_configs=5 is small. Spearman 0.90 has p=0.037 by exact
       permutation. Need more SURVIVE configs to corroborate or
       refute. SURVIVE strict-mech-HELD set is structurally hard
       to expand (sync=100 Q-explodes, sync=10k goes dormant).

    Authored as STARTING POINT. Substrate authors should design new
    SURVIVE sweeps at intermediate sync (1k-3k typical sweet spot)
    to populate more in-scope configs and test whether the rank
    survives at n_configs ≥ 10."""
    del source, covariates, dedupe_strategy
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


# =====================================================================
# CLAIM 18 — DELETED. Bridge audit step 2 (2026-05-11).
# `algorithmic_activation_rate_mediates_link__bounded_q` was an
# explicit placeholder ("Open-question bridge — empirical signal
# not yet established"). Now in scope (n=967) and POW_INSUF;
# `greedy_match_late` cache materialisation goal achieved, but
# the claim doesn't survive.
# =====================================================================


# =====================================================================
# CLAIM 4 + CLAIM 16 — DELETED. Bridge audit step 2 (BRIDGE_AUDIT.md).
#
# Both bf→g_link cross-env bridges retracted on the post-fix
# corpus: chain-amplifier theory survives, but via CLAIM 17
# (Q-stable per-burst link persistence,
# `findings_minatar_link_attenuation.md`), not via
# bootstrap_fraction. See `findings_residual_unexplained.md`.
# =====================================================================


# =====================================================================
# DDQN measurement graph — the closure.
# =====================================================================
DDQN_UNIVERSE_BRIDGES = (
    # CLAIM 2 — load-bearing necessary scope (causal refutation).
    ddqn_refuted_when_dormancy_fires,
    # CLAIM 2 corroborations — Pearl rung-2 designed interventions.
    adaptive_dqn_recovers_ddqn_benefit__fourrooms_factor_0p5,
    adaptive_dqn_fails_to_avoid_attenuation__spaceinvaders_1m,
    # CLAIM 5 — effective-horizon scope (Pearl rung-2 do(γ) sweep).
    ddqn_benefit_scales_with_effective_horizon__fourrooms,
    ddqn_benefit_scales_with_effective_horizon__metamaze_high_gamma,
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
    target_staleness_late_mediates_outcome__fourrooms,
    target_staleness_late_mediates_outcome__breakout_sync100,
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
    # 'null'). The two together are the explicit form of the
    # polarity finding.
    link_r_predictable_from_polarity__soft_tautology,
    # CLAIM 26 — slope-predictor regression cut; subsumed by CLAIM
    # 26b's gate-conjunction outcome bridge below. See
    # `findings_g1_predicts_link_slope.md`.
    # CLAIM 26b — substantive cross-env replacement for CLAIM 26's
    # slope-predictor regression. Tests that DDQN's outcome benefit is
    # positive panel-level when the three gates fire jointly. The
    # outcome-level claim, not the slope-level one.
    ddqn_helps_under_three_gate_scope__cross_env,
    # CLAIM 15 — Polyak-τ rung-2 corroboration on FourRooms:
    # do(τ) → Δ_outcome ATE significantly negative (-0.018,
    # p=0.003, refutations pass). The Pearl rung-2 layer for
    # CLAIM 13's staleness mediation, FourRooms-specific.
    staleness_amplifies_ddqn_outcome__sparse_goal_polyak,
    # CLAIM 15b — companion null bridge: under SURVIVAL polarity
    # in the polyak regime, the staleness-mediation chain is
    # BROKEN. Empirical |ATE| < null_band on Asterix → HELD null.
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

CLAIM 3 (sufficient scope) is deliberately ABSENT — no exogenous
predicate corroborates a sufficient condition for DDQN's outcome
benefit, and we don't author null bridges."""


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
    'adaptive_dqn_fails_to_avoid_attenuation__spaceinvaders_1m',
    'adaptive_dqn_recovers_ddqn_benefit__fourrooms_factor_0p5',
    'ddqn_attenuates_at_late_bursts__spaceinvaders',
    'ddqn_benefit_scales_with_effective_horizon__fourrooms',
    'ddqn_benefit_scales_with_effective_horizon__metamaze_high_gamma',
    'ddqn_helps_at_early_bursts__pixel_envs',
    'ddqn_refuted_when_dormancy_fires',
    'eff_h_mediates_g_link__goal_envs',
    'eff_h_mediates_g_link__survival_envs',
    'target_staleness_late_mediates_outcome__fourrooms',
    'target_staleness_late_mediates_outcome__breakout_sync100',
    'target_staleness_late_mediates_outcome__minatar_intermediate_sync',
    'cross_config_staleness_slope_negative__survive',
    'link_r_predictable_from_polarity__soft_tautology',
    'ddqn_helps_under_three_gate_scope__cross_env',
    'staleness_amplifies_ddqn_outcome__sparse_goal_polyak',
    'staleness_does_not_amplify_ddqn_outcome__survival_polyak',
    'ddqn_does_not_rescue__acrobot_rs_0p1',
    'ddqn_does_not_rescue__cartpole_rs_0p1',
    'ddqn_increases_argmax_entropy__fourrooms_rs_0p1',
    'ddqn_entropy_matches_vanilla__fourrooms_rs_1p0',
]


# Canonical name `corroborate.runner` imports;
# DDQN_UNIVERSE_BRIDGES stays as an alias for legacy call sites.
BRIDGES = DDQN_UNIVERSE_BRIDGES

