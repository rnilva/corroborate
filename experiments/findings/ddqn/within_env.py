"""Within-env do() probes — γ on FourRooms + MetaMaze.

- `ddqn_benefit_scales_with_effective_horizon__fourrooms` (CLAIM 5):
  FR γ-sweep, per-γ stratum-Cohen's d on outcome, Pearson r against
  γ tests chain-depth scaling. AWAITING DATA (γ=0.999 FR cells
  absent post-rebuild).
- `metamaze_link_steeper_at_high_gamma` (CLAIM 24): on MetaMaze n_γ=2
  ({0.99, 0.999}), Δ_outcome should AMPLIFY at high γ if chain-depth
  is the lever. Currently REFUTED — was paired-Δ init-correlation.
"""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.stratum_effect_panel import StratumEffectPanel
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.stats import MetaRegressionResult

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn._scope import (
    G1_VANILLA_CONFIG_PREMISE_ACTIVE,
    VANILLA_JENS_NOISE_FLOOR,
)
from experiments.findings.ddqn._verdicts import (
    meta_regression_coefficient_verdict,
)


# Per-γ effective_horizon on FourRooms (empirical means at each γ
# on the current ddqn cache after `gamma_sweep_fourrooms` ingest
# 2026-05-12). Pinned for CLAIM 5's multi-stratum random-effects
# meta-regression on `effective_horizon` slope across γ-strata.
_FOURROOMS_EFFECTIVE_HORIZON_PER_GAMMA: dict[object, dict[str, float]] = {
    0.99: {'effective_horizon': 37.3},
    0.995: {'effective_horizon': 80.6},
    0.999: {'effective_horizon': 235.6},
}


# CLAIM 5 — within-env do(γ) on FourRooms.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & pl.col('gamma').is_in([0.99, 0.995, 0.999])
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
        & pl.col('target_sync.tau').is_null()
    ),
    predicted_direction='a_gt_b',
)
def ddqn_benefit_scales_with_effective_horizon__fourrooms(
    meta_regression_unpaired_d: MetaRegressionResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = (
        'gamma', 'total_steps', 'reward_scale',
    ),
    covariate_key_field: str = 'gamma',
    covariates_per_key: dict[object, dict[str, float]] = (
        _FOURROOMS_EFFECTIVE_HORIZON_PER_GAMMA
    ),
    scope_predictor: str = 'jensen_gap',
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    slope_threshold: float = 0.01,
    min_strata: int = 3,
) -> Verdict:
    """Within-FR do(γ) chain-depth scaling probe. Per-(γ, config)
    independent-samples Cohen's d → random-effects meta-regression
    on `effective_horizon` (env-derived from γ). HELD when β_eff_h
    ≥ `slope_threshold` AND significant.

    Post-roast issue 7 refactor (2026-05-12): replaced
    `stratum_id_scaling_verdict` (Pearson r on per-γ cohen_d
    panel) with the multi-stratum meta-regression shape used by
    CLAIM 19. The previous form inherited the n=3 envs Pearson r
    brittleness (`findings_n3_pearson_brittle`) — at n_strata=2
    (current cache γ=0.99 only), Pearson r is degenerate; even at
    n=3 a 1-SE perturbation could swing r between +1 and -1. The
    meta-regression form expands the panel via within-γ config
    replicates (`(γ, total_steps, reward_scale)` strata), giving
    proper SE on the slope coefficient.

    `slope_threshold=0.01` is the substrate-meaningful magnitude
    (calibrated like CLAIM 19): observed eff_h range across FR's
    γ values ≈ 42 units (27.6 at γ=0.99 → ~70 at γ=0.999);
    threshold 0.01 corresponds to |Δd| ≥ 0.42 across the span —
    Cohen's "small effect" magnitude.

    Cache has only γ=0.99 FR cells in three sub-corpora →
    n_strata ≤ 3, covariate is constant across all strata →
    meta-regression unidentified → POW_INSUF. Once γ=0.999 FR
    cells land, the multi-stratum form has between-γ variation
    AND within-γ replicates → proper test of the chain-depth
    amplification claim documented in
    `findings_gamma_sweep_three_regimes.md`."""
    del treatment_arm, baseline_arm, source, stratify_by
    del covariate_key_field, covariates_per_key
    del scope_predictor, min_vanilla_predictor
    return meta_regression_coefficient_verdict(
        meta_regression_unpaired_d,
        'effective_horizon',
        sign=1,
        threshold=slope_threshold,
        min_strata=min_strata,
    )


# CLAIM 24 — Within-MetaMaze do(γ): link slope steepens?
_METAMAZE_GAMMA_SCOPE = (
    (pl.col('env_name') == 'MetaMaze-misc')
    & pl.col('gamma').is_in([0.99, 0.999])
    & G1_VANILLA_CONFIG_PREMISE_ACTIVE
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    & pl.col('target_sync.tau').is_null()
)


def _metamaze_amplification_verdict(
    panel: StratumEffectPanel,
    *,
    high_gamma: float,
    low_gamma: float,
    high_floor: float,
    amplification_ratio_min: float,
) -> Verdict:
    """Shared decision logic for the γ-amplification bridge.
    HELD when per-step-normalized high-γ stratum Δ ≥ `high_floor`
    AND ≥ `amplification_ratio_min` × per-step-normalized low-γ
    Δ (or low-γ ≤ 0 trivially).

    **Discount-horizon normalization** (post-anomaly 2026-05-12):
    `eval_best_burst_mean` is the γ-discounted return; on dense-
    reward envs (MetaMaze ≈ +1 per cell, 200-step cap) the raw
    metric doubles γ=0.99 → γ=0.999 purely from
    `Σ γ^t = (1-γ^T)/(1-γ)` scaling. Comparing raw Δs across γ
    confounds discount-scale with policy quality. We rescale by
    `(1 - γ)` to recover a per-step-equivalent reward (the
    invariant under infinite-horizon discounting); the
    amplification claim now tests whether DDQN's per-step benefit
    grows with γ, not whether the discount horizon does."""
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
            gamma_f = float(gamma_val)
            # Discount-horizon normalization: Σ γ^t = 1/(1-γ);
            # multiply Δ by (1-γ) to recover per-step-equivalent.
            normalized = delta * (1.0 - gamma_f)
            if math.isclose(gamma_f, high_gamma, rel_tol=1e-6):
                high_delta = normalized
            elif math.isclose(gamma_f, low_gamma, rel_tol=1e-6):
                low_delta = normalized
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


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_METAMAZE_GAMMA_SCOPE,
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
    high_floor: float = 0.01,
    amplification_ratio_min: float = 1.5,
) -> Verdict:
    """Within-MetaMaze do(γ): n_γ=2 amplification test on
    **per-step-normalized** Δ_outcome (raw `eval_best_burst_mean`
    scaled by `(1-γ)`). HELD when per-step high-γ Δ ≥
    `high_floor` AND ≥ `amplification_ratio_min` × low-γ Δ (or
    low-γ ≤ 0).

    Discount-normalization rationale (post-anomaly investigation
    2026-05-12): raw `eval_best_burst_mean` doubles on MetaMaze
    from γ=0.99 (~27) → γ=0.999 (~56) — that's
    `Σ γ^t = 1/(1-γ)` scaling on a dense-reward env, NOT improved
    policy quality. Per-step equivalent is ~0.31 reward/step in
    both regimes. Comparing raw Δs across γ confounded
    discount-horizon with amplification. With normalization, at
    γ=0.999 Δ_per_step ≈ -0.002 (vanilla SLIGHTLY beats DDQN);
    at γ=0.99 Δ_per_step ≈ +0.015. No amplification — verdict
    NO_EFFECT survives the metric fix.

    `high_floor=0.01` calibrated to per-step scale: ~3% of
    MetaMaze's typical per-step reward (~0.3); a meaningful
    high-γ amplification would push DDQN's per-step benefit
    above 0.01 (≈ Cohen's small at this scale). Pre-fix
    `high_floor=0.5` was for raw-discounted units."""
    del measurables, stratify_by, min_seeds_per_arm
    return _metamaze_amplification_verdict(
        stratum_effect_panel,
        high_gamma=high_gamma, low_gamma=low_gamma,
        high_floor=high_floor,
        amplification_ratio_min=amplification_ratio_min,
    )


# Per-env q_autocorr_late mean on vanilla canonical-config baseline
# cells. Lag-1 autocorrelation of `online_max_q_per_step` over late
# 50% of training — proxy for function-approximator spatial
# coherence: how strongly the FA enforces Q(s,a) ≈ Q(s',a') for
# consecutive trajectory states. Slow-drift envs (FR maze, CartPole
# balancing, MetaMaze) approach 1.0; fast-dynamics envs (Acrobot
# pendulum) approach 0. Empirical means computed 2026-05-12 from
# post-fix vanilla cells (n_envs=8 strata); see
# `findings_fa_coherence_bias.md` for the full panel + r=+0.71
# cross-env correlation with log(jens/σ_Q) that motivated this
# bridge.
_AUTOCORR_PER_ENV: dict[object, dict[str, float]] = {
    'FourRooms-misc':   {'q_autocorr_vanilla': 0.99},
    'CartPole-v1':      {'q_autocorr_vanilla': 0.76},
    'MetaMaze-misc':    {'q_autocorr_vanilla': 0.72},
    'Breakout-MinAtar': {'q_autocorr_vanilla': 0.74},
    'MountainCar-v0':   {'q_autocorr_vanilla': 0.59},
    'PacMan-jumanji':   {'q_autocorr_vanilla': 0.35},
    'Acrobot-v1':       {'q_autocorr_vanilla': 0.07},
}


# CLAIM 27 — Cross-env: DDQN's bias-reduction scales with FA-coherence.
@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        pl.col('env_name').is_in(tuple(_AUTOCORR_PER_ENV.keys()))
        & (pl.col('gamma') == 0.99)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
        & pl.col('jensen_gap').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_bias_reduction_scales_with_fa_coherence__cross_env(
    meta_regression_unpaired_d: MetaRegressionResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'jensen_gap',
    # Stratify by (env, config) — different configs at the same
    # env land in distinct strata. Independent-samples Cohen's d
    # per (env, config), no seed-pairing (per
    # `feedback_paired_g_in_rl`: RL seed-pairing inflates SE
    # without genuine within-subject correlation; trajectories
    # diverge by the first decision step).
    stratify_by: tuple[str, ...] = (
        'env_name', 'total_steps', 'replay.capacity', 'sync_period',
    ),
    covariate_key_field: str = 'env_name',
    covariates_per_key: dict[object, dict[str, float]] = (
        _AUTOCORR_PER_ENV
    ),
    scope_predictor: str = 'jensen_gap',
    min_vanilla_predictor: float = VANILLA_JENS_NOISE_FLOOR,
    slope_threshold: float = 0.3,
    min_strata: int = 4,
) -> Verdict:
    """Cross-env do(DDQN) probe at the BIAS level: per-(env, burst)
    Δ_jens = jens_DDQN − jens_vanilla → meta-regress on
    `q_autocorr_vanilla`. The bias-reduction grows (becomes more
    negative) as FA-coherence increases. HELD when β_autocorr ≤
    −`slope_threshold` AND significant.

    Per-burst over per-cell: per-cell scalars collapse the bias
    trajectory and lose the early-vs-late phase structure that the
    FA-amplification mechanism operates through. At γ=0.99 most
    envs are partly-mech-dormant at best-burst (jens collapses
    late) — the per-burst panel captures the WHOLE bias trajectory
    so the autocorr signal isn't washed out by phase mismatch
    (cf. CLAUDE.md § per-burst-canonical rule).

    Theoretical motivation: high q_autocorr means the FA enforces
    Q(s,a) ≈ Q(s',a') for s ≈ s' along trajectory; an overestimate
    at one state propagates spatially via shared trunk gradients,
    amplifying argmax-bias coverage. DDQN's argmax-decorrelation
    breaks this loop. Prediction: Δ_jens (DDQN−vanilla) should be
    near-zero on low-autocorr envs (Acrobot 0.07) and large-
    negative on high-autocorr envs (FR 0.99).

    Empirical motivation: `findings_fa_coherence_bias.md` —
    cross-env r(q_autocorr_late, log(jens/σ_Q)) = +0.71, p=0.003,
    n=15 strata 8 envs at the vanilla descriptive level. This
    bridge is the do(DDQN) interventional sibling: per-(env, burst)
    paired-g of jens, regressed on autocorr.

    `slope_threshold=0.3` is calibrated to the cross-env autocorr
    range [0.07, 0.99] ≈ 0.92 units — a slope of −0.3 corresponds
    to ≈ −0.28 Cohen's g shift across the full range (Cohen's
    "small"). A meaningful FA-coherence-driven bias reduction.

    The complementary OUTCOME bridge — DDQN's outcome benefit
    scaling with autocorr — runs null cross-env at γ=0.99
    because the bias-→-outcome translation is gated by per-env
    G3 outcome-headroom (different envs have different ceilings;
    CartPole sees its outcome plummet at high-jens regardless
    of DDQN). The bias-level claim is the cleaner test of the
    mechanism."""
    del treatment_arm, baseline_arm, source, stratify_by
    del covariate_key_field, covariates_per_key
    del scope_predictor, min_vanilla_predictor
    return meta_regression_coefficient_verdict(
        meta_regression_unpaired_d,
        'q_autocorr_vanilla',
        sign=-1,
        threshold=slope_threshold,
        min_strata=min_strata,
    )


# Per-cell scope filters for the FA-degeneracy conjunction
# (`findings_unified_degeneracy_theory.md`). Each bridge uses
# `reward_nonzero_frac` and / or `q_autocorr_late` as PER-CELL
# scope predicates (NOT per-env dict constants), which makes
# them transitively required by the bridge → `--ingest-all`
# walks the trace columns + computes the measurable per cell.
#
# Thresholds derived from FR γ=0.999 vanilla-collapse condition
# (factorial 2026-05-13):
#  - bare FR (sparse): reward_nonzero_frac ≈ 0.005 → 0.05 cutoff
#  - deep MLP[64,64]:  q_autocorr_late ≈ 0.7-0.99 → 0.5 cutoff
# Scope-restricted bridges fire on cells where the theory's
# axes are ACTIVE; null bridges on the same scope would
# falsify the theory in the rescue regime.

# Sparse-reward scope: vanilla's per-step reward signal is near-
# zero (uninf-r axis active).
_SPARSE_REWARD_SCOPE = pl.col('reward_nonzero_frac') < 0.05
# High-FA-coherence scope: vanilla's online_max_Q autocorr is
# high (FA-coherence axis active).
_HIGH_AUTOCORR_SCOPE = pl.col('q_autocorr_late') > 0.5
# Canonical-config gates shared across the three new bridges.
_CANONICAL_CONFIG_SCOPE = (
    (pl.col('gamma') == 0.99)
    & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
    & pl.col('action_duplicate_k').is_null()
    & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    & pl.col('jensen_gap').is_finite()
)


# CLAIM 28 — Sparse-reward scope: DDQN's bias-reduction is large
# where per-step reward is uninformative (axis iii active).
@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _SPARSE_REWARD_SCOPE
        & _CANONICAL_CONFIG_SCOPE
        & pl.col('reward_nonzero_frac').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_bias_reduction_under_sparse_reward_scope(
    stratified_arm_diff_pooled: object,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    measurables: tuple[str, ...] = ('jensen_gap',),
    stratify_by: tuple[str, ...] = (
        'env_name', 'total_steps', 'replay.capacity', 'sync_period',
    ),
    min_seeds_per_arm: int = 10,
    effect_threshold: float = 0.3,
) -> Verdict:
    """Per-cell SCOPE: cells with `reward_nonzero_frac < 0.05`
    (sparse-reward, theory's axis-iii ACTIVE). Pool Cohen's d on
    `jensen_gap` per (env, config) via independent-samples →
    random-effects pool. HELD when DL pooled d ≤ −0.3 AND
    heterogeneity-not-flagged.

    **Per-cell scope (not per-env dict)**: the bridge filters
    cells INDIVIDUALLY by their measured `reward_nonzero_frac`.
    Each cell flowing through the bridge has had its trace
    re-walked at `--ingest-all` time to compute this measurable,
    so the runner backfills it for any corpus where traces are
    locally / cloud-restorable.

    This shape is more honest than the env-level pinning:
    `reward_nonzero_frac` is (env, policy)-dependent — a vanilla
    cell on FR γ=0.999 that collapses may have density near 0
    while a converged vanilla cell on FR γ=0.99 sees the goal
    reward more often, density ~ 0.005. Per-cell filtering
    captures the regime each cell is actually IN; per-env mean
    averages over both regimes.

    Empirical anchor (`findings_unified_degeneracy_theory.md`):
    shaping FR via `PotentialReward` makes density → 1 and rescues
    vanilla collapse (0.21 → 61.5 outcome at γ=0.999). Cells with
    density > 0.05 should ESCAPE this bridge's scope; cells in
    scope should show large bias-reduction under DDQN."""
    del treatment_arm, baseline_arm, measurables, stratify_by
    del min_seeds_per_arm, effect_threshold
    # Use stratified_arm_diff_pooled's verdict directly; the
    # primitive emits HELD / NO_EFFECT / POWER_INSUFFICIENT /
    # HELD_WITH_SCOPE_FLAG based on the DL pooled-d magnitude +
    # heterogeneity. Pass-through.
    if not hasattr(stratified_arm_diff_pooled, 'verdict'):
        return Verdict.POWER_INSUFFICIENT
    return getattr(stratified_arm_diff_pooled, 'verdict')


# CLAIM 29 — High-FA-coherence scope: DDQN's bias-reduction is
# large where the FA over-smooths Q across nearby states (axis i
# active). Standalone test of axis (i) at per-cell resolution.
@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _HIGH_AUTOCORR_SCOPE
        & _CANONICAL_CONFIG_SCOPE
        & pl.col('q_autocorr_late').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_bias_reduction_under_high_fa_coherence_scope(
    stratified_arm_diff_pooled: object,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    measurables: tuple[str, ...] = ('jensen_gap',),
    stratify_by: tuple[str, ...] = (
        'env_name', 'total_steps', 'replay.capacity', 'sync_period',
    ),
    min_seeds_per_arm: int = 10,
    effect_threshold: float = 0.3,
) -> Verdict:
    """Per-cell SCOPE: cells with `q_autocorr_late > 0.5`. Same
    pool-and-DL shape as the sparse-reward bridge. Sibling test
    of the FA-coherence axis (i).

    The original `ddqn_bias_reduction_scales_with_fa_coherence__cross_env`
    bridge uses a per-env DICT covariate — captures the cross-env
    slope but not the per-cell regime structure. This bridge
    asks: of cells in the high-autocorr regime (regardless of
    env), does DDQN reduce bias significantly? Per-cell scope is
    the post-`feedback_endogenous_scope_predicates` shape: env-
    feature predicates over env-name predicates."""
    del treatment_arm, baseline_arm, measurables, stratify_by
    del min_seeds_per_arm, effect_threshold
    if not hasattr(stratified_arm_diff_pooled, 'verdict'):
        return Verdict.POWER_INSUFFICIENT
    return getattr(stratified_arm_diff_pooled, 'verdict')


# CLAIM 30 — Full conjunction scope: axes (i) ∧ (iii) BOTH
# active. The load-bearing test of the multiplicative theory.
@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _SPARSE_REWARD_SCOPE
        & _HIGH_AUTOCORR_SCOPE
        & _CANONICAL_CONFIG_SCOPE
        & pl.col('reward_nonzero_frac').is_finite()
        & pl.col('q_autocorr_late').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_bias_reduction_under_full_degeneracy_scope(
    stratified_arm_diff_pooled: object,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    measurables: tuple[str, ...] = ('jensen_gap',),
    stratify_by: tuple[str, ...] = (
        'env_name', 'total_steps', 'replay.capacity', 'sync_period',
    ),
    min_seeds_per_arm: int = 10,
    effect_threshold: float = 0.3,
) -> Verdict:
    """Per-cell CONJUNCTION SCOPE: cells with BOTH `reward_nonzero_frac
    < 0.05` AND `q_autocorr_late > 0.5`. The load-bearing test of
    the FA-degeneracy theory's multiplicative structure: only
    cells where ALL endogenous axes fire should show large DDQN
    bias-reduction.

    Compared to the standalone (sparsity-only / autocorr-only)
    bridges, the conjunction scope is STRICTLY smaller. The
    multiplicative theory predicts:
    - On conjunction-scope cells: DDQN reduces bias substantially
      (pooled d ≤ −0.3, HELD).
    - On standalone-scope-but-not-conjunction cells (axis (i) OR
      (iii) but not both): DDQN benefit attenuates.

    The standalone bridges measure pooled effect on UNION; this
    bridge measures pooled effect on INTERSECTION. If the
    multiplicative theory holds, the conjunction's d-magnitude
    should be larger than either standalone's d on cells outside
    the conjunction (post-hoc decomposition once cells land)."""
    del treatment_arm, baseline_arm, measurables, stratify_by
    del min_seeds_per_arm, effect_threshold
    if not hasattr(stratified_arm_diff_pooled, 'verdict'):
        return Verdict.POWER_INSUFFICIENT
    return getattr(stratified_arm_diff_pooled, 'verdict')


# CLAIM 31/32/33 — Outcome-target siblings of the mech bridges
# above. Same per-cell scopes; ask whether DDQN's bias-reduction
# TRANSLATES into outcome benefit. Together with the mech bridges,
# these form a mech-link decoupling cluster: same scope, two
# verdicts (mech-side, outcome-side). The contrast IS the finding
# wherever they differ.
#
# Predicted_direction='a_gt_b' (DDQN > vanilla on outcome). The
# pre-registration: theory predicts DDQN should improve outcome
# whenever mech HELDs on the same scope. Empirical reality may
# differ (cf. `findings_q_amplification_cartpole.md` +
# `findings_ddqn_variance_injection.md`: on some envs DDQN reduces
# bias but increases per-seed σ_Q, hurting outcome).


# CLAIM 31 — Sparse-reward scope on OUTCOME.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _SPARSE_REWARD_SCOPE
        & _CANONICAL_CONFIG_SCOPE
        & pl.col('reward_nonzero_frac').is_finite()
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def ddqn_outcome_benefit_under_sparse_reward_scope(
    stratified_arm_diff_pooled: object,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    measurables: tuple[str, ...] = ('eval_best_burst_raw_mean',),
    stratify_by: tuple[str, ...] = (
        'env_name', 'total_steps', 'replay.capacity', 'sync_period',
    ),
    min_seeds_per_arm: int = 10,
    effect_threshold: float = 0.3,
) -> Verdict:
    """OUTCOME-side counterpart of CLAIM 28. Same per-cell
    sparse-reward scope; tests whether DDQN's bias-reduction
    translates into outcome benefit at the raw-return scale.

    Sibling-to-CLAIM-28 verdict CONTRAST is the finding: any
    drift between mech-side HELD and outcome-side non-HELD on
    identical scope is exactly the mech↛link decoupling the
    framework's three-verdict architecture was designed to
    surface (PAPER §3.4)."""
    del treatment_arm, baseline_arm, measurables, stratify_by
    del min_seeds_per_arm, effect_threshold
    if not hasattr(stratified_arm_diff_pooled, 'verdict'):
        return Verdict.POWER_INSUFFICIENT
    return getattr(stratified_arm_diff_pooled, 'verdict')


# CLAIM 32 — High-FA-coherence scope on OUTCOME.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _HIGH_AUTOCORR_SCOPE
        & _CANONICAL_CONFIG_SCOPE
        & pl.col('q_autocorr_late').is_finite()
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def ddqn_outcome_benefit_under_high_fa_coherence_scope(
    stratified_arm_diff_pooled: object,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    measurables: tuple[str, ...] = ('eval_best_burst_raw_mean',),
    stratify_by: tuple[str, ...] = (
        'env_name', 'total_steps', 'replay.capacity', 'sync_period',
    ),
    min_seeds_per_arm: int = 10,
    effect_threshold: float = 0.3,
) -> Verdict:
    """OUTCOME-side counterpart of CLAIM 29."""
    del treatment_arm, baseline_arm, measurables, stratify_by
    del min_seeds_per_arm, effect_threshold
    if not hasattr(stratified_arm_diff_pooled, 'verdict'):
        return Verdict.POWER_INSUFFICIENT
    return getattr(stratified_arm_diff_pooled, 'verdict')


# CLAIM 33 — Full conjunction scope on OUTCOME.
@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _SPARSE_REWARD_SCOPE
        & _HIGH_AUTOCORR_SCOPE
        & _CANONICAL_CONFIG_SCOPE
        & pl.col('reward_nonzero_frac').is_finite()
        & pl.col('q_autocorr_late').is_finite()
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def ddqn_outcome_benefit_under_full_degeneracy_scope(
    stratified_arm_diff_pooled: object,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    measurables: tuple[str, ...] = ('eval_best_burst_raw_mean',),
    stratify_by: tuple[str, ...] = (
        'env_name', 'total_steps', 'replay.capacity', 'sync_period',
    ),
    min_seeds_per_arm: int = 10,
    effect_threshold: float = 0.3,
) -> Verdict:
    """OUTCOME-side counterpart of CLAIM 30 (the conjunction
    bridge). Paired with CLAIM 30 this is the LOAD-BEARING test
    of FA-degeneracy theory at the OUTCOME layer:
    - CLAIM 30 HELD + CLAIM 33 HELD → theory fully corroborated
    - CLAIM 30 HELD + CLAIM 33 NO_EFFECT → mech↛link decoupling
      (theory predicts mech but link breaks — premise insufficient)
    - CLAIM 30 NO_EFFECT → theory's mech-axis prediction refuted

    Pre-registered: signed-direction expectation a_gt_b (DDQN
    raises outcome). The framework's PI-based verdict applies."""
    del treatment_arm, baseline_arm, measurables, stratify_by
    del min_seeds_per_arm, effect_threshold
    if not hasattr(stratified_arm_diff_pooled, 'verdict'):
        return Verdict.POWER_INSUFFICIENT
    return getattr(stratified_arm_diff_pooled, 'verdict')


BRIDGES = (
    # γ-sweep bridges moved to `ddqn_sweeps.within_env_sweeps` —
    # canonical pins γ=0.99 so they fire empty here:
    #   ddqn_benefit_scales_with_effective_horizon__fourrooms (CLAIM 5)
    #   metamaze_link_steeper_at_high_gamma (CLAIM 24)
    ddqn_bias_reduction_scales_with_fa_coherence__cross_env,
    ddqn_bias_reduction_under_sparse_reward_scope,
    ddqn_bias_reduction_under_high_fa_coherence_scope,
    ddqn_bias_reduction_under_full_degeneracy_scope,
    ddqn_outcome_benefit_under_sparse_reward_scope,
    ddqn_outcome_benefit_under_high_fa_coherence_scope,
    ddqn_outcome_benefit_under_full_degeneracy_scope,
)
