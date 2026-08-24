"""`stratified_arm_diff_pooled` — per-stratum independent-samples
Cohen's d + DerSimonian-Laird random-effects pooling.

The principled cross-env aggregator (per `findings_within_stratum_primitives.md`):
strata are the unit of inference, not cells. 30 seeds in MountainCar
aren't 30 independent observations of "DDQN's effect on MountainCar"; they
are 30 noisy measurements of the same stratum-level effect. Aggregate seeds
first → one `(arm_mean, arm_sd, n_seeds)` per `(stratum, arm)`. Then:

  1. Per stratum, compute independent-samples Cohen's d + SE (Hedges 1981
     small-sample formula).
  2. Apply stratum-level scope filters on VANILLA-arm aggregates (e.g.
     `vanilla_mean_jens > 0.05`) — both arms in a stratum survive together,
     no asymmetric filtering bias.
  3. Pool per-stratum effects via DerSimonian-Laird random-effects
     (existing `random_effects_summary` in `corroborate.stats.effect_size`).
  4. Report pooled estimate + heterogeneity (τ², I²) + per-stratum panel.

This is conceptually distinct from `paired_g`: paired_g pairs cells across
arms by `pair_by` tuple, computes per-pair Δs, treats them as i.i.d.
samples (pseudo-replication when seeds within a stratum aren't independent
draws of "the env's effect"). This primitive treats each stratum as ONE
observation of the effect, with within-stratum precision tracked via SE.

Honest about heterogeneity: when between-stratum variance is large (I² high),
the pooled estimate has wide CIs — that's a feature. Cross-env effects ARE
heterogeneous; obscuring it via per-pair pooling was misleading.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from corroborate.bridge.analysis import analysis
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.stats.effect_size import (
    PooledStats, PredictedDirection,
    random_effects_summary, random_effects_verdict,
)

from corroborate.analyses.panel.stratum_panel import stratum_panel


@dataclass(frozen=True, slots=True)
class StratumDiff:
    """Per-stratum independent-samples mean diff + Cohen's d.

    `stratum_id` — tuple of values for the stratify_by keys.
    `mean_diff` — `mean(treatment) - mean(baseline)` on raw scale.
    `cohen_d` — Cohen's d (Hedges 1981 small-sample form):
      `mean_diff / sqrt((sd_t² + sd_b²)/2)`. Standardized,
      sign-stable across heterogeneous-scale envs.
    `cohen_se` — Approximate SE of Cohen's d:
      `sqrt((n_t + n_b)/(n_t * n_b) + d²/(2*(n_t + n_b - 2)))`.
    `baseline_predictor` — the vanilla-arm aggregate of the
      stratum-level scope-filter predictor (e.g. mean jens). Used
      to apply scope-at-stratum-level. NaN if not requested.
    """
    stratum_id: tuple[object, ...]
    mean_diff: float
    cohen_d: float
    cohen_se: float
    arm_mean_treatment: float
    arm_mean_baseline: float
    arm_sd_treatment: float
    arm_sd_baseline: float
    n_seeds_treatment: int
    n_seeds_baseline: int
    baseline_predictor: float


@dataclass(frozen=True, slots=True)
class StratifiedArmDiffPooledResult:
    """Output of stratified per-arm difference + DL pooling.

    `pooled_d` / `pooled_se` — DerSimonian-Laird random-effects
    pooled Cohen's d + SE (across in-scope strata). `tau2` /
    `i_squared` / `q_statistic` — heterogeneity diagnostics.
    `n_strata` — count of strata that contributed (post stratum-
    level scope filter).

    `per_stratum` — tuple of `StratumDiff` for every in-scope
    stratum (sorted by `stratum_id`). Lets bridges build per-env
    panels without re-aggregating.

    `pooled` — full `PooledStats` carrying the prediction
    interval (pi_lo / pi_hi) used by `random_effects_verdict`.
    Surfaced as a field rather than recomputed from
    `(pooled_d, pooled_se)` so bridges can inspect the
    heterogeneity diagnostics + PI bounds without re-running
    `random_effects_summary`.

    `verdict` / `refutation` — heterogeneity-flagged verdict
    via `random_effects_verdict`. `HELD_WITH_SCOPE_FLAG` when
    PI excludes zero in the predicted direction AND
    `I² ≥ I2_THRESHOLD` (default 0.5) — the scope-discovery
    trigger documented in `verdict.py` and ANALYSIS_RECIPE.md
    §1.5. Bridges that fixture this analysis can return
    `(result.verdict, result.refutation)` directly to enforce
    the heterogeneity discipline at the verdict layer rather
    than re-implementing it in each bridge body.

    `pooled_p_value` — two-sided p-value for `pooled_d != 0` from
    `pooled_d / pooled_se` under normal approximation.
    """
    pooled_d: float
    pooled_se: float
    tau2: float
    i_squared: float
    q_statistic: float
    n_strata: int
    per_stratum: tuple[StratumDiff, ...]
    stratify_by: tuple[str, ...]
    measurable: str
    treatment_arm: str
    baseline_arm: str
    arm_field: str
    scope_predictor: str
    pooled: PooledStats
    verdict: Verdict
    refutation: RefutationClass | None

    @property
    def pooled_p_value(self) -> float:
        if (
            math.isnan(self.pooled_d)
            or math.isnan(self.pooled_se)
            or self.pooled_se == 0.0
        ):
            return float('nan')
        z = abs(self.pooled_d / self.pooled_se)
        return float(math.erfc(z / math.sqrt(2)))

    @property
    def pooled_ci_lo(self) -> float:
        if math.isnan(self.pooled_d) or math.isnan(self.pooled_se):
            return float('nan')
        return self.pooled_d - 1.96 * self.pooled_se

    @property
    def pooled_ci_hi(self) -> float:
        if math.isnan(self.pooled_d) or math.isnan(self.pooled_se):
            return float('nan')
        return self.pooled_d + 1.96 * self.pooled_se


# Removed `_per_stratum_aggregate` and `_cohen_d_indep_samples`
# in Phase 5 migration (2026-05-13): panel-build + Cohen's d
# computation moved to `stratum_panel.cohen_d` / `.cohen_se`.


@analysis
def stratified_arm_diff_pooled(
    cells: pl.DataFrame,
    *,
    source: str,
    treatment_arm: str,
    baseline_arm: str,
    stratify_by: tuple[str, ...] = (
        'env_name', 'sync_period', 'gamma', 'action_duplicate_k',
    ),
    arm_field: str = 'arm_key',
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 0.05,
    min_seeds_per_arm: int = 5,
    predicted_direction: PredictedDirection | None = None,
) -> StratifiedArmDiffPooledResult:
    """Per-stratum independent-samples Cohen's d, DL-pooled.

    Phase 5 migration (2026-05-13): delegates to `stratum_panel`
    for the cell→panel build, then builds the rich
    `StratifiedArmDiffPooledResult` from panel data + applies the
    `min_baseline_predictor` stratum-level filter +
    `random_effects_summary` for the DL pool. Result type and
    semantics unchanged — verdict-preserving.

    Strata are formed by `stratify_by` tuple (default:
    `(env_name, sync_period, gamma, action_duplicate_k)`). Per
    stratum:
      1. Compute per-arm `mean(source)`, `sd(source)`, n_seeds.
      2. Compute independent-samples Cohen's d + SE.
      3. Compute vanilla-arm `mean(scope_predictor)` (default
         `jensen_gap`).

    Stratum-level scope filter (BOTH arms in or out together):
      - `n_seeds_treatment >= min_seeds_per_arm`
      - `n_seeds_baseline >= min_seeds_per_arm`
      - `baseline_predictor > min_baseline_predictor`

    Per-bridge cell-level filters (env, config exclusions) belong
    on `Bridge.scope` upstream; this primitive only handles
    stratum-level filters that depend on per-arm aggregates.

    DerSimonian-Laird random-effects pooling on the in-scope
    per-stratum (cohen_d, cohen_se) pairs (existing
    `corroborate.stats.effect_size.random_effects_summary`).

    Returns pooled estimate + heterogeneity + per-stratum panel.
    `pooled_d` is in Cohen's d units (sign-stable, scale-free).
    NaN-pooled when fewer than 2 strata pass filters.
    """
    # Phase 5: delegate panel-build to `stratum_panel`. We need
    # both `source` and `scope_predictor` on the panel so the
    # `min_baseline_predictor` stratum-level filter can read the
    # baseline arm's `scope_predictor` mean per stratum.
    measurables_for_panel: tuple[str, ...] = (
        (source,) if source == scope_predictor
        else (source, scope_predictor)
    )
    panel = stratum_panel.fn(
        cells,
        measurables=measurables_for_panel,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        stratify_by=stratify_by,
        arm_field=arm_field,
        min_seeds_per_arm=min_seeds_per_arm,
    )

    cohen_d_per_stratum = panel.cohen_d(source)
    cohen_se_per_stratum = panel.cohen_se(source)

    per_stratum: list[StratumDiff] = []
    for i, stratum_id in enumerate(panel.strata):
        # n_seeds reflects FINITE-VALUE counts for `source`,
        # matching legacy semantics where cells with NaN source
        # were excluded before counting.
        n_t = panel.n_treatment_per_measurable[source][i]
        n_b = panel.n_baseline_per_measurable[source][i]
        if n_t < min_seeds_per_arm or n_b < min_seeds_per_arm:
            continue
        # Stratum-level scope filter on vanilla aggregate
        v_predictor_mean = panel.means_baseline[scope_predictor][i]
        if math.isnan(v_predictor_mean):
            continue
        if v_predictor_mean <= min_baseline_predictor:
            continue
        mean_t = panel.means_treatment[source][i]
        mean_b = panel.means_baseline[source][i]
        sd_t = panel.stds_treatment[source][i]
        sd_b = panel.stds_baseline[source][i]
        per_stratum.append(StratumDiff(
            stratum_id=stratum_id,
            mean_diff=mean_t - mean_b,
            cohen_d=cohen_d_per_stratum[i],
            cohen_se=cohen_se_per_stratum[i],
            arm_mean_treatment=mean_t,
            arm_mean_baseline=mean_b,
            arm_sd_treatment=sd_t,
            arm_sd_baseline=sd_b,
            n_seeds_treatment=n_t,
            n_seeds_baseline=n_b,
            baseline_predictor=v_predictor_mean,
        ))

    # DL pooling on (cohen_d, cohen_se)
    obs = [
        (s.cohen_d, s.cohen_se) for s in per_stratum
        if not math.isnan(s.cohen_d) and not math.isnan(s.cohen_se)
        and s.cohen_se > 0
    ]
    if len(obs) >= 2:
        pooled = random_effects_summary(obs)
        pooled_d = pooled.pooled_g
        pooled_se = pooled.se_pooled
        tau2 = pooled.tau2
        i_squared = pooled.I2
        q_stat = pooled.Q
    else:
        pooled = PooledStats(
            pooled_g=float('nan'), se_pooled=float('nan'),
            tau2=float('nan'), I2=float('nan'), Q=float('nan'),
            pi_lo=float('nan'), pi_hi=float('nan'),
            empirical_min_g=float('nan'), empirical_max_g=float('nan'),
            n_cells=len(obs), assumption_violations=(),
        )
        pooled_d = float('nan')
        pooled_se = float('nan')
        tau2 = float('nan')
        i_squared = float('nan')
        q_stat = float('nan')

    verdict, refutation = random_effects_verdict(
        pooled, predicted_direction=predicted_direction,
    )

    return StratifiedArmDiffPooledResult(
        pooled_d=pooled_d,
        pooled_se=pooled_se,
        tau2=tau2,
        i_squared=i_squared,
        q_statistic=q_stat,
        n_strata=len(per_stratum),
        per_stratum=tuple(per_stratum),
        stratify_by=stratify_by,
        measurable=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        arm_field=arm_field,
        scope_predictor=scope_predictor,
        pooled=pooled,
        verdict=verdict,
        refutation=refutation,
    )


__all__ = [
    'StratumDiff',
    'StratifiedArmDiffPooledResult',
    'stratified_arm_diff_pooled',
]
