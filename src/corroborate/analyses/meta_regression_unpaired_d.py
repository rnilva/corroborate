"""`meta_regression_unpaired_d` — per-stratum independent-samples
Cohen's d, then random-effects meta-regression on env-level
covariates.

Sibling of `meta_regression_paired_g`: that primitive computes
per-env Hedges' g via seed-pairing, then meta-regresses on
env-level covariates. This primitive uses per-stratum
independent-samples Cohen's d (via `stratified_arm_diff_pooled`)
— the seed-pairing-free form per CLAUDE.md's
`feedback_paired_g_in_rl` rule.

Within-env strata (configs at different `total_steps`,
`reward_scale`, `sync_period`, etc.) contribute multiple panel
rows for the same env. The meta-regression's between-stratum
variance captures within-env config heterogeneity properly; the
env-level covariate slope is estimated from between-env
variation with SE that accounts for the within-env replicates.

The composition: cells →
`stratified_arm_diff_pooled(stratify_by=...)` → per-stratum
(`cohen_d`, `cohen_se`) panel → `StratumG` records →
`meta_regress_panel(covariates_per_stratum=...)` →
`MetaRegressionResult`. NaN strata (e.g. saturated outcome →
NaN d) and mech-dormant strata (filtered via
`min_vanilla_predictor`) are dropped before meta-regression.

Returns NaN-coefficient `MetaRegressionResult` when the panel
is too small for OLS (n_strata ≤ 1 + n_covariates) or design is
singular — bridges checking `coef is None` (or NaN coefficient)
naturally fall through to POW_INSUF."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from corroborate.analyses.stratified_arm_diff_pooled import (
    stratified_arm_diff_pooled,
)
from corroborate.bridge.analysis import analysis
from corroborate.corpus.schema import StratumG
from corroborate.stats import MetaRegressionResult, meta_regress_panel
from corroborate.stats.meta_regression import Pool


@analysis
def meta_regression_unpaired_d(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    source: str,
    covariates_per_env: Mapping[str, Mapping[str, float]],
    stratify_by: tuple[str, ...] = (
        'env_name', 'total_steps', 'reward_scale',
    ),
    scope_predictor: str = 'jensen_gap',
    min_vanilla_predictor: float = 0.05,
    min_seeds_per_arm: int = 5,
    alpha: float = 0.05,
    pool: Pool = 'random',
) -> MetaRegressionResult:
    """Per-stratum Cohen's d → random-effects meta-regression on
    env-level covariates.

    First dimension of `stratify_by` MUST be `'env_name'` — the
    analysis broadcasts `covariates_per_env[env_name]` to all
    strata with that env via `stratum_id[0]`. Remaining
    dimensions (`total_steps`, `reward_scale`, etc.) just add
    panel rows.

    Empty / underpowered panel → NaN-coefficient result (empty
    coefficients tuple, intercept=NaN). Bridges should check
    `coef is None` or `math.isnan(coef.coefficient)` for graceful
    POW_INSUF fallthrough."""
    if not stratify_by or stratify_by[0] != 'env_name':
        raise ValueError(
            "meta_regression_unpaired_d: stratify_by must start with "
            f"'env_name'; got {stratify_by!r}"
        )
    cells_list = list(cells)
    pooled = stratified_arm_diff_pooled.fn(
        cells_list,
        source=source,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        stratify_by=stratify_by,
        scope_predictor=scope_predictor,
        min_vanilla_predictor=min_vanilla_predictor,
        min_seeds_per_arm=min_seeds_per_arm,
    )
    panel: list[StratumG[tuple[object, ...]]] = []
    cps: dict[tuple[object, ...], Mapping[str, float]] = {}
    for s in pooled.per_stratum:
        if not s.stratum_id:
            continue
        env = s.stratum_id[0]
        if not isinstance(env, str):
            continue
        env_covs = covariates_per_env.get(env)
        if env_covs is None:
            continue
        if math.isnan(s.cohen_d) or math.isnan(s.cohen_se):
            continue
        if s.cohen_se <= 0.0:
            continue
        panel.append(StratumG(
            stratum_id=s.stratum_id,
            g=s.cohen_d,
            se=s.cohen_se,
            n_pairs=s.n_seeds_treatment + s.n_seeds_baseline,
        ))
        cps[s.stratum_id] = env_covs
    try:
        return meta_regress_panel(
            panel, covariates_per_stratum=cps, alpha=alpha, pool=pool,
        )
    except ValueError:
        # NaN everything sensitive to "did the regression run?"
        # so downstream readers see "unfit" not "homogeneous fit."
        # `pool` and `i_squared` would otherwise default to misleading
        # values (`pool='fixed'`, `i_squared=0.0`) per
        # MetaRegressionResult's frozen-dataclass defaults.
        return MetaRegressionResult(
            n_strata=len(panel),
            intercept=float('nan'),
            coefficients=(),
            r_squared=float('nan'),
            intercept_se=float('nan'),
            intercept_ci_lo=float('nan'),
            intercept_ci_hi=float('nan'),
            intercept_p_value=float('nan'),
            tau_sq=float('nan'),
            q_statistic=float('nan'),
            i_squared=float('nan'),
            pool=pool,
        )


__all__ = ['meta_regression_unpaired_d']
