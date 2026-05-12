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
    covariates_per_key: Mapping[object, Mapping[str, float]],
    covariate_key_field: str = 'env_name',
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
    key-level covariates.

    `covariate_key_field` (default `'env_name'`) names which
    `stratify_by` dimension keys `covariates_per_key`. The
    analysis broadcasts `covariates_per_key[stratum[i]]` to all
    strata where `stratum_id[i]` matches a key (and `i` is the
    position of `covariate_key_field` in `stratify_by`).
    `covariate_key_field` MUST appear in `stratify_by`.

    Common shapes:
    - **Cross-env scaling**: `covariate_key_field='env_name'`,
      `covariates_per_key={'Acrobot-v1': {'eff_h': 49}, ...}`.
    - **Within-env γ scaling (CLAIM 5)**:
      `covariate_key_field='gamma'`,
      `covariates_per_key={0.99: {'eff_h': 28}, 0.999: {'eff_h': 70}}`,
      `stratify_by=('gamma', 'total_steps', 'reward_scale')` (env
      pinned by the bridge's scope predicate).

    Empty / underpowered panel → NaN-coefficient result (empty
    coefficients tuple, intercept=NaN). Bridges should check
    `coef is None` or `math.isnan(coef.coefficient)` for graceful
    POW_INSUF fallthrough."""
    if not stratify_by or covariate_key_field not in stratify_by:
        raise ValueError(
            f"meta_regression_unpaired_d: covariate_key_field "
            f"{covariate_key_field!r} must appear in stratify_by; "
            f"got {stratify_by!r}"
        )
    key_position = stratify_by.index(covariate_key_field)
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
        if not s.stratum_id or len(s.stratum_id) <= key_position:
            continue
        key = s.stratum_id[key_position]
        key_covs = covariates_per_key.get(key)
        if key_covs is None:
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
        cps[s.stratum_id] = key_covs
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
