"""`paired_link_per_env` — per-env scalar paired-link r, then meta-
regress on a per-env moderator.

Where `paired_link_per_burst` consumes per-burst-vector Measurables
to expose the temporal structure of the link, this primitive collapses
to one paired link r per env (using scalar measurables) and exposes
the cross-env relationship between that link and an env-level
moderator.

Schematically, for each env in scope:

    pair (treatment, baseline) on `pair_by`
    Δ_target    = target(treatment)    − target(baseline)
    Δ_predictor = predictor(treatment) − predictor(baseline)
    link_r_env  = Pearson r(Δ_target, Δ_predictor) across pairs

    moderator_env = mean over cells of a per-cell scalar
                    (e.g. `env_reward_polarity`)

Then meta-regress link_r_env on moderator_env across envs:

    atanh(link_r_env) ~ β₀ + β_M · moderator_env

Returns `MetaRegressionResult` with the moderator's coefficient.
The Fisher-z transform stabilises the SE: `se = 1/sqrt(n_pairs − 3)`.

The canonical use: testing whether `r(Δ_eff_h, Δ_outcome)` per env
is predictable from `env_reward_polarity` — the explicit form of the
"polarity predicts link sign" soft tautology (CLAIM 14 in
`experiments/findings/ddqn/`).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from collections import defaultdict

import numpy as np

from corroborate.analyses.paired_g import resolve_value as _resolve_value
from corroborate.bridge.analysis import analysis
from corroborate.corpus.schema import StratumG
from corroborate.stats import MetaRegressionResult, meta_regress_panel
from corroborate.stats.meta_regression import Pool


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with degenerate-input guards. Returns NaN on n<3,
    zero variance, or non-finite output. The call site projects to
    Fisher z, where r=±1 is clamped at the panel-build boundary."""
    if len(x) < 3:
        return float('nan')
    sx = float(x.std(ddof=0))
    sy = float(y.std(ddof=0))
    if sx == 0 or sy == 0:
        return float('nan')
    r = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(r):
        return float('nan')
    return r


def _build_per_env_link_panel(
    cells: Sequence[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    target: str,
    predictor: str,
    pair_by: tuple[str, ...],
    env_filter: tuple[str, ...],
    arm_field: str,
) -> tuple[tuple[str, float, int], ...]:
    """Returns a panel of `(env, link_r, n_pairs)` tuples.

    The shape is intentionally raw (not yet `StratumG`) so the
    caller can choose its own SE policy (Fisher-z, bootstrap, etc.)
    when projecting to the panel `meta_regress_panel` consumes."""
    by_env: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    env_set = set(env_filter) if env_filter else None
    for cell in cells:
        env = cell.get('env_name')
        if not isinstance(env, str):
            continue
        if env_set is not None and env not in env_set:
            continue
        by_env[env].append(cell)

    panel: list[tuple[str, float, int]] = []
    for env, env_cells in sorted(by_env.items()):
        # Pair (treatment, baseline) on pair_by.
        keyed: dict[
            tuple[object, ...],
            dict[str, Mapping[str, object]],
        ] = defaultdict(dict)
        for c in env_cells:
            arm = c.get(arm_field)
            if arm not in (treatment_arm, baseline_arm):
                continue
            key = tuple(c.get(k) for k in pair_by)
            arm_str = treatment_arm if arm == treatment_arm else baseline_arm
            keyed[key][arm_str] = c
        d_target: list[float] = []
        d_predictor: list[float] = []
        for k, arms in keyed.items():
            del k
            t = arms.get(treatment_arm)
            b = arms.get(baseline_arm)
            if t is None or b is None:
                continue
            t_target = _resolve_value(t, target)
            b_target = _resolve_value(b, target)
            t_pred = _resolve_value(t, predictor)
            b_pred = _resolve_value(b, predictor)
            if any(math.isnan(x) for x in (t_target, b_target, t_pred, b_pred)):
                continue
            d_target.append(t_target - b_target)
            d_predictor.append(t_pred - b_pred)
        if len(d_target) < 3:
            continue
        r = _pearson_r(np.asarray(d_predictor), np.asarray(d_target))
        panel.append((env, r, len(d_target)))
    return tuple(panel)


def _build_per_env_moderator_means(
    cells: Sequence[Mapping[str, object]],
    *,
    moderator: str,
    env_filter: tuple[str, ...],
) -> dict[str, float]:
    """Per-env mean of the moderator measurable across cells. NaNs
    skipped; envs with all-NaN values omitted from the dict."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    env_set = set(env_filter) if env_filter else None
    for cell in cells:
        env = cell.get('env_name')
        if not isinstance(env, str):
            continue
        if env_set is not None and env not in env_set:
            continue
        v = _resolve_value(cell, moderator)
        if math.isnan(v):
            continue
        sums[env] += v
        counts[env] += 1
    return {
        env: sums[env] / counts[env]
        for env in sums
        if counts[env] > 0
    }


@analysis
def paired_link_per_env(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    target: str,
    predictor: str,
    moderator: str,
    pair_by: tuple[str, ...] = ('seed',),
    env_filter: tuple[str, ...] = (),
    arm_field: str = 'arm_key',
    pool: Pool = 'random',
    alpha: float = 0.05,
) -> MetaRegressionResult:
    """Per-env paired link r(Δ_target, Δ_predictor) → meta-regress
    on `moderator`'s per-env mean. Returns the meta-regression
    result; the bridge reads `coefficients[moderator]` for the
    cross-env relationship.

    **Sign convention (NOT the same as `paired_link_per_burst`).**
    Per-env r is computed on RAW Δ vectors — `Δ_predictor` is NOT
    negated. The sign of r therefore depends on which direction
    the substrate author named as `predictor` and `target`:
    - if `predictor` is "lower-is-better mediator residual" (e.g.
      `jensen_gap`) and `target` is "higher-is-better outcome", an
      *active* link reads NEGATIVE r (more reduction → more gain
      = Δ_predictor↓ ⇒ Δ_target↑).
    - if both are oriented the same way, an active link reads
      POSITIVE r.
    `paired_link_per_burst` flips `Δ_predictor` so positive-r
    always reads "active link"; this primitive does not. Bridge
    authors gating on `coefficients[moderator]` must commit
    explicitly to which sign they predict.

    Fisher-z projection: each env's link r is mapped to atanh(r)
    with se = 1 / sqrt(n_pairs − 3). r values within ±1e-6 of ±1
    clamp to ±0.999999 before atanh to avoid `inf`. Envs with
    n_pairs < 4, NaN r, or NaN/missing moderator mean drop from
    the panel.

    `meta_regress_panel`'s contract requires `n_pairs ≥ 2` per
    stratum AND `se > 0`; both implied by `n_pairs ≥ 4` (so
    Fisher-z se is finite and positive)."""
    cells_list = list(cells)
    panel_raw = _build_per_env_link_panel(
        cells_list,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        target=target,
        predictor=predictor,
        pair_by=pair_by,
        env_filter=env_filter,
        arm_field=arm_field,
    )
    moderator_means = _build_per_env_moderator_means(
        cells_list,
        moderator=moderator,
        env_filter=env_filter,
    )

    # Project to Fisher-z + StratumG panel.
    panel: list[StratumG[str]] = []
    covariates: dict[str, dict[str, float]] = {}
    for env, r, n_pairs in panel_raw:
        if n_pairs < 4 or math.isnan(r):
            continue
        if env not in moderator_means:
            continue
        r_clamped = max(-0.999999, min(0.999999, r))
        z = float(math.atanh(r_clamped))
        se = 1.0 / math.sqrt(n_pairs - 3)
        panel.append(StratumG[str](
            stratum_id=env, g=z, se=se, n_pairs=n_pairs,
        ))
        covariates[env] = {moderator: moderator_means[env]}

    return meta_regress_panel(
        panel,
        covariates_per_stratum=covariates,
        alpha=alpha,
        pool=pool,
    )


__all__ = ['paired_link_per_env']
