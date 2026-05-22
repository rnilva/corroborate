"""Vanilla-predictor → Δ-outcome link backdoor + refutations.

Tests the *common-claim* of "reducing bias causes higher return"
empirically via DoWhy backdoor on per-(env, config) **stratum-
level** rows. At each stratum:

    baseline_predictor[stratum] = mean over baseline seeds of `predictor`
    Δ_target[stratum]         = mean_T(target) − mean_B(target)
                                (independent-samples; no seed pairing)

The predictor (e.g., `jensen_gap` measured on the baseline arm)
captures the algorithm's premise magnitude — "vanilla overestimates
by this much in this regime." The target (e.g.,
`eval_best_burst_raw_mean`) is the policy-quality outcome.

Why this shape and not `stratum_delta_link_dowhy`: that sibling
correlates Δ_predictor (= Δ_Q − Δ_MC by definition when
predictor = `jensen_gap` and target = `mc_return`) with Δ_target
(= Δ_MC), exposing Δ_MC on both sides and producing a
near-tautological positive r regardless of mechanism. The
present primitive uses **only the baseline arm's predictor** (so
Δ_MC contributes nothing to the predictor side) and the
cross-arm Δ on target — empirically independent quantities.

Env one-hot is the adjustment set (envs differ in reward scale
and base bias magnitude; without adjustment a strong cross-env
positive slope appears that's purely scale-confound). The
within-env test is the load-bearing one: do configs where
vanilla overestimates more correspond to configs where DDQN
gains more outcome?

Output mirrors `StratumDeltaLinkDowhyResult` so the same
verdict helpers (`dowhy_backdoor_verdict`, `dowhy_placebo_verdict`,
`dowhy_rcc_verdict`) compose without adaptation."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from corroborate.analyses._dowhy_internal import backdoor_with_refutations
from corroborate.analyses.dowhy import (
    BackdoorResult,
    RefutationResult,
)
from corroborate.analyses.panel.stratum_panel import stratum_panel
from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class StratumBaselinePredictorLinkDowhyResult:
    """Backdoor + placebo + RCC refutations of `baseline_predictor
    → Δ_target` on per-(env, config) **stratum-level** rows.
    Adjusts for env (one-hot)."""
    backdoor: BackdoorResult
    placebo: RefutationResult
    random_common_cause: RefutationResult
    n_strata: int
    treatment_col: str
    outcome_col: str


# Removed `_build_stratum_panel` in Phase 6 migration (2026-05-13):
# panel-build moved to `stratum_panel`; the body's per-stratum row
# assembly + env one-hot encoding now lives directly in
# `stratum_baseline_predictor_link_dowhy`.


def _nan_backdoor(
    treatment_col: str, outcome_col: str, method_name: str,
) -> BackdoorResult:
    return BackdoorResult(
        ate=float('nan'),
        identified=False,
        estimand_str='',
        method_name=method_name,
        treatment=treatment_col,
        outcome=outcome_col,
        n_rows=0,
    )


def _nan_refutation(
    treatment_col: str, outcome_col: str, method_name: str,
    refuter_name: str,
) -> RefutationResult:
    return RefutationResult(
        real_ate=float('nan'),
        refuted_ate=float('nan'),
        drift=float('nan'),
        method_name=method_name,
        refuter_name=refuter_name,
        treatment=treatment_col,
        outcome=outcome_col,
        n_rows=0,
    )


@analysis
def stratum_baseline_predictor_link_dowhy(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    predictor_col: str = 'jensen_gap',
    target_col: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = (
        'env_name', 'gamma', 'sync_period', 'total_steps',
        'optimizer.inner.lr', 'optimizer.inner.weight_decay',
        'replay.capacity',
    ),
    arm_field: str = 'arm_key',
    method_name: str = 'backdoor.linear_regression',
    min_seeds_per_arm: int = 5,
    min_baseline_predictor: float = 0.05,
    random_state: int = 0,
) -> StratumBaselinePredictorLinkDowhyResult:
    """Test `baseline_predictor → Δ_target` on stratum-level rows
    via DoWhy backdoor + refutations, adjusting for env.

    Phase 6 migration (2026-05-13): delegates panel-build to
    `stratum_panel`. The per-stratum rows + env-one-hot DAG
    construction stay; the cell→stratum aggregation moves into
    the shared panel primitive. Result type and semantics
    unchanged — verdict-preserving.

    Each stratum: pool seeds within each arm INDEPENDENTLY, take
    baseline-arm-mean of `predictor_col` as the treatment value
    and (treatment-arm-mean − baseline-arm-mean) of `target_col`
    as the outcome value. No seed pairing.

    Strata where vanilla mean predictor < `min_baseline_predictor`
    (premise inactive) drop before DoWhy.

    Empty panel (no stratum survives filters) yields a
    NaN-everywhere result."""
    cells_list = list(cells)
    treatment_col = 'v_pred'
    outcome_col = 'd_out'
    measurables_for_panel: tuple[str, ...] = (
        (predictor_col,) if predictor_col == target_col
        else (predictor_col, target_col)
    )
    panel = stratum_panel.fn(
        cells_list,
        measurables=measurables_for_panel,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        stratify_by=stratify_by,
        arm_field=arm_field,
        min_seeds_per_arm=min_seeds_per_arm,
    )
    rows: list[Mapping[str, object]] = []
    envs_seen: list[str] = []
    try:
        env_idx = stratify_by.index('env_name')
    except ValueError:
        env_idx = None
    for i, stratum_id in enumerate(panel.strata):
        # Match legacy semantics: require ≥ `min_seeds_per_arm`
        # FINITE-VALUE cells for predictor (on baseline) and
        # target (on both arms) before computing means.
        n_b_pred = panel.n_baseline_per_measurable[predictor_col][i]
        n_t_target = panel.n_treatment_per_measurable[target_col][i]
        n_b_target = panel.n_baseline_per_measurable[target_col][i]
        if (
            n_b_pred < min_seeds_per_arm
            or n_t_target < min_seeds_per_arm
            or n_b_target < min_seeds_per_arm
        ):
            continue
        v_pred = panel.means_baseline[predictor_col][i]
        if math.isnan(v_pred) or v_pred <= min_baseline_predictor:
            continue
        v_target = panel.means_baseline[target_col][i]
        t_target = panel.means_treatment[target_col][i]
        if math.isnan(v_target) or math.isnan(t_target):
            continue
        env_v: object | None = (
            stratum_id[env_idx] if env_idx is not None else None
        )
        if not isinstance(env_v, str):
            continue
        if env_v not in envs_seen:
            envs_seen.append(env_v)
        rows.append({
            'env_name': env_v,
            treatment_col: v_pred,
            outcome_col: t_target - v_target,
        })

    if rows:
        envs_sorted = sorted(envs_seen)
        env_dummy_cols = [f'__env__{e}' for e in envs_sorted[1:]]
        rows = [
            {
                **r,
                **{
                    col: (1.0 if col == f'__env__{r["env_name"]}' else 0.0)
                    for col in env_dummy_cols
                },
            }
            for r in rows
        ]
        dag: list[tuple[str, str]] = []
        for c in env_dummy_cols:
            dag.append((c, treatment_col))
            dag.append((c, outcome_col))
        dag.append((treatment_col, outcome_col))
    else:
        dag = []
    if not rows:
        return StratumBaselinePredictorLinkDowhyResult(
            backdoor=_nan_backdoor(
                treatment_col, outcome_col, method_name,
            ),
            placebo=_nan_refutation(
                treatment_col, outcome_col, method_name,
                refuter_name='placebo_treatment_refuter',
            ),
            random_common_cause=_nan_refutation(
                treatment_col, outcome_col, method_name,
                refuter_name='random_common_cause',
            ),
            n_strata=0,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
        )

    backdoor, placebo, rcc = backdoor_with_refutations(
        rows, treatment=treatment_col, outcome=outcome_col,
        dag=dag, method_name=method_name,
        random_state=random_state,
    )
    return StratumBaselinePredictorLinkDowhyResult(
        backdoor=backdoor,
        placebo=placebo,
        random_common_cause=rcc,
        n_strata=len(rows),
        treatment_col=treatment_col,
        outcome_col=outcome_col,
    )


__all__ = [
    'StratumBaselinePredictorLinkDowhyResult',
    'stratum_baseline_predictor_link_dowhy',
]
