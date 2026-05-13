"""Vanilla-predictor → Δ-outcome link backdoor + refutations.

Tests the *common-claim* of "reducing bias causes higher return"
empirically via DoWhy backdoor on per-(env, config) **stratum-
level** rows. At each stratum:

    vanilla_predictor[stratum] = mean over baseline seeds of `predictor`
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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from corroborate.analyses.dowhy import (
    BackdoorResult,
    RefutationResult,
    backdoor_ate,
    placebo_refutation,
    random_common_cause_refutation,
)
from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class StratumVanillaPredictorLinkDowhyResult:
    """Backdoor + placebo + RCC refutations of `vanilla_predictor
    → Δ_target` on per-(env, config) **stratum-level** rows.
    Adjusts for env (one-hot)."""
    backdoor: BackdoorResult
    placebo: RefutationResult
    random_common_cause: RefutationResult
    n_strata: int
    treatment_col: str
    outcome_col: str


def _build_stratum_panel(
    cells: Sequence[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    predictor_col: str,
    target_col: str,
    stratify_by: tuple[str, ...],
    arm_field: str,
    min_seeds_per_arm: int,
    min_vanilla_predictor: float,
) -> tuple[
    list[Mapping[str, object]],
    list[tuple[str, str]],
    str,
    str,
]:
    """Build per-stratum rows: one row per unique `stratify_by`
    tuple. Each row carries `vanilla_predictor` (mean over
    baseline cells of `predictor_col`) and `delta_target`
    (mean_T − mean_B of `target_col`). Env one-hot dummies are
    added for backdoor adjustment."""
    treatment_col = 'v_pred'
    outcome_col = 'd_out'

    # Group cells by (stratum_key, arm). Stratum key = tuple of
    # values pulled from `stratify_by` columns.
    by_stratum_arm: dict[
        tuple[object, ...],
        dict[str, list[Mapping[str, object]]],
    ] = {}
    for cell in cells:
        arm = cell.get(arm_field)
        if not isinstance(arm, str):
            continue
        if arm not in (treatment_arm, baseline_arm):
            continue
        try:
            key = tuple(cell.get(k) for k in stratify_by)
        except (TypeError, KeyError):
            continue
        if any(v is None for v in key):
            continue
        by_stratum_arm.setdefault(key, {}).setdefault(arm, []).append(cell)

    stratum_rows: list[dict[str, object]] = []
    envs_seen: list[str] = []
    for key, arms_dict in by_stratum_arm.items():
        t_cells = arms_dict.get(treatment_arm, [])
        b_cells = arms_dict.get(baseline_arm, [])
        if len(t_cells) < min_seeds_per_arm:
            continue
        if len(b_cells) < min_seeds_per_arm:
            continue
        b_pred_vals: list[float] = []
        for c in b_cells:
            v = c.get(predictor_col)
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                b_pred_vals.append(float(v))
        t_target_vals: list[float] = []
        for c in t_cells:
            v = c.get(target_col)
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                t_target_vals.append(float(v))
        b_target_vals: list[float] = []
        for c in b_cells:
            v = c.get(target_col)
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                b_target_vals.append(float(v))
        if (
            len(b_pred_vals) < min_seeds_per_arm
            or len(t_target_vals) < min_seeds_per_arm
            or len(b_target_vals) < min_seeds_per_arm
        ):
            continue
        vanilla_pred = sum(b_pred_vals) / len(b_pred_vals)
        if vanilla_pred <= min_vanilla_predictor:
            continue
        delta_target = (
            sum(t_target_vals) / len(t_target_vals)
            - sum(b_target_vals) / len(b_target_vals)
        )
        # env is the first element of stratify_by by convention;
        # if not present, fall back to 'env_name' lookup on a cell.
        if 'env_name' in stratify_by:
            env_idx = stratify_by.index('env_name')
            env_v = key[env_idx]
        else:
            env_v = b_cells[0].get('env_name')
        if not isinstance(env_v, str):
            continue
        if env_v not in envs_seen:
            envs_seen.append(env_v)
        stratum_rows.append({
            'env_name': env_v,
            treatment_col: vanilla_pred,
            outcome_col: delta_target,
        })

    if not stratum_rows:
        return [], [], treatment_col, outcome_col

    # Env one-hot, drop first env to avoid collinearity.
    envs_sorted = sorted(envs_seen)
    env_dummy_cols = [f'__env__{e}' for e in envs_sorted[1:]]
    rows: list[Mapping[str, object]] = []
    for r in stratum_rows:
        row = dict(r)
        e = r['env_name']
        for col in env_dummy_cols:
            row[col] = 1.0 if col == f'__env__{e}' else 0.0
        rows.append(row)

    dag: list[tuple[str, str]] = []
    for c in env_dummy_cols:
        dag.append((c, treatment_col))
        dag.append((c, outcome_col))
    dag.append((treatment_col, outcome_col))
    return rows, dag, treatment_col, outcome_col


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
def stratum_vanilla_predictor_link_dowhy(
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
    min_vanilla_predictor: float = 0.05,
) -> StratumVanillaPredictorLinkDowhyResult:
    """Test `vanilla_predictor → Δ_target` on stratum-level rows
    via DoWhy backdoor + refutations, adjusting for env.

    Each stratum: pool seeds within each arm INDEPENDENTLY, take
    baseline-arm-mean of `predictor_col` as the treatment value
    and (treatment-arm-mean − baseline-arm-mean) of `target_col`
    as the outcome value. No seed pairing.

    Strata where vanilla mean predictor < `min_vanilla_predictor`
    (premise inactive) drop before DoWhy.

    Empty panel (no stratum survives filters) yields a
    NaN-everywhere result."""
    cells_list = list(cells)
    rows, dag, treatment_col, outcome_col = _build_stratum_panel(
        cells_list,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        predictor_col=predictor_col,
        target_col=target_col,
        stratify_by=stratify_by,
        arm_field=arm_field,
        min_seeds_per_arm=min_seeds_per_arm,
        min_vanilla_predictor=min_vanilla_predictor,
    )
    if not rows:
        return StratumVanillaPredictorLinkDowhyResult(
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

    backdoor = backdoor_ate.fn(
        rows, treatment=treatment_col, outcome=outcome_col,
        dag=dag, method_name=method_name,
    )
    placebo = placebo_refutation.fn(
        rows, treatment=treatment_col, outcome=outcome_col,
        dag=dag, method_name=method_name,
    )
    rcc = random_common_cause_refutation.fn(
        rows, treatment=treatment_col, outcome=outcome_col,
        dag=dag, method_name=method_name,
    )
    return StratumVanillaPredictorLinkDowhyResult(
        backdoor=backdoor,
        placebo=placebo,
        random_common_cause=rcc,
        n_strata=len(rows),
        treatment_col=treatment_col,
        outcome_col=outcome_col,
    )


__all__ = [
    'StratumVanillaPredictorLinkDowhyResult',
    'stratum_vanilla_predictor_link_dowhy',
]
