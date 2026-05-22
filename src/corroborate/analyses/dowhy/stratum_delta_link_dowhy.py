"""Stratum-Δ link backdoor + refutations.

Tests the *continuous* mech→outcome link via DoWhy backdoor on
per-(env, burst) **stratum-level** Δ rows. At each (env, burst):
pool DDQN seeds, pool vanilla seeds INDEPENDENTLY, then

    Δ_predictor[env, b] = mean_T(predictor[seed, b]) − mean_B(predictor[seed, b])
    Δ_target[env, b]    = mean_T(target[seed, b])    − mean_B(target[seed, b])

No seed pairing. Each (env, burst) row is one independent-samples
estimate of the within-stratum treatment-effect Δ. (Replaces the
deleted seed-paired sibling `paired_delta_link_dowhy`, which
built rows per-(env, burst, seed) — pseudo-replicating each
stratum by N seeds where N_effective_per_stratum is 1.)

Mech conditioning is built in: strata where vanilla's mean
predictor is below `min_baseline_predictor` (G1 premise inactive)
are skipped before they reach DoWhy. A link-bridge consuming this
fixture is automatically conditioned on mech-premise-active strata.

DAG: burst-dummies → Δ_predictor, burst-dummies → Δ_target,
Δ_predictor → Δ_target. Burst one-hot encoding drops the first
burst to avoid collinearity (standard treatment).

Output (`StratumDeltaLinkDowhyResult`) carries `n_strata`
(NOT `n_pairs`) to be honest about the unit of inference —
strata, not seed-pairs."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from corroborate.analyses._dowhy_internal import backdoor_with_refutations
from corroborate.analyses.dowhy import (
    BackdoorResult,
    RefutationResult,
)
from corroborate.analyses._cell_value import evaluate_per_burst_source
from corroborate.bridge.analysis import analysis
from corroborate.measurables import Measurable


@dataclass(frozen=True, slots=True)
class StratumDeltaLinkDowhyResult:
    """Backdoor + placebo + RCC refutations of `Δ_predictor →
    Δ_target` on per-(env, burst) **stratum-level** Δ rows.
    Adjusts for burst (one-hot); env can be filtered to a single
    substrate."""
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
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    env_filter: tuple[str, ...],
    arm_field: str,
    min_baseline_predictor: float,
) -> tuple[
    list[Mapping[str, object]],
    list[tuple[str, str]],
    str,
    str,
]:
    """Build per-(env, burst) **stratum-level** Δ panel rows + DAG."""
    treatment_col = 'djens'
    outcome_col = 'dout'

    by_env_arm: dict[
        tuple[str, str],
        list[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]],
    ] = {}
    env_filter_set = set(env_filter) if env_filter else None
    for cell in cells:
        env = cell.get('env_name')
        arm = cell.get(arm_field)
        if not isinstance(env, str) or not isinstance(arm, str):
            continue
        if env_filter_set is not None and env not in env_filter_set:
            continue
        if arm not in (treatment_arm, baseline_arm):
            continue
        target_v = evaluate_per_burst_source(link_target, cell)
        predictor_v = evaluate_per_burst_source(link_predictor, cell)
        if target_v.size == 0 or predictor_v.size == 0:
            continue
        if target_v.shape[0] != predictor_v.shape[0]:
            continue
        by_env_arm.setdefault((env, arm), []).append(
            (target_v, predictor_v),
        )

    stratum_rows: list[dict[str, object]] = []
    bursts_seen: set[int] = set()
    envs = sorted({e for (e, _) in by_env_arm})
    for env in envs:
        t_cells = by_env_arm.get((env, treatment_arm), [])
        b_cells = by_env_arm.get((env, baseline_arm), [])
        if not t_cells or not b_cells:
            continue
        # Per (env, burst): pool DDQN seeds, pool vanilla seeds,
        # independent-samples Δ. burst index must exist in both arms.
        t_min_bursts = min(c[0].shape[0] for c in t_cells)
        b_min_bursts = min(c[0].shape[0] for c in b_cells)
        common_bursts = min(t_min_bursts, b_min_bursts)
        if common_bursts == 0:
            continue
        for b in range(common_bursts):
            t_target_b = np.array(
                [float(c[0][b]) for c in t_cells], dtype=np.float64,
            )
            t_pred_b = np.array(
                [float(c[1][b]) for c in t_cells], dtype=np.float64,
            )
            b_target_b = np.array(
                [float(c[0][b]) for c in b_cells], dtype=np.float64,
            )
            b_pred_b = np.array(
                [float(c[1][b]) for c in b_cells], dtype=np.float64,
            )
            # NaN-aware means; require ≥1 finite per arm
            mean_t_target = float(np.nanmean(t_target_b))
            mean_t_pred = float(np.nanmean(t_pred_b))
            mean_b_target = float(np.nanmean(b_target_b))
            mean_b_pred = float(np.nanmean(b_pred_b))
            if any(math.isnan(v) for v in (
                mean_t_target, mean_t_pred,
                mean_b_target, mean_b_pred,
            )):
                continue
            # Mech conditioning: skip stratum if vanilla's mean
            # predictor is below the premise-active floor.
            if mean_b_pred <= min_baseline_predictor:
                continue
            d_target = mean_t_target - mean_b_target
            d_pred = mean_t_pred - mean_b_pred
            bursts_seen.add(b)
            stratum_rows.append({
                'env_name': env,
                'burst_index': b,
                outcome_col: d_target,
                treatment_col: d_pred,
            })

    if not stratum_rows:
        return [], [], treatment_col, outcome_col

    bursts_sorted = sorted(bursts_seen)
    burst_dummy_cols = [f'__burst__{b}' for b in bursts_sorted[1:]]
    rows: list[Mapping[str, object]] = []
    for r in stratum_rows:
        row = dict(r)
        b = r['burst_index']
        for col in burst_dummy_cols:
            row[col] = 1.0 if col == f'__burst__{b}' else 0.0
        rows.append(row)

    dag: list[tuple[str, str]] = []
    for c in burst_dummy_cols:
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
def stratum_delta_link_dowhy(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    env_filter: tuple[str, ...] = (),
    arm_field: str = 'arm_key',
    method_name: str = 'backdoor.linear_regression',
    min_baseline_predictor: float = 0.05,
    random_state: int = 0,
) -> StratumDeltaLinkDowhyResult:
    """Test `Δ_predictor → Δ_target` on stratum-level (env, burst)
    Δ rows via DoWhy backdoor + refutations.

    Pool seeds within each arm INDEPENDENTLY at each (env, burst);
    Δ at that stratum is the difference of arm-means. No per-pair
    structure. Strata where vanilla mean predictor <
    `min_baseline_predictor` (mech premise inactive) are dropped
    before they enter DoWhy.

    Empty panel (no env survives filter, no paired strata, or all
    strata fail mech filter) yields a NaN-everywhere result."""
    cells_list = list(cells)
    rows, dag, treatment_col, outcome_col = _build_stratum_panel(
        cells_list,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        link_predictor=link_predictor,
        link_target=link_target,
        env_filter=env_filter,
        arm_field=arm_field,
        min_baseline_predictor=min_baseline_predictor,
    )
    if not rows:
        return StratumDeltaLinkDowhyResult(
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
    return StratumDeltaLinkDowhyResult(
        backdoor=backdoor,
        placebo=placebo,
        random_common_cause=rcc,
        n_strata=len(rows),
        treatment_col=treatment_col,
        outcome_col=outcome_col,
    )


__all__ = [
    'StratumDeltaLinkDowhyResult',
    'stratum_delta_link_dowhy',
]
