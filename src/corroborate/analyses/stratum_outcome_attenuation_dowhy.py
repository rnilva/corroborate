"""Stratum-level outcome-attenuation backdoor + refutations.

Tests whether a binary attenuator (env-mean of some column over
a threshold) reduces per-(env, burst) **stratum-Δ outcome** via
DoWhy backdoor with env one-hot adjustment.

The sibling `link_attenuation_dowhy` uses within-(env, burst)
Pearson r between seed-paired Δs as the outcome — which has
seeds-as-units inside the per-stratum statistic. This analysis
replaces that with an independent-samples stratum-Δ: at each
(env, burst), pool DDQN seeds and pool vanilla seeds
INDEPENDENTLY, then Δ_outcome = mean_T(outcome) − mean_B(outcome).
No seed pairing.

The trade: the claim moves from "Q-divergence weakens the
LINK between Δ_jens and Δ_outcome" to "Q-divergence attenuates
Δ_outcome itself" — simpler shape that fits stratum-Δ form
cleanly. The mediation framing is dropped; the attenuation
claim remains.

Mech conditioning: strata where vanilla mean predictor is below
`min_vanilla_predictor` (G1 premise inactive) are skipped.

DAG: env-dummies → binary_treatment, env-dummies → Δ_outcome,
binary_treatment → Δ_outcome. Env one-hot drops the first env
to avoid collinearity. Identified iff at least 2 envs are above
threshold AND at least 2 are below (so the binary contrast has
both groups present)."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from corroborate.analyses.dowhy import (
    BackdoorResult,
    RefutationResult,
    backdoor_ate,
    placebo_refutation,
    random_common_cause_refutation,
)
from corroborate.analyses.paired_g_per_burst import (
    evaluate_per_burst_source,
)
from corroborate.bridge.analysis import analysis
from corroborate.measurables import Measurable


@dataclass(frozen=True, slots=True)
class StratumOutcomeAttenuationDowhyResult:
    """Backdoor + placebo + RCC refutations of a binary attenuator
    on per-(env, burst) **stratum-Δ outcome**. Adjusts for env
    one-hot. `n_above` / `n_below` characterise the env-split."""
    backdoor: BackdoorResult
    placebo: RefutationResult
    random_common_cause: RefutationResult
    n_strata: int
    n_above: int
    n_below: int
    binary_threshold: float
    attenuator: str
    treatment_col: str
    outcome_col: str


def _env_means(
    cells: Sequence[Mapping[str, object]],
    *,
    confounder: str,
    attenuator: str,
) -> dict[str, float]:
    """Per-env mean of `attenuator`. Cells lacking the confounder,
    a numeric attenuator, or a non-NaN value are excluded."""
    bins: dict[str, list[float]] = {}
    for cell in cells:
        env = cell.get(confounder)
        if not isinstance(env, str):
            continue
        v = cell.get(attenuator)
        if not isinstance(v, (int, float)):
            continue
        f = float(v)
        if math.isnan(f):
            continue
        bins.setdefault(env, []).append(f)
    return {e: sum(vs) / len(vs) for e, vs in bins.items() if vs}


def _build_stratum_panel(
    cells: Sequence[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    attenuator: str,
    binary_threshold: float,
    confounder: str,
    arm_field: str,
    min_vanilla_predictor: float,
) -> tuple[
    list[Mapping[str, object]],
    list[tuple[str, str]],
    str,
    str,
    int,
    int,
]:
    """Build per-(env, burst) stratum-Δ panel rows + DoWhy DAG.

    Returns (panel_rows, dag_edges, treatment_col, outcome_col,
    n_above, n_below). Empty when no env survives the split or all
    strata fail mech filter."""
    treatment_col = f'{attenuator}__above_threshold'
    outcome_col = 'd_outcome'

    by_env_arm: dict[
        tuple[str, str],
        list[tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]],
    ] = {}
    for cell in cells:
        env = cell.get(confounder)
        arm = cell.get(arm_field)
        if not isinstance(env, str) or not isinstance(arm, str):
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

    env_attenuator = _env_means(
        cells, confounder=confounder, attenuator=attenuator,
    )
    panel_envs = sorted({
        e for (e, _) in by_env_arm if e in env_attenuator
    })
    # Envs need BOTH arms with data.
    panel_envs = [
        e for e in panel_envs
        if (e, treatment_arm) in by_env_arm
        and (e, baseline_arm) in by_env_arm
    ]
    if not panel_envs:
        return [], [], treatment_col, outcome_col, 0, 0

    # Build stratum rows.
    stratum_rows: list[dict[str, object]] = []
    n_above = 0
    n_below = 0
    for env in panel_envs:
        env_mean = env_attenuator[env]
        treated_env = 1.0 if env_mean > binary_threshold else 0.0
        t_cells = by_env_arm[(env, treatment_arm)]
        b_cells = by_env_arm[(env, baseline_arm)]
        common_bursts = min(
            min(c[0].shape[0] for c in t_cells),
            min(c[0].shape[0] for c in b_cells),
        )
        if common_bursts == 0:
            continue
        env_strata_added = 0
        for b in range(common_bursts):
            t_target_b = np.array(
                [float(c[0][b]) for c in t_cells], dtype=np.float64,
            )
            b_target_b = np.array(
                [float(c[0][b]) for c in b_cells], dtype=np.float64,
            )
            b_pred_b = np.array(
                [float(c[1][b]) for c in b_cells], dtype=np.float64,
            )
            mean_t_target = float(np.nanmean(t_target_b))
            mean_b_target = float(np.nanmean(b_target_b))
            mean_b_pred = float(np.nanmean(b_pred_b))
            if math.isnan(mean_t_target) or math.isnan(mean_b_target):
                continue
            if math.isnan(mean_b_pred):
                continue
            if mean_b_pred <= min_vanilla_predictor:
                continue
            d_outcome = mean_t_target - mean_b_target
            stratum_rows.append({
                confounder: env,
                'burst_index': b,
                outcome_col: d_outcome,
                treatment_col: treated_env,
            })
            env_strata_added += 1
        if env_strata_added > 0:
            if treated_env >= 0.5:
                n_above += 1
            else:
                n_below += 1

    if not stratum_rows:
        return [], [], treatment_col, outcome_col, 0, 0

    # Env one-hot encode (drop-first to avoid collinearity).
    envs_in_panel: list[str] = sorted({
        e for r in stratum_rows
        if isinstance((e := r[confounder]), str)
    })
    env_cols = [f'__env__{e}' for e in envs_in_panel[1:]]

    rows: list[Mapping[str, object]] = []
    for r in stratum_rows:
        row = dict(r)
        env = r[confounder]
        for c in env_cols:
            row[c] = 1.0 if c == f'__env__{env}' else 0.0
        rows.append(row)

    dag: list[tuple[str, str]] = []
    for c in env_cols:
        dag.append((c, treatment_col))
        dag.append((c, outcome_col))
    dag.append((treatment_col, outcome_col))
    return rows, dag, treatment_col, outcome_col, n_above, n_below


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
def stratum_outcome_attenuation_dowhy(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    attenuator: str,
    binary_threshold: float,
    confounder: str = 'env_name',
    arm_field: str = 'arm_key',
    method_name: str = 'backdoor.linear_regression',
    min_vanilla_predictor: float = 0.05,
) -> StratumOutcomeAttenuationDowhyResult:
    """Test whether `attenuator > binary_threshold` reduces
    per-(env, burst) **stratum-Δ outcome** via DoWhy backdoor +
    refutations.

    Independent-samples Δ at each stratum: pool DDQN seeds + pool
    vanilla seeds, take mean difference on `link_target`. No seed
    pairing. Mech conditioning via `min_vanilla_predictor` —
    strata where vanilla mean of `link_predictor` is below the
    floor are dropped before DoWhy.

    Identified iff ≥ 2 envs above-threshold AND ≥ 2 below. Empty
    panel yields NaN-everywhere result; bridges check `n_strata`
    and `backdoor.identified` for power before reading ATEs."""
    cells_list = list(cells)
    rows, dag, t_col, o_col, n_above, n_below = _build_stratum_panel(
        cells_list,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        link_target=link_target,
        link_predictor=link_predictor,
        attenuator=attenuator,
        binary_threshold=binary_threshold,
        confounder=confounder,
        arm_field=arm_field,
        min_vanilla_predictor=min_vanilla_predictor,
    )
    if not rows:
        return StratumOutcomeAttenuationDowhyResult(
            backdoor=_nan_backdoor(t_col, o_col, method_name),
            placebo=_nan_refutation(
                t_col, o_col, method_name,
                refuter_name='placebo_treatment_refuter',
            ),
            random_common_cause=_nan_refutation(
                t_col, o_col, method_name,
                refuter_name='random_common_cause',
            ),
            n_strata=0, n_above=0, n_below=0,
            binary_threshold=binary_threshold,
            attenuator=attenuator,
            treatment_col=t_col,
            outcome_col=o_col,
        )

    backdoor = backdoor_ate.fn(
        rows, treatment=t_col, outcome=o_col,
        dag=dag, method_name=method_name,
    )
    placebo = placebo_refutation.fn(
        rows, treatment=t_col, outcome=o_col,
        dag=dag, method_name=method_name,
    )
    rcc = random_common_cause_refutation.fn(
        rows, treatment=t_col, outcome=o_col,
        dag=dag, method_name=method_name,
    )
    return StratumOutcomeAttenuationDowhyResult(
        backdoor=backdoor,
        placebo=placebo,
        random_common_cause=rcc,
        n_strata=len(rows),
        n_above=n_above,
        n_below=n_below,
        binary_threshold=binary_threshold,
        attenuator=attenuator,
        treatment_col=t_col,
        outcome_col=o_col,
    )


__all__ = [
    'StratumOutcomeAttenuationDowhyResult',
    'stratum_outcome_attenuation_dowhy',
]
