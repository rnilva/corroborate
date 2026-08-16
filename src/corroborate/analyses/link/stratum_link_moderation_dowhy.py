"""Stratum-level link moderation: does a binary attenuator weaken
the `Δ_predictor → Δ_target` slope?

# UNCONSUMED — REVISIT 2026-11-17. No bridge currently consumes
# this fixture. Kept because (a) the method is RL-substrate-safe
# (independent-samples per stratum, no seed pairing) and (b) it
# answers a structurally distinct question (moderation, not
# mediation) that docs/HYPOTHESIS_AS_GRAPH.md §3b's scope-cluster
# pattern naturally invokes. If no consumer adopts it by the
# revisit date, demote to DELETE.

The slope-moderation form is the right shape when the science
is a mediation claim ("extreme Q-divergence attenuates the
bias→outcome link") — Δ_target alone can't see the link's
existence.

Independent-samples per stratum (no seed pairing): at each
(env, burst) stratum, pool DDQN seeds, pool vanilla seeds,
take mean difference for both predictor and target.

Identification under colinearity. Binary attenuator is computed
as `1[env_mean(attenuator) > threshold]` — deterministic in env.
Including binary_attenuator's main effect alongside env one-hot
would be rank-deficient (env determines binary_attenuator).
Resolution: include the interaction term `Δ_predictor ×
binary_attenuator` WITHOUT the binary_attenuator main effect.
The interaction varies WITHIN env (because Δ_predictor varies
across bursts within env), so β_int is identified — it captures
the difference in link slope between above-threshold and
below-threshold envs.

Model fit by DoWhy backdoor.linear_regression:

    Δ_target ~ β_0 + β_link · Δ_pred
                   + β_int · (Δ_pred × 1[env above-thresh])
                   + Σ_e γ_e · env_dummy_e + ε

`backdoor` carries β_int (the moderation ATE). β_int > 0 means
above-threshold envs have a less-negative link slope → link
attenuated. Placebo + RCC refutations are run on the
interaction-term treatment under env-one-hot adjustment.

Mech conditioning: strata where vanilla mean predictor is below
`min_baseline_predictor` (G1 premise inactive) are dropped before
the regression — the link is moot if the mech isn't firing.

Identified iff ≥ 1 env above-threshold AND ≥ 1 below; bridges
should check `n_envs_above >= 1 AND n_envs_below >= 1` and
`n_strata > p` (≥ 5 + env-dummies) for power."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

import polars as pl

from corroborate._internals.polars import as_rows
from corroborate.analyses.dowhy import (
    BackdoorResult,
    RefutationResult,
    backdoor_ate,
    placebo_refutation,
    random_common_cause_refutation,
)
from corroborate.analyses._cell_value import evaluate_per_burst_source
from corroborate.bridge.analysis import analysis
from corroborate.measurables import Measurable


@dataclass(frozen=True, slots=True)
class StratumLinkModerationDowhyResult:
    """Backdoor + placebo + RCC refutations of the interaction
    `Δ_predictor × binary_attenuator → Δ_target` under env-one-hot
    adjustment. The interaction coefficient is the link
    moderation effect — sign of attenuation depends on the
    Δ_predictor → Δ_target sign in the below-threshold group."""
    backdoor: BackdoorResult
    placebo: RefutationResult
    random_common_cause: RefutationResult
    n_strata: int
    n_envs_above: int
    n_envs_below: int
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


def _build_panel(
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
    attenuator: str,
    binary_threshold: float,
    confounder: str,
    arm_field: str,
    min_baseline_predictor: float,
) -> tuple[
    list[Mapping[str, object]],
    list[tuple[str, str]],
    str, str, int, int,
]:
    """Build per-(env, burst) stratum-Δ panel rows + DoWhy DAG."""
    interaction_col = 'd_pred_x_above'
    pred_col = 'd_pred'
    outcome_col = 'd_target'

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

    env_atten = _env_means(
        cells, confounder=confounder, attenuator=attenuator,
    )
    panel_envs = sorted({
        e for (e, _) in by_env_arm if e in env_atten
    })
    panel_envs = [
        e for e in panel_envs
        if (e, treatment_arm) in by_env_arm
        and (e, baseline_arm) in by_env_arm
    ]
    if not panel_envs:
        return [], [], interaction_col, outcome_col, 0, 0

    raw_rows: list[dict[str, object]] = []
    for env in panel_envs:
        env_mean = env_atten[env]
        above = 1.0 if env_mean > binary_threshold else 0.0
        t_cells = by_env_arm[(env, treatment_arm)]
        b_cells = by_env_arm[(env, baseline_arm)]
        common_bursts = min(
            min(c[0].shape[0] for c in t_cells),
            min(c[0].shape[0] for c in b_cells),
        )
        for b in range(common_bursts):
            t_target = np.array(
                [float(c[0][b]) for c in t_cells], dtype=np.float64,
            )
            t_pred = np.array(
                [float(c[1][b]) for c in t_cells], dtype=np.float64,
            )
            b_target = np.array(
                [float(c[0][b]) for c in b_cells], dtype=np.float64,
            )
            b_pred = np.array(
                [float(c[1][b]) for c in b_cells], dtype=np.float64,
            )
            mt_t = float(np.nanmean(t_target))
            mt_p = float(np.nanmean(t_pred))
            mb_t = float(np.nanmean(b_target))
            mb_p = float(np.nanmean(b_pred))
            if any(math.isnan(v) for v in (mt_t, mt_p, mb_t, mb_p)):
                continue
            if mb_p <= min_baseline_predictor:
                continue
            d_pred = mt_p - mb_p
            d_target = mt_t - mb_t
            raw_rows.append({
                confounder: env,
                'burst_index': b,
                pred_col: d_pred,
                outcome_col: d_target,
                interaction_col: d_pred * above,
                '__above': above,
            })

    if not raw_rows:
        return [], [], interaction_col, outcome_col, 0, 0

    envs_above = sorted({
        r[confounder] for r in raw_rows
        if isinstance(r[confounder], str) and float(r['__above']) >= 0.5  # type: ignore[arg-type]
    })
    envs_below = sorted({
        r[confounder] for r in raw_rows
        if isinstance(r[confounder], str) and float(r['__above']) < 0.5  # type: ignore[arg-type]
    })
    n_above = len(envs_above)
    n_below = len(envs_below)
    if n_above < 1 or n_below < 1:
        return [], [], interaction_col, outcome_col, n_above, n_below

    envs_in_panel: list[str] = sorted({
        e for r in raw_rows
        if isinstance((e := r[confounder]), str)
    })
    env_cols = [f'__env__{e}' for e in envs_in_panel[1:]]

    rows: list[Mapping[str, object]] = []
    for r in raw_rows:
        row = dict(r)
        env_v = r[confounder]
        for c in env_cols:
            row[c] = 1.0 if c == f'__env__{env_v}' else 0.0
        row.pop('__above', None)
        rows.append(row)

    dag: list[tuple[str, str]] = []
    for c in env_cols:
        dag.append((c, pred_col))
        dag.append((c, outcome_col))
        dag.append((c, interaction_col))
    dag.append((pred_col, outcome_col))
    dag.append((pred_col, interaction_col))
    dag.append((interaction_col, outcome_col))
    return rows, dag, interaction_col, outcome_col, n_above, n_below


def _nan_backdoor(t: str, o: str, m: str) -> BackdoorResult:
    return BackdoorResult(
        ate=float('nan'), identified=False, estimand_str='',
        method_name=m, treatment=t, outcome=o, n_rows=0,
    )


def _nan_refutation(t: str, o: str, m: str, name: str) -> RefutationResult:
    return RefutationResult(
        real_ate=float('nan'), refuted_ate=float('nan'),
        drift=float('nan'), method_name=m, refuter_name=name,
        treatment=t, outcome=o, n_rows=0,
    )


@analysis
def stratum_link_moderation_dowhy(
    cells: pl.DataFrame | Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    attenuator: str,
    binary_threshold: float,
    confounder: str = 'env_name',
    arm_field: str = 'arm_key',
    method_name: str = 'backdoor.linear_regression',
    min_baseline_predictor: float = 0.05,
    random_state: int = 0,
) -> StratumLinkModerationDowhyResult:
    """Test whether `attenuator > binary_threshold` moderates the
    `Δ_predictor → Δ_target` slope via DoWhy backdoor on the
    interaction term.

    Independent-samples Δ at each (env, burst) stratum;
    interaction term `Δ_predictor × 1[env above threshold]`
    treated as the causal target. Backdoor adjustment via env
    one-hot + Δ_predictor main effect. The interaction
    coefficient's ATE IS the link attenuation: sign > 0 means
    above-threshold envs have a less-negative link slope (link
    weakened); sign < 0 means link strengthened.

    Mech conditioning via `min_baseline_predictor` drops strata
    where vanilla mean predictor is below the G1-premise floor.

    Empty panel (no env above OR below threshold, no admitted
    strata) yields NaN-everywhere. Bridges should check
    `n_strata > p_covariates` and `n_envs_above ≥ 1 AND
    n_envs_below ≥ 1` for identification."""
    cells = as_rows(cells)
    cells_list = list(cells)
    rows, dag, t_col, o_col, n_above, n_below = _build_panel(
        cells_list,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        link_predictor=link_predictor,
        link_target=link_target,
        attenuator=attenuator,
        binary_threshold=binary_threshold,
        confounder=confounder,
        arm_field=arm_field,
        min_baseline_predictor=min_baseline_predictor,
    )
    if not rows:
        return StratumLinkModerationDowhyResult(
            backdoor=_nan_backdoor(t_col, o_col, method_name),
            placebo=_nan_refutation(
                t_col, o_col, method_name,
                name='placebo_treatment_refuter',
            ),
            random_common_cause=_nan_refutation(
                t_col, o_col, method_name,
                name='random_common_cause',
            ),
            n_strata=0,
            n_envs_above=n_above,
            n_envs_below=n_below,
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
        random_state=random_state,
    )
    rcc = random_common_cause_refutation.fn(
        rows, treatment=t_col, outcome=o_col,
        dag=dag, method_name=method_name,
        random_state=random_state,
    )
    return StratumLinkModerationDowhyResult(
        backdoor=backdoor,
        placebo=placebo,
        random_common_cause=rcc,
        n_strata=len(rows),
        n_envs_above=n_above,
        n_envs_below=n_below,
        binary_threshold=binary_threshold,
        attenuator=attenuator,
        treatment_col=t_col,
        outcome_col=o_col,
    )


__all__ = [
    'StratumLinkModerationDowhyResult',
    'stratum_link_moderation_dowhy',
]
