"""Stratum-panel JCI Spearman for mediation falsification.

Builds per-(env, config) stratum-level rows (one observation per
config, scalar predictor + scalar target) and runs three causal-
discovery primitives:

1. **Marginal Spearman ρ(predictor, Δ_target)** — cross-stratum
   rank correlation, pooled across envs. Vulnerable to scale
   confound (env-size dominates).
2. **JCI-stratified Spearman ρ(predictor, Δ_target | env)** —
   per-env Spearman, Fisher-z-pooled. The within-env link
   pooled honestly.
3. **JCI + partial Spearman ρ(predictor, Δ_target | v_target,
   env)** — same as (2) but with vanilla's outcome (config-
   quality proxy) partial-correlated out. Tests whether the
   link survives controlling for the third-variable
   "config-quality" confound.

Why this primitive: the existing `stratum_delta_link_dowhy`
correlates Δ_predictor with Δ_target — vulnerable to algebraic
tautology when predictor ⊃ target (e.g., jens = Q − MC and
target = MC). This primitive uses **one-arm scalar predictor**
(mean over baseline-arm cells) and **cross-arm Δ target**,
algebraically independent when the predictor doesn't share a
constituent with the target.

The result is consumable by `partial_spearman_null_verdict` and
`partial_spearman_signed_verdict` from the existing verdict
helpers — bridges author null-form or signed claims directly."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

import polars as pl

from corroborate._internals.polars import as_rows
from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import (
    stratified_partial_spearman_rho,
    stratified_spearman_rho,
)


@dataclass(frozen=True, slots=True)
class StratumPanelJciResult:
    """Three Spearman rhos on the per-stratum panel.

    `predictor`, `target`, `stratify_by` echo the inputs for
    introspection. `n_strata` is the number of (env, config)
    rows that survived `min_seeds_per_arm` per-arm and
    `min_baseline_predictor` filtering.

    - `rho_marginal` / `p_marginal`: Spearman across all strata,
      no env adjustment.
    - `rho_stratified` / `p_stratified`: JCI Spearman, per-env
      Fisher-z-pooled.
    - `rho_partial_stratified` / `p_partial_stratified`: same as
      stratified but with `partial_z` (vanilla target mean)
      partial-correlated out per env.

    NaN rho/p when no stratum reaches `min_seeds_per_arm`."""
    predictor: str
    target: str
    stratify_by: str
    partial_z: str
    n_strata: int
    rho_marginal: float
    p_marginal: float
    rho_stratified: float
    p_stratified: float
    rho_partial_stratified: float
    p_partial_stratified: float


def _build_panel_arrays(
    cells: Sequence[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    predictor_col: str,
    target_col: str,
    stratify_by: tuple[str, ...],
    arm_field: str,
    min_seeds_per_arm: int,
    min_baseline_predictor: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[object],
]:
    """Build per-stratum scalars: (baseline_predictor_mean,
    delta_target, vanilla_target_mean) per row, plus the env-
    bucket key (first element of `stratify_by`, or pulled from
    `env_name` field on the cell)."""
    by_key_arm: dict[
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
        by_key_arm.setdefault(key, {}).setdefault(arm, []).append(cell)

    predictor_vals: list[float] = []
    delta_target_vals: list[float] = []
    vanilla_target_vals: list[float] = []
    envs: list[object] = []
    for key, arms_dict in by_key_arm.items():
        t_cells = arms_dict.get(treatment_arm, [])
        b_cells = arms_dict.get(baseline_arm, [])
        if len(t_cells) < min_seeds_per_arm:
            continue
        if len(b_cells) < min_seeds_per_arm:
            continue
        def _finite_floats(
            cells_in: list[Mapping[str, object]], col: str,
        ) -> list[float]:
            out: list[float] = []
            for c in cells_in:
                v = c.get(col)
                if isinstance(v, (int, float)):
                    f = float(v)
                    if not math.isnan(f):
                        out.append(f)
            return out
        b_pred = _finite_floats(b_cells, predictor_col)
        b_target = _finite_floats(b_cells, target_col)
        t_target = _finite_floats(t_cells, target_col)
        if (
            len(b_pred) < min_seeds_per_arm
            or len(b_target) < min_seeds_per_arm
            or len(t_target) < min_seeds_per_arm
        ):
            continue
        v_pred = sum(b_pred) / len(b_pred)
        if v_pred <= min_baseline_predictor:
            continue
        v_target = sum(b_target) / len(b_target)
        d_target = sum(t_target) / len(t_target) - v_target
        if 'env_name' in stratify_by:
            env_idx = stratify_by.index('env_name')
            env_v = key[env_idx]
        else:
            env_v = b_cells[0].get('env_name')
        if not isinstance(env_v, str):
            continue
        predictor_vals.append(v_pred)
        delta_target_vals.append(d_target)
        vanilla_target_vals.append(v_target)
        envs.append(env_v)

    return (
        np.asarray(predictor_vals, dtype=np.float64),
        np.asarray(delta_target_vals, dtype=np.float64),
        np.asarray(vanilla_target_vals, dtype=np.float64),
        envs,
    )


def _nan_result(
    predictor: str, target: str, stratify_by: str, partial_z: str,
) -> StratumPanelJciResult:
    return StratumPanelJciResult(
        predictor=predictor, target=target,
        stratify_by=stratify_by, partial_z=partial_z,
        n_strata=0,
        rho_marginal=float('nan'), p_marginal=float('nan'),
        rho_stratified=float('nan'), p_stratified=float('nan'),
        rho_partial_stratified=float('nan'),
        p_partial_stratified=float('nan'),
    )


@analysis
def stratum_panel_jci_spearman(
    cells: pl.DataFrame | Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    predictor_col: str,
    target_col: str = 'eval_best_burst_raw_mean',
    stratify_by: tuple[str, ...] = (
        'env_name', 'gamma', 'sync_period', 'total_steps',
        'optimizer.inner.lr', 'optimizer.inner.weight_decay',
        'replay.capacity',
    ),
    arm_field: str = 'arm_key',
    env_field: str = 'env_name',
    min_seeds_per_arm: int = 5,
    min_baseline_predictor: float = 0.0,
) -> StratumPanelJciResult:
    """Per-stratum-panel JCI Spearman tests of `predictor →
    Δ_target` mediation.

    Builds (predictor=baseline-arm mean, Δ_target=cross-arm Δ on
    `target_col`, partial_z=baseline-arm target mean) per (env,
    config) and runs three Spearman variants. The third
    (JCI + partial) is the strongest empirical falsification
    surface — it controls for both env (scale confound) and
    baseline arm's outcome (config-quality confound)."""
    cells = as_rows(cells)
    cells_list = list(cells)
    x_arr, dy_arr, vy_arr, envs = _build_panel_arrays(
        cells_list,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        predictor_col=predictor_col,
        target_col=target_col,
        stratify_by=stratify_by,
        arm_field=arm_field,
        min_seeds_per_arm=min_seeds_per_arm,
        min_baseline_predictor=min_baseline_predictor,
    )
    del env_field
    n = int(x_arr.size)
    if n < 4:
        return _nan_result(
            predictor_col, target_col, 'env_name', target_col,
        )
    # Marginal
    if float(np.std(x_arr)) == 0.0 or float(np.std(dy_arr)) == 0.0:
        r_m, p_m = float('nan'), float('nan')
    else:
        r_raw, p_raw = spearmanr(x_arr, dy_arr)
        r_m = float(r_raw)
        p_m = float(p_raw)
    # JCI stratified
    rho_s, p_s = stratified_spearman_rho(x_arr, dy_arr, envs)
    # JCI partial (control for vanilla target)
    rho_p, p_p = stratified_partial_spearman_rho(
        x_arr, dy_arr, vy_arr, envs,
    )
    return StratumPanelJciResult(
        predictor=predictor_col, target=target_col,
        stratify_by='env_name', partial_z=target_col,
        n_strata=n,
        rho_marginal=r_m, p_marginal=p_m,
        rho_stratified=rho_s, p_stratified=p_s,
        rho_partial_stratified=rho_p, p_partial_stratified=p_p,
    )


__all__ = [
    'StratumPanelJciResult',
    'stratum_panel_jci_spearman',
]
