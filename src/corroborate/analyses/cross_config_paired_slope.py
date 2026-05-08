"""`cross_config_paired_slope` — cross-config Spearman ρ between
per-config (treatment − baseline) Δs of two measurables.

Each config (a unique combination of `config_keys`) contributes
one (mean_Δ_predictor, mean_Δ_target) pair, computed by averaging
seed-level paired differences. Spearman ρ is computed across
configs.

Distinct from:
- `proportion_mediated` — within-cell linear mediation
- `paired_link_per_burst` — per-burst link r within an arm
- `paired_link_per_env` — per-env link r meta-regressed on a
  per-env moderator
- `meta_regression_paired_g` — per-env g with env-mean covariates

Used when the question is "across env/intervention conditions, do
conditions with bigger Δ_predictor also have bigger Δ_target?".
The unit of analysis is the config; cross-config slope captures
SCOPE-level relationships that don't reduce to within-cell
mediation.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import scipy.stats as stats

from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class CrossConfigPairedSlopeResult:
    """Cross-config Spearman ρ between per-config mean Δ_predictor
    and per-config mean Δ_target. `n_configs` is the number of
    configs that contributed (each had ≥ `min_pairs_per_config`
    seed-pairs)."""
    rho: float
    p_value: float
    n_configs: int
    config_means_predictor: tuple[float, ...]
    config_means_target: tuple[float, ...]


@analysis
def cross_config_paired_slope(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    target: str,
    predictor: str,
    config_keys: tuple[str, ...],
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
    min_pairs_per_config: int = 5,
    min_configs: int = 4,
) -> CrossConfigPairedSlopeResult:
    """Compute Spearman ρ across configs of (mean Δ_target,
    mean Δ_predictor).

    Returns NaN ρ/p when n_configs < `min_configs`. Configs with
    fewer than `min_pairs_per_config` paired seeds are dropped.
    NaN values in either measurable drop the affected pair.
    """
    cells_list = list(cells)
    config_arm: dict[
        tuple[object, ...],
        dict[str, dict[tuple[object, ...], Mapping[str, object]]],
    ] = defaultdict(dict)
    for c in cells_list:
        config_key = tuple(c.get(k) for k in config_keys)
        arm = c.get(arm_field)
        if arm not in (treatment_arm, baseline_arm):
            continue
        if not isinstance(arm, str):
            continue
        pair_key = tuple(c.get(k) for k in pair_by)
        config_arm[config_key].setdefault(arm, {})[pair_key] = c

    pred_means: list[float] = []
    target_means: list[float] = []
    for arms_map in config_arm.values():
        van_pairs = arms_map.get(baseline_arm, {})
        ddq_pairs = arms_map.get(treatment_arm, {})
        common = set(van_pairs.keys()) & set(ddq_pairs.keys())
        if len(common) < min_pairs_per_config:
            continue
        d_pred: list[float] = []
        d_target: list[float] = []
        for pk in common:
            van_c = van_pairs[pk]
            ddq_c = ddq_pairs[pk]
            try:
                vp = float(van_c[predictor])  # type: ignore[arg-type]
                dp = float(ddq_c[predictor])  # type: ignore[arg-type]
                vt = float(van_c[target])  # type: ignore[arg-type]
                dt = float(ddq_c[target])  # type: ignore[arg-type]
            except (KeyError, TypeError, ValueError):
                continue
            if any(math.isnan(v) for v in (vp, dp, vt, dt)):
                continue
            d_pred.append(dp - vp)
            d_target.append(dt - vt)
        if len(d_pred) < min_pairs_per_config:
            continue
        pred_means.append(float(np.mean(d_pred)))
        target_means.append(float(np.mean(d_target)))

    n = len(pred_means)
    if n < min_configs:
        return CrossConfigPairedSlopeResult(
            rho=float('nan'), p_value=float('nan'),
            n_configs=n,
            config_means_predictor=tuple(pred_means),
            config_means_target=tuple(target_means),
        )

    rho_raw, p_raw = stats.spearmanr(pred_means, target_means)
    return CrossConfigPairedSlopeResult(
        rho=float(rho_raw),
        p_value=float(p_raw),
        n_configs=n,
        config_means_predictor=tuple(pred_means),
        config_means_target=tuple(target_means),
    )
