"""`cross_stratum_arm_diff_slope` — independent-samples cross-
stratum Spearman ρ between per-stratum arm-mean Δs.

Each stratum (a unique combination of `stratify_by` values)
contributes one (Δ_predictor, Δ_target) point where:
  Δ_x = mean(treatment x | stratum) − mean(baseline x | stratum)

Both arm means are computed independently across each arm's
cells in the stratum — no pair-key cross-referencing. Spearman
ρ is computed across strata.

**Sibling to `cross_config_paired_slope`**: the paired form
invokes `pair_by=('seed',)` which is operationally vacuous on
this aggregate (`mean(Δ over paired seeds) = mean(t) − mean(b)`
by linearity) but carries "paired" nomenclature that misleads
about same-pair-key semantics. Substrate authors who don't have
a causal pairing claim should reach for this primitive instead.

Distinct from:
- `cross_config_paired_slope` — paired-Δ form (legacy name; same
  result on corpora where every pair-key appears in both arms).
- `proportion_mediated` — within-cell linear mediation.
- `stratified_arm_diff_pooled` — within-stratum Cohen's d,
  DL-pooled across strata. Different question (effect-size
  meta-analysis vs cross-stratum dose-response slope).
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
class CrossStratumArmDiffSlopeResult:
    """Cross-stratum Spearman ρ between per-stratum arm-mean Δs.

    `n_strata` is the number of strata that contributed (both
    arms had ≥ `min_seeds_per_arm` finite-valued cells on both
    predictor and target)."""
    rho: float
    p_value: float
    n_strata: int
    arm_diff_predictor: tuple[float, ...]
    arm_diff_target: tuple[float, ...]


@analysis
def cross_stratum_arm_diff_slope(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    target: str,
    predictor: str,
    stratify_by: tuple[str, ...],
    arm_field: str = 'arm_key',
    min_seeds_per_arm: int = 3,
    min_strata: int = 4,
) -> CrossStratumArmDiffSlopeResult:
    """Compute Spearman ρ across strata of (Δ_predictor, Δ_target)
    where Δ_x = mean_treatment(x) − mean_baseline(x).

    Per-stratum: requires ≥ `min_seeds_per_arm` finite-valued
    cells in BOTH arms on BOTH predictor and target. Returns NaN
    ρ/p when n_strata < `min_strata`. NaN values drop their cell;
    pure independent-samples per arm (no pair-key cross-reference).
    """
    per_stratum_arm: dict[
        tuple[object, ...], dict[str, list[Mapping[str, object]]],
    ] = defaultdict(lambda: defaultdict(list))
    for c in cells:
        arm = c.get(arm_field)
        if arm not in (treatment_arm, baseline_arm):
            continue
        if not isinstance(arm, str):
            continue
        sk = tuple(c.get(k) for k in stratify_by)
        per_stratum_arm[sk][arm].append(c)

    def _finite_values(
        cs: list[Mapping[str, object]], key: str,
    ) -> list[float]:
        out: list[float] = []
        for c in cs:
            v = c.get(key)
            if not isinstance(v, (int, float)):
                continue
            f = float(v)
            if math.isnan(f):
                continue
            out.append(f)
        return out

    diff_predictor: list[float] = []
    diff_target: list[float] = []
    for arms_map in per_stratum_arm.values():
        t_cells = arms_map.get(treatment_arm, [])
        b_cells = arms_map.get(baseline_arm, [])
        t_pred = _finite_values(t_cells, predictor)
        b_pred = _finite_values(b_cells, predictor)
        t_tgt = _finite_values(t_cells, target)
        b_tgt = _finite_values(b_cells, target)
        if (
            len(t_pred) < min_seeds_per_arm
            or len(b_pred) < min_seeds_per_arm
            or len(t_tgt) < min_seeds_per_arm
            or len(b_tgt) < min_seeds_per_arm
        ):
            continue
        diff_predictor.append(float(np.mean(t_pred) - np.mean(b_pred)))
        diff_target.append(float(np.mean(t_tgt) - np.mean(b_tgt)))

    n = len(diff_predictor)
    if n < min_strata:
        return CrossStratumArmDiffSlopeResult(
            rho=float('nan'), p_value=float('nan'),
            n_strata=n,
            arm_diff_predictor=tuple(diff_predictor),
            arm_diff_target=tuple(diff_target),
        )

    rho_raw, p_raw = stats.spearmanr(diff_predictor, diff_target)
    return CrossStratumArmDiffSlopeResult(
        rho=float(rho_raw),
        p_value=float(p_raw),
        n_strata=n,
        arm_diff_predictor=tuple(diff_predictor),
        arm_diff_target=tuple(diff_target),
    )


__all__ = [
    'CrossStratumArmDiffSlopeResult',
    'cross_stratum_arm_diff_slope',
]
