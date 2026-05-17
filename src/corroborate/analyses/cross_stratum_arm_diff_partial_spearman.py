"""`cross_stratum_arm_diff_partial_spearman` — partial Spearman
ρ(Δ_predictor, Δ_target | Δ_confound) across strata.

Each stratum (a unique combination of `stratify_by` values)
contributes one triple `(Δ_predictor, Δ_target, Δ_confound)` where
`Δ_x = mean(treatment x | stratum) − mean(baseline x | stratum)`,
computed independently across each arm's cells in the stratum
(no pair-key cross-reference). Partial Spearman ρ via the
closed-form three-rank-correlation identity is then computed
across strata.

**The disambiguation question this primitive answers.** When
DDQN (or any intervention) moves a candidate-mediator AND a
known mediator in the same direction within a single stratum,
`stratified_partial_spearman(M, Y | Z)` cannot tell whether M
is an independent channel or Z's shadow — within-stratum the
arm contrast collapses both. Sibling
`cross_stratum_arm_diff_slope(M, Y)` measures whether
Δ_M scales with Δ_Y across strata but cannot condition on Δ_Z.
This primitive is the missing combination: across strata, does
Δ_M predict Δ_Y *after* controlling for Δ_Z?

Concrete case: at Asterix γ=0.999 single-env, DDQN reduces
`q_inter_state_grad_overlap` AND `jensen_gap` ~50% in lockstep;
`stratified_partial_spearman(smoothness, outcome | jens)`
returned ρ=-0.088 — null, but unable to distinguish "smoothness
is jens-shadow" from "underpowered". A 6-stratum panel
(Asterix/Breakout/Freeway × γ=0.95/0.999) with this primitive
discriminates: if Δ_smoothness ≈ k · Δ_jens across strata then
partial-r → 0; if smoothness has independent variance then
partial-r ≠ 0.

Distinct from:
- `cross_stratum_arm_diff_slope` — marginal cross-stratum ρ
  (no conditioning).
- `cross_stratum_property_slope` — single-arm property vs
  outcome (not arm-diff).
- `stratified_partial_spearman` — per-cell partial-r within
  stratum, Fisher-z pooled. Doesn't answer the cross-stratum
  dose-response question.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import scipy.stats as stats

from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import partial_spearman_rho


@dataclass(frozen=True, slots=True)
class CrossStratumArmDiffPartialSpearmanResult:
    """Cross-stratum partial Spearman ρ(Δ_predictor, Δ_target |
    Δ_confound).

    `n_strata` is the number of strata that contributed (both
    arms had ≥ `min_seeds_per_arm` finite-valued cells on
    predictor, target, AND confound).

    `arm_diff_*` carry the per-stratum Δs for inspection. The
    marginal Spearman ρ(Δ_predictor, Δ_target) is also reported
    as `rho_marginal` so callers can compare partial-vs-marginal
    without an extra primitive call.
    """
    rho: float
    p_value: float
    rho_marginal: float
    n_strata: int
    arm_diff_predictor: tuple[float, ...]
    arm_diff_target: tuple[float, ...]
    arm_diff_confound: tuple[float, ...]


@analysis
def cross_stratum_arm_diff_partial_spearman(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    target: str,
    predictor: str,
    confound: str,
    stratify_by: tuple[str, ...],
    arm_field: str = 'arm_key',
    min_seeds_per_arm: int = 3,
    min_strata: int = 5,
) -> CrossStratumArmDiffPartialSpearmanResult:
    """Compute partial Spearman ρ(Δ_predictor, Δ_target |
    Δ_confound) across strata.

    Per-stratum: requires ≥ `min_seeds_per_arm` finite-valued
    cells in BOTH arms on ALL of predictor, target, confound.
    Returns NaN ρ/p when n_strata < `min_strata` (default 5; the
    closed-form partial-r requires n ≥ 5 — see
    `partial_spearman_rho`).

    The closed-form first-order partial Spearman is computed via
    `graph.discovery.partial_spearman_rho` — rank-transform each
    Δ vector, pairwise Spearman, partial-correlation combination.
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

    def _finite(
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
    diff_confound: list[float] = []
    for arms_map in per_stratum_arm.values():
        t_cells = arms_map.get(treatment_arm, [])
        b_cells = arms_map.get(baseline_arm, [])
        t_p = _finite(t_cells, predictor)
        b_p = _finite(b_cells, predictor)
        t_t = _finite(t_cells, target)
        b_t = _finite(b_cells, target)
        t_c = _finite(t_cells, confound)
        b_c = _finite(b_cells, confound)
        if any(
            len(arr) < min_seeds_per_arm
            for arr in (t_p, b_p, t_t, b_t, t_c, b_c)
        ):
            continue
        diff_predictor.append(float(np.mean(t_p) - np.mean(b_p)))
        diff_target.append(float(np.mean(t_t) - np.mean(b_t)))
        diff_confound.append(float(np.mean(t_c) - np.mean(b_c)))

    n = len(diff_predictor)
    if n < min_strata:
        return CrossStratumArmDiffPartialSpearmanResult(
            rho=float('nan'), p_value=float('nan'),
            rho_marginal=float('nan'),
            n_strata=n,
            arm_diff_predictor=tuple(diff_predictor),
            arm_diff_target=tuple(diff_target),
            arm_diff_confound=tuple(diff_confound),
        )

    x_arr = np.asarray(diff_predictor, dtype=np.float64)
    y_arr = np.asarray(diff_target, dtype=np.float64)
    z_arr = np.asarray(diff_confound, dtype=np.float64)
    rho, p = partial_spearman_rho(x_arr, y_arr, z_arr)
    # Marginal Spearman for the caller's marginal-vs-partial diff.
    rho_m_raw, _ = stats.spearmanr(diff_predictor, diff_target)
    return CrossStratumArmDiffPartialSpearmanResult(
        rho=float(rho),
        p_value=float(p),
        rho_marginal=float(rho_m_raw),
        n_strata=n,
        arm_diff_predictor=tuple(diff_predictor),
        arm_diff_target=tuple(diff_target),
        arm_diff_confound=tuple(diff_confound),
    )


__all__ = [
    'CrossStratumArmDiffPartialSpearmanResult',
    'cross_stratum_arm_diff_partial_spearman',
]
