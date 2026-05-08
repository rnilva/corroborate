"""`proportion_mediated` — DEPRECATED linear-mediation summary.

**DEPRECATED — DO NOT GATE LOAD-BEARING BRIDGES ON THIS RESULT.**

For new code, prefer:
- `corroborate.graph.discovery.partial_spearman_rho(X, Y, Z)` —
  non-parametric "does conditioning on Z kill the X→Y
  association?" test. Has an honest SE (Fisher-z, df = n − 4),
  no functional-form assumption, and rank-based so robust to
  saturation / heavy tails.
- `stratified_partial_spearman_rho` (same module) for the JCI
  per-stratum Fisher-z-pooled form when env is a confounder.
- DoWhy's `mediation` estimator (not yet wrapped) when an
  explicit causal-mediation NDE/NIE is needed.

This module is kept for back-compat reads of past `RunRow.measurements`
columns that stored a `proportion_mediated` scalar. Calling the
function emits a `DeprecationWarning` and the result should be
treated as a **descriptive heuristic only**, not a verdict input.

---

**Why deprecated.** Three structural problems:

1. **Ratio explodes near the denominator zero.** `proportion =
   indirect / mean(ΔY)`. When the total effect's magnitude is
   small but nonzero, a noisy slope estimate produces a wildly
   variable proportion. The existing `<1e-12` guard catches
   exact zero; everything between zero and `SE(mean(ΔY))` is
   essentially noise. The result has NO SE, so bridges cannot
   tell signal from noise on the ratio.

2. **Can land outside `[0, 1]` without action.** Direct/indirect
   with opposite signs (suppression) or sampling noise produces
   `proportion ∈ (-∞, ∞)`. The `in_unit_interval` flag REPORTS
   the failure but doesn't change the verdict — bridges can
   still gate on a number that's structurally meaningless.

3. **First-difference identification ≠ population M→Y slope.**
   The estimator `β_YM = Cov(ΔY, ΔM) / Var(ΔM)` recovers the
   structural M→Y slope only if M→Y is linear AND homogeneous
   across pairs. The Q-explosion regimes the framework regularly
   visits (Asterix sync=100, Breakout late bursts) are exactly
   where M→outcome is non-monotone — the very scenario where
   per-burst link analyses replaced scalar slopes because the
   scalar form silently combines causally-opposite phases.

The `partial_spearman_rho` family doesn't share these problems
(non-parametric, has SE, doesn't conflate phases under sign
flip — though signing requires per-burst stratification). It is
the framework's recommended primitive for "is M a mediator?"
verdicts.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from corroborate.analyses.paired_g import resolve_value as _paired_g_resolve_value
from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class ProportionMediatedResult:
    """Linear-mediation decomposition.

    `proportion`: the (indirect / total) share — the fraction of
    the treatment's effect on `target` that flows through
    `mediator`. NaN when total effect is too small to estimate
    against (|mean(Δ_Y)| < 1e-12), or when n_pairs < 3.

    `total`, `direct`, `indirect`: the underlying point estimates
    in the same units as `target`. `total = direct + indirect`
    by construction.

    `slope_y_on_m`: the β_YM coefficient (slope of Δ_Y on Δ_M).

    `in_unit_interval`: True iff `proportion ∈ [0, 1]`. False
    indicates linear-mediation assumptions have failed —
    interpret with caution, escalate to counterfactual
    decomposition.

    `n_pairs`: number of paired observations contributing.
    `target`, `mediator`, `treatment_arm`, `baseline_arm`,
    `pair_by`: provenance of the call."""
    proportion: float
    total: float
    direct: float
    indirect: float
    slope_y_on_m: float
    in_unit_interval: bool
    n_pairs: int
    target: str
    mediator: str
    treatment_arm: str
    baseline_arm: str
    pair_by: tuple[str, ...]


def _key_tuple(
    record: Mapping[str, object], pair_by: tuple[str, ...],
) -> tuple[object, ...]:
    return tuple(record[k] for k in pair_by)


def _resolve_value(record: Mapping[str, object], source: str) -> float:
    """Wrap `paired_g.resolve_value` with NaN-on-error semantics.

    `paired_g.resolve_value` raises `KeyError` when `source` is
    absent and no measurable is registered, and `TypeError` when
    a measurable returns a non-scalar. `proportion_mediated` is
    deprecated and prefers to NaN-skip such pairs rather than
    crash; catch the canonical errors here. This keeps the
    resolution rule (record-first, registry-fallback) DRY with
    the canonical implementation."""
    try:
        return _paired_g_resolve_value(record, source)
    except (KeyError, TypeError):
        return float('nan')


@analysis
def proportion_mediated(
    cells: Iterable[Mapping[str, object]],
    *,
    target: str,
    mediator: str,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
    upstream_source: str | None = None,
    upstream_max_delta: float | None = None,
    upstream_min_delta: float | None = None,
) -> ProportionMediatedResult:
    """Linear proportion of the treatment effect on `target` that
    is mediated by `mediator`.

    Pairs cells on `pair_by` (typically `('seed',)`), computes
    per-pair `Δ_target` and `Δ_mediator`, fits the OLS slope of
    Δ_Y on Δ_M, then reports

        proportion = β_YM · mean(Δ_M) / mean(Δ_Y).

    `target` and `mediator` resolve through the measurable
    registry first, falling back to field-path reads on the cell
    record (same convention as `paired_g.source`).

    **DEPRECATED.** Emits `DeprecationWarning` on call. Prefer
    `corroborate.graph.discovery.partial_spearman_rho(target,
    mediator, conditioning=treatment_indicator)` for
    "does conditioning on M kill T→Y?" verdicts; see this
    module's docstring for the rationale and alternatives. The
    function is kept for back-compat reads of past corpora that
    persisted a `proportion_mediated` scalar.

    `upstream_source` enables **conditioning on the upstream step
    of a mediation chain**. When the chain is
    `do(treatment) → upstream → mediator → target`, pairs where
    the treatment-induced Δ_upstream did not move in the predicted
    direction are not part of the active link — pooling them
    dilutes the mediator's measured contribution. Pass the
    upstream measurable's name (e.g. `'jensen_gap'` for
    DDQN's bias-correction step) and one of:

      `upstream_max_delta`: keep pairs where Δ_upstream < this
        (e.g. `0.0` to keep only pairs where DDQN reduced bias);
      `upstream_min_delta`: keep pairs where Δ_upstream > this
        (e.g. `0.0` to keep pairs where the upstream step
        increased — Q-amplification regimes).

    The two are mutually exclusive — pass at most one. When
    both are None (default), no upstream conditioning is applied
    and the analysis reduces to the standard 2-variable form.

    The result's `in_unit_interval` flag is the diagnostic for
    linear-mediation assumption failure — see this module's
    docstring §"Why deprecated" item 2."""
    warnings.warn(
        'proportion_mediated is deprecated and structurally '
        'fragile (ratio-of-noisy-means; can land outside [0, 1]; '
        'first-difference identification ≠ population slope). '
        'Use `corroborate.graph.discovery.partial_spearman_rho` '
        'for mediation hypothesis tests; see '
        '`corroborate.analyses.proportion_mediated`\'s module '
        'docstring for the full alternatives list.',
        DeprecationWarning,
        stacklevel=2,
    )
    if upstream_max_delta is not None and upstream_min_delta is not None:
        raise ValueError(
            'proportion_mediated: pass at most one of '
            '`upstream_max_delta` / `upstream_min_delta`',
        )
    if (upstream_max_delta is not None or upstream_min_delta is not None) \
            and upstream_source is None:
        raise ValueError(
            'proportion_mediated: `upstream_max_delta` / '
            '`upstream_min_delta` require `upstream_source`',
        )

    treatment_y: dict[tuple[object, ...], float] = {}
    treatment_m: dict[tuple[object, ...], float] = {}
    treatment_u: dict[tuple[object, ...], float] = {}
    baseline_y: dict[tuple[object, ...], float] = {}
    baseline_m: dict[tuple[object, ...], float] = {}
    baseline_u: dict[tuple[object, ...], float] = {}
    for cell in cells:
        arm = cell.get(arm_field)
        if arm not in (treatment_arm, baseline_arm):
            continue
        key = _key_tuple(cell, pair_by)
        y = _resolve_value(cell, target)
        m = _resolve_value(cell, mediator)
        if math.isnan(y) or math.isnan(m):
            continue
        u = (
            _resolve_value(cell, upstream_source)
            if upstream_source is not None else 0.0
        )
        if upstream_source is not None and math.isnan(u):
            continue
        if arm == treatment_arm:
            treatment_y[key] = y
            treatment_m[key] = m
            treatment_u[key] = u
        else:
            baseline_y[key] = y
            baseline_m[key] = m
            baseline_u[key] = u

    paired_keys_all = sorted(set(treatment_y) & set(baseline_y))

    # Apply upstream conditioning per pair.
    paired_keys: list[tuple[object, ...]]
    if upstream_source is not None and (
        upstream_max_delta is not None or upstream_min_delta is not None
    ):
        paired_keys = []
        for k in paired_keys_all:
            d_u = treatment_u[k] - baseline_u[k]
            if upstream_max_delta is not None and d_u >= upstream_max_delta:
                continue
            if upstream_min_delta is not None and d_u <= upstream_min_delta:
                continue
            paired_keys.append(k)
    else:
        paired_keys = paired_keys_all

    n_pairs = len(paired_keys)
    if n_pairs < 3:
        return _nan_result(
            target=target, mediator=mediator,
            treatment_arm=treatment_arm, baseline_arm=baseline_arm,
            pair_by=pair_by, n_pairs=n_pairs,
        )

    delta_y = [treatment_y[k] - baseline_y[k] for k in paired_keys]
    delta_m = [treatment_m[k] - baseline_m[k] for k in paired_keys]

    # OLS slope of Δ_Y on Δ_M with intercept.
    n = float(n_pairs)
    mean_y = sum(delta_y) / n
    mean_m = sum(delta_m) / n
    cov_ym = sum(
        (y - mean_y) * (m - mean_m) for y, m in zip(delta_y, delta_m)
    )
    var_m = sum((m - mean_m) ** 2 for m in delta_m)
    if var_m < 1e-18:
        return _nan_result(
            target=target, mediator=mediator,
            treatment_arm=treatment_arm, baseline_arm=baseline_arm,
            pair_by=pair_by, n_pairs=n_pairs,
        )
    slope = cov_ym / var_m

    indirect = slope * mean_m
    direct = mean_y - indirect
    if abs(mean_y) < 1e-12:
        return ProportionMediatedResult(
            proportion=float('nan'),
            total=mean_y, direct=direct, indirect=indirect,
            slope_y_on_m=slope, in_unit_interval=False,
            n_pairs=n_pairs,
            target=target, mediator=mediator,
            treatment_arm=treatment_arm, baseline_arm=baseline_arm,
            pair_by=pair_by,
        )
    proportion = indirect / mean_y
    in_unit_interval = 0.0 <= proportion <= 1.0
    return ProportionMediatedResult(
        proportion=proportion,
        total=mean_y, direct=direct, indirect=indirect,
        slope_y_on_m=slope,
        in_unit_interval=in_unit_interval,
        n_pairs=n_pairs,
        target=target, mediator=mediator,
        treatment_arm=treatment_arm, baseline_arm=baseline_arm,
        pair_by=pair_by,
    )


def _nan_result(
    *, target: str, mediator: str,
    treatment_arm: str, baseline_arm: str,
    pair_by: tuple[str, ...], n_pairs: int,
) -> ProportionMediatedResult:
    return ProportionMediatedResult(
        proportion=float('nan'),
        total=float('nan'), direct=float('nan'), indirect=float('nan'),
        slope_y_on_m=float('nan'),
        in_unit_interval=False,
        n_pairs=n_pairs,
        target=target, mediator=mediator,
        treatment_arm=treatment_arm, baseline_arm=baseline_arm,
        pair_by=pair_by,
    )


__all__ = ['ProportionMediatedResult', 'proportion_mediated']
