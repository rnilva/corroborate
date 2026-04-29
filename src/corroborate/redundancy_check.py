"""Tautology audit — three independent checks for whether a
candidate mediator can carry causal information in an HP→outcome
analysis.

Three distinct tautology kinds, each with its own check:

- **Outcome-tautological** (`flagged_outcome`): the mediator reads
  from the same trace columns the outcome aggregates from. Example:
  `learning_curve_auc` integrates `mc_return`; the outcome path
  averages `mc_return` over eval bursts. The mediator's "g of solved
  vs unsolved" is then a re-statement of the outcome at a different
  aggregation, not a causal mediator. **Detection**: jaccard on the
  measurable's `reads` field vs the outcome's source columns.

- **HP-deterministic** (`flagged_hp`): the mediator's per-cell value
  is a deterministic function of the HP. Example: at steady state
  with uniform sampling, `mean_replay_sample_age = capacity / 2`.
  **Detection**: OLS R² of mediator on HP ≥ threshold (default 0.95).

- **HP-shadow / no-residual-signal** (`flagged_no_residual_signal`):
  the mediator's apparent association with the outcome is *entirely
  mediated by the HP*. After controlling for HP via stratified
  Spearman ρ(mediator, outcome | HP-stratum), the within-stratum
  pooled correlation is near zero. The mediator doesn't contribute
  solve-prediction beyond what the HP itself provides. Example:
  `td_within_batch_var_late` on the CartPole HP corpus — strongly
  HP-conditioned (10× difference cap=10k vs cap=50k), but within
  each capacity stratum it doesn't predict solving.

  Implementation note: uses `stratified_spearman_rho` (per-stratum
  ρ pooled via Fisher z, weighted by n−3) rather than residual-
  partial-Spearman, because the latter is unstable when within-
  stratum noise is small relative to between-stratum signal — rank
  residuals collapse and the resulting Pearson is noisy. Stratified
  ρ is more robust at the discrete-HP-grid corpora the framework
  typically operates on. Only one HP axis is used as the stratum
  key (the first in `hp_axes`); for multi-axis HP-shadow detection
  the analyst can collapse the HP grid into a single configuration
  ID upstream.

A clean mediator passes all three: structurally independent of the
outcome's source columns, not deterministic in any HP, and adds
residual signal beyond the HP."""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from corroborate.causal_discovery import stratified_spearman_rho
from corroborate.schema import RunRow


class _NameReadsProtocol(Protocol):
    """Structural shape of a Measurable for tautology audit —
    only `.name` and `.reads` are needed. Used as the panel
    parameter type so heterogeneous Measurable[R, T] instances
    can be passed without invariance friction."""
    @property
    def name(self) -> str: ...
    @property
    def reads(self) -> tuple[str, ...]: ...


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """`|A ∩ B| / |A ∪ B|`. Returns 0 when both sets are empty
    (vacuous-overlap convention; not informative)."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def reads_overlap(
    a: _NameReadsProtocol, b: _NameReadsProtocol,
) -> float:
    """Jaccard between two measurables' leaf-reads sets."""
    return jaccard(frozenset(a.reads), frozenset(b.reads))


def is_outcome_tautological(
    mediator: _NameReadsProtocol,
    outcome_reads: frozenset[str],
    *,
    threshold: float = 0.5,
) -> bool:
    """Reads-set jaccard between the mediator and the outcome's
    source columns. ≥ `threshold` flags the mediator as essentially
    a re-encoding of the outcome at a different aggregation.

    `outcome_reads` is provided by the caller — for path-keyed
    outcomes (`outcome.eval_best_burst_mean`, etc.), this is the
    trace columns the cell-runner aggregates from to produce that
    path. Most RL outcome paths derive from `mc_return` per eval
    episode."""
    return jaccard(frozenset(mediator.reads), outcome_reads) >= threshold


def _r_squared(
    x: Sequence[float], y: Sequence[float],
) -> float:
    """OLS R² of `y ~ x`. Returns NaN when n < 2 or x has zero
    variance. R² = 1 - SS_res / SS_tot."""
    n = len(x)
    if n < 2 or len(y) != n:
        return float('nan')
    finite = [
        (xi, yi) for xi, yi in zip(x, y)
        if not math.isnan(xi) and not math.isnan(yi)
    ]
    if len(finite) < 2:
        return float('nan')
    xs = [p[0] for p in finite]
    ys = [p[1] for p in finite]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    var_x = sum((xi - x_mean) ** 2 for xi in xs)
    if var_x == 0.0:
        return float('nan')
    cov_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in finite)
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    ss_tot = sum((yi - y_mean) ** 2 for yi in ys)
    if ss_tot == 0.0:
        return 1.0
    ss_res = sum(
        (yi - (intercept + slope * xi)) ** 2 for xi, yi in finite
    )
    return max(0.0, 1.0 - ss_res / ss_tot)


def is_hp_tautological(
    mediator_per_cell: Sequence[float],
    hp_per_cell: Sequence[float],
    *,
    threshold: float = 0.95,
) -> bool:
    """OLS R² of mediator on HP. ≥ `threshold` flags the mediator
    as a deterministic function of the HP — the mediator IS the HP,
    not a measurement of consequence.

    Caveat: this is per-axis empirical R². A mediator deterministic
    in two HPs jointly (`f(lr, batch)`) won't be caught by checking
    each axis individually. For multi-axis joint R², use the audit
    panel."""
    r2 = _r_squared(hp_per_cell, mediator_per_cell)
    if math.isnan(r2):
        return False
    return r2 >= threshold


# ============ Corpus-level audit ============

@dataclass(frozen=True, slots=True)
class TautologyReport:
    """Per-measurable audit summary against a (corpus, outcome,
    HP-axes) trio. Three flags, each from an independent check:

    `outcome_jaccard` / `flagged_outcome` — structural: reads-set
      overlap with the outcome's source columns.
    `hp_r_squared` / `flagged_hp` — empirical: per-HP-axis OLS R².
      Flagged axes are those where the mediator is essentially a
      deterministic function of the HP.
    `outcome_stratified_rho` / `outcome_stratified_p` /
    `flagged_no_residual_signal` — empirical: stratified Spearman
      ρ(mediator, outcome | hp_stratum_axis). Flagged when |ρ| <
      threshold AND p ≥ alpha — meaning within-stratum correlation
      between mediator and outcome is null after pooling, so the
      mediator's apparent outcome association is HP-mediated. NaN
      when `outcome_path` or `hp_stratum_axis` wasn't provided."""
    measurable_name: str
    outcome_jaccard: float
    hp_r_squared: Mapping[str, float]
    outcome_stratified_rho: float
    outcome_stratified_p: float
    flagged_outcome: bool
    flagged_hp: tuple[str, ...]
    flagged_no_residual_signal: bool

    @property
    def is_clean(self) -> bool:
        """Mediator passes all three checks: structurally independent
        of the outcome, not deterministic in any HP, and adds residual
        signal beyond HP."""
        return (
            not self.flagged_outcome
            and not self.flagged_hp
            and not self.flagged_no_residual_signal
        )


def audit_mediator_panel(
    measurables: Sequence[_NameReadsProtocol],
    runs: Sequence[RunRow],
    *,
    outcome_reads: frozenset[str],
    hp_axes: tuple[str, ...],
    outcome_path: str | None = None,
    hp_stratum_axis: str | None = None,
    mediator_path_for: Mapping[str, str] | None = None,
    outcome_jaccard_threshold: float = 0.5,
    hp_r_squared_threshold: float = 0.95,
    stratified_rho_threshold: float = 0.1,
    stratified_alpha: float = 0.05,
    stratified_min_stratum_size: int = 4,
) -> tuple[TautologyReport, ...]:
    """Audit a panel of mediators with all three checks.

    `outcome_reads` — trace-column names the outcome aggregates from.
      Used for the structural jaccard check.
    `hp_axes` — `RunRow.measurements` keys naming the HPs.
    `outcome_path` — flat path to a per-cell outcome scalar. When
      provided alongside `hp_stratum_axis`, the audit computes
      stratified Spearman ρ(mediator, outcome | hp_stratum) for
      each measurable.
    `hp_stratum_axis` — single HP axis to stratify by for the
      partial-correlation check. Defaults to the first axis in
      `hp_axes`. (Multi-axis stratification means more strata with
      fewer cells each — typically not power-feasible.)
    `mediator_path_for` — optional override mapping
      `measurable.name → RunRow.measurements key`.
    `stratified_rho_threshold` — `|ρ| < threshold` flags HP-shadow.
    `stratified_alpha` — significance threshold for the test.
    `stratified_min_stratum_size` — strata with fewer cells are
      skipped (Fisher-z floor at n=4).

    Three flags per measurable:
    - `flagged_outcome`: jaccard ≥ outcome_jaccard_threshold.
    - `flagged_hp`: HP axes where R² ≥ hp_r_squared_threshold.
    - `flagged_no_residual_signal`: |stratified_ρ| < threshold AND
      p ≥ alpha — within each HP-stratum the mediator doesn't
      correlate with the outcome, so the marginal correlation is
      HP-mediated."""
    reports: list[TautologyReport] = []
    for m in measurables:
        # Outcome-side: structural reads-jaccard.
        oj = jaccard(frozenset(m.reads), outcome_reads)
        flagged_o = oj >= outcome_jaccard_threshold

        # HP-side: empirical R² per axis.
        path = (
            mediator_path_for[m.name]
            if mediator_path_for and m.name in mediator_path_for
            else f'mediator.{m.name}'
        )
        mediator_vals: list[float] = []
        outcome_vals: list[float] = []
        hp_vals_by_axis: dict[str, list[float]] = {a: [] for a in hp_axes}
        for r in runs:
            mv = r.measurements.get(path)
            if not isinstance(mv, (int, float)) or isinstance(mv, bool):
                continue
            mvf = float(mv)
            if math.isnan(mvf) or math.isinf(mvf):
                continue
            ov: float | None = None
            if outcome_path is not None:
                ov_raw = r.measurements.get(outcome_path)
                if (
                    not isinstance(ov_raw, (int, float))
                    or isinstance(ov_raw, bool)
                ):
                    continue
                ov_f = float(ov_raw)
                if math.isnan(ov_f) or math.isinf(ov_f):
                    continue
                ov = ov_f
            row_hp_vals: dict[str, float] = {}
            ok = True
            for axis in hp_axes:
                hv = r.measurements.get(axis)
                if not isinstance(hv, (int, float)) or isinstance(hv, bool):
                    ok = False
                    break
                hvf = float(hv)
                if math.isnan(hvf) or math.isinf(hvf):
                    ok = False
                    break
                row_hp_vals[axis] = hvf
            if not ok:
                continue
            mediator_vals.append(mvf)
            if ov is not None:
                outcome_vals.append(ov)
            for axis in hp_axes:
                hp_vals_by_axis[axis].append(row_hp_vals[axis])

        hp_r2: dict[str, float] = {}
        flagged_hp: list[str] = []
        for axis in hp_axes:
            r2 = _r_squared(hp_vals_by_axis[axis], mediator_vals)
            hp_r2[axis] = r2
            if not math.isnan(r2) and r2 >= hp_r_squared_threshold:
                flagged_hp.append(axis)

        # Stratified-correlation check. Skipped when outcome_path
        # is missing, when no HP-stratum axis is available, or when
        # data is too thin per stratum.
        stratum_axis = (
            hp_stratum_axis if hp_stratum_axis is not None
            else (hp_axes[0] if hp_axes else None)
        )
        strat_rho = float('nan')
        strat_p = float('nan')
        flagged_no_residual = False
        if (
            outcome_path is not None
            and stratum_axis is not None
            and len(outcome_vals) == len(mediator_vals)
            and len(mediator_vals) >= 2 * stratified_min_stratum_size
        ):
            x_arr = np.asarray(mediator_vals, dtype=np.float64)
            y_arr = np.asarray(outcome_vals, dtype=np.float64)
            stratum_vals = hp_vals_by_axis.get(stratum_axis, [])
            if len(stratum_vals) == len(mediator_vals):
                strat_rho, strat_p = stratified_spearman_rho(
                    x_arr, y_arr, stratum_vals,
                    min_stratum_size=stratified_min_stratum_size,
                )
                if (
                    not math.isnan(strat_rho)
                    and abs(strat_rho) < stratified_rho_threshold
                    and not math.isnan(strat_p)
                    and strat_p >= stratified_alpha
                ):
                    flagged_no_residual = True

        reports.append(TautologyReport(
            measurable_name=m.name,
            outcome_jaccard=oj,
            hp_r_squared=hp_r2,
            outcome_stratified_rho=strat_rho,
            outcome_stratified_p=strat_p,
            flagged_outcome=flagged_o,
            flagged_hp=tuple(flagged_hp),
            flagged_no_residual_signal=flagged_no_residual,
        ))
    return tuple(reports)
