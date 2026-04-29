"""Tautology audit — does a candidate mediator share information
with the outcome (reads-set jaccard) or is it deterministic from
the HP (regression R²)?

Two distinct tautology kinds:

- **Outcome-tautological**: the mediator reads from the same trace
  columns the outcome aggregates from. Example: `learning_curve_auc`
  integrates `mc_return` over training; `outcome.eval_best_burst_mean`
  averages `mc_return` over eval bursts. The mediator's "g of solved
  vs unsolved" is then a re-statement of the outcome at a different
  aggregation, not a causal mediator. Detected by jaccard on the
  measurable's `reads` field vs the outcome's source columns.

- **HP-tautological**: the mediator's per-cell value is a
  deterministic function of the HP. Example: at steady state with
  uniform sampling, `mean_replay_sample_age = capacity / 2`. The
  mediator then *encodes* the HP rather than mediating its effect.
  Detected empirically: regress the mediator's per-cell value on the
  HP axis; R² ≥ threshold flags determinism.

Both checks gate whether a measurable can serve as a *causal*
mediator in HP→outcome analyses. A measurable that's flagged on
either check should not be reported as a mediator without explicit
acknowledgment — its appearance as a "predictor of solve" is
mechanical, not causal."""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

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
    HP-axes) trio.

    `outcome_jaccard` — reads-set jaccard with `outcome_reads`.
    `hp_r_squared` — per-HP-axis R² of the mediator regressed on
      that axis individually.
    `flagged_outcome` — `outcome_jaccard ≥ outcome_threshold`.
    `flagged_hp` — tuple of HP axis names where R² ≥ hp_threshold."""
    measurable_name: str
    outcome_jaccard: float
    hp_r_squared: Mapping[str, float]
    flagged_outcome: bool
    flagged_hp: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        """Mediator is independent of both outcome and HPs."""
        return not self.flagged_outcome and not self.flagged_hp


def audit_mediator_panel(
    measurables: Sequence[_NameReadsProtocol],
    runs: Sequence[RunRow],
    *,
    outcome_reads: frozenset[str],
    hp_axes: tuple[str, ...],
    mediator_path_for: Mapping[str, str] | None = None,
    outcome_jaccard_threshold: float = 0.5,
    hp_r_squared_threshold: float = 0.95,
) -> tuple[TautologyReport, ...]:
    """Audit a panel of mediators against an outcome's reads-set
    and a set of HP axes.

    `outcome_reads` — the set of trace-column names the outcome
      aggregates from. Caller-provided because outcome paths are
      flat strings on `RunRow.measurements`, not Measurable
      instances.
    `hp_axes` — sequence of `RunRow.measurements` keys that name the
      HPs (e.g. `('replay.capacity', 'replay.batch_size', ...)`).
    `mediator_path_for` — optional override mapping
      `measurable.name → RunRow.measurements key`. Defaults to
      `f'mediator.{measurable.name}'` (the cell-runner's projection
      convention). Override per-measurable when the path differs."""
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
        hp_vals_by_axis: dict[str, list[float]] = {a: [] for a in hp_axes}
        for r in runs:
            mv = r.measurements.get(path)
            if not isinstance(mv, (int, float)) or isinstance(mv, bool):
                continue
            mvf = float(mv)
            if math.isnan(mvf) or math.isinf(mvf):
                continue
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
            for axis in hp_axes:
                hp_vals_by_axis[axis].append(row_hp_vals[axis])

        hp_r2: dict[str, float] = {}
        flagged_hp: list[str] = []
        for axis in hp_axes:
            r2 = _r_squared(hp_vals_by_axis[axis], mediator_vals)
            hp_r2[axis] = r2
            if not math.isnan(r2) and r2 >= hp_r_squared_threshold:
                flagged_hp.append(axis)

        reports.append(TautologyReport(
            measurable_name=m.name,
            outcome_jaccard=oj,
            hp_r_squared=hp_r2,
            flagged_outcome=flagged_o,
            flagged_hp=tuple(flagged_hp),
        ))
    return tuple(reports)
