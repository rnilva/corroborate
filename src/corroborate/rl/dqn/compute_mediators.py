"""Compute panel of Measurables on a corpus of cells by streaming
traces.

The substrate's per-cell trace store carries the per-step series
each Measurable reads (`mc_return`, `td_error`, `online_q_per_action`,
state_hash`, etc.). At analysis time we want scalar reductions
(`mediator.greedy_match_late`, `mediator.q_gap_late`, ...) per
cell. This primitive streams the trace store, evaluates each
panel Measurable per cell, and returns RunRows with the scalars
injected into `measurements` under the `mediator.{name}` key.

Use:
    from corroborate.rl.dqn.compute_mediators import (
        compute_mediator_panel, DEFAULT_PANEL,
    )
    from corroborate.persistence import read_runrows
    runs = read_runrows(Path('experiments/data/X/runs.parquet'))
    enriched = compute_mediator_panel(
        runs,
        traces_paths=[Path('experiments/data/X/traces.parquet')],
        panel=DEFAULT_PANEL,
    )
    # Each enriched RunRow has `mediator.greedy_match_late`,
    # `mediator.q_gap_late`, etc. in .measurements.

Cells without a matching trace are silently dropped — the
streaming contract is "only cells with traces produce enriched
rows". Cells whose Measurable evaluation raises NaN keep that NaN
in the output (downstream filters drop it)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from corroborate.measurables import Measurable
from corroborate.persistence import iter_trace_records
from corroborate.rl.dqn.measurables import (
    greedy_match_late,
    learning_curve_auc,
    plateau_slope_late,
    q_gap_growth,
    q_gap_late,
    q_max_growth,
    return_at_25pct_steps,
    state_coverage_kl_uniform_late,
    state_visit_entropy_late,
    td_residual_late,
    td_within_batch_var_late,
    time_to_threshold,
    v_vs_max_delta_late,
)
from corroborate.schema import RunRow


# All record-only measurables (not requiring leaf injection).
# fill_ratio_late requires a `capacity` kwarg → handled by the
# substrate at construction time if needed.
DEFAULT_PANEL: tuple[Measurable[Mapping[str, object], float], ...] = (
    q_gap_late, q_gap_growth, q_max_growth,
    v_vs_max_delta_late,
    td_residual_late, td_within_batch_var_late,
    greedy_match_late,
    learning_curve_auc, time_to_threshold,
    return_at_25pct_steps, plateau_slope_late,
    state_visit_entropy_late, state_coverage_kl_uniform_late,
)


def _columns_for_panel(
    panel: Sequence[Measurable[Mapping[str, object], float]],
) -> tuple[str, ...]:
    """Default column set: 'id' + union of panel.reads."""
    cols: set[str] = {'id'}
    for m in panel:
        cols.update(m.reads)
    return tuple(sorted(cols))


def compute_mediator_panel(
    runs: Sequence[RunRow],
    traces_paths: Sequence[Path],
    *,
    panel: Sequence[Measurable[Mapping[str, object], float]] = DEFAULT_PANEL,
    columns: Sequence[str] | None = None,
) -> list[RunRow]:
    """Stream traces from each path, evaluate `panel` per cell,
    return RunRows with `mediator.{name}` injected.

    - `runs`: cells to enrich (typically loaded from runs.parquet).
    - `traces_paths`: trace parquet(s); per-arm stored separately
       is fine — pass the whole list. Cells in `runs` without a
       matching trace are silently dropped.
    - `panel`: Measurable instances to evaluate (default:
       DEFAULT_PANEL — all record-only measurables in the dqn
       panel).
    - `columns`: trace columns to read (default: derived from
       panel.reads ∪ {'id'}).

    Memory bounded by `iter_trace_records`'s batch_size."""
    runs_by_id = {r.id: r for r in runs}
    cols = tuple(columns) if columns is not None else _columns_for_panel(panel)
    enriched: list[RunRow] = []
    for trace_path in traces_paths:
        for record in iter_trace_records(trace_path, columns=cols):
            cell_id = record.get('id')
            if not isinstance(cell_id, str) or cell_id not in runs_by_id:
                continue
            run = runs_by_id[cell_id]
            # `panel` measurables return `float` (Measurable[..., float]
            # bound at the panel-element type), which is one of the
            # `MeasurementLeaf` arms — type the dict accordingly so
            # `replace(run, measurements=...)` lands without a cast.
            from corroborate.schema import MeasurementLeaf
            new_meas: dict[str, MeasurementLeaf] = dict(run.measurements)
            for m in panel:
                new_meas[f'mediator.{m.name}'] = m.fn(record)
            enriched.append(replace(run, measurements=new_meas))
    return enriched


__all__ = ['DEFAULT_PANEL', 'compute_mediator_panel']
