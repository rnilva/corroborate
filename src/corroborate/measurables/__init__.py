"""Measurables — typed reductions of substrate records into
scalar / array measurements consumed by analyses + bridges.

Three sub-modules:
- `measurables.core` — the `Measurable[R, T]` type, the `@measurable`
  decorator + registry, transitive read-resolution, and the
  evaluator that runs registered measurables on a corpus.
- `measurables.reductions` — substrate-neutral reduction
  factories: `from_key`, `reduce_axis`, `slice_axis`,
  `mean_window`, `late_window_mean`, `masked_window_mean`,
  `growth_window`, `max_abs`, `mean_peak_window`,
  `peak_centered_window`, `log_safe`, `cv_safe`. Composition
  primitives — substrate authors compose these with named
  measurables to build the analysis-input shape.
- `measurables.redundancy_check` — three-check tautology audit
  (HP-shadow / outcome-leak / convergence), used by the
  `tautology_audit` analysis.

Consumers `from corroborate.measurables import X` for the public
surface."""
from corroborate.measurables.measurable import (
    Measurable,
    compute_missing_columns,
    evaluate_with_measurables,
    get_registered,
    measurable,
    register,
    registered_names,
    transitive_measurables,
    transitive_reads,
)
from corroborate.measurables.redundancy_check import (
    TautologyReport,
    audit_mediator_panel,
    is_hp_tautological,
    is_outcome_tautological,
    jaccard,
    reads_overlap,
)
from corroborate.measurables.reductions import (
    Reduction,
    cv_safe,
    from_key,
    growth_window,
    late_window_mean,
    log_safe,
    masked_window_mean,
    max_abs,
    mean_peak_window,
    mean_window,
    peak_centered_window,
    reduce_axis,
    slice_axis,
)

__all__ = [
    'Measurable',
    'Reduction',
    'TautologyReport',
    'audit_mediator_panel',
    'compute_missing_columns',
    'cv_safe',
    'evaluate_with_measurables',
    'from_key',
    'get_registered',
    'growth_window',
    'is_hp_tautological',
    'is_outcome_tautological',
    'jaccard',
    'late_window_mean',
    'log_safe',
    'masked_window_mean',
    'max_abs',
    'mean_peak_window',
    'mean_window',
    'measurable',
    'peak_centered_window',
    'reads_overlap',
    'reduce_axis',
    'register',
    'registered_names',
    'slice_axis',
    'transitive_measurables',
    'transitive_reads',
]
