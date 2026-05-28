"""Measurables — typed reductions of substrate records into
scalar / array measurements consumed by analyses + bridges.

Three sub-modules:
- `measurables.measurable` — `Measurable[R, T]` type, `@measurable`
  decorator + registry, transitive read-resolution, evaluator
  that runs registered measurables on a corpus.
- `measurables.reductions` — substrate-neutral reduction
  factories (`from_key`, `reduce_axis`, `slice_axis`,
  `mean_window`, `late_window_mean`, `masked_window_mean`,
  `growth_window`, `max_abs`, `mean_peak_window`,
  `peak_centered_window`, `log_safe`, `cv_safe`).
- `measurables.redundancy_check` — three-check tautology audit;
  consumed by the `tautology_audit` analysis. Internal — access
  via the submodule path if needed.

Public surface re-exported here: the type, the decorator, the
registry accessors, the evaluator, the reduction factories.
Tautology-audit machinery and internal helpers live on the
submodule path."""
from corroborate.measurables.measurable import (
    Measurable,
    audit_measurable_registry,
    compute_missing_columns,
    evaluate_with_measurables,
    get_registered,
    measurable,
    register,
    register_as,
    registered_names,
    transitive_measurables,
    transitive_reads,
)
from corroborate.measurables.reductions import (
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
    'audit_measurable_registry',
    'compute_missing_columns',
    'cv_safe',
    'evaluate_with_measurables',
    'from_key',
    'get_registered',
    'growth_window',
    'late_window_mean',
    'log_safe',
    'masked_window_mean',
    'max_abs',
    'mean_peak_window',
    'mean_window',
    'measurable',
    'peak_centered_window',
    'reduce_axis',
    'register',
    'register_as',
    'registered_names',
    'slice_axis',
    'transitive_measurables',
    'transitive_reads',
]
