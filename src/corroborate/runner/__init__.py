"""Runner — sweep dispatch + analysis registry + config loading.

Five sub-modules:

- `runner.runner` — `run_hypotheses` corpus runner +
  `HypothesisModule` Protocol + `run_module`. The end-to-end
  driver authors invoke from sweep scripts.
- `runner.sweep` — `Runner[R]` Protocol, `sweep` driver,
  `SweepResult`, `SweepCellResult`, `CellFailure`.
- `runner.registry` — substrate-facing string→handle
  `Registry` (FnClaim + frozen-dataclass auto-discovery).
- `runner.analysis` — `@analysis` decorator + `Analysis`
  wrapper + analysis registry + `resolve_for_holds_when`
  fixture-injection.
- `runner.config_loader` — YAML → Hypothesis loader +
  `resolve`, `is_str_keyed_mapping`, `build_hypothesis_from_mapping`.

Consumers `from corroborate.runner import X`."""
from corroborate.runner.analysis import (
    Analysis,
    analysis,
    get_registered,
    registered_names,
    resolve_for_holds_when,
    run_for,
)
from corroborate.runner.config_loader import (
    build_hypothesis_from_mapping,
    is_str_keyed_mapping,
    load_hypothesis,
    resolve,
)
from corroborate.runner.registry import Registry
from corroborate.runner.runner import (
    HypothesisModule,
    run_module,
)
from corroborate.runner.sweep import (
    CellFailure,
    Runner,
    SweepCellResult,
    SweepResult,
    empty_graph,
    run_hypotheses,
    sweep,
)

__all__ = [
    'Analysis',
    'CellFailure',
    'HypothesisModule',
    'Registry',
    'Runner',
    'SweepCellResult',
    'SweepResult',
    'analysis',
    'build_hypothesis_from_mapping',
    'empty_graph',
    'get_registered',
    'is_str_keyed_mapping',
    'load_hypothesis',
    'registered_names',
    'resolve',
    'resolve_for_holds_when',
    'run_for',
    'run_hypotheses',
    'run_module',
    'sweep',
]
