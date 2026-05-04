"""Runner — sweep dispatch + corpus runner + config loading.

Four sub-modules:

- `runner.runner` — `run_module` end-to-end driver +
  `HypothesisModule` Protocol. Authors invoke this from sweep
  scripts.
- `runner.sweep` — `Runner[R]` Protocol, `sweep` driver,
  `run_hypotheses`, `SweepResult`, `SweepCellResult`,
  `CellFailure`, `empty_graph`.
- `runner.registry` — substrate-facing string→handle
  `Registry` (FnClaim + frozen-dataclass auto-discovery).
- `runner.config_loader` — YAML → Hypothesis loader +
  `resolve`, `is_str_keyed_mapping`,
  `build_hypothesis_from_mapping`.

The `@analysis` decorator + `resolve_for_holds_when` fixture
injection live in `corroborate.bridge.analysis` — they are the
fixture-injection glue for `claim_bridge`, not a runner concern.

Consumers `from corroborate.runner import X`."""
from corroborate.runner.config_loader import (
    build_hypothesis_from_mapping,
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
    'CellFailure',
    'HypothesisModule',
    'Registry',
    'Runner',
    'SweepCellResult',
    'SweepResult',
    'build_hypothesis_from_mapping',
    'empty_graph',
    'load_hypothesis',
    'resolve',
    'run_hypotheses',
    'run_module',
    'sweep',
]
