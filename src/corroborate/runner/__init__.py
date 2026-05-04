"""Runner — sweep dispatch + corpus runner + config loading.

Four sub-modules:

- `runner.runner` — `run_module` end-to-end driver +
  `HypothesisModule` Protocol. Authors invoke this from sweep
  scripts.
- `runner.sweep` — `Runner[R]` Protocol, `run_intervention`
  paired-sweep driver, `SweepResult`, `SweepCellResult`,
  `CellFailure`.
- `runner.registry` — substrate-facing string→handle
  `Registry` (FnClaim + frozen-dataclass auto-discovery).
- `runner.config_loader` — YAML → Hypothesis loader.

`@analysis` + fixture injection live in
`corroborate.bridge.analysis` — they're the glue for
`claim_bridge`, not a runner concern.

Public surface re-exported here: the corpus runner entry
points, `Registry`, the sweep types. Internal sweep helpers
(`empty_graph`), config-loader helpers (`resolve`,
`build_hypothesis_from_mapping`, `is_str_keyed_mapping`) live
on the submodule path."""
from corroborate.runner.config_loader import (
    load_hypothesis,
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
    run_intervention,
)

__all__ = [
    'CellFailure',
    'HypothesisModule',
    'Registry',
    'Runner',
    'SweepCellResult',
    'SweepResult',
    'load_hypothesis',
    'run_intervention',
    'run_module',
]
