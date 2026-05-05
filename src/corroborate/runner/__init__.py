"""Runner — sweep dispatch + corpus runner.

Three sub-modules:

- `runner.runner` — `run` end-to-end driver. Authors invoke this
  from sweep scripts; consumes the `Hypothesis` Protocol
  (`corroborate.core.hypothesis.Hypothesis`).
- `runner.sweep` — `Runner[R]` Protocol, `run_intervention`
  paired-sweep driver, `SweepResult`, `SweepCellResult`,
  `CellFailure`.
- `runner.registry` — substrate-facing string→handle
  `Registry` (FnClaim + frozen-dataclass auto-discovery).

`@analysis` + fixture injection live in
`corroborate.bridge.analysis` — they're the glue for
`claim_bridge`, not a runner concern.

YAML-loaded `HypothesisConfig` (the substrate-coupled
intermediate that decomposes into a Hypothesis Protocol-conformer
+ a `base` callable) lives substrate-side; the framework's
hypothesis surface is the `Hypothesis` Protocol in
`corroborate.core.hypothesis`."""
from corroborate.runner.registry import Registry
from corroborate.runner.runner import collect_bridges, run
from corroborate.runner.sweep import (
    CellFailure,
    Runner,
    SweepCellResult,
    SweepResult,
    run_intervention,
)

__all__ = [
    'CellFailure',
    'Registry',
    'Runner',
    'SweepCellResult',
    'SweepResult',
    'collect_bridges',
    'run',
    'run_intervention',
]
