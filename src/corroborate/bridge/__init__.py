"""Bridge — verdict layer.

A `Bridge` is a typed edge in the framework's claim graph: a
`@claim_bridge`-decorated function whose body resolves to a
`Verdict` for one (treatment, baseline) cell pair. The framework
runs the bridge over a corpus via `evaluate(b, cells)`. Bridges
consume `@analysis`-registered fixtures via name-keyed parameter
injection — the analysis-registry + fixture injection live
alongside Bridge in this subpackage.

Sub-modules:
- `bridge.verdict` — `Verdict` enum, `RefutationClass` enum.
- `bridge.bridge` — `Bridge` typed edge, `@claim_bridge`
  decorator, `evaluate`, `BridgeEvaluation`.
- `bridge.admission` — admission-gate types (`AdmissionGate`,
  `GateLevel`, `GateResult`) and the framework's auto-gates
  (`distinct_arms`, `exogenous_source`, `exogenous_scope`,
  `no_predicted_direction`).
- `bridge.analysis` — `@analysis` decorator, `Analysis`
  wrapper, analysis registry, `resolve_for_holds_when`
  fixture-injection.
- `bridge.predicates` — NaN-safe polars expression helpers
  for `Bridge.scope=` (encapsulates polars's NaN-comparison
  and NaN-propagation gotchas).

Consumers `from corroborate.bridge import X`."""
from corroborate.bridge.admission import (
    AdmissionGate,
    GateLevel,
    GateResult,
    distinct_arms,
    exogenous_scope,
    exogenous_source,
    is_endogenous,
    no_predicted_direction,
    resolved_source,
)
from corroborate.bridge.analysis import (
    Analysis,
    analysis,
)
from corroborate.bridge.bridge import (
    Bridge,
    BridgeEvaluation,
    claim_bridge,
    evaluate,
)
from corroborate.bridge.predicates import (
    finite,
    finite_between,
    finite_ge,
    finite_gt,
    finite_le,
    finite_lt,
    partition_aggregate,
)
from corroborate.bridge.verdict import (
    RefutationClass,
    Verdict,
)

__all__ = [
    'AdmissionGate',
    'Analysis',
    'Bridge',
    'BridgeEvaluation',
    'GateLevel',
    'GateResult',
    'RefutationClass',
    'Verdict',
    'analysis',
    'claim_bridge',
    'distinct_arms',
    'evaluate',
    'exogenous_scope',
    'exogenous_source',
    'finite',
    'finite_between',
    'finite_ge',
    'finite_gt',
    'finite_le',
    'finite_lt',
    'is_endogenous',
    'no_predicted_direction',
    'partition_aggregate',
    'resolved_source',
]
