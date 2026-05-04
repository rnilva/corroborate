"""Bridge — verdict layer.

A `Bridge` is a typed edge in the framework's claim graph: a
`@claim_bridge`-decorated function whose body resolves to a
`Verdict` for one (treatment, baseline) cell pair. The framework
runs the bridge over a corpus, derives a per-bridge verdict, and
folds the bridge graph into a `HypothesisVerdict`. Bridges
consume `@analysis`-registered fixtures via name-keyed
parameter injection — the analysis-registry + fixture
injection live alongside Bridge in this subpackage.

Sub-modules:
- `bridge.verdict` — `Verdict` enum, `RefutationClass` enum.
- `bridge.claim_bridge` — `Bridge` typed edge, `@claim_bridge`
  decorator, `evaluate`, `BridgeEvaluation`.
- `bridge.analysis` — `@analysis` decorator, `Analysis`
  wrapper, analysis registry, `resolve_for_holds_when`
  fixture-injection.
- `bridge.hypothesis_verdict` — `HypothesisVerdict`, the
  bridge-graph → verdict roll-up via
  `hypothesis_subgraph_verdict`.
- `bridge.scope` — `Scope` dataclass + `build_scope`
  (per-hypothesis scope claim).

Consumers `from corroborate.bridge import X`."""
from corroborate.bridge.analysis import (
    Analysis,
    analysis,
)
from corroborate.bridge.claim_bridge import (
    Bridge,
    BridgeEvaluation,
    claim_bridge,
    evaluate,
)
from corroborate.bridge.hypothesis_verdict import (
    HypothesisVerdict,
)
from corroborate.bridge.scope import (
    Scope,
    build_scope,
)
from corroborate.bridge.verdict import (
    RefutationClass,
    Verdict,
)

__all__ = [
    'Analysis',
    'Bridge',
    'BridgeEvaluation',
    'HypothesisVerdict',
    'RefutationClass',
    'Scope',
    'Verdict',
    'analysis',
    'build_scope',
    'claim_bridge',
    'evaluate',
]
