"""Bridge — verdict layer.

A `Bridge` is a typed edge in the framework's claim graph: a
declarative `claim_bridge`-decorated function whose body resolves
to a `Verdict` for one (treatment, baseline) cell pair. The
framework runs the bridge over a corpus, derives a per-bridge
verdict, and folds the bridge graph into a `HypothesisVerdict`.

Three sub-modules:
- `bridge.verdict` — `Verdict` enum, `RefutationClass` enum.
- `bridge.claim_bridge` — `Bridge`, `@claim_bridge` decorator,
  `evaluate`, `BridgeEvaluation`.
- `bridge.hypothesis_verdict` — `HypothesisVerdict`, the bridge-
  graph→verdict roll-up via `hypothesis_subgraph_verdict`.
- `bridge.scope` — `Scope` dataclass + `build_scope` (the per-
  hypothesis scope claim).

Consumers `from corroborate.bridge import X`."""
from corroborate.bridge.claim_bridge import (
    Bridge,
    BridgeEvaluation,
    claim_bridge,
    evaluate,
    measurable_names_for_bridges,
)
from corroborate.bridge.hypothesis_verdict import (
    HypothesisVerdict,
    hypothesis_subgraph_verdict,
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
    'Bridge',
    'BridgeEvaluation',
    'HypothesisVerdict',
    'RefutationClass',
    'Scope',
    'Verdict',
    'build_scope',
    'claim_bridge',
    'evaluate',
    'hypothesis_subgraph_verdict',
    'measurable_names_for_bridges',
]
