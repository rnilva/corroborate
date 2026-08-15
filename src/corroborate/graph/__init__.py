"""Graph — typed graph primitives.

Six sub-modules, each addressable via the qualified path
(`corroborate.graph.X.Y`):

- `graph.graph` — generic `Graph[N, M]` + `Edge[N, M]`.
- `graph.causal` — `CausalGraph` (Pearl-shape, Tier-typed) +
  `BridgeEdge`, `Tier`, `Direction`.
- `graph.computation` — `ComputationGraph` extracted from
  `CallRecord` traces; `build_computation_graph`.
- `graph.correlation` — measurable cross-correlation graph +
  diff machinery.
- `graph.discovery` — PC-style adjacency discovery + orientation.
- `graph.render` — deterministic DOT/SVG renderers for computation and
  evaluated evidence graphs; `render_evidence` is the one-call entry
  that draws a hypothesis run's evidence graph with every display
  decision defaulted from the run. (Not re-exported here:
  `graph.render` imports the bridge layer, which itself imports this
  package — the qualified path avoids the cycle.)

The package surface re-exports only the most commonly-consumed
types. Helpers, internal statistical primitives, diff/PC
machinery, and result containers are accessible via the
submodule path.
"""
from corroborate.graph.causal import (
    BridgeEdge,
    CausalGraph,
    ClusterVerdict,
    Direction,
    EvidentiaryLevel,
    PostEvalEntry,
    Tier,
    cluster_verdict,
    clusters_by_extent,
    evaluated_graph,
)
from corroborate.graph._extent import stable_extent_hash
from corroborate.graph.computation import (
    ComputationGraph,
)
from corroborate.graph.graph import (
    Edge,
    Graph,
)

__all__ = [
    'BridgeEdge',
    'CausalGraph',
    'ClusterVerdict',
    'ComputationGraph',
    'Direction',
    'Edge',
    'EvidentiaryLevel',
    'Graph',
    'PostEvalEntry',
    'Tier',
    'cluster_verdict',
    'clusters_by_extent',
    'evaluated_graph',
    'stable_extent_hash',
]
