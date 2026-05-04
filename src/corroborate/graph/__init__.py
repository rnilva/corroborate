"""Graph — typed graph primitives used across the framework.

Four purposes co-located here:

- `graph.graph` — generic `Graph[N, M]` + `Edge[N, M]` types
  (parametric in node/metadata). The substrate for all the
  specialised graph types below.
- `graph.causal` — `CausalGraph` (Tier-typed Pearl-shape graph)
  and the `BridgeEdge` carried on its edges.
- `graph.computation` — `ComputationGraph` extracted from
  `CallRecord` traces; the runtime composition graph.
- `graph.measurable` — measurable-cross-correlation graph + the
  diff against the substrate's column-role mapping.
- `graph.discovery` — PC-style adjacency discovery + the
  `discover_adjacency` / `orient_adjacency` API.

Consumers `from corroborate.graph import X` for the public
surface; sub-modules are accessible via the explicit path
(`corroborate.graph.discovery.discover_adjacency`)."""
from corroborate.graph.causal import (
    BridgeEdge,
    CausalGraph,
    Direction,
    Tier,
    chain_tier,
)
from corroborate.graph.computation import (
    ComputationEdge,
    ComputationGraph,
    build_computation_graph,
    extract_raw_edges,
    measurables_by_attachment,
    producing_paths,
)
from corroborate.graph.discovery import (
    DiscoveredAdjacency,
    EdgeDiff,
    OrientedAdjacency,
    VariableScope,
    assert_stratification_admissible,
    classify_variable_scope,
    compare_pc_depths,
    discover_adjacency,
    orient_adjacency,
    partial_spearman_rho,
    partial_spearman_rho_multi,
    stratified_partial_spearman_rho,
    stratified_spearman_rho,
)
from corroborate.graph.graph import (
    Edge,
    Graph,
    GraphDiff,
)
from corroborate.graph.measurable import (
    ColumnRole,
    Correlation,
    DiffCategory,
    DiffEdge,
    correlation_matrix_table,
    diff_against_claim_graph,
    explained_by_claim_graph,
    pairwise_correlations,
)

__all__ = [
    'BridgeEdge',
    'CausalGraph',
    'ColumnRole',
    'ComputationEdge',
    'ComputationGraph',
    'Correlation',
    'DiffCategory',
    'DiffEdge',
    'Direction',
    'DiscoveredAdjacency',
    'Edge',
    'EdgeDiff',
    'Graph',
    'GraphDiff',
    'OrientedAdjacency',
    'Tier',
    'VariableScope',
    'assert_stratification_admissible',
    'build_computation_graph',
    'chain_tier',
    'classify_variable_scope',
    'compare_pc_depths',
    'correlation_matrix_table',
    'diff_against_claim_graph',
    'discover_adjacency',
    'explained_by_claim_graph',
    'extract_raw_edges',
    'measurables_by_attachment',
    'orient_adjacency',
    'pairwise_correlations',
    'partial_spearman_rho',
    'partial_spearman_rho_multi',
    'producing_paths',
    'stratified_partial_spearman_rho',
    'stratified_spearman_rho',
]
