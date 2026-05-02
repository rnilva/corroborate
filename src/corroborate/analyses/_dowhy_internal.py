"""DoWhy private helpers — DAG coercion + CausalModel build +
refuter-effect extraction.

Phase 4F of the Bridge-collapse refactor moved these out of the
deleted `bridges_dowhy.py` (the per-record `Bridge[R]`-shaped
DoWhy wrappers). The `@analysis`-shaped versions in
`analyses/dowhy.py` consume the same machinery; this module is
the shared private home so a future move to a richer DoWhy
adapter doesn't fan out across `analyses/`.

Lazy imports — DoWhy / pandas / networkx are imported inside the
helper bodies, not at module top, so corroborate's spine imports
cleanly without them. ImportError surfaces at call time."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx
    import pandas as pd
    from dowhy import CausalModel

    from corroborate.causal_graph import CausalGraph


# Type alias — DAG accepted shapes. Coerced to `nx.DiGraph[str]`
# by `_to_networkx` at call time. PEP 695 lazy alias, so the
# TYPE_CHECKING imports above are sufficient.
type DAGLike = (
    'nx.DiGraph[str] | CausalGraph | list[tuple[str, str]]'
)


def _to_networkx(graph: DAGLike) -> 'nx.DiGraph[str]':
    """Coerce a graph spec into `nx.DiGraph[str]`. Accepts an
    `nx.DiGraph` directly; corroborate's `CausalGraph`
    (`Graph[str, BridgeEdge]`); or a `list[(source, target)]` edge
    tuple list."""
    import networkx as nx

    if isinstance(graph, nx.DiGraph):
        return graph

    g: nx.DiGraph[str] = nx.DiGraph()
    if isinstance(graph, list):
        for src, tgt in graph:
            g.add_edge(src, tgt)
        return g

    # corroborate's CausalGraph is `Graph[str, BridgeEdge]` —
    # frozen dataclass with `nodes: frozenset[str]` and
    # `edges: tuple[Edge[str, BridgeEdge], ...]`.
    for n in graph.nodes:
        g.add_node(n)
    for e in graph.edges:
        g.add_edge(e.source, e.target)
    return g


def _build_causal_model(
    df: 'pd.DataFrame', treatment: str, outcome: str, graph: DAGLike,
) -> 'CausalModel':
    """Construct a DoWhy `CausalModel`. Validates that treatment
    and outcome appear as both DAG nodes and DataFrame columns."""
    from dowhy import CausalModel

    nx_graph = _to_networkx(graph)
    for nm, role in ((treatment, 'treatment'), (outcome, 'outcome')):
        if nm not in nx_graph.nodes:
            raise ValueError(
                f'{role} {nm!r} is not a node in the supplied DAG. '
                f'DAG nodes: {sorted(nx_graph.nodes)!r}.',
            )
        if nm not in df.columns:
            raise ValueError(
                f'{role} {nm!r} is not a column in the record. '
                f'Record columns: {sorted(df.columns)!r}.',
            )
    return CausalModel(
        data=df, treatment=treatment, outcome=outcome, graph=nx_graph,
    )


def _refuter_effect(refuter: object) -> float:
    """Extract the post-refutation effect. DoWhy renamed this
    attribute between versions: older expose `estimated_effect`,
    newer expose `new_effect`. Try both."""
    for attr in ('new_effect', 'estimated_effect'):
        if hasattr(refuter, attr):
            return float(getattr(refuter, attr))  # pyright: ignore[reportAny]
    raise AttributeError(
        f'DoWhy refuter {type(refuter).__name__} has neither '
        f'`new_effect` nor `estimated_effect` — version mismatch?',
    )


def _record_keys_for(graph: DAGLike) -> list[str]:
    """Return the variable names in `graph` — the DataFrame columns
    we need to project from the record."""
    nx_graph = _to_networkx(graph)
    return list(nx_graph.nodes)


__all__ = [
    'DAGLike',
    '_build_causal_model',
    '_record_keys_for',
    '_refuter_effect',
    '_to_networkx',
]
