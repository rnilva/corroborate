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

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx
    import pandas as pd
    from dowhy import CausalModel

    from corroborate.graph.causal import CausalGraph


# Type alias — DAG accepted shapes. Coerced to `nx.DiGraph[str]`
# by `_to_networkx` at call time. PEP 695 lazy alias, so the
# TYPE_CHECKING imports above are sufficient.
#
# Sequence-of-edges accepts both `list` and `tuple` — bridges
# pass `tuple[tuple[str, str], ...]` as a default arg (immutable
# class-level constants), while script callers tend to build
# `list[tuple[str, str]]` inline.
type DAGLike = (
    'nx.DiGraph[str] | CausalGraph '
    '| list[tuple[str, str]] | tuple[tuple[str, str], ...]'
)


def _to_networkx(graph: DAGLike) -> 'nx.DiGraph[str]':
    """Coerce a graph spec into `nx.DiGraph[str]`. Accepts an
    `nx.DiGraph` directly; corroborate's `CausalGraph`
    (`Graph[str, BridgeEdge]`); or a sequence of `(source,
    target)` edge 2-tuples (list or tuple — bridge defaults are
    typically tuples for immutability)."""
    import networkx as nx

    if isinstance(graph, nx.DiGraph):
        return graph

    g: nx.DiGraph[str] = nx.DiGraph()
    if isinstance(graph, (list, tuple)):
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
            return float(getattr(refuter, attr))
    raise AttributeError(
        f'DoWhy refuter {type(refuter).__name__} has neither '
        f'`new_effect` nor `estimated_effect` — version mismatch?',
    )


def _record_keys_for(graph: DAGLike) -> list[str]:
    """Return the variable names in `graph` — the DataFrame columns
    we need to project from the record."""
    nx_graph = _to_networkx(graph)
    return list(nx_graph.nodes)


def _cells_to_dataframe(
    cells: Iterable[Mapping[str, object]],
    keys: list[str],
) -> 'pd.DataFrame':
    """Project the cell collection to a pandas DataFrame: one
    row per cell, columns = `keys`. Cells missing any required
    key are skipped (so partial corpora don't crash). Non-scalar
    values are skipped.

    Lifted from `analyses.dowhy` so sibling analyses (e.g.
    `mediation_dowhy`) can reuse without crossing the
    public/private boundary on a sibling-module helper."""
    import pandas as pd

    rows: list[dict[str, float]] = []
    for cell in cells:
        row: dict[str, float] = {}
        complete = True
        for k in keys:
            v = cell.get(k)
            if isinstance(v, bool):
                row[k] = float(v)
            elif isinstance(v, (int, float)):
                row[k] = float(v)
            else:
                complete = False
                break
        if complete:
            rows.append(row)
    return pd.DataFrame(rows)


def _backdoor_estimate(
    cells: Iterable[Mapping[str, object]],
    treatment: str,
    outcome: str,
    dag: DAGLike,
    method_name: str,
) -> tuple[
    'pd.DataFrame',
    object,
    object | None,
]:
    """Build DataFrame + CausalModel + run identification +
    (when identified) estimation. Helper shared by all DoWhy-
    consuming analyses so model construction is consistent."""
    df = _cells_to_dataframe(cells, _record_keys_for(dag))
    model = _build_causal_model(df, treatment, outcome, dag)
    identified = model.identify_effect(
        proceed_when_unidentifiable=False,
    )
    if (
        getattr(identified, 'no_directed_path', False)
        or not getattr(identified, 'estimands', None)
    ):
        return df, identified, None
    estimate = model.estimate_effect(
        identified, method_name=method_name,
    )
    return df, identified, estimate


def backdoor_with_refutations(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment: str,
    outcome: str,
    dag: DAGLike,
    method_name: str = 'backdoor.linear_regression',
) -> tuple[
    'BackdoorResult', 'RefutationResult', 'RefutationResult',
]:
    """Bundle the 3-call DoWhy chain: backdoor ATE + placebo +
    RCC refutations, all on the same `(treatment, outcome, dag)`
    triple. Used by `stratum_*_link_dowhy` analyses that always
    run the three together — the refutation chain is the
    methodologically required corroboration, not a separable
    choice (CLAUDE.md §"Mediation recipe" steps 1+5)."""
    from corroborate.analyses.dowhy import (
        backdoor_ate, placebo_refutation,
        random_common_cause_refutation,
    )
    cells_list = list(cells)
    backdoor = backdoor_ate.fn(
        cells_list, treatment=treatment, outcome=outcome,
        dag=dag, method_name=method_name,
    )
    placebo = placebo_refutation.fn(
        cells_list, treatment=treatment, outcome=outcome,
        dag=dag, method_name=method_name,
    )
    rcc = random_common_cause_refutation.fn(
        cells_list, treatment=treatment, outcome=outcome,
        dag=dag, method_name=method_name,
    )
    return backdoor, placebo, rcc


if TYPE_CHECKING:
    from corroborate.analyses.dowhy import (
        BackdoorResult, RefutationResult,
    )


__all__ = [
    'DAGLike',
    '_backdoor_estimate',
    '_build_causal_model',
    '_cells_to_dataframe',
    '_record_keys_for',
    '_refuter_effect',
    '_to_networkx',
    'backdoor_with_refutations',
]
