"""Measurable graph — statistical structure derived from trace returns.

Dual to `computation_graph`: the claim graph gives mechanistic
dataflow (which `@claim`'s output feeds which Claim's input by
object identity); the measurable graph gives statistical structure
(which scalar measurables co-vary across many steps).

Both are derived from `@claim` returns. The difference is the
reduction:

- claim graph: `id()` matching on ONE eager call — a structural
  graph (`computation_graph.build_*`).
- measurable graph: scalar-value Pearson correlation across MANY
  steps — a statistical graph.

The minimum viable version (this module): take a
`Mapping[str, ArrayLike]` of per-step scalar measurables (e.g. a
record dict from a scan output), build a `Graph[str, Correlation]`
where edges carry pairwise Pearson r. Derived per-arm post-trace.

The `explained_by_claim_graph(corr_edge, computation_graph)`
diagnostic separates statistical edges *predicted by* the
mechanistic skeleton from *emergent* correlations: a corr-edge
(a, b) where the computation graph reaches a↔b via Claim dataflow
is explained; one without a Claim path is unexplained — either
mediated (path > 1 in some unmodeled measurable) or spurious.

Later extensions (deferred): partial correlations conditioning on
claim-graph mediators; interventional Δs across arms;
multi-edge stratification by Pearl-ladder tier."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import override

import numpy as np
import numpy.typing as npt

from corroborate.graph.computation import ComputationGraph
from corroborate.graph.graph import Graph


@dataclass(frozen=True, slots=True)
class Correlation:
    """Pearson r between two scalar series. NaN when either series
    is constant or shorter than 2."""
    r: float

    @override
    def __str__(self) -> str:
        if not np.isfinite(self.r):
            return 'r=nan'
        return f'r={self.r:+.3f}'


type MeasurableGraph = Graph[str, Correlation]


def _safe_pearson(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64],
) -> float:
    """Pearson's r with safe handling: NaN when either series is
    constant (zero std), or has fewer than 2 samples, or the two
    series have mismatched lengths (e.g. record fields collected
    at different observation cadences)."""
    if len(a) < 2 or len(a) != len(b):
        return float('nan')
    s1 = float(np.std(a))
    s2 = float(np.std(b))
    if s1 == 0.0 or s2 == 0.0:
        return float('nan')
    centred = (a - float(np.mean(a))) * (b - float(np.mean(b)))
    return float(np.mean(centred) / (s1 * s2))


def pairwise_correlations(
    metrics: Mapping[str, object],
) -> MeasurableGraph:
    """Every pair of scalar measurable series → one undirected
    edge in a `Graph[str, Correlation]`.

    `metrics` keys become graph nodes; values must be 1-D
    array-likes (per-step scalar series). Non-1-D fields are
    silently skipped — only scalar trajectories form the
    statistical graph.

    Pearson is symmetric, so we add ONE edge per unordered pair
    (lexically-first key as source). Callers querying reachability
    treat the graph as undirected; `MeasurableGraph` itself
    inherits directed-multigraph semantics from
    `corroborate.graph.Graph` for consistency."""
    clean: dict[str, npt.NDArray[np.float64]] = {}
    for name, series in metrics.items():
        arr = np.asarray(series)
        if arr.ndim != 1:
            continue
        clean[name] = arr.astype(np.float64)

    g: MeasurableGraph = Graph()
    for name in clean:
        g = g.with_node(name)

    keys = sorted(clean.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            r = _safe_pearson(clean[k1], clean[k2])
            g = g.with_edge(k1, k2, Correlation(r=r))
    return g


def correlation_matrix_table(
    g: MeasurableGraph, threshold: float = 0.0,
) -> list[str]:
    """Render the graph's edges as text lines sorted by |r|
    descending. `threshold` filters out |r| below the cutoff and
    NaN entries."""
    rows: list[str] = []
    finite_edges = [
        e for e in g.edges
        if np.isfinite(e.metadata.r) and abs(e.metadata.r) >= threshold
    ]
    finite_edges.sort(key=lambda e: -abs(e.metadata.r))
    for e in finite_edges:
        rows.append(
            f'  {e.source:24s} <-> {e.target:24s}  r = {e.metadata.r:+.3f}'
        )
    return rows


def explained_by_claim_graph(
    a: str, b: str, claim_graph: ComputationGraph,
) -> bool:
    """True iff `a` and `b` are connected (either direction) in
    `claim_graph`'s undirected projection.

    Distinguishes:
    - **Explained edges** — corr(a, b) and a↔b reachable in the
      claim graph. The mechanistic skeleton predicts the
      statistical edge; no surprise.
    - **Unexplained edges** — corr(a, b) but no claim path. Either
      mediated through an unmodeled measurable, or spurious. Both
      are diagnostically interesting: the framework's recommendation
      is to either extend the claim graph (declare the mediator)
      or treat the correlation as spurious until shown otherwise.

    Uses undirected reachability — the @claim Protocol expresses
    one-step dataflow in one direction, but for the
    "statistical-edge-explained-by-mechanism" question, either
    direction of dataflow connects the two measurables."""
    if a == b:
        return a in claim_graph.nodes
    if a not in claim_graph.nodes or b not in claim_graph.nodes:
        return False
    seen = {a}
    frontier = [a]
    while frontier:
        cur = frontier.pop()
        if cur == b:
            return True
        # Both directions for undirected reachability.
        neighbours = (
            claim_graph.successors(cur) | claim_graph.predecessors(cur)
        )
        for nxt in neighbours:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return False


class DiffCategory(Enum):
    """Why a discovered edge is or isn't in the claim graph.

    `EXPLAINED` — the column→claim-node bridge maps both endpoints
      to claim graph node sets, and at least one cross-pair is
      reachable via `explained_by_claim_graph`. The mechanism
      skeleton predicts the statistical edge.

    `UNREACHABLE_IN_CLAIM` — both endpoints map to non-empty
      claim node sets but no cross-pair is reachable. The
      claim graph asserts independence; PC says otherwise.
      Strongest missing-edge signal: the authored mechanism is
      genuinely incomplete here.

    `UNMAPPED_OUTCOME` — at least one endpoint is an outcome
      aggregate with no claim-node projection. Outcomes are
      Measurables (observation reductions), not Claims —
      mechanism-vs-observation is a framework-design separation.
      The implementation is expected to project a Measurable to its
      claim-graph entry points via `transitive_reads → claims
      that write those keys`. Until the implementation supplies that
      projection, the diff cannot bridge namespaces.

    `UNMAPPED_DERIVED` — at least one endpoint is a derived
      cross-leaf feature (e.g. `effective_horizon = γ × bf`)
      computed at analysis time, not produced by a claim's
      record_call. Derived features could be promoted to typed
      reductions in the claim graph; until then they live
      outside it.

    `UNMAPPED_OTHER` — both endpoints are unmapped and not
      flagged above; typically leaf↔leaf cross-correlations
      arising from corpus HP-by-env design choices, not a
      framework gap."""
    EXPLAINED = 'explained'
    UNMAPPED_OUTCOME = 'unmapped_outcome'
    UNMAPPED_DERIVED = 'unmapped_derived'
    UNMAPPED_OTHER = 'unmapped_other'
    UNREACHABLE_IN_CLAIM = 'unreachable_in_claim'


@dataclass(frozen=True, slots=True)
class DiffEdge:
    """One discovered edge categorized against a claim graph."""
    a: str
    b: str
    category: DiffCategory
    reason: str


@dataclass(frozen=True, slots=True)
class ColumnRole:
    """Substrate-supplied bridge between PC column names and the
    per-step claim graph + role taxonomy.

    `claim_nodes` — set of claim graph entry points this column
      depends on. For a column representing a single claim's
      output, this is `frozenset({claim_name})`. For a column
      representing a Measurable observation/reduction, this is
      the set of claim graph nodes whose `record_call` outputs
      contribute to the measurable's `transitive_reads` — i.e.
      the implementation's projection of "which claims feed this
      measurable" via the trace. Empty when the column doesn't
      map to the claim graph (leaves, structural HPs).
    `role` — one of `'outcome'`, `'derived'`, `'leaf'`, `'claim'`,
      `'measurable'`. Drives which `DiffCategory` an unmapped or
      unreachable edge falls into.

    Why a SET of claim nodes, not a single one: Measurables are
    observation reductions — by design separate from mechanism
    claims (the framework's typed×open shape). A measurable's
    bridge into the claim graph is the trace keys it reads,
    which can come from multiple claim nodes. The diff tool asks
    "is any pair (a' ∈ a.claim_nodes, b' ∈ b.claim_nodes)
    connected?" — `EXPLAINED` iff at least one pair is reachable.
    This keeps the framework's mechanism-vs-observation separation
    intact while letting the diff tool bridge namespaces."""
    claim_nodes: frozenset[str]
    role: str


def diff_against_claim_graph(
    edges: Iterable[tuple[str, str]],
    claim_graph: ComputationGraph,
    column_roles: Mapping[str, ColumnRole],
) -> tuple[DiffEdge, ...]:
    """Categorize each PC-discovered edge against the per-step
    claim graph using the implementation's column→claim-node bridge.

    Counterpart to `explained_by_claim_graph`: takes a SET of
    edges + a typed bridge, returns the diff partitioned by
    `DiffCategory`. The `UNMAPPED_OUTCOME` category surfaces the
    training-loop integration axis specifically — these are
    edges the per-step claim graph cannot represent by
    construction. A framework extension that surfaces the loop
    axis as a typed structural element would convert these to
    `EXPLAINED`.

    `column_roles` is the implementation's bridge mapping; typically
    supplied by the substrate-side module that authored the
    columns.

    Edges are normalized to `(min, max)` ordering."""
    out: list[DiffEdge] = []
    for raw_a, raw_b in edges:
        a, b = sorted((raw_a, raw_b))
        ra = column_roles.get(a)
        rb = column_roles.get(b)
        nodes_a = ra.claim_nodes if ra else frozenset()
        nodes_b = rb.claim_nodes if rb else frozenset()
        if nodes_a and nodes_b:
            # Any-pair reachability: a column may project to
            # multiple claim graph entry points (measurables read
            # from multiple trace keys). Edge is EXPLAINED iff at
            # least one cross-pair is connected.
            connected_pair: tuple[str, str] | None = None
            for na in nodes_a:
                for nb in nodes_b:
                    if explained_by_claim_graph(na, nb, claim_graph):
                        connected_pair = (na, nb)
                        break
                if connected_pair is not None:
                    break
            if connected_pair is not None:
                na, nb = connected_pair
                out.append(DiffEdge(
                    a=a, b=b, category=DiffCategory.EXPLAINED,
                    reason=f'{na} ↔ {nb} reachable',
                ))
            else:
                out.append(DiffEdge(
                    a=a, b=b,
                    category=DiffCategory.UNREACHABLE_IN_CLAIM,
                    reason=f'projections {sorted(nodes_a)} and '
                           f'{sorted(nodes_b)} both populated but '
                           f'no cross-pair reachable',
                ))
            continue
        # At least one side has no claim-graph projection.
        roles = {ra.role if ra else 'unknown',
                 rb.role if rb else 'unknown'}
        if 'outcome' in roles:
            out.append(DiffEdge(
                a=a, b=b, category=DiffCategory.UNMAPPED_OUTCOME,
                reason='outcome aggregate has no measurable→claim '
                       'projection (substrate must supply '
                       'transitive_reads → claim mapping)',
            ))
        elif 'derived' in roles:
            out.append(DiffEdge(
                a=a, b=b, category=DiffCategory.UNMAPPED_DERIVED,
                reason='derived cross-leaf feature',
            ))
        else:
            out.append(DiffEdge(
                a=a, b=b, category=DiffCategory.UNMAPPED_OTHER,
                reason=f'roles={sorted(roles)}',
            ))
    return tuple(out)


__all__ = [
    'ColumnRole',
    'Correlation',
    'DiffCategory',
    'DiffEdge',
    'correlation_matrix_table',
    'diff_against_claim_graph',
    'explained_by_claim_graph',
    'pairwise_correlations',
]
