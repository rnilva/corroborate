"""Generic directed labeled multigraph primitive.

Multigraph = same `(source, target)` pair may carry multiple edges
with distinct metadata. This is load-bearing for the framework's
graph layer: the same pair of measurables can carry an associational
correlation edge AND an interventional bridge edge with different
tier / evidentiary level — multi-edges are how the graph stratifies
evidence.

Two specialisations live alongside this module:

- `computation_graph.ComputationGraph`: nodes are Claim names, edges
  are `ComputationEdge`s with the reader's kwarg name and source
  output path. Derived from `trace_context()` records.
- (later) `causal_graph.CausalGraph`: nodes are measurable names,
  edges are bridge-derived `BridgeEdge`s with Pearl-ladder tier.

Both inherit (multi-edge aware where it matters):
- Reachability (`reachable(s, t)`, `successors(n)`, `predecessors(n)`).
- Path enumeration (`first_path(s, t)`).
- Multi-edge access (`edges_between(s, t)` returns ALL edges).
- Subgraph (`subgraph(keep_nodes)`).
- Pretty-print (`to_tree()`).
- Structural diff (`diff(other)` — multiset symmetric difference of
  edges, plus node-set differences).

Frozen dataclass for immutability — graph mutations return new
instances. For corroborate-scale toy graphs (≤30 nodes, ≤50 edges)
the tuple-rebuilding cost is negligible."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, replace


@dataclass(frozen=True, slots=True)
class Edge[N, M]:
    """One directed edge: (source, target, metadata). Metadata is
    typed at the graph level (`Graph[N, M]`). Multiple edges with
    the same source+target may exist — multigraph semantics."""
    source: N
    target: N
    metadata: M


@dataclass(frozen=True)
class Graph[N, M]:
    """Directed labeled multigraph. Nodes form a frozenset; edges
    form a tuple. Both immutable. Modifications return new Graph
    instances. Multiple edges between the same `(source, target)`
    pair are first-class (multigraph semantics)."""
    nodes: frozenset[N] = field(default_factory=frozenset)
    edges: tuple[Edge[N, M], ...] = ()

    # ============ Construction ============

    def with_node(self, n: N) -> Graph[N, M]:
        if n in self.nodes:
            return self
        return replace(self, nodes=self.nodes | {n})

    def with_edge(
        self, source: N, target: N, metadata: M,
    ) -> Graph[N, M]:
        """Add an edge. Does NOT dedupe against existing identical
        edges — multigraph semantics. If the caller wants to avoid
        duplicates, they check `edges_between(source, target)` first."""
        return replace(
            self,
            nodes=self.nodes | {source, target},
            edges=self.edges + (
                Edge(source=source, target=target, metadata=metadata),
            ),
        )

    # ============ Adjacency ============

    def successors(self, n: N) -> set[N]:
        """Nodes reachable in one step. Multi-edges to the same
        successor count once (set semantics — adjacency is binary)."""
        return {e.target for e in self.edges if e.source == n}

    def predecessors(self, n: N) -> set[N]:
        return {e.source for e in self.edges if e.target == n}

    def edges_from(self, n: N) -> tuple[Edge[N, M], ...]:
        """All outgoing edges, including multi-edges to the same
        target."""
        return tuple(e for e in self.edges if e.source == n)

    def edges_into(self, n: N) -> tuple[Edge[N, M], ...]:
        """All incoming edges, including multi-edges from the same
        source."""
        return tuple(e for e in self.edges if e.target == n)

    def edges_between(
        self, source: N, target: N,
    ) -> tuple[Edge[N, M], ...]:
        """All edges with this exact source/target pair. Returns
        more than one entry under multigraph semantics — useful for
        reading evidence stratified at a pair (associational +
        interventional + refuter, etc.)."""
        return tuple(
            e for e in self.edges
            if e.source == source and e.target == target
        )

    # ============ Reachability + paths ============

    def reachable(self, source: N, target: N) -> bool:
        if source == target:
            return source in self.nodes
        if source not in self.nodes or target not in self.nodes:
            return False
        seen = {source}
        frontier = [source]
        while frontier:
            cur = frontier.pop()
            for nxt in self.successors(cur):
                if nxt == target:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return False

    def first_path(
        self, source: N, target: N,
    ) -> tuple[Edge[N, M], ...] | None:
        """Depth-first path; returns the first edge sequence found,
        or None if unreachable. Empty tuple when source == target.
        Under multi-edge semantics, picks an arbitrary edge from
        each multi-set traversed; callers needing all paths should
        walk `edges_between` themselves."""
        if source == target:
            return () if source in self.nodes else None
        if source not in self.nodes or target not in self.nodes:
            return None
        stack: list[tuple[N, tuple[Edge[N, M], ...]]] = [(source, ())]
        seen = {source}
        while stack:
            cur, path = stack.pop()
            for e in self.edges_from(cur):
                new_path = path + (e,)
                if e.target == target:
                    return new_path
                if e.target not in seen:
                    seen.add(e.target)
                    stack.append((e.target, new_path))
        return None

    # ============ Subgraph + diff ============

    def subgraph(self, keep_nodes: Iterable[N]) -> Graph[N, M]:
        keep = frozenset(keep_nodes)
        kept_edges = tuple(
            e for e in self.edges
            if e.source in keep and e.target in keep
        )
        return Graph(nodes=keep & self.nodes, edges=kept_edges)

    def diff(self, other: Graph[N, M]) -> GraphDiff[N, M]:
        """Multiset symmetric difference of edges (preserves
        multigraph semantics) plus node-set differences.

        For each unique `Edge(source, target, metadata)` value,
        count occurrences in self vs other; the difference goes to
        `edges_only_in_self` or `edges_only_in_other` with its
        multiplicity. An edge matched 1-1 across both graphs is
        gone from the diff."""
        self_counter: Counter[Edge[N, M]] = Counter(self.edges)
        other_counter: Counter[Edge[N, M]] = Counter(other.edges)

        only_self_counter = self_counter - other_counter
        only_other_counter = other_counter - self_counter

        only_self = tuple(only_self_counter.elements())
        only_other = tuple(only_other_counter.elements())

        nodes_only_self = self.nodes - other.nodes
        nodes_only_other = other.nodes - self.nodes

        return GraphDiff(
            edges_only_in_self=only_self,
            edges_only_in_other=only_other,
            nodes_only_in_self=nodes_only_self,
            nodes_only_in_other=nodes_only_other,
        )

    # ============ Pretty-print ============

    def to_tree(self, root: N | None = None) -> str:
        """Sources-first BFS render. If `root` is given, walk from
        that node; otherwise list components by source-roots."""
        if not self.nodes:
            return '(empty graph)'
        lines: list[str] = []
        seen: set[N] = set()
        queue: list[tuple[N, int]] = []
        if root is not None:
            queue.append((root, 0))
        for n in self.nodes:
            if not self.predecessors(n) and n not in seen:
                queue.append((n, 0))
        while queue:
            cur, depth = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            indent = '  ' * depth
            lines.append(f'{indent}{cur}')
            for e in self.edges_from(cur):
                lines.append(
                    f'{indent}  → {e.target} [{e.metadata}]',
                )
                if e.target not in seen:
                    queue.append((e.target, depth + 1))
        return '\n'.join(lines)


@dataclass(frozen=True, slots=True)
class GraphDiff[N, M]:
    """Result of `Graph.diff(other)`. Edges-only-in-self and
    edges-only-in-other are TUPLES (not sets) — they preserve the
    multiplicity of multi-edge differences."""
    edges_only_in_self: tuple[Edge[N, M], ...]
    edges_only_in_other: tuple[Edge[N, M], ...]
    nodes_only_in_self: frozenset[N]
    nodes_only_in_other: frozenset[N]

    def is_empty(self) -> bool:
        return not (
            self.edges_only_in_self or self.edges_only_in_other
            or self.nodes_only_in_self or self.nodes_only_in_other
        )

    def conflicting_pairs(self) -> dict[
        tuple[N, N],
        tuple[tuple[Edge[N, M], ...], tuple[Edge[N, M], ...]],
    ]:
        """Pairs `(source, target)` where both diff sides have some
        edge — i.e. self and other agree the pair exists but
        disagree on the multi-edge content. Useful for reporting
        evidence-stratification mismatches across cycles."""
        self_by_pair: dict[tuple[N, N], list[Edge[N, M]]] = {}
        for e in self.edges_only_in_self:
            self_by_pair.setdefault(
                (e.source, e.target), [],
            ).append(e)
        other_by_pair: dict[tuple[N, N], list[Edge[N, M]]] = {}
        for e in self.edges_only_in_other:
            other_by_pair.setdefault(
                (e.source, e.target), [],
            ).append(e)

        conflicts: dict[
            tuple[N, N],
            tuple[tuple[Edge[N, M], ...], tuple[Edge[N, M], ...]],
        ] = {}
        for pair in set(self_by_pair) & set(other_by_pair):
            conflicts[pair] = (
                tuple(self_by_pair[pair]),
                tuple(other_by_pair[pair]),
            )
        return conflicts
