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

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from corroborate.computation_graph import ComputationGraph
from corroborate.graph import Graph


@dataclass(frozen=True, slots=True)
class Correlation:
    """Pearson r between two scalar series. NaN when either series
    is constant or shorter than 2."""
    r: float

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
    series have mismatched lengths (e.g. train-step vs eval-burst
    cadences in the same record dict)."""
    if len(a) < 2 or len(a) != len(b):
        return float('nan')
    s1 = float(a.std())
    s2 = float(b.std())
    if s1 == 0.0 or s2 == 0.0:
        return float('nan')
    centred = (a - float(a.mean())) * (b - float(b.mean()))
    return float(centred.mean() / (s1 * s2))


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
