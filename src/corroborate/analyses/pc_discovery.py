"""`pc_discovery` — @analysis wrapper around conservative-PC
adjacency discovery + Meek/v-structure orientation.

Conservative-PC (Ramsey-Zhang-Spirtes 2006) on a cell-keyed
joint distribution: discovers the CPDAG over named variables,
returns surviving skeleton + directed/undirected/ambiguous
edges + per-edge separating sets. The framework's CI primitive
(`graph.discovery.partial_spearman_rho`) drives the test.

Use when:
- multiple candidate mediators are in play, individual partial-r
  bridges return UNDERPOWERED, and a graph-shape verdict is
  more useful than per-edge p-values.
- you want a *structural* refutation: "candidate X is NOT a
  mediator" is `arm ⫫ X` (marginal) or `arm ⫫ outcome | X` (X
  is in the screening set with adequate power).
- you need to surface a non-orientable collinear cluster
  rather than guess one mediator from many lockstep movers.

Bridges declare the parameter as `pc_discovery: PCDiscoveryResult`
and override `nodes`, `alpha`, `max_conditioning`, `stratify_by`
via kwargs. The result type carries helper methods for the
common queries:
- `is_in_skeleton(a, b)` — adjacency check
- `is_marginally_independent(a, b)` — `{}` in separating sets
- `separated_by(a, b)` — frozenset of all minimal sepsets
- `is_directed(a, b)` — oriented edge present

The cells iterable is projected to a polars DataFrame at the
analysis boundary (`_cells_to_polars_dataframe`); non-finite
values drop the cell. Stratify_by (e.g. `env_name`) becomes the
JCI context node — NOT included in the graph, but the
within-stratum CI tests are pooled via Fisher z.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import polars as pl

from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import (
    DiscoveredAdjacency,
    OrientedAdjacency,
    discover_adjacency,
    orient_adjacency,
)


@dataclass(frozen=True, slots=True)
class PCDiscoveryResult:
    """Conservative-PC CPDAG + diagnostics over named variables.

    `n_cells` is the row count after non-finite filtering. Bridges
    should check `n_cells >= min_cells` before consuming the
    graph — small samples produce sparse skeletons that look like
    refutations but are actually underpowered."""
    variables: tuple[str, ...]
    n_cells: int
    alpha: float
    max_conditioning: int
    stratify_by: str | None
    skeleton: DiscoveredAdjacency
    oriented: OrientedAdjacency

    def is_in_skeleton(self, a: str, b: str) -> bool:
        """True if `{a, b}` is an edge in the discovered skeleton."""
        return frozenset({a, b}) in self.skeleton.edges

    def is_marginally_independent(self, a: str, b: str) -> bool:
        """True if PC found `a ⫫ b` unconditionally — `{}` is in
        the separating sets for `{a, b}`. Surfaces "X is not
        arm-affected" or "Y is unrelated to outcome" verdicts."""
        seps = self.skeleton.separating_sets.get(
            frozenset({a, b}), frozenset(),
        )
        return frozenset() in seps

    def separated_by(
        self, a: str, b: str,
    ) -> frozenset[frozenset[str]]:
        """All Z that PC found to satisfy `a ⫫ b | Z`. Empty
        frozenset is returned for edges in the skeleton (no Z
        screens them at the tested conditioning depth)."""
        return self.skeleton.separating_sets.get(
            frozenset({a, b}), frozenset(),
        )

    def has_separating_set_containing(
        self, a: str, b: str, *required: str,
    ) -> bool:
        """True if SOME sepset Z of `a ⫫ b | Z` contains all of
        `required`. Asks "does {required} (alone or with extras)
        screen a from b?". Use to ratify mediator candidates:
        `arm ⫫ outcome | {jens}` HELDs as evidence that jens is
        sufficient to mediate."""
        target = frozenset(required)
        for z in self.separated_by(a, b):
            if target <= z:
                return True
        return False

    def is_directed(self, source: str, target: str) -> bool:
        """True if the CPDAG has the oriented edge `source →
        target`."""
        return (source, target) in self.oriented.directed_edges

    def is_undirected_adjacent(self, a: str, b: str) -> bool:
        """True if `{a, b}` is an undirected (non-orientable)
        edge in the CPDAG. Surfaces lockstep collinear pairs PC
        couldn't direct from data."""
        return frozenset({a, b}) in self.oriented.undirected_edges


def _cells_to_polars(
    cells: Iterable[Mapping[str, object]],
    keys: tuple[str, ...],
    *,
    stratify_by: str | None,
    indicators: Mapping[str, tuple[str, str]] = {},
) -> pl.DataFrame:
    """Project cells → polars DataFrame with one column per key
    plus the stratifier (if any). Cells with any non-finite key
    are dropped — closed-form CI tests need every row to carry
    every variable.

    `bool` is coerced to float (True → 1.0); ints stay ints
    until polars columns coerce them. Non-numeric keys (str, None)
    drop the cell."""
    cols_needed = list(keys)
    if stratify_by is not None and stratify_by not in cols_needed:
        cols_needed.append(stratify_by)
    indicator_outputs = set(indicators.keys())
    rows: list[dict[str, object]] = []
    for cell in cells:
        row: dict[str, object] = {}
        complete = True
        for k in cols_needed:
            # Synthetic indicator column derived from a string-typed
            # source field (e.g. arm_key → arm_ddqn_indicator).
            if k in indicator_outputs:
                source_key, match_value = indicators[k]
                src = cell.get(source_key)
                if src is None:
                    complete = False
                    break
                row[k] = 1.0 if src == match_value else 0.0
                continue
            v = cell.get(k)
            if k == stratify_by:
                if v is None:
                    complete = False
                    break
                row[k] = v
                continue
            if isinstance(v, bool):
                row[k] = float(v)
            elif isinstance(v, (int, float)):
                f = float(v)
                if math.isnan(f) or math.isinf(f):
                    complete = False
                    break
                row[k] = f
            else:
                complete = False
                break
        if complete:
            rows.append(row)
    if not rows:
        # Empty df with schema — polars accepts list[dict] but
        # crashes on empty list, so build explicitly.
        schema: dict[str, pl.DataType | type[pl.DataType]] = {
            k: pl.Float64 for k in keys
        }
        if stratify_by is not None:
            schema[stratify_by] = pl.Utf8
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows)


@analysis
def pc_discovery(
    cells: Iterable[Mapping[str, object]],
    *,
    nodes: tuple[str, ...],
    alpha: float = 0.05,
    max_conditioning: int = 2,
    conservative: bool = True,
    stratify_by: str | None = None,
    indicators: Mapping[str, tuple[str, str]] = {},
) -> PCDiscoveryResult:
    """Run conservative-PC on the named `nodes` over the cell
    collection.

    `nodes`: ordered tuple of column names to include as graph
    variables. Each name must be a scalar key resolvable on each
    cell (registered measurable name or flat record key). Order
    doesn't affect surviving adjacency under conservative-PC.

    `alpha`: significance level. Edge survives iff every tested
    conditioning set yields `p < alpha`.

    `max_conditioning`: cap on |Z| for CI tests. 0 = marginal
    only; 1 = single-Z partial; ≥2 multi-Z residual regression.

    `conservative`: if True (default), v-structure orientation
    requires the candidate collider Z to be in *no* separating
    set of (X, Y); otherwise Z's collider status is marked
    ambiguous. If False, orient on the first sepset (vanilla PC).

    `stratify_by`: JCI categorical context (e.g. `env_name`).
    Within-stratum CI tests Fisher-z pooled; the stratifier is
    NOT a graph node.

    `indicators`: optional mapping `{output_name: (source_key,
    match_value)}` that synthesises 0/1 indicator columns from
    string-valued cell fields. Use for arm encoding:
    `{'arm_ddqn_indicator': ('arm_key', DDQN_ARM)}` adds a float
    column to each cell so PC can treat the categorical arm as a
    graph node.

    Returns `PCDiscoveryResult` carrying the skeleton, oriented
    CPDAG, and cell count post-finite-filtering."""
    df = _cells_to_polars(
        cells, nodes, stratify_by=stratify_by, indicators=indicators,
    )
    skeleton = discover_adjacency(
        df,
        variables=nodes,
        alpha=alpha,
        max_conditioning=max_conditioning,
        stratify_by=stratify_by,
    )
    oriented = orient_adjacency(skeleton, conservative=conservative)
    return PCDiscoveryResult(
        variables=tuple(nodes),
        n_cells=df.height,
        alpha=alpha,
        max_conditioning=max_conditioning,
        stratify_by=stratify_by,
        skeleton=skeleton,
        oriented=oriented,
    )


__all__ = [
    'PCDiscoveryResult',
    'pc_discovery',
]
