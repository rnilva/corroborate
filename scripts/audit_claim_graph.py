"""Audit the authored claim-graph topology per hypothesis module.

Wires the unconsumed `corroborate.graph.causal.authored_graph(BRIDGES)`
into a concrete consumer (round 1 of `UNCONSUMED_PRIMITIVES_AUDIT.md`).

For each module exporting a `BRIDGES` tuple, materialises the topology
WITHOUT running bridges (so no cache load, no fixture evaluation), then
reports:

- node / edge counts
- multi-edge pairs (same source + target with multiple bridges):
  candidate "structural pair" sites — falsification companion, polarity-
  stratified sibling, mean/median aggregator pair, etc. These are the
  bridges the archived primitives manifest tried to type as a new
  relationship enum; the topology already groups them.
- fan-out leaders (sources with many outgoing bridges): central
  intervention contrasts, the do() roots.
- fan-in leaders (targets with many incoming bridges): the most-tested
  outcome nodes.
- tier distribution per module.

Run: `uv run python scripts/audit_claim_graph.py`

No arguments. Discovers hypothesis modules under `experiments/findings/`
automatically (any module with a `BRIDGES` attribute that is a tuple
of `Bridge`).
"""
from __future__ import annotations

import importlib
import pkgutil
from collections import Counter
from typing import cast

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import (
    BridgeEdge,
    CausalGraph,
    Tier,
    authored_graph,
)


def discover_hypothesis_modules() -> list[tuple[str, tuple[Bridge, ...]]]:
    """Return (module_name, BRIDGES) for every module under
    `experiments.findings` that exports a `BRIDGES` tuple of `Bridge`."""
    import experiments.findings  # type: ignore[import-not-found]

    hits: list[tuple[str, tuple[Bridge, ...]]] = []
    pkg = experiments.findings
    for info in pkgutil.iter_modules(pkg.__path__, prefix=pkg.__name__ + '.'):
        if info.ispkg:
            continue
        try:
            mod = importlib.import_module(info.name)
        except Exception as exc:
            print(f'  skip {info.name}: import error ({type(exc).__name__})')
            continue
        bridges = getattr(mod, 'BRIDGES', None)
        if not isinstance(bridges, tuple) or not bridges:
            continue
        if not all(isinstance(b, Bridge) for b in bridges):
            continue
        hits.append((info.name, cast(tuple[Bridge, ...], bridges)))
    return hits


def report_multi_edges(g: CausalGraph) -> None:
    """Pairs with ≥2 edges — candidate structural pairs."""
    by_pair: dict[tuple[str, str], list[BridgeEdge]] = {}
    for e in g.edges:
        by_pair.setdefault((e.source, e.target), []).append(e.metadata)
    multi = {p: edges for p, edges in by_pair.items() if len(edges) >= 2}
    if not multi:
        print('  multi-edge pairs: none')
        return
    print(f'  multi-edge pairs: {len(multi)} (candidate structural pairs)')
    for (src, tgt), edges in sorted(multi.items(), key=lambda x: -len(x[1])):
        names = [m.bridge_name for m in edges]
        tiers = Counter(m.tier.name for m in edges)
        tier_str = ', '.join(f'{n}×{t.lower()}' for t, n in tiers.items())
        src_short = src if len(src) <= 60 else src[:57] + '...'
        print(f'    {src_short:<60s} → {tgt}  ({tier_str})')
        for n in names:
            print(f'      · {n}')


def report_fan(g: CausalGraph) -> None:
    """Fan-out (per source) + fan-in (per target) distributions."""
    out_deg: Counter[str] = Counter()
    in_deg: Counter[str] = Counter()
    for e in g.edges:
        out_deg[e.source] += 1
        in_deg[e.target] += 1
    print('  fan-out leaders (intervention sources with most bridges):')
    for src, n in out_deg.most_common(5):
        if n < 2:
            break
        src_short = src if len(src) <= 60 else src[:57] + '...'
        print(f'    {n:>3}  {src_short}')
    print('  fan-in leaders (outcome targets with most bridges):')
    for tgt, n in in_deg.most_common(5):
        if n < 2:
            break
        print(f'    {n:>3}  {tgt}')


def report_tier_distribution(g: CausalGraph) -> None:
    """Edge counts by tier."""
    tiers = Counter(e.metadata.tier.name for e in g.edges)
    parts = [f'{n}×{t.lower()}' for t, n in tiers.most_common()]
    print(f'  tier distribution: {", ".join(parts)}')


def report_isolated_targets(g: CausalGraph) -> None:
    """Targets that never appear as sources (terminal outcome nodes)
    and sources that never appear as targets (root interventions).
    Useful for spotting chain breaks."""
    sources = {e.source for e in g.edges}
    targets = {e.target for e in g.edges}
    only_source = sorted(sources - targets)
    only_target = sorted(targets - sources)
    if only_source:
        print(f'  root nodes (sourced only, never targeted): {len(only_source)}')
        for n in only_source[:5]:
            n_short = n if len(n) <= 60 else n[:57] + '...'
            print(f'    {n_short}')
        if len(only_source) > 5:
            print(f'    ... ({len(only_source) - 5} more)')
    if only_target:
        print(f'  terminal nodes (targeted only, never sourced): {len(only_target)}')
        for n in only_target[:5]:
            print(f'    {n}')
        if len(only_target) > 5:
            print(f'    ... ({len(only_target) - 5} more)')


def audit_module(module_name: str, bridges: tuple[Bridge, ...]) -> None:
    """Build the authored graph and emit the report."""
    g = authored_graph(bridges)
    short_name = module_name.split('.')[-1]
    print()
    print(f'=== {short_name} ===')
    print(
        f'  bridges: {len(bridges)};  graph nodes: {len(g.nodes)};  '
        f'edges: {len(g.edges)}'
    )
    report_tier_distribution(g)
    report_multi_edges(g)
    report_fan(g)
    report_isolated_targets(g)


def main() -> None:
    modules = discover_hypothesis_modules()
    if not modules:
        print('No hypothesis modules with BRIDGES discovered.')
        return
    print(f'Discovered {len(modules)} hypothesis module(s) with BRIDGES.')
    for name, bridges in modules:
        audit_module(name, bridges)


if __name__ == '__main__':
    main()
