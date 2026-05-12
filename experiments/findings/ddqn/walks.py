"""Graph walks as executable claim queries — demo.

This is an EXPERIMENT, not a framework primitive. It builds a
post-evaluated `CausalGraph` from `BRIDGES` + the snapshot's per-
bridge `extent_hash` values, then walks the graph to report on
cluster-shaped claims.

Cluster identity = `(source, target, extent_hash)`. Two bridges
that admitted identical cell-sets on the cache that produced the
snapshot share a cluster — the framework derives cluster identity
empirically, no author labels needed. The refutation-cluster
pattern (≥2 INTERVENTIONAL HELDs on the same cluster) is a
structural query, not a baked-in graph mutation.

If this pattern earns its keep across two or three more findings
notes, the `evaluated_graph` + `clusters_by_extent` helpers
graduate into `corroborate.graph.causal`. Today they live here
so authors can copy + adapt without committing the framework yet."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from enum import Enum
from pathlib import Path

from corroborate.bridge.bridge import Bridge
from corroborate.bridge.verdict import Verdict
from corroborate.graph.causal import (
    BridgeEdge, CausalGraph, EvidentiaryLevel, Tier, authored_graph,
)
from corroborate.graph.graph import Edge


# ============ Snapshot loader ============


def load_post_eval(
    run_json: Path,
) -> Mapping[str, tuple[Verdict, int]]:
    """Read a `*.run.json` snapshot, return
    `{bridge_name: (Verdict, extent_hash)}`."""
    d = json.loads(run_json.read_text())
    return {
        b['bridge_name']: (Verdict(b['verdict']), int(b['extent_hash']))
        for b in d['bridges']
    }


# ============ Evidentiary-level stamper ============


def _stamp_level(meta: BridgeEdge, verdict: Verdict) -> EvidentiaryLevel:
    """Map (verdict, tier) → evidentiary_level. Single source of
    truth for "what does this edge mean post-evaluation"."""
    if verdict == Verdict.HELD:
        if meta.tier == Tier.INTERVENTIONAL:
            return 'causal_one_sided'
        return 'correlational'  # ASSOCIATIONAL and INVARIANT admits
    if verdict == Verdict.NO_EFFECT:
        return 'refuted'
    return 'unevaluated'  # POWER_INSUFFICIENT / INVARIANT_VIOLATION / INADMISSIBLE


def evaluated_graph(
    bridges: tuple[Bridge, ...],
    post_eval: Mapping[str, tuple[Verdict, int]],
) -> CausalGraph:
    """Build the authored graph then stamp each edge's
    `evidentiary_level` AND `extent_hash` from the snapshot's
    per-bridge `(Verdict, extent_hash)` tuple. Bridges absent
    from `post_eval` stay `'unevaluated'` with `extent_hash=0`."""
    g = authored_graph(bridges)
    new_edges: list[Edge[str, BridgeEdge]] = []
    for e in g.edges:
        pe = post_eval.get(e.metadata.bridge_name)
        if pe is None:
            new_edges.append(e)
            continue
        verdict, ehash = pe
        new_meta = replace(
            e.metadata,
            evidentiary_level=_stamp_level(e.metadata, verdict),
            extent_hash=ehash,
        )
        new_edges.append(replace(e, metadata=new_meta))
    return replace(g, edges=tuple(new_edges))


# ============ Cluster discovery + verdict ============


class ClusterVerdict(Enum):
    """Verdict on a refutation cluster (multi-bridge edge group).
    Mirrors the framework's three-verdict-not-binary discipline at
    the cluster level."""
    SUPPORTED = 'supported'
    REFUTED = 'refuted'
    UNDERPOWERED = 'underpowered'
    EMPTY_EXTENT = 'empty_extent'


_EMPTY_EXTENT_HASH = hash(frozenset[str]())


def clusters_by_extent(
    g: CausalGraph,
) -> dict[tuple[str, str, int], tuple[BridgeEdge, ...]]:
    """Group every edge in `g` by `(source, target, extent_hash)`.
    Multi-edge groups are refutation clusters; singletons are
    standalone bridges on that edge under that extent. Empty-extent
    edges all share `hash(frozenset())` per cluster identity."""
    buckets: dict[tuple[str, str, int], list[BridgeEdge]] = {}
    for e in g.edges:
        key = (e.source, e.target, e.metadata.extent_hash)
        buckets.setdefault(key, []).append(e.metadata)
    return {k: tuple(v) for k, v in buckets.items()}


def cluster_verdict(members: tuple[BridgeEdge, ...]) -> ClusterVerdict:
    """Compose member edges' evidentiary_level into a cluster
    verdict. Empty-extent is its own bucket (corpus can't test).
    REFUTED if any member refutes. SUPPORTED if every member
    admits (non-empty extent). Otherwise UNDERPOWERED."""
    if not members:
        return ClusterVerdict.UNDERPOWERED
    if all(m.extent_hash == _EMPTY_EXTENT_HASH for m in members):
        return ClusterVerdict.EMPTY_EXTENT
    levels = {m.evidentiary_level for m in members}
    if 'refuted' in levels:
        return ClusterVerdict.REFUTED
    if levels.issubset({'correlational', 'causal_one_sided'}):
        return ClusterVerdict.SUPPORTED
    return ClusterVerdict.UNDERPOWERED


# ============ Demo runner ============


def _main() -> None:
    from experiments.findings import ddqn
    repo_root = Path(__file__).resolve().parents[3]
    post_eval = load_post_eval(repo_root / 'experiments/findings/ddqn.run.json')
    g = evaluated_graph(ddqn.BRIDGES, post_eval)

    clusters = clusters_by_extent(g)
    multi = {k: v for k, v in clusters.items() if len(v) >= 2}
    print(f'Evaluated graph: {len(g.nodes)} nodes, {len(g.edges)} edges, '
          f'{len(multi)} multi-bridge clusters.')
    print()
    print('Cluster verdicts:')
    for (src, tgt, _), members in sorted(
        multi.items(), key=lambda x: (-len(x[1]), x[0][:2]),
    ):
        v = cluster_verdict(members)
        src_short = src if len(src) < 50 else src[:47] + '...'
        print(f'  {v.value:14s}  ({len(members)} bridges)  {src_short} → {tgt}')
        for m in members:
            print(f'      {m.bridge_name}')


if __name__ == '__main__':
    _main()
