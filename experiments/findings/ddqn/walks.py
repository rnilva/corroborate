"""Graph walks as executable claim queries — demo.

The framework primitives `evaluated_graph`, `clusters_by_extent`,
`cluster_verdict`, `ClusterVerdict` realize the HYPOTHESIS_AS_GRAPH
principle in `corroborate.graph.causal`. This module is the
demo: load a snapshot, build the post-evaluated graph, walk it
to report cluster-shaped claims.

Cluster identity = `(source, target, extent_hash)`. Two bridges
that admitted identical cell-sets on the cache that produced the
snapshot share a cluster — the framework derives cluster identity
empirically, no author labels needed."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from corroborate.bridge.verdict import Verdict
from corroborate.graph.causal import (
    cluster_verdict,
    clusters_by_extent,
    evaluated_graph,
)


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
