"""Phase E smoke — typed Hypothesis subgraph verdict on DDQN
corpus.

Demonstrates the typed-edge pipeline end-to-end:

1. Author a `Hypothesis` with `intervention_arms` + `edges` (3
   typed `ClaimedEdge`s: mechanism, outcome, link).
2. `hypothesis_subgraph_verdict` walks the subgraph: per-edge
   BridgeResults via `hypothesis_comparison_from_cells` (mechanism
   / outcome / refuter) or Pearson-r over per-group g (link).
3. The returned `HypothesisVerdict` carries a typed `CausalGraph`
   built directly from the per-edge BridgeResults (Tier-typed
   `BridgeEdge`s with `promote_bridged_evidence` post-pass for
   co-paired admits).

The bridges attached to each edge are stubs in this smoke — the
measurement paths (`mechanism.jensen_gap`,
`outcome.late_window_mean`) are populated by cell_runner's
projections, not by per-cell bridge invocations. The test
happens at the `from_cells` layer reading those measurements,
which is exactly the verdict-walk's dispatch path."""
from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import cast

import polars as pl

from corroborate.bridge import Bridge, BridgeResult, bridge as bridge_decorator
from corroborate.claimed_edge import (
    link_edge,
    mechanism_edge,
    outcome_edge,
)
from corroborate.hypothesis import Hypothesis
from corroborate.hypothesis_verdict import hypothesis_subgraph_verdict
from corroborate.intervention import Intervention
from corroborate.persistence import read_runrows
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.schema import RunRow
from corroborate.verdict import Verdict


# ============ Stub bridges — measurement paths come from substrate ============

def _path_finite_bridge(path: str) -> Bridge[Mapping[str, object]]:
    """Stub bridge that returns HELD iff `record[path]` is a
    finite scalar. Used as the `bridge` field on ClaimedEdges
    when the actual cross-arm test happens at the `from_cells`
    layer (reading the measurement directly from RunRows)."""
    @bridge_decorator(targets=(path,), name=f'path_finite({path})')
    def _b(record: Mapping[str, object]) -> BridgeResult:
        v = record.get(path)
        finite = isinstance(v, (int, float)) and math.isfinite(float(v))
        return BridgeResult(
            verdict=(
                Verdict.HELD if finite else Verdict.POWER_INSUFFICIENT
            ),
            reason='', stats={},
            name=f'path_finite({path})',
            targets=(path,),
        )
    return _b


# ============ Hypothesis construction ============

def _ddqn_hypothesis() -> Hypothesis[Mapping[str, object]]:
    """DDQN hypothesis as a typed 3-edge subgraph claim:

    - mechanism: do(arm) → mechanism.jensen_gap (a_lt_b, INTERVENTIONAL)
    - outcome:   do(arm) → outcome.late_window_mean (a_gt_b, INTERVENTIONAL)
    - link:      mechanism.jensen_gap → outcome.late_window_mean (a_gt_b, ASSOCIATIONAL)
    """
    return Hypothesis(
        name='ddqn',
        intervention={},
        intervention_arms=(
            Intervention(
                slot_path='bootstrap',
                replacement=partial(
                    bootstrap, greedification=double_greedify,
                ),
            ),
        ),
        edges=(
            mechanism_edge(
                target='mechanism.jensen_gap',
                predicted_direction='a_lt_b',
                bridge=_path_finite_bridge('mechanism.jensen_gap'),
            ),
            outcome_edge(
                target='outcome.late_window_mean',
                predicted_direction='a_gt_b',
                bridge=_path_finite_bridge('outcome.late_window_mean'),
            ),
            link_edge(
                source='mechanism.jensen_gap',
                target='outcome.late_window_mean',
                predicted_direction='a_gt_b',
                bridge=_path_finite_bridge('outcome.late_window_mean'),
            ),
        ),
    )


def _vanilla_hypothesis() -> Hypothesis[Mapping[str, object]]:
    return Hypothesis(
        name='vanilla_dqn', intervention={}, intervention_arms=(),
    )


# ============ Smoke entry ============

def main() -> None:
    runs_path = Path(
        '/workspace/corroborate/experiments/data/ddqn/'
        'runs_with_mediators.parquet'
    )
    if not runs_path.exists():
        print(f'[Phase E] corpus not found: {runs_path}')
        sys.exit(0)
    df = pl.read_parquet(runs_path)
    if 'arm_key' not in df.columns:
        print(
            f'[Phase E] {runs_path.name} has no `arm_key` column — '
            f'run `migrate_runs_inject_arm_key.py` first'
        )
        sys.exit(0)

    treatment_h = _ddqn_hypothesis()
    baseline_h = _vanilla_hypothesis()
    treatment_arm_key = treatment_h.arm_key()
    baseline_arm_key = baseline_h.arm_key()

    runs: list[RunRow] = read_runrows(runs_path)
    treatment = [r for r in runs if r.arm_key == treatment_arm_key]
    baseline = [r for r in runs if r.arm_key == baseline_arm_key]

    print(f'[Phase E] read {len(runs)} runs from {runs_path.name}')
    print(f'  treatment ({treatment_arm_key!r}): {len(treatment)}')
    print(f'  baseline ({baseline_arm_key!r}): {len(baseline)}')
    print()

    verdict = hypothesis_subgraph_verdict(
        treatment_h, treatment, baseline,
        pair_by=('seed',),
        group_by='env_name',
        baseline_h=baseline_h,
    )

    pattern = verdict.pattern()
    print(
        f'[Phase E] §3 pattern (mechanism, outcome, link): '
        f'{tuple(v.value for v in pattern)}'
    )
    print()

    for edge in treatment_h.edges:
        br = verdict.bridge_results.get((edge.source, edge.target))
        if br is None:
            continue
        if edge.target in verdict.comparison_rows:
            row = verdict.comparison_rows[edge.target]
            g = row.effect_size_g
            i2 = row.pooled.I2 if row.pooled is not None else float('nan')
            print(
                f'  {edge.role:<10} {edge.source!r:<25} → '
                f'{edge.target!r:<35} '
                f'verdict={br.verdict.value:<22} '
                f'g={g if g is not None else float("nan"):+.3f}  '
                f'I²={i2:.3f}'
            )
        else:
            rho = br.stats.get('rho')
            p = br.stats.get('pvalue')
            n = br.stats.get('n_groups')
            rho_v = float(rho) if isinstance(rho, (int, float)) else float('nan')
            p_v = float(p) if isinstance(p, (int, float)) else float('nan')
            n_v = int(n) if isinstance(n, int) else 0
            print(
                f'  {edge.role:<10} {edge.source!r:<25} → '
                f'{edge.target!r:<35} '
                f'verdict={br.verdict.value:<22} '
                f'r={rho_v:+.3f}  p={p_v:.3f}  n_groups={n_v}'
            )

    print()
    print(f'[Phase E] typed CausalGraph from verdict')
    g = verdict.graph
    print(f'  graph nodes: {sorted(g.nodes)}')
    print(f'  graph edges: {len(g.edges)}')
    for ge in g.edges:
        meta = ge.metadata
        print(
            f'    {ge.source!r} → {ge.target!r}  '
            f'tier={meta.tier.name:<14} '
            f'level={meta.evidentiary_level:<18} '
            f'name={meta.bridge_name}'
        )


if __name__ == '__main__':
    main()
    sys.exit(0)


_ = cast  # quiet unused-import lint when only used in annotations
