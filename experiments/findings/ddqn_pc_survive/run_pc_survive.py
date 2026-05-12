"""PC adjacency discovery on the SURVIVE cohort (per-cell, JCI by env).

Sibling of `ddqn_pc_reach/run_pc_reach.py` — same variable
set, same algorithm, different cohort (positive-polarity envs in
the bounded-Q regime).

Cohort: SURVIVE envs ∩ scope:
  env_reward_polarity > 0.3  (positive polarity)
  ∧ q_divergence_score < 1.0  (Q bounded, no Q-explosion regime)
  ∧ jensen_dormancy_gap < 0.05  (mech premise active)
  ∧ standard config (n_step=1, no action duplication, rs=1.0,
                     no polyak τ).

Q-bounded restriction is load-bearing on SURVIVE: at sync=10k
MinAtar the silent-inversion regime kicks in (`findings_sync_curve_
goldilocks.md`), and at sync ≥ 1k Q-explosion regime makes Δ_jens
flip sign. Within q_div < 1.0 we expect mech HELD ∧ link active.

Per-cell variables — same set as REACH for direct comparison:

  is_ddqn      — binary treatment indicator
  jens         — jensen_gap
  q_div        — q_divergence_score
  argmax_H     — argmax_entropy_late
  stale        — target_staleness_late
  eff_h        — effective_horizon
  outcome      — eval_best_burst_mean

JCI by `env_name`, conservative-PC at depth ≤ 1, α=0.05.
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate analysis registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry
from corroborate.graph.discovery import (
    discover_adjacency, orient_adjacency,
)
from corroborate.runner.runner import (
    _load_directory,  # pyright: ignore[reportPrivateUsage]
    _compute_measurables,  # pyright: ignore[reportPrivateUsage]
)


DDQN_ARM = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASELINE_ARM = 'baseline'

VARS = (
    'is_ddqn',
    'jensen_gap',
    'q_divergence_score',
    'argmax_entropy_late',
    'target_staleness_late',
    'effective_horizon',
    'eval_best_burst_mean',
)
VAR_SHORT = {
    'is_ddqn': 'is_ddqn',
    'jensen_gap': 'jens',
    'q_divergence_score': 'q_div',
    'argmax_entropy_late': 'argmax_H',
    'target_staleness_late': 'stale',
    'effective_horizon': 'eff_h',
    'eval_best_burst_mean': 'outcome',
}


def _scoped_cohort(runs: pl.DataFrame) -> pl.DataFrame:
    """SURVIVE polarity + Q-bounded + mech-active scope."""
    scope = (
        pl.col('arm_key').is_in([DDQN_ARM, BASELINE_ARM])
        & pl.col('env_reward_polarity').is_finite()
        & (pl.col('env_reward_polarity') > 0.3)
        & pl.col('q_divergence_score').is_finite()
        & (pl.col('q_divergence_score') < 1.0)
        & pl.col('jensen_dormancy_gap').is_finite()
        & (pl.col('jensen_dormancy_gap') < 0.05)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    )
    return runs.filter(scope)


def main() -> None:
    required = (
        'jensen_gap', 'q_divergence_score', 'argmax_entropy_late',
        'target_staleness_late', 'effective_horizon',
        'eval_best_burst_mean', 'jensen_dormancy_gap',
        'env_reward_polarity',
    )
    runs = _load_directory(
        _REPO_ROOT / 'experiments' / 'data',
        restore_from_cloud=False, required=required, bridges=(),
    )
    runs = _compute_measurables(runs, required)

    cohort = _scoped_cohort(runs)
    cohort = cohort.with_columns(
        (pl.col('arm_key') == DDQN_ARM).cast(pl.Float64).alias('is_ddqn'),
    )

    available = [v for v in VARS if v in cohort.columns]
    panel = cohort.select([*available, 'env_name']).filter(
        pl.all_horizontal([pl.col(v).is_finite() for v in available])
    )
    print(f'In-scope SURVIVE cells: {len(panel)}')
    print('Env counts:')
    for r in panel.group_by('env_name').len().sort('env_name').iter_rows(named=True):
        print(f'  {r["env_name"]:<25s}: {r["len"]}')
    print(f'\nVariables: {[VAR_SHORT[v] for v in available]}')
    print()

    adj = discover_adjacency(
        panel, variables=available,
        alpha=0.05, max_conditioning=1,
        stratify_by='env_name',
    )

    print(f'=== PC adjacency (conservative, depth ≤ 1, JCI by env) ===\n')
    n_possible = len(available) * (len(available) - 1) // 2
    print(f'variables: {len(adj.variables)}, edges remaining: '
          f'{len(adj.edges)} / {n_possible} possible\n')
    print('Edges retained (no separating Z found):')
    for edge in sorted(adj.edges, key=lambda e: tuple(sorted(e))):
        x, y = sorted(edge)
        print(f'  {VAR_SHORT[x]:>10s} — {VAR_SHORT[y]:<10s}')
    print()
    print('Edges removed (with separating set Z):')
    all_pairs = {frozenset(e) for e in combinations(available, 2)}
    for edge in sorted(all_pairs - adj.edges,
                       key=lambda e: tuple(sorted(e))):
        x, y = sorted(edge)
        sepsets = adj.separating_sets.get(edge, frozenset())
        if not sepsets:
            sepset_str = '(no test)'
        else:
            sepset_str = ', '.join(
                '∅' if not s else '{' + ', '.join(
                    VAR_SHORT[v] for v in sorted(s)
                ) + '}'
                for s in sepsets
            )
        print(f'  {VAR_SHORT[x]:>10s} ⊥ {VAR_SHORT[y]:<10s} | {sepset_str}')

    oriented = orient_adjacency(adj)
    print()
    print('=== CPDAG (v-structure detection + Meek rules) ===\n')
    print(f'directed edges ({len(oriented.directed_edges)}):')
    for src, tgt in sorted(oriented.directed_edges):
        print(f'  {VAR_SHORT[src]:>10s} → {VAR_SHORT[tgt]:<10s}')
    print()
    print(f'undirected edges ({len(oriented.undirected_edges)}):')
    for edge in sorted(oriented.undirected_edges, key=lambda e: tuple(sorted(e))):
        x, y = sorted(edge)
        print(f'  {VAR_SHORT[x]:>10s} — {VAR_SHORT[y]:<10s}')
    if oriented.ambiguous_triples:
        print()
        print(f'ambiguous triples ({len(oriented.ambiguous_triples)}):')
        for triple in sorted(oriented.ambiguous_triples):
            x, z, y = triple
            print(f'  {VAR_SHORT[x]:>10s} — {VAR_SHORT[z]:^10s} — {VAR_SHORT[y]:<10s}  '
                  f'(Z undetermined)')

    out_path = (
        _REPO_ROOT / 'experiments' / 'findings' /
        'ddqn_pc_survive' / 'pc_adjacency.json'
    )
    out = {
        'cohort': {
            'scope': 'env_reward_polarity > 0.3 ∧ q_div < 1.0 ∧ '
                     'jensen_dormancy_gap < 0.05 ∧ standard config',
            'n_cells_total': panel.height,
            'n_per_env': {
                r['env_name']: r['len']
                for r in panel.group_by('env_name').len().iter_rows(named=True)
            },
        },
        'method': {
            'algorithm': 'conservative-PC',
            'alpha': 0.05,
            'max_conditioning': 1,
            'stratify_by': 'env_name',
            'ci_test': 'stratified_partial_spearman_rho (rank-based, Fisher-z pool)',
        },
        'variables': sorted(VAR_SHORT[v] for v in adj.variables),
        'edges_undirected_retained': [
            sorted(VAR_SHORT[v] for v in e) for e in adj.edges
        ],
        'edges_directed': [
            [VAR_SHORT[src], VAR_SHORT[tgt]]
            for src, tgt in sorted(oriented.directed_edges)
        ],
        'edges_remaining_undirected': [
            sorted(VAR_SHORT[v] for v in e)
            for e in oriented.undirected_edges
        ],
        'separating_sets': {
            f'{VAR_SHORT[sorted(k)[0]]}|{VAR_SHORT[sorted(k)[1]]}': [
                sorted(VAR_SHORT[v] for v in s) for s in vals
            ]
            for k, vals in adj.separating_sets.items()
        },
        'ambiguous_triples': [
            [VAR_SHORT[v] for v in t]
            for t in oriented.ambiguous_triples
        ],
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote: {out_path}')


if __name__ == '__main__':
    main()
