"""PC causal-graph discovery on the paired-Δ panel, JCI-stratified by
(env, sync, total_steps).

Variables in the graph (all paired-Δ values, DDQN − baseline):

  Δ_jens       — mechanism step (bias correction firing)
  Δ_stale      — target staleness (the candidate mediator from CLAIM 13)
  Δ_eff_h      — effective horizon (length-channel proxy)
  Δ_bf         — bootstrap fraction (length-channel proxy)
  Δ_q_div      — Q-divergence score
  Δ_outcome    — eval_best_burst_mean (target)

JCI stratifier: `(env, sync, total_steps)` — within-stratum partial CI
tests, Fisher-z pooled across strata.

The output:
  1. Adjacency (PC at depth ≤ 1, conservative-PC, α=0.05)
  2. Oriented CPDAG via v-structure detection + Meek rules
  3. Per-stratum panel of stratum-level marginal correlations (so we
     can see the regime-specific structure underneath the pool)
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

import corroborate_rl.dqn.measurables  # register
from corroborate.graph.discovery import (
    discover_adjacency, orient_adjacency,
)

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
ENVS = (
    'Acrobot-v1', 'Asterix-MinAtar', 'Breakout-MinAtar',
    'CartPole-v1', 'FourRooms-misc', 'Freeway-MinAtar',
    'MetaMaze-misc', 'MountainCar-v0', 'SpaceInvaders-MinAtar',
)

CANDIDATES = (
    'jensen_gap',
    'target_staleness_late',
    'effective_horizon',
    'bootstrap_fraction',
    'q_divergence_score',
)


def _build_paired_panel(df: pl.DataFrame) -> pl.DataFrame:
    """Pair (DDQN, baseline) cells per (env, corpus, gamma, sync,
    total_steps, seed) and return a flat DataFrame of paired Δ
    values + a stratum tag."""
    cells = df.filter(
        pl.col('env_name').is_in(list(ENVS))
        & pl.col('arm_key').is_in(['baseline', DDQN])
    )
    pair_keys = ['env_name', 'corpus', 'gamma', 'total_steps', 'sync_period', 'seed']
    select_cols = pair_keys + ['eval_best_burst_mean'] + [
        c for c in CANDIDATES if c in cells.columns
    ]
    v = cells.filter(pl.col('arm_key') == 'baseline').select(select_cols).rename(
        {c: f'{c}_v' for c in select_cols if c not in pair_keys}
    )
    d = cells.filter(pl.col('arm_key') == DDQN).select(select_cols).rename(
        {c: f'{c}_d' for c in select_cols if c not in pair_keys}
    )
    j = v.join(d, on=pair_keys, how='inner').filter(
        pl.col('eval_best_burst_mean_v').is_finite()
        & pl.col('eval_best_burst_mean_d').is_finite()
    )
    rename_map = {'eval_best_burst_mean': 'outcome'}
    rename_map.update({
        'jensen_gap': 'jens',
        'target_staleness_late': 'stale',
        'effective_horizon': 'eff_h',
        'bootstrap_fraction': 'bf',
        'q_divergence_score': 'q_div',
    })
    deltas = []
    for orig_name, short_name in rename_map.items():
        if f'{orig_name}_d' in j.columns:
            deltas.append(
                (pl.col(f'{orig_name}_d') - pl.col(f'{orig_name}_v')).alias(f'd_{short_name}')
            )
    j = j.with_columns(deltas)
    j = j.with_columns(
        pl.concat_str([
            pl.col('env_name'), pl.lit('|'),
            pl.col('sync_period').cast(pl.Utf8), pl.lit('|'),
            pl.col('total_steps').cast(pl.Utf8),
        ]).alias('stratum')
    )
    return j


def main() -> None:
    df = pl.read_parquet('experiments/data/cache/ddqn.parquet')
    paired = _build_paired_panel(df)
    print(f'paired cells: {paired.height} across '
          f'{paired["stratum"].n_unique()} strata (env|sync|steps)', flush=True)

    var_names = ['d_jens', 'd_stale', 'd_eff_h', 'd_bf', 'd_q_div', 'd_outcome']
    available = [v for v in var_names if v in paired.columns]
    panel = paired.select([*available, 'stratum']).filter(
        pl.all_horizontal([pl.col(v).is_finite() for v in available])
    )
    print(f'after dropping non-finite: {panel.height} cells × {len(available)} vars',
          flush=True)

    # 1. PC adjacency at depth ≤ 1 with JCI stratification by stratum.
    adj = discover_adjacency(
        panel, variables=available,
        alpha=0.05, max_conditioning=1,
        stratify_by='stratum',
    )
    print()
    print('=== PC adjacency (conservative, depth ≤ 1, JCI by stratum) ===\n')
    print(f'variables: {sorted(adj.variables)}')
    print(f'edges remaining: {len(adj.edges)} of {len(available) * (len(available)-1) // 2} possible')
    print()
    print('Edges (separating sets shown for retained):')
    for edge in sorted(adj.edges, key=lambda e: tuple(sorted(e))):
        x, y = sorted(edge)
        print(f'  {x} — {y}', flush=True)
    print()
    print('Removed edges (with separating set Z):')
    for edge in sorted(set(frozenset(e) for e in combinations(available, 2)) - adj.edges,
                       key=lambda e: tuple(sorted(e))):
        x, y = sorted(edge)
        sepsets = adj.separating_sets.get(edge, frozenset())
        sepset_strs = [
            f'∅' if not s else '{' + ', '.join(sorted(s)) + '}'
            for s in sepsets
        ]
        print(f'  {x} ⊥ {y} | {", ".join(sepset_strs) or "(no test)"}', flush=True)

    # 2. Orient adjacency
    oriented = orient_adjacency(adj)
    print()
    print('=== CPDAG (after v-structure detection + Meek rules) ===\n')
    print(f'directed edges ({len(oriented.directed_edges)}):')
    for src, tgt in sorted(oriented.directed_edges):
        print(f'  {src} → {tgt}', flush=True)
    print()
    print(f'undirected edges ({len(oriented.undirected_edges)}):')
    for edge in sorted(oriented.undirected_edges, key=lambda e: tuple(sorted(e))):
        x, y = sorted(edge)
        print(f'  {x} — {y}', flush=True)
    if oriented.ambiguous_triples:
        print()
        print(f'ambiguous triples ({len(oriented.ambiguous_triples)}):')
        for triple in sorted(oriented.ambiguous_triples):
            x, z, y = triple
            print(f'  {x} — {z} — {y}  (Z={z} undetermined collider)', flush=True)

    # ASCII visual
    print()
    print('=== ASCII visual ===\n')
    for src, tgt in sorted(oriented.directed_edges):
        arrow = '→'
        print(f'  {src:<10} {arrow}  {tgt}', flush=True)
    for edge in sorted(oriented.undirected_edges, key=lambda e: tuple(sorted(e))):
        x, y = sorted(edge)
        print(f'  {x:<10} —  {y}', flush=True)

    # 3. Save
    out_path = Path('experiments/findings/sync_curve_breakout/pc_causal_graph.json')
    out = {
        'variables': sorted(adj.variables),
        'edges_undirected': [sorted(e) for e in adj.edges],
        'edges_directed': [list(e) for e in sorted(oriented.directed_edges)],
        'edges_remaining_undirected': [
            sorted(e) for e in oriented.undirected_edges
        ],
        'separating_sets': {
            f'{sorted(k)[0]}|{sorted(k)[1]}': [sorted(s) for s in v]
            for k, v in adj.separating_sets.items()
        },
        'ambiguous_triples': [list(t) for t in oriented.ambiguous_triples],
        'n_observations': panel.height,
        'n_strata': paired['stratum'].n_unique(),
        'alpha': 0.05,
        'max_conditioning': 1,
        'stratify_by': '(env, sync, total_steps)',
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote: {out_path}', flush=True)


if __name__ == '__main__':
    main()
