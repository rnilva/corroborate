"""PC adjacency discovery on the REACH cohort (per-cell, JCI by env).

Cohort: `{FourRooms-misc, Acrobot-v1, MountainCar-v0, MetaMaze-misc}`
∩ `_DDQN_RELEVANT_SCOPE` (G1∧G2 active, standard config).

Per-cell variables (each cell is one observation; NO per-pair Δs):

  is_ddqn      — binary treatment indicator (1 = DDQN, 0 = vanilla)
  jensen_gap   — premise activity (G1)
  q_div        — q_divergence_score (algebraic shadow of jens)
  argmax_H     — argmax_entropy_late (empirical shadow of jens)
  stale        — target_staleness_late (empirical mediator)
  eff_h        — effective_horizon (length-polarity projection)
  outcome      — eval_best_burst_mean

JCI stratifier: `env_name` — within-env CI tests, Fisher-z pooled.
α = 0.05, max_conditioning = 1 (depth-1 partial Spearman).

The output:
  1. PC adjacency (conservative-PC at depth ≤ 1)
  2. Oriented CPDAG via v-structure detection + Meek rules
  3. JSON dump + ASCII visual

Expected per CLAIM 26b / three-gate framework:
  - is_ddqn  → jens     (treatment reduces bias)
  - jens     → outcome  (premise drives effect)
  - q_div    ⊥ outcome | jens  (algebraic shadow)
  - argmax_H ⊥ outcome | jens  (empirical shadow)
  - eff_h    ⊥ outcome | jens  (per JCI eff_h migration: |ρ|≈0.6
    survives, so this edge LIKELY survives PC at depth 1 too —
    pre-emptive note: rank-coupling survives, not necessarily
    direct edge).
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import polars as pl

# Project root on path so `experiments.findings.*` is importable.
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


REACH_ENVS = (
    'FourRooms-misc',
    'Acrobot-v1',
    'MountainCar-v0',
    'MetaMaze-misc',
)
DDQN_ARM = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASELINE_ARM = 'baseline'

# Variables to discover edges among (per-cell, NOT per-pair-Δ).
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
    """Apply REACH cohort + relaxed scope (G1 only — drops the
    dormancy gate since `jensen_dormancy_gap` requires trace-derived
    `online_std_q_per_step`, which is evicted for most postfix
    corpora). G1 alone (jens > 0.05) captures "premise active";
    the dormancy floor refinement is omitted in this descriptive
    PC discovery."""
    scope = (
        pl.col('env_name').is_in(list(REACH_ENVS))
        & pl.col('arm_key').is_in([DDQN_ARM, BASELINE_ARM])
        & pl.col('jensen_gap').is_finite()
        & (pl.col('jensen_gap') > 0.05)
        & pl.col('n_actions').is_finite()
        & (pl.col('n_actions') >= 3)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    )
    return runs.filter(scope)


def main() -> None:
    required = (
        'jensen_gap', 'q_divergence_score', 'argmax_entropy_late',
        'target_staleness_late', 'effective_horizon',
        'eval_best_burst_mean',
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
    print(f'In-scope REACH cells: {len(panel)}')
    print('Env counts:')
    for r in panel.group_by('env_name').len().sort('env_name').iter_rows(named=True):
        print(f'  {r["env_name"]:<20s}: {r["len"]}')
    print(f'\nVariables: {[VAR_SHORT[v] for v in available]}')
    print()

    # 1. PC adjacency at depth ≤ 1, JCI by env_name.
    adj = discover_adjacency(
        panel, variables=available,
        alpha=0.05, max_conditioning=1,
        stratify_by='env_name',
    )

    print(f'=== PC adjacency (conservative, depth ≤ 1, JCI by env) ===\n')
    n_possible = len(available) * (len(available) - 1) // 2
    print(f'variables: {len(adj.variables)}, edges remaining: '
          f'{len(adj.edges)} / {n_possible} possible')
    print()
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

    # 2. Orient adjacency.
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
        'ddqn_universe_pc_reach' / 'pc_adjacency.json'
    )
    out = {
        'cohort': {
            'reach_envs': list(REACH_ENVS),
            'scope': '_DDQN_RELEVANT_SCOPE (G1 ∧ G2 + standard config)',
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
