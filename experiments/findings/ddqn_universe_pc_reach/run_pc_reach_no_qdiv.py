"""Sibling PC discovery on REACH excluding q_divergence_score.

The original `run_pc_reach.py` removed two `is_ddqn → ·` edges
because conditioning on qdiv made them independent:
  is_ddqn ⊥ jens  | {q_div}
  is_ddqn ⊥ q_div | {jens}

The tautology_audit (`run_tautology_audit.py`) showed within-γ
ρ(qdiv, jens) = +0.974 — the algebraic shadow
`qdiv = jens / (R/(1−γ))` is near-perfect collinearity. So
those PC removals partly follow from algebra, not empirical
indep.

This script re-runs PC with qdiv excluded from the variable
set. The is_ddqn→jens edge then gets a clean conditional-
independence test against the remaining mediator candidates
(argmax_H, stale, eff_h, outcome). Result documents whether
the "is_ddqn graph-disconnected on REACH" finding survives
the algebraic-shadow removal.

Same cohort (REACH ∩ `_DDQN_RELEVANT_SCOPE` G1∧G2 standard config,
n≈346), same algorithm (conservative-PC depth ≤ 1, JCI by
env_name, α=0.05).
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

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate registry
from corroborate.graph.discovery import (
    discover_adjacency, orient_adjacency,
)
from corroborate.runner.runner import (
    _load_directory,  # pyright: ignore[reportPrivateUsage]
    _compute_measurables,  # pyright: ignore[reportPrivateUsage]
)


REACH_ENVS = (
    'FourRooms-misc', 'Acrobot-v1', 'MountainCar-v0', 'MetaMaze-misc',
)
DDQN_ARM = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASELINE_ARM = 'baseline'

# Variable set — same as `run_pc_reach.py` MINUS q_divergence_score.
VARS = (
    'is_ddqn',
    'jensen_gap',
    'argmax_entropy_late',
    'target_staleness_late',
    'effective_horizon',
    'eval_best_burst_mean',
)
VAR_SHORT = {
    'is_ddqn': 'is_ddqn',
    'jensen_gap': 'jens',
    'argmax_entropy_late': 'argmax_H',
    'target_staleness_late': 'stale',
    'effective_horizon': 'eff_h',
    'eval_best_burst_mean': 'outcome',
}


def _scoped_cohort(runs: pl.DataFrame) -> pl.DataFrame:
    scope = (
        pl.col('env_name').is_in(list(REACH_ENVS))
        & pl.col('arm_key').is_in([DDQN_ARM, BASELINE_ARM])
        & pl.col('jensen_gap').is_finite()
        & (pl.col('jensen_gap') > 0.05)
        & pl.col('jensen_dormancy_gap').is_finite()
        & (pl.col('jensen_dormancy_gap') < 0.05)
        & pl.col('n_actions').is_finite()
        & (pl.col('n_actions') >= 3)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
    )
    return runs.filter(scope)


def main() -> None:
    required = (
        'jensen_gap', 'argmax_entropy_late',
        'target_staleness_late', 'effective_horizon',
        'eval_best_burst_mean', 'jensen_dormancy_gap',
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
    print(f'In-scope REACH cells (qdiv-excluded): {len(panel)}')
    print('Env counts:')
    for r in panel.group_by('env_name').len().sort('env_name').iter_rows(named=True):
        print(f'  {r["env_name"]:<20s}: {r["len"]}')
    print(f'\nVariables: {[VAR_SHORT[v] for v in available]}')
    print()

    adj = discover_adjacency(
        panel, variables=available,
        alpha=0.05, max_conditioning=1,
        stratify_by='env_name',
    )

    print(f'=== PC adjacency (qdiv excluded, depth ≤ 1, JCI by env) ===\n')
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
        'ddqn_universe_pc_reach' / 'pc_adjacency_no_qdiv.json'
    )
    out = {
        'note': 'q_divergence_score excluded — algebraic shadow of jens (qdiv = jens/(R/(1−γ))); within-γ ρ(qdiv, jens) = +0.974 per `tautology_audit.json`. Excluding it gives a clean test of the empirical is_ddqn → jens edge that the original run conflated with the algebraic shadow.',
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
