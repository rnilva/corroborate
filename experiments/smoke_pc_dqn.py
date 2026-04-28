"""§4 acceptance — PC discovery on the DDQN corpus.

Replicates the structural finding from PAPER_NOTES.md §4: at
α=0.05 with JCI stratification on `env_name`, PC discovers no
surviving edge between the binary intervention (`arm_ddqn`) and
any outcome variable. The Hasselt mechanism→outcome bridge is
data-level rejected, not just "underpowered."

Variables (6, vs v10 §4.2's 11 — we don't yet have v9's invariant
measures like `q_peak_lags_divergence_peak`):

- `arm_ddqn`            — binary 0/1 (vanilla / DDQN)
- `mechanism.jensen_gap` — proximal mechanism (DDQN should reduce)
- `outcome.late_window_mean`
- `outcome.eval_final_mean`
- `outcome.eval_best_burst_mean`
- `outcome.eval_best_burst_step`

Stratifier: `env_name` (17 envs in the migrated corpus).

§4-acceptance gate: NO edge between `arm_ddqn` and any
`outcome.*` variable in the discovered adjacency. The mechanism
chain may stay intact (`arm_ddqn → mechanism.jensen_gap` is
expected since DDQN's slot swap drops the gap on a subset of
envs), but the outcome side disconnects.

Run: `JAX_PLATFORMS=cpu uv run python experiments/smoke_pc_dqn.py`."""
from __future__ import annotations

# CPU-only: PC + scipy is pure-numpy, no JAX needed.
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from pathlib import Path

import polars as pl

from corroborate.causal_discovery import (
    discover_adjacency,
    orient_adjacency,
)


_RUNS_PATH = Path(__file__).parent / 'data' / 'ddqn' / 'runs.parquet'


_VARIABLES: list[str] = [
    'arm_ddqn',
    'mechanism.jensen_gap',
    'outcome.late_window_mean',
    'outcome.eval_final_mean',
    'outcome.eval_best_burst_mean',
    'outcome.eval_best_burst_step',
]

_OUTCOME_VARS: frozenset[str] = frozenset(
    v for v in _VARIABLES if v.startswith('outcome.')
)


def _prepare_dataframe() -> pl.DataFrame:
    """Load runs.parquet, encode arm_ddqn as 0/1, drop rows with
    NaN in any PC variable. Returns a DataFrame ready for
    `discover_adjacency`."""
    df = pl.read_parquet(_RUNS_PATH)
    df = df.with_columns(
        (pl.col('intervention_name') == 'ddqn').cast(pl.Int64).alias('arm_ddqn'),
    )
    # Drop NaN rows in PC variables; PC needs complete cases.
    df = df.drop_nulls(subset=_VARIABLES)
    df = df.filter(
        ~pl.any_horizontal([pl.col(v).is_nan() for v in _VARIABLES if df[v].dtype.is_float()])
    )
    return df


def main() -> None:
    df = _prepare_dataframe()
    n_obs = df.height
    n_envs = df['env_name'].n_unique()
    print(f'corpus: {n_obs} cells × {n_envs} envs '
          f'(loaded {_RUNS_PATH.name})')
    print(f'PC variables: {_VARIABLES}')
    print(f'JCI stratifier: env_name')
    print(f'α=0.05, max_conditioning=1')
    print()

    adj = discover_adjacency(
        df,
        variables=_VARIABLES,
        alpha=0.05,
        max_conditioning=1,
        stratify_by='env_name',
    )
    oriented = orient_adjacency(adj)

    print('=' * 72)
    print('Discovered adjacency (after Meek-rule orientation):')
    print('=' * 72)
    print()
    if oriented.directed_edges:
        print(f'Directed edges ({len(oriented.directed_edges)}):')
        for src, tgt in sorted(oriented.directed_edges):
            print(f'  {src:<32} → {tgt}')
    if oriented.undirected_edges:
        print(f'\nUndirected edges ({len(oriented.undirected_edges)}):')
        for edge in sorted(
            (tuple(sorted(e)) for e in oriented.undirected_edges),
        ):
            print(f'  {edge[0]:<32} ─ {edge[1]}')
    if not oriented.directed_edges and not oriented.undirected_edges:
        print('  (no surviving edges)')
    if oriented.ambiguous_triples:
        print(f'\nAmbiguous triples ({len(oriented.ambiguous_triples)}):')
        for trip in sorted(oriented.ambiguous_triples):
            print(f'  {trip[0]:<28} − {trip[1]:<28} − {trip[2]}')

    print()
    print('=' * 72)
    print('Removed edges + separating sets:')
    print('=' * 72)
    print()
    for edge, sepsets in sorted(
        adj.separating_sets.items(),
        key=lambda kv: tuple(sorted(kv[0])),
    ):
        x, y = sorted(edge)
        sepset_str = '; '.join(
            (f'{{{", ".join(sorted(s))}}}' if s else '{}')
            for s in sepsets
        )
        print(f'  {x:<32} ⫫ {y:<32}  by  {sepset_str}')

    print()
    print('=' * 72)
    print('§4 acceptance gate')
    print('=' * 72)
    print()
    arm_outcome_edges: list[tuple[str, str]] = []
    for edge in adj.edges:
        a, b = sorted(edge)
        if a == 'arm_ddqn' and b in _OUTCOME_VARS:
            arm_outcome_edges.append((a, b))
        if b == 'arm_ddqn' and a in _OUTCOME_VARS:
            arm_outcome_edges.append((b, a))

    if arm_outcome_edges:
        print(f'  ✗ FAIL: {len(arm_outcome_edges)} surviving edges between')
        print(f'    arm_ddqn and outcome variables — Hasselt link NOT')
        print(f'    rejected at this corpus / α=0.05.')
        for a, b in arm_outcome_edges:
            print(f'    {a} ─ {b}')
    else:
        print(f'  ✓ HELD: no surviving edge between arm_ddqn and any')
        print(f'    outcome variable. The mechanism→outcome bridge is')
        print(f'    data-level rejected at α=0.05 with JCI on env_name.')


if __name__ == '__main__':
    main()
