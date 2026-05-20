"""Causal-discovery + mediation panel at SI γ=0.999 (n=60, both arms).

Question. Under do(DDQN) at SI γ=0.999, are the per-cell shifts in
(jensen_gap, policy_growth_fraction, policy_churn_late,
state_hash_entropy_late, outcome) connected by a single mediation
chain, or are they parallel side-effects of the intervention?

Two passes:

1. PC adjacency discovery (`graph.discovery.discover_adjacency`)
   over the 6-variable panel.
2. Marginal vs single-Z / multi-Z conditional Spearman ρ(arm, outcome).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import experiments.findings.ddqn_three_conditions  # populate registries

from corroborate.graph.discovery import (
    discover_adjacency,
    partial_spearman_rho,
    partial_spearman_rho_multi,
    _spearman_marginal,
)


_SI_G999 = (
    (pl.col('env_name') == 'SpaceInvaders-MinAtar')
    & (pl.col('gamma') == 0.999)
    & pl.col('policy_growth_fraction').is_finite()
    & pl.col('policy_churn_late').is_finite()
    & pl.col('state_hash_entropy_late').is_finite()
    & pl.col('jensen_gap').is_finite()
    & pl.col('eval_best_burst_raw_mean').is_finite()
)


def _col(df: pl.DataFrame, name: str) -> np.ndarray:
    return np.asarray(df[name].to_list(), dtype=np.float64)


def main() -> None:
    df = pl.read_parquet('experiments/data/cache/ddqn_three_conditions.parquet')
    cells = df.filter(_SI_G999).with_columns(
        (pl.col('arm_key') != 'baseline').cast(pl.Float64).alias('arm'),
    )
    print(f'SI γ=0.999 cells: {len(cells)}')
    print(f'  baseline n: {(cells["arm"] == 0).sum()}, DDQN n: {(cells["arm"] == 1).sum()}')
    print()

    variables: tuple[str, ...] = (
        'arm',
        'jensen_gap',
        'policy_growth_fraction',
        'policy_churn_late',
        'state_hash_entropy_late',
        'eval_best_burst_raw_mean',
    )

    print('=== Pass 1 — PC adjacency at depth ≤ 2 ===')
    print()
    adj = discover_adjacency(cells, variables=variables, max_conditioning=2, alpha=0.05)
    print(f'Surviving edges after depth-2 conditioning ({len(adj.edges)} of {len(variables)*(len(variables)-1)//2} possible):')
    for e in sorted(adj.edges, key=lambda fs: tuple(sorted(fs))):
        a, b = sorted(e)
        sep = adj.separating_sets.get(e, frozenset())
        sep_str = ' | '.join(
            '{' + ', '.join(sorted(s)) + '}'
            for s in sep
        ) if sep else 'NONE (marginal)'
        print(f'  {a:30s} ↔ {b:30s}  sep sets: {sep_str}')
    print()
    arm_outcome_edge = frozenset({'arm', 'eval_best_burst_raw_mean'})
    if arm_outcome_edge in adj.edges:
        sep = adj.separating_sets.get(arm_outcome_edge, frozenset())
        print('arm — outcome edge SURVIVED depth-2 conditioning')
        print(f'  separating sets tested but failed to separate: {sep or "none"}')
    else:
        sep = adj.separating_sets.get(arm_outcome_edge, frozenset())
        print('arm — outcome edge REMOVED by some conditioning set:')
        for s in sep:
            print(f'  separator: {sorted(s)}')
    print()

    print('=== Pass 2 — Marginal vs conditional Spearman ρ(arm, outcome) ===')
    print()
    x_arm = _col(cells, 'arm')
    y_out = _col(cells, 'eval_best_burst_raw_mean')
    rho_m, p_m = _spearman_marginal(x_arm, y_out)
    print(f'Marginal: ρ(arm, outcome) = {rho_m:+.3f}, p = {p_m:.4g}, n = {len(cells)}')
    print()

    print('Single-mediator partial:')
    for med in ('jensen_gap', 'policy_growth_fraction', 'policy_churn_late', 'state_hash_entropy_late'):
        z = _col(cells, med)
        rho_c, p_c = partial_spearman_rho(x_arm, y_out, z)
        shrink = abs(rho_c / rho_m) if abs(rho_m) > 1e-9 else float('nan')
        print(
            f'  ρ(arm, outcome | {med:30s}) = {rho_c:+.3f}, p = {p_c:.4g}, '
            f'shrink = {shrink:.2f}× of |marginal|'
        )
    print()

    print('Multi-mediator partial (OLS residual form):')
    candidates: tuple[tuple[str, ...], ...] = (
        ('jensen_gap', 'policy_growth_fraction'),
        ('jensen_gap', 'state_hash_entropy_late'),
        ('policy_growth_fraction', 'state_hash_entropy_late'),
        ('policy_growth_fraction', 'policy_churn_late'),
        ('state_hash_entropy_late', 'policy_churn_late'),
        ('jensen_gap', 'policy_growth_fraction', 'state_hash_entropy_late'),
        ('jensen_gap', 'policy_growth_fraction', 'policy_churn_late', 'state_hash_entropy_late'),
    )
    for med_set in candidates:
        z_mat = np.stack([_col(cells, m) for m in med_set], axis=1)
        rho_c, p_c = partial_spearman_rho_multi(x_arm, y_out, z_mat)
        shrink = abs(rho_c / rho_m) if abs(rho_m) > 1e-9 else float('nan')
        z_label = ' + '.join(med_set)
        print(
            f'  ρ(arm, outcome | {z_label})\n'
            f'    = {rho_c:+.3f}, p = {p_c:.4g}, shrink = {shrink:.2f}× of |marginal|'
        )
    print()


if __name__ == '__main__':
    main()
