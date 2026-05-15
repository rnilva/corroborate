"""Audit MetaMaze's DDQN-translation puzzle across the existing
γ × FA-depth factorial corpus (`ddqn_axis_probes_metamaze_1m`).

Canonical scope (γ=0.99, MLP[64,64], 1M) gave the "mech-without-
translation" diagnostic per memory
`findings_ddqn_mediator_heterogeneity`. The 480-cell factorial
varies γ ∈ {0.99, 0.999} × FA ∈ {linear, MLP[64,64]} at the
canonical 1M / 60 seeds-per-arm. Question: does the
mech-without-translation pattern hold across all four cells, or
does it flip with γ / depth?

For each (γ, FA) cell, report arm-mean deltas (DDQN − vanilla)
of: outcome (raw + disc), jens, q_late_mean, q_divergence_score,
bg_magnitude, argmax_entropy, target_staleness_late.

Independent-samples Cohen's d using pooled-SD. n=60 per arm, so
|d| ≥ 0.4 is well-powered at α=0.05.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src/corroborate_rl'))

PROBE = REPO / 'experiments' / 'probes' / 'ddqn_axis_probes_metamaze_1m' / 'runs.parquet'

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASE = 'baseline'

MEASURABLES: tuple[str, ...] = (
    'eval_best_burst_mean',         # disc outcome
    'eval_best_burst_raw_mean',     # raw outcome (γ-invariant)
    'jensen_gap',
    'q_late_mean',
    'target_staleness_late',
    'q_action_grad_overlap_late',
    'ddqn_bootstrap_gap_late',
)


def cell_summary(df: pl.DataFrame, gamma: float, hidden: str) -> dict[str, float]:
    sub = df.filter(
        (pl.col('gamma') == gamma) & (pl.col('q_network.hidden') == hidden),
    )
    ddqn = sub.filter(pl.col('arm_key') == DDQN)
    base = sub.filter(pl.col('arm_key') == BASE)
    out: dict[str, float] = {}
    for m in MEASURABLES:
        if m not in df.columns:
            continue
        dv = ddqn[m].drop_nulls().drop_nans().to_numpy()
        bv = base[m].drop_nulls().drop_nans().to_numpy()
        if dv.size < 3 or bv.size < 3:
            continue
        mu_d, mu_b = float(np.mean(dv)), float(np.mean(bv))
        sd = math.sqrt(
            (np.var(dv, ddof=1) + np.var(bv, ddof=1)) / 2
        )
        d_cohen = (mu_d - mu_b) / sd if sd > 0 else float('nan')
        out[f'{m}__ddqn'] = mu_d
        out[f'{m}__base'] = mu_b
        out[f'{m}__delta'] = mu_d - mu_b
        out[f'{m}__d'] = d_cohen
    return out


def main() -> None:
    df = pl.read_parquet(str(PROBE))
    print(f'Loaded {df.shape[0]} cells from {PROBE.name}')

    cells = [(g, h) for g in (0.99, 0.999) for h in ('()', '(64,64)')]
    summaries = {cell: cell_summary(df, *cell) for cell in cells}

    # Print a comparison table across cells: each row = measurable,
    # columns = the four cells.
    print(f'\n{"measurable":<32s}', end='')
    for g, h in cells:
        print(f'  γ={g:<6} {h:<10}', end='')
    print()
    print('-' * 110)
    for m in MEASURABLES:
        # ddqn / base / delta / d for each cell
        for what in ('ddqn', 'base', 'delta', 'd'):
            row = f'{m}__{what}'
            print(f'{row:<32s}', end='')
            for cell in cells:
                v = summaries[cell].get(row, float('nan'))
                print(f'  {v:>16.3f}', end='')
            print()
        print()


if __name__ == '__main__':
    main()
