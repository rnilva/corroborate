"""Per-burst link analysis on expectile_3way: does expectile show
the same per-burst (Δbias, Δret) coupling as DDQN?

The framework's recipe (FINDINGS.md ninth revision) found that on
FourRooms, scalar mech-link slopes silently average early bias-
reduction with late Q-explosion to mask phase structure. Per-burst
r(Δbias, Δret) was negative at every burst (-0.34 to -0.95) — the
within-pair coupling holds throughout training.

If expectile-vs-vanilla shows the same per-burst pattern as DDQN-vs-
vanilla, the residual `bf → g_link | g_mech` is structural to the env
(both arms see the same per-burst coupling). If they differ, the
residual is mechanism-specific.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr

from corroborate.analyses.paired_link_per_burst import (
    paired_link_per_burst, phase_link_consistency,
)
from corroborate.measurables.reductions import from_key

DATA = Path('experiments/data/expectile_3way')
BASELINE = 'baseline'
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
EXPECTILE = 'bootstrap=partial(Claim:bootstrap;greedification=partial(Claim:expectile_greedify;tau=0.7))'


def _per_burst_means(col: pl.Series) -> list[list[float]]:
    out: list[list[float]] = []
    for row in col:
        if row is None:
            out.append([])
            continue
        arr = np.asarray(row.to_list(), dtype=np.float64)
        if arr.ndim != 2:
            out.append([])
            continue
        out.append(arr.mean(axis=1).tolist())
    return out


def _to_cells(df: pl.DataFrame) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for row in df.iter_rows(named=True):
        cells.append({
            'env_name': row['env_name'],
            'arm_key': row['arm_key'],
            'seed': row['seed'],
            'mc_per_burst': row['mc_per_burst'],
            'bias_per_burst': row['bias_per_burst'],
        })
    return cells


def main() -> None:
    print('Loading expectile_3way runs + traces...')
    runs = pl.read_parquet(DATA / 'runs.parquet', columns=['id', 'env_name', 'arm_key', 'seed'])
    traces = pl.read_parquet(DATA / 'traces.parquet', columns=['id', 'mc_return', 'predicted_q_at_start'])
    df = runs.join(traces, on='id', how='inner')
    print(f'cells: {len(df)}')

    mc_pb = _per_burst_means(df['mc_return'])
    qs_pb = _per_burst_means(df['predicted_q_at_start'])
    bias_pb = [
        [q - m for q, m in zip(q_row, mc_row)] for mc_row, q_row in zip(mc_pb, qs_pb)
    ]
    df = df.with_columns(
        mc_per_burst=pl.Series(mc_pb, dtype=pl.List(pl.Float64)),
        bias_per_burst=pl.Series(bias_pb, dtype=pl.List(pl.Float64)),
    )

    cells = _to_cells(df)
    target = from_key('mc_per_burst')
    predictor = from_key('bias_per_burst')

    contrast_pairs = [('DDQN', DDQN), ('EXPECTILE', EXPECTILE)]
    print()
    for env in sorted({c['env_name'] for c in cells}):
        print('=' * 100)
        print(f'env = {env}')
        for label, treatment in contrast_pairs:
            res = paired_link_per_burst.fn(
                cells,
                treatment_arm=treatment, baseline_arm=BASELINE,
                target=target, predictor=predictor,
                pair_by=('seed',), env_name=env,
            )
            plc = phase_link_consistency(res, env_name=env)
            print(f'  {label:<10} plc={plc:.2f}, n_strata={res.n_strata}')
            print(f'    {"burst":>5} | {"r":>7} {"p":>9} {"slope":>10} {"mean_dpred":>12} {"mean_dtarget":>13} {"sd_dt":>8} {"n":>4}')
            for s in res.strata:
                sig = '*' if s.p < 0.05 else ' '
                print(f'    {s.burst_index:>5} | {s.r:>+7.2f} {s.p:>9.2g} {s.slope:>+10.4f} {s.mean_d_predictor:>+12.3f} {s.mean_d_target:>+13.3f} {s.sd_d_target:>8.3f} {s.n_pairs:>4}{sig}')

    # Save for later inspection
    out_data: dict[str, dict[str, dict]] = {}
    for env in sorted({c['env_name'] for c in cells}):
        out_data[env] = {}
        for label, treatment in contrast_pairs:
            res = paired_link_per_burst.fn(
                cells, treatment_arm=treatment, baseline_arm=BASELINE,
                target=target, predictor=predictor, pair_by=('seed',), env_name=env,
            )
            plc = phase_link_consistency(res, env_name=env)
            out_data[env][label] = {
                'plc': plc,
                'strata': [
                    {'burst': s.burst_index, 'r': s.r, 'p': s.p, 'slope': s.slope,
                     'mean_d_pred': s.mean_d_predictor, 'mean_d_target': s.mean_d_target,
                     'sd_d_target': s.sd_d_target, 'n_pairs': s.n_pairs}
                    for s in res.strata
                ],
            }
    out = Path('experiments/findings/sync_curve_breakout/expectile_per_burst_panel.json')
    out.write_text(json.dumps(out_data, indent=2))
    print()
    print(f'wrote: {out}')


if __name__ == '__main__':
    main()
