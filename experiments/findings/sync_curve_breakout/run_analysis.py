"""Sync-curve link-attenuation analysis on Breakout-MinAtar.

Tests the prediction from `findings_minatar_link_attenuation.md`:
at sync=100 the per-burst link dies after Phase 1 (plc=0.05);
sync=10000 should keep the link active throughout because
Q-explosion is suppressed. The intermediate sync ∈ {1000, 3000}
points complete the curve.

Sources (Breakout-MinAtar only):
- sync=100:    minatar_1M/{runs, per_burst_arrays}.parquet
- sync=1000:   minatar_sync_curve/ddqn_sync1k/{runs, traces}.parquet
- sync=3000:   minatar_sync_curve/ddqn_sync3k/{runs, traces}.parquet
- sync=10000:  minatar_sync_intervention/runs_with_bridge_cache.parquet

Per-burst arrays (mc_per_burst, q_per_burst, bias_per_burst) are
extracted via mean-over-episodes of the (n_bursts, n_eps_per_burst)
list-of-list columns where not pre-computed.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from corroborate.analyses.link.paired_link_per_burst import (
    paired_link_per_burst, phase_link_consistency,
)
from corroborate.measurables.reductions import from_key


ENV = 'Breakout-MinAtar'
DATA = Path('experiments/data')
TREATMENT_ARM = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASELINE_ARM = 'baseline'


def _per_burst_means(col: pl.Series) -> list[list[float]]:
    """For each row's list[list[f64]] (n_bursts × n_eps), return
    a list of per-burst means (length n_bursts)."""
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


def _load_sync_100_breakout() -> list[Mapping[str, object]]:
    runs = pl.read_parquet(DATA / 'minatar_1M' / 'runs.parquet')
    pba = pl.read_parquet(DATA / 'minatar_1M' / 'per_burst_arrays.parquet')
    bk = runs.filter(pl.col('env_name') == ENV).join(pba, on='id', how='inner')
    bk = bk.with_columns(
        # bias_per_burst already has Q − MC; q_per_burst = mc + bias.
        q_per_burst=pl.struct(['mc_per_burst', 'bias_per_burst']).map_elements(
            lambda s: [m + b for m, b in zip(s['mc_per_burst'], s['bias_per_burst'])],
            return_dtype=pl.List(pl.Float64),
        ),
    )
    return _to_cells(bk, sync=100)


def _load_sync_from_traces(corpus: str, sync: int) -> list[Mapping[str, object]]:
    runs_path = DATA / corpus / 'runs.parquet'
    traces_path = DATA / corpus / 'traces.parquet'
    runs = pl.read_parquet(runs_path).filter(pl.col('env_name') == ENV)
    traces = pl.read_parquet(
        traces_path,
        columns=['id', 'mc_return', 'predicted_q_at_start'],
    )
    bk = runs.join(traces, on='id', how='inner')
    mc_pb = _per_burst_means(bk['mc_return'])
    q_pb = _per_burst_means(bk['predicted_q_at_start'])
    bias_pb = [
        [q - m for q, m in zip(q_row, mc_row)]
        for mc_row, q_row in zip(mc_pb, q_pb)
    ]
    bk = bk.with_columns(
        mc_per_burst=pl.Series(mc_pb, dtype=pl.List(pl.Float64)),
        q_per_burst=pl.Series(q_pb, dtype=pl.List(pl.Float64)),
        bias_per_burst=pl.Series(bias_pb, dtype=pl.List(pl.Float64)),
    )
    return _to_cells(bk, sync=sync)


def _load_sync_from_bridge_cache(
    corpus: str, sync: int,
) -> list[Mapping[str, object]]:
    df = pl.read_parquet(
        DATA / corpus / 'runs_with_bridge_cache.parquet',
    ).filter(pl.col('env_name') == ENV)
    mc_pb = _per_burst_means(df['mc_return'])
    q_pb = _per_burst_means(df['predicted_q_at_start'])
    bias_pb = [
        [q - m for q, m in zip(q_row, mc_row)]
        for mc_row, q_row in zip(mc_pb, q_pb)
    ]
    df = df.with_columns(
        mc_per_burst=pl.Series(mc_pb, dtype=pl.List(pl.Float64)),
        q_per_burst=pl.Series(q_pb, dtype=pl.List(pl.Float64)),
        bias_per_burst=pl.Series(bias_pb, dtype=pl.List(pl.Float64)),
    )
    return _to_cells(df, sync=sync)


def _to_cells(df: pl.DataFrame, *, sync: int) -> list[Mapping[str, object]]:
    cells: list[Mapping[str, object]] = []
    for row in df.iter_rows(named=True):
        cells.append({
            'env_name': row['env_name'],
            'arm_key': row['arm_key'],
            'seed': row['seed'],
            'sync_period': sync,
            'mc_per_burst': row['mc_per_burst'],
            'bias_per_burst': row['bias_per_burst'],
            'q_per_burst': row['q_per_burst'],
        })
    return cells


# `from_key` returns a Measurable that reads the named column.
# evaluate_per_burst_source's cache-first dispatch matches
# `source.name` against the cell column, so this hits without
# evaluating the callback.
mc_pb_source = from_key('mc_per_burst')
bias_pb_source = from_key('bias_per_burst')


def main() -> None:
    print(f'Loading {ENV} cells across 4 sync values...')
    cohorts = {
        100:   _load_sync_100_breakout(),
        1000:  _load_sync_from_traces('minatar_sync_curve/ddqn_sync1k', 1000),
        3000:  _load_sync_from_traces('minatar_sync_curve/ddqn_sync3k', 3000),
        10000: _load_sync_from_bridge_cache('minatar_sync_intervention', 10000),
    }
    for sync, cells in cohorts.items():
        arms = sorted({c['arm_key'] for c in cells})
        seeds = sorted({c['seed'] for c in cells})
        print(f'  sync={sync:>5}  cells={len(cells):3d}  arms={len(arms)}  seeds={len(seeds)}')

    print()
    print(f'Per-burst link analysis: r(Δ_mc, -Δ_bias) per burst, paired by seed.')
    print(f'  treatment_arm = {TREATMENT_ARM!r}')
    print(f'  baseline_arm  = {BASELINE_ARM!r}')
    print()

    results: dict[int, object] = {}
    for sync, cells in cohorts.items():
        result = paired_link_per_burst.fn(
            cells,
            treatment_arm=TREATMENT_ARM,
            baseline_arm=BASELINE_ARM,
            target=mc_pb_source,
            predictor=bias_pb_source,
            pair_by=('seed',),
            env_name=ENV,
        )
        plc = phase_link_consistency(result, env_name=ENV)
        results[sync] = result
        n_bursts = len(result.strata)
        print(f'sync={sync:>5}  plc={plc:.3f}  n_bursts={n_bursts}')

    print()
    print('Per-burst β = Δ_outcome / -Δ_bias (note: predictor is negated):')
    print(f'  {"burst":>5} | ' + ' | '.join(f'sync={s:>5}: r,β,p,n' for s in cohorts))
    for b in range(20):
        line = f'  {b:>5} |'
        for sync in cohorts:
            res = cast(object, results[sync])
            stratum = next(
                (s for s in res.strata if s.burst_index == b),  # type: ignore[attr-defined]
                None,
            )
            if stratum is None:
                line += '              -                |'
                continue
            line += f' r={stratum.r:+.2f},β={stratum.slope:+.4g},p={stratum.p:.2g},n={stratum.n_pairs} |'
        print(line)

    out_dir = Path('experiments/findings/sync_curve_breakout')
    summary = {
        sync: {
            'plc': phase_link_consistency(res, env_name=ENV),
            'strata': [
                {
                    'burst': s.burst_index,
                    'r': s.r,
                    'p': s.p,
                    'slope_beta': s.slope,
                    'mean_d_predictor': s.mean_d_predictor,
                    'mean_d_target': s.mean_d_target,
                    'sd_d_target': s.sd_d_target,
                    'n_pairs': s.n_pairs,
                }
                for s in res.strata  # type: ignore[attr-defined]
            ],
        }
        for sync, res in results.items()
    }
    (out_dir / 'sync_curve_panel.json').write_text(json.dumps(summary, indent=2))
    print()
    print(f'Wrote: {out_dir/"sync_curve_panel.json"}')


if __name__ == '__main__':
    main()
