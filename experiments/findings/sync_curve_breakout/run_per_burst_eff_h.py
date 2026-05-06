"""Per-burst eff_h trajectory + per-burst link analysis on FourRooms
and Acrobot from expectile_3way corpus (paired DDQN vs vanilla).

For each cell, compute per-burst bf = 1 − mean(done over burst's
training-step window). Per-burst eff_h_b = 1/(1 − γ·bf_b). Then ask:
- does DDQN's per-burst eff_h trajectory differ from vanilla's?
- does per-burst eff_h trajectory predict per-burst β-link strength
  (the slope from paired_link_per_burst on bias-as-mediator)?
- when does the DDQN-vs-vanilla eff_h difference emerge in training?
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from scipy.stats import pearsonr

DATA = Path('experiments/data/expectile_3way')
BASELINE = 'baseline'
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
ENVS = ('Acrobot-v1', 'FourRooms-misc')


def _per_burst_bf(done_arr: np.ndarray, n_bursts: int) -> np.ndarray:
    """Bin `done` (length=total_steps) into n_bursts windows, return
    per-burst bf = 1 − mean(done) per window."""
    total = len(done_arr)
    sps = total // n_bursts
    a = done_arr[: n_bursts * sps].reshape(n_bursts, sps)
    return 1.0 - a.mean(axis=1)


def _per_burst_means(col: pl.Series) -> list[list[float]]:
    """For each row's list[list[f64]] (n_bursts × n_eps), return
    a list of per-burst means."""
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


def _analyze(env: str) -> dict[str, object]:
    runs = pl.read_parquet(DATA / 'runs.parquet', columns=['id', 'env_name', 'arm_key', 'seed', 'gamma', 'total_steps', 'eval_every'])
    runs = runs.filter(pl.col('env_name') == env)
    print(f'\n=== env = {env} ===')
    if len(runs) == 0:
        print('  no cells')
        return {'env': env, 'cells': 0}
    gamma = float(runs['gamma'].drop_nans().mean() or 0.99)
    total_steps = int(runs['total_steps'].mean() or 0)
    eval_every = int(runs['eval_every'].mean() or 50000)
    n_bursts = total_steps // eval_every if eval_every > 0 else 10
    print(f'  cells={len(runs)}, γ={gamma}, total_steps={total_steps}, eval_every={eval_every}, n_bursts={n_bursts}')

    traces = pl.read_parquet(
        DATA / 'traces.parquet',
        columns=['id', 'done', 'mc_return', 'predicted_q_at_start'],
    )
    df = runs.join(traces, on='id', how='inner')

    # Per-burst eff_h, mc, bias for each cell
    eff_h_pb_list: list[list[float]] = []
    bf_pb_list: list[list[float]] = []
    mc_pb = _per_burst_means(df['mc_return'])
    qs_pb = _per_burst_means(df['predicted_q_at_start'])
    bias_pb = [
        [q - m for q, m in zip(q_row, mc_row)] for mc_row, q_row in zip(mc_pb, qs_pb)
    ]
    for done_row in df['done']:
        if done_row is None:
            eff_h_pb_list.append([])
            bf_pb_list.append([])
            continue
        done_arr = np.asarray(done_row.to_list(), dtype=np.float64)
        bf_pb = _per_burst_bf(done_arr, n_bursts)
        bf_pb_list.append(bf_pb.tolist())
        eff_h_pb = 1.0 / (1.0 - gamma * bf_pb)
        eff_h_pb_list.append(eff_h_pb.tolist())
    df = df.with_columns(
        eff_h_pb=pl.Series(eff_h_pb_list, dtype=pl.List(pl.Float64)),
        bf_pb=pl.Series(bf_pb_list, dtype=pl.List(pl.Float64)),
        mc_pb=pl.Series(mc_pb, dtype=pl.List(pl.Float64)),
        bias_pb=pl.Series(bias_pb, dtype=pl.List(pl.Float64)),
    )

    # Pair DDQN vs vanilla per seed; aggregate per-burst arrays
    van = df.filter(pl.col('arm_key') == BASELINE).select(['seed', 'eff_h_pb', 'bf_pb', 'mc_pb', 'bias_pb']).rename({'eff_h_pb':'eh_v','bf_pb':'bf_v','mc_pb':'mc_v','bias_pb':'bias_v'})
    ddqn = df.filter(pl.col('arm_key') == DDQN).select(['seed', 'eff_h_pb', 'bf_pb', 'mc_pb', 'bias_pb']).rename({'eff_h_pb':'eh_d','bf_pb':'bf_d','mc_pb':'mc_d','bias_pb':'bias_d'})
    paired = van.join(ddqn, on='seed', how='inner')
    print(f'  paired seeds: {len(paired)}')

    rows = paired.to_dicts()
    eh_van = np.array([r['eh_v'] for r in rows])  # (n_seeds, n_bursts)
    eh_ddqn = np.array([r['eh_d'] for r in rows])
    bias_van = np.array([r['bias_v'] for r in rows])
    bias_ddqn = np.array([r['bias_d'] for r in rows])
    mc_van = np.array([r['mc_v'] for r in rows])
    mc_ddqn = np.array([r['mc_d'] for r in rows])
    delta_eh = eh_ddqn - eh_van
    delta_mc = mc_ddqn - mc_van
    delta_bias = bias_ddqn - bias_van

    print()
    print(f'  Per-burst panel ({n_bursts} bursts × {eh_van.shape[0]} paired seeds):')
    print(f'    {"b":>2} {"mean eh_v":>10} {"mean eh_d":>10} {"Δ eh":>9} {"mean Δbias":>11} {"mean Δmc":>10} {"r(Δeh,Δmc)":>11} {"r(Δbias,Δmc)":>13}')
    for b in range(n_bursts):
        eh_v_b = eh_van[:, b].mean()
        eh_d_b = eh_ddqn[:, b].mean()
        d_eh_b = delta_eh[:, b].mean()
        d_bias_b = delta_bias[:, b].mean()
        d_mc_b = delta_mc[:, b].mean()
        if delta_eh[:, b].std() > 0 and delta_mc[:, b].std() > 0:
            r_eh, _ = pearsonr(delta_eh[:, b], delta_mc[:, b])
        else:
            r_eh = float('nan')
        if delta_bias[:, b].std() > 0 and delta_mc[:, b].std() > 0:
            r_bias, _ = pearsonr(delta_bias[:, b], delta_mc[:, b])
        else:
            r_bias = float('nan')
        print(f'    {b:>2} {eh_v_b:>10.2f} {eh_d_b:>10.2f} {d_eh_b:>+9.3f} {d_bias_b:>+11.3f} {d_mc_b:>+10.3f} {r_eh:>+11.3f} {r_bias:>+13.3f}')

    return {
        'env': env,
        'gamma': gamma,
        'n_bursts': n_bursts,
        'n_paired': len(paired),
        'per_burst': [
            {
                'b': b,
                'mean_eh_v': float(eh_van[:, b].mean()),
                'mean_eh_d': float(eh_ddqn[:, b].mean()),
                'mean_d_eh': float(delta_eh[:, b].mean()),
                'mean_d_bias': float(delta_bias[:, b].mean()),
                'mean_d_mc': float(delta_mc[:, b].mean()),
                'r_d_eh_d_mc': float(pearsonr(delta_eh[:, b], delta_mc[:, b])[0]) if delta_eh[:, b].std() > 0 and delta_mc[:, b].std() > 0 else float('nan'),
                'r_d_bias_d_mc': float(pearsonr(delta_bias[:, b], delta_mc[:, b])[0]) if delta_bias[:, b].std() > 0 and delta_mc[:, b].std() > 0 else float('nan'),
            }
            for b in range(n_bursts)
        ],
    }


def main() -> None:
    out: dict[str, object] = {}
    for env in ENVS:
        out[env] = _analyze(env)
    out_path = Path('experiments/findings/sync_curve_breakout/per_burst_eff_h_panel.json')
    out_path.write_text(json.dumps(out, indent=2))
    print()
    print(f'wrote: {out_path}')


if __name__ == '__main__':
    main()
