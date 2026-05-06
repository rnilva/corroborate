"""Trace-derived mediator audit on Acrobot + MountainCar from
the `expectile_3way` corpus (3 arms × 30 seeds × 5 envs).

Extends the cross-env target_staleness_late panel beyond
FourRooms (capacity_sweep) and Breakout sync=100 (minatar_1M).
Both Acrobot and MountainCar are GOAL-polarity envs, so we
expect the mediation share to be similar to FourRooms (~0.27)
under the Hasselt-bias-correction → low-staleness mechanism.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
from corroborate_rl.dqn import measurables as _dqn_measurables  # noqa: F401
from corroborate.analyses.proportion_mediated import proportion_mediated as _pm
from corroborate.measurables import get_registered

proportion_mediated = _pm.fn

CORPUS_DIR = Path('experiments/data/expectile_3way')
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'

# expectile_3way: fixed gamma + total_steps + sync_period; pair on seed.
PAIR_BY = ('seed',)

PER_STEP_CANDIDATES = [
    'q_max_growth',
    'target_staleness_late',
    'target_staleness_early',
    'v_vs_max_delta_late',
    'td_residual_late',
    'greedy_match_late',
]
SHORT_LIST_CANDIDATES = [
    'q_mc_calibration_pearson',
    'learning_curve_auc',
    'return_at_25pct_steps',
]
SCALAR_TARGETS = ['eval_best_burst_mean', 'jensen_gap', 'env_reward_polarity']
SHORT_TRACE_COLS = ['id', 'episode_length', 'mc_return', 'predicted_q_at_start']


def _mean_window(arr2d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    n = arr2d.shape[1]
    if n == 0:
        return np.full(arr2d.shape[0], np.nan)
    i_lo = int(lo * n)
    i_hi = max(int(hi * n), i_lo + 1)
    return arr2d[:, i_lo:i_hi].mean(axis=1)


def _load_array(traces: pl.DataFrame, col: str) -> np.ndarray:
    list_len = traces[col].list.len()[0]
    casted = traces.select(pl.col(col).cast(pl.Array(pl.Float64, list_len)))
    return casted[col].to_numpy()


def compute_per_step_measurables(
    runs: pl.DataFrame,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    schema = pl.scan_parquet(CORPUS_DIR / 'traces.parquet').collect_schema()
    avail = set(schema.names())

    print('  per-step pass 1: online_max + target_max...', flush=True)
    t = time.monotonic()
    cols = [c for c in ['online_max_q_per_step', 'target_max_q_per_step'] if c in avail]
    if 'online_max_q_per_step' not in avail or 'target_max_q_per_step' not in avail:
        print(f'    (missing; skipping target_staleness)', flush=True)
        nan = np.full(len(runs), np.nan)
        out['q_max_growth'] = nan.copy()
        out['target_staleness_late'] = nan.copy()
        out['target_staleness_early'] = nan.copy()
    else:
        traces = pl.read_parquet(CORPUS_DIR / 'traces.parquet', columns=['id'] + cols)
        traces = runs.select('id').join(traces, on='id', how='inner')
        online_max = _load_array(traces, 'online_max_q_per_step')
        target_max = _load_array(traces, 'target_max_q_per_step')
        early_q = _mean_window(online_max, 0.0, 0.25)
        late_q = _mean_window(online_max, 0.75, 1.0)
        out['q_max_growth'] = late_q / np.maximum(np.abs(early_q), 1e-9)
        n_steps = online_max.shape[1]
        for name, lo, hi in (('target_staleness_late', 0.5, 1.0),
                             ('target_staleness_early', 0.0, 0.25)):
            i_lo, i_hi = int(lo * n_steps), max(int(hi * n_steps), int(lo * n_steps) + 1)
            slc_o = online_max[:, i_lo:i_hi]
            slc_t = target_max[:, i_lo:i_hi]
            abs_gap = np.abs(slc_o - slc_t)
            denom = np.maximum.reduce([
                np.abs(slc_o), np.abs(slc_t), np.full_like(slc_o, 1e-6),
            ])
            out[name] = (abs_gap / denom).mean(axis=1)
        del online_max, target_max, traces
        print(f'    {time.monotonic()-t:.1f}s, shape (n_cells, n_steps)', flush=True)

    if 'online_mean_q_per_step' in avail and 'online_max_q_per_step' in avail:
        print('  per-step pass 2: v_vs_max_delta_late...', flush=True)
        t = time.monotonic()
        traces = pl.read_parquet(
            CORPUS_DIR / 'traces.parquet',
            columns=['id', 'online_max_q_per_step', 'online_mean_q_per_step'],
        )
        traces = runs.select('id').join(traces, on='id', how='inner')
        online_max = _load_array(traces, 'online_max_q_per_step')
        online_mean = _load_array(traces, 'online_mean_q_per_step')
        out['v_vs_max_delta_late'] = _mean_window(online_max - online_mean, 0.5, 1.0)
        del online_max, online_mean, traces
        print(f'    {time.monotonic()-t:.1f}s', flush=True)
    else:
        out['v_vs_max_delta_late'] = np.full(len(runs), np.nan)

    if 'td_error' in avail:
        print('  per-step pass 3: td_residual_late...', flush=True)
        t = time.monotonic()
        traces = pl.read_parquet(CORPUS_DIR / 'traces.parquet', columns=['id', 'td_error'])
        traces = runs.select('id').join(traces, on='id', how='inner')
        td = _load_array(traces, 'td_error')
        out['td_residual_late'] = _mean_window(np.abs(td), 0.5, 1.0)
        del td, traces
        print(f'    {time.monotonic()-t:.1f}s', flush=True)
    else:
        out['td_residual_late'] = np.full(len(runs), np.nan)

    if 'online_argmax_per_step' in avail and 'target_argmax_per_step' in avail:
        print('  per-step pass 4: greedy_match_late...', flush=True)
        t = time.monotonic()
        traces = pl.read_parquet(
            CORPUS_DIR / 'traces.parquet',
            columns=['id', 'online_argmax_per_step', 'target_argmax_per_step'],
        )
        traces = runs.select('id').join(traces, on='id', how='inner')
        on_am = _load_array(traces, 'online_argmax_per_step')
        tg_am = _load_array(traces, 'target_argmax_per_step')
        matches = (on_am == tg_am).astype(np.float64)
        out['greedy_match_late'] = _mean_window(matches, 0.5, 1.0)
        del on_am, tg_am, matches, traces
        print(f'    {time.monotonic()-t:.1f}s', flush=True)
    else:
        out['greedy_match_late'] = np.full(len(runs), np.nan)

    return out


def compute_short_list_measurables(
    runs: pl.DataFrame,
) -> dict[str, list[float]]:
    print('  short-list...', flush=True)
    traces = pl.read_parquet(CORPUS_DIR / 'traces.parquet', columns=SHORT_TRACE_COLS)
    traces = runs.select('id').join(traces, on='id', how='inner')
    joined = runs.join(traces, on='id', how='inner')

    cands = list(dict.fromkeys(SHORT_LIST_CANDIDATES + SCALAR_TARGETS))
    out: dict[str, list[float]] = {c: [] for c in cands}
    t = time.monotonic()
    for row in joined.iter_rows(named=True):
        for cand in cands:
            m = get_registered(cand)
            try:
                v = m.fn(row) if m is not None else float('nan')
                out[cand].append(float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float('nan'))
            except Exception:
                out[cand].append(float('nan'))
    print(f'    {time.monotonic()-t:.1f}s, {len(joined)} rows × {len(cands)} measurables', flush=True)
    return out


def audit_env(env_name: str, all_runs: pl.DataFrame) -> list[dict]:
    print(f'\n=== {env_name} ===', flush=True)
    env_runs = all_runs.filter(pl.col('env_name') == env_name)
    print(f'cells: {len(env_runs)}', flush=True)

    per_step = compute_per_step_measurables(env_runs)
    short = compute_short_list_measurables(env_runs)

    extra: list[pl.Series] = []
    for k, v in per_step.items():
        extra.append(pl.Series(name=k, values=v.tolist()))
    for k, v in short.items():
        extra.append(pl.Series(name=k, values=v))
    small = env_runs.with_columns(extra)
    cells = small.filter(pl.col('arm_key').is_in(['baseline', DDQN])).to_dicts()
    print(f'cells (baseline+DDQN): {len(cells)}', flush=True)

    print()
    hdr = ('mediator', 'n_pairs', 'proportion', 'in_unit', 'indirect', 'direct', 'verdict')
    fmt = '{:<28} {:>8} {:>11} {:>8} {:>10} {:>10} {:>12}'
    print(fmt.format(*hdr))
    print('-' * 100, flush=True)

    all_candidates = PER_STEP_CANDIDATES + SHORT_LIST_CANDIDATES
    results = []
    for mediator in all_candidates:
        result = proportion_mediated(
            cells=cells,
            target='eval_best_burst_mean',
            mediator=mediator,
            treatment_arm=DDQN,
            baseline_arm='baseline',
            pair_by=PAIR_BY,
            upstream_source='jensen_gap',
            upstream_max_delta=0.0,
        )
        prop = result.proportion
        if math.isnan(prop):
            verdict = 'NaN'
        elif not result.in_unit_interval:
            verdict = 'INVALID'
        elif prop >= 0.2:
            verdict = '** HELD **'
        else:
            verdict = 'NULL'
        prop_str = f'{prop:+.3f}' if not math.isnan(prop) else 'nan'
        in_unit = 'T' if result.in_unit_interval else 'F'
        print(fmt.format(
            mediator, str(result.n_pairs), prop_str, in_unit,
            f'{result.indirect:+.3f}' if not math.isnan(result.indirect) else 'nan',
            f'{result.direct:+.3f}' if not math.isnan(result.direct) else 'nan',
            verdict,
        ), flush=True)
        results.append({
            'env': env_name, 'mediator': mediator, 'n_pairs': result.n_pairs,
            'proportion': prop, 'in_unit_interval': result.in_unit_interval,
            'indirect': result.indirect, 'direct': result.direct,
            'total': result.total, 'slope': result.slope_y_on_m,
        })
    return results


def main() -> None:
    runs_full = pl.read_parquet(CORPUS_DIR / 'runs.parquet')
    runs = runs_full.select(['id', 'env_name', 'arm_key', 'seed', 'gamma', 'total_steps', 'sync_period'])
    print(f'corpus: {len(runs)} cells', flush=True)

    target_envs = ['Acrobot-v1', 'MountainCar-v0']
    all_results = []
    for env in target_envs:
        all_results.extend(audit_env(env, runs))

    out = Path('experiments/findings/sync_curve_breakout/mediator_audit_expectile_3way.json')
    out.write_text(json.dumps(all_results, indent=2))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
