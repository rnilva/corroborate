"""Trace-derived mediator audit on Breakout-MinAtar sync=100 (SURVIVAL pool).

Source: minatar_1M corpus, restored Breakout traces from S3.
  tmp/arm002__Breakout-MinAtar__vanilla_dqn__traces.parquet  (1.78 GB)
  tmp/arm003__Breakout-MinAtar__ddqn__traces.parquet         (1.78 GB)

Counterpart to FourRooms audit. Tests whether `target_staleness_late`
is a substrate-level mediator (cross-env) or a FourRooms-specific
finding. Q-explosion regime on Breakout-MinAtar makes the staleness
mechanism's logic strongest here.
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

CORPUS_DIR = Path('experiments/data/minatar_1M')
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
ENV = 'Breakout-MinAtar'

# Breakout: 30 seeds × 1 sync (100) × 1 total_steps (1M) × 1 gamma (0.99)
PAIR_BY = ('seed',)

VANILLA_SHARD = CORPUS_DIR / 'tmp' / 'arm002__Breakout-MinAtar__vanilla_dqn__traces.parquet'
DDQN_SHARD = CORPUS_DIR / 'tmp' / 'arm003__Breakout-MinAtar__ddqn__traces.parquet'

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
    """Cast list-typed col to fixed-width Array(Float64) + numpy 2D."""
    list_len = traces[col].list.len()[0]
    casted = traces.select(pl.col(col).cast(pl.Array(pl.Float64, list_len)))
    return casted[col].to_numpy()


def _load_breakout_traces(cols: list[str], runs: pl.DataFrame) -> pl.DataFrame:
    """Load + concat the two arm shards, restrict to needed cols.

    `cols` MUST include 'id'. The arm shards' IDs are env+arm-specific,
    so we filter via runs.id. Returns a frame in runs-order (matches
    runs.parquet row order so per-cell numpy arrays align)."""
    if 'id' not in cols:
        cols = ['id'] + cols
    print(f'    [load] vanilla shard ({VANILLA_SHARD.stat().st_size / 1e9:.1f}GB), cols={cols}', flush=True)
    t = time.monotonic()
    vanilla = pl.read_parquet(VANILLA_SHARD, columns=cols)
    print(f'    [load] {time.monotonic()-t:.1f}s; vanilla rows: {len(vanilla)}', flush=True)
    t = time.monotonic()
    ddqn = pl.read_parquet(DDQN_SHARD, columns=cols)
    print(f'    [load] {time.monotonic()-t:.1f}s; ddqn rows: {len(ddqn)}', flush=True)
    combined = pl.concat([vanilla, ddqn])
    # Filter to ids that are in our runs frame
    runs_ids = runs['id'].to_list()
    filtered = combined.filter(pl.col('id').is_in(runs_ids))
    # Re-order to match runs row order
    aligned = runs.select('id').join(filtered, on='id', how='inner')
    print(f'    [load] aligned rows: {len(aligned)}', flush=True)
    return aligned


def compute_per_step_measurables(
    runs: pl.DataFrame,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    print('  per-step pass 1: online_max + target_max...', flush=True)
    t = time.monotonic()
    traces = _load_breakout_traces(
        ['online_max_q_per_step', 'target_max_q_per_step'], runs,
    )
    online_max = _load_array(traces, 'online_max_q_per_step')
    target_max = _load_array(traces, 'target_max_q_per_step')
    print(f'    arrays: {online_max.shape}, {time.monotonic()-t:.1f}s', flush=True)

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

    print('  per-step pass 2: online_max + online_mean...', flush=True)
    t = time.monotonic()
    traces = _load_breakout_traces(
        ['online_max_q_per_step', 'online_mean_q_per_step'], runs,
    )
    online_max = _load_array(traces, 'online_max_q_per_step')
    online_mean = _load_array(traces, 'online_mean_q_per_step')
    out['v_vs_max_delta_late'] = _mean_window(online_max - online_mean, 0.5, 1.0)
    del online_max, online_mean, traces
    print(f'    {time.monotonic()-t:.1f}s', flush=True)

    print('  per-step pass 3: td_error...', flush=True)
    t = time.monotonic()
    traces = _load_breakout_traces(['td_error'], runs)
    td = _load_array(traces, 'td_error')
    out['td_residual_late'] = _mean_window(np.abs(td), 0.5, 1.0)
    del td, traces
    print(f'    {time.monotonic()-t:.1f}s', flush=True)

    print('  per-step pass 4: argmax cols...', flush=True)
    t = time.monotonic()
    traces = _load_breakout_traces(
        ['online_argmax_per_step', 'target_argmax_per_step'], runs,
    )
    on_am = _load_array(traces, 'online_argmax_per_step')
    tg_am = _load_array(traces, 'target_argmax_per_step')
    matches = (on_am == tg_am).astype(np.float64)
    out['greedy_match_late'] = _mean_window(matches, 0.5, 1.0)
    del on_am, tg_am, matches, traces
    print(f'    {time.monotonic()-t:.1f}s', flush=True)

    return out


def compute_short_list_measurables(
    runs: pl.DataFrame,
) -> dict[str, list[float]]:
    print('  short-list: load short cols...', flush=True)
    traces = _load_breakout_traces(SHORT_TRACE_COLS, runs)
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
    print(f'    [{time.monotonic()-t:.1f}s] {len(joined)} rows × {len(cands)} measurables', flush=True)
    return out


def main() -> None:
    t0 = time.monotonic()

    # Load runs, filter to Breakout
    runs_full = pl.read_parquet(CORPUS_DIR / 'runs.parquet')
    runs = runs_full.filter(pl.col('env_name') == ENV).select(
        list(dict.fromkeys(['id', 'arm_key'] + list(PAIR_BY)))
    )
    print(f'[{time.monotonic()-t0:.1f}s] Breakout cells: {len(runs)} (expect 60: 30 baseline + 30 ddqn)', flush=True)

    print('=== Pass 1: per-step measurables (numpy) ===', flush=True)
    per_step = compute_per_step_measurables(runs)
    for k, v in per_step.items():
        n_finite = int(np.isfinite(v).sum())
        print(f'  {k}: {n_finite}/{len(v)} finite', flush=True)

    print('=== Pass 2: short-list measurables ===', flush=True)
    short = compute_short_list_measurables(runs)
    for k, v in short.items():
        n_finite = sum(1 for x in v if isinstance(x, float) and not math.isnan(x))
        print(f'  {k}: {n_finite}/{len(v)} finite', flush=True)

    extra: list[pl.Series] = []
    for k, v in per_step.items():
        extra.append(pl.Series(name=k, values=v.tolist()))
    for k, v in short.items():
        extra.append(pl.Series(name=k, values=v))
    small = runs.with_columns(extra)
    print(f'[{time.monotonic()-t0:.1f}s] small frame: {small.shape}', flush=True)

    cells = small.filter(pl.col('arm_key').is_in(['baseline', DDQN])).to_dicts()
    print(f'cells (baseline+DDQN): {len(cells)}', flush=True)

    print('\n=== Mediator audit on Breakout sync=100 (mech-HELD: Δ_jens<0) ===\n', flush=True)
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
            'mediator': mediator, 'n_pairs': result.n_pairs,
            'proportion': prop, 'in_unit_interval': result.in_unit_interval,
            'indirect': result.indirect, 'direct': result.direct,
            'total': result.total, 'slope': result.slope_y_on_m,
        })

    out = Path('experiments/findings/sync_curve_breakout/mediator_audit_breakout.json')
    out.write_text(json.dumps(results, indent=2))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
