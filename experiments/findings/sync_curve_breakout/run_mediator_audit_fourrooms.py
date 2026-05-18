"""Trace-derived mediator audit on FourRooms (GOAL pool).

Approach:
  - Load runs columns (small, scalar HPs).
  - Load trace list-columns; cast each to fixed-width Array(Float64, N)
    so polars→numpy gives a clean (n_cells, n_steps) 2D ndarray
    without Python list overhead.
  - Compute per-step reductions (target_staleness, q_max_growth, etc.)
    directly in numpy on the 2D arrays.
  - Compute short-list scalars (q_mc_calibration, jensen_gap,
    eval_best_burst_mean, env_reward_polarity, learning_curve_auc) via
    iter_rows on a SECOND join with only the short cols.
  - Combine into a small per-cell scalars frame, run proportion_mediated
    with each candidate as mediator under mech-HELD conditioning.
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
# Force-load substrate measurables.
from corroborate_rl.dqn import measurables as _dqn_measurables  # noqa: F401
from corroborate.analyses.paired.proportion_mediated import proportion_mediated as _pm
from corroborate.measurables import get_registered

proportion_mediated = _pm.fn

CORPUS_DIR = Path('experiments/data/capacity_sweep_fourrooms')
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'

# corpus-level pair_by — capacity_sweep_fourrooms varies on
# replay.capacity + seed at fixed gamma + total_steps.
PAIR_BY = ('gamma', 'total_steps', 'seed', 'replay.capacity')

# Short-list candidates (fast, ~50 floats per cell).
SHORT_LIST_CANDIDATES = [
    'q_mc_calibration_pearson',
    'learning_curve_auc',
    'return_at_25pct_steps',
]
SCALAR_TARGETS = ['eval_best_burst_mean', 'jensen_gap', 'env_reward_polarity']

SHORT_TRACE_COLS = ['id', 'episode_length', 'mc_return', 'predicted_q_at_start']

# Per-step candidates (computed via numpy on 2D arrays).
PER_STEP_CANDIDATES = [
    'q_max_growth',           # late_q / early_q of online_max_q
    'target_staleness_late',  # mean late 50% |online-target| / max
    'target_staleness_early', # mean early 25% |online-target| / max
    'v_vs_max_delta_late',    # mean late 50% (online_max - online_mean)
    'td_residual_late',       # mean late 50% |td_error|
    'greedy_match_late',      # mean late 50% (online_argmax == target_argmax)
]


def _mean_window(arr2d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Mean over fractional window [lo, hi] of axis=1. Returns shape (n_cells,)."""
    n = arr2d.shape[1]
    if n == 0:
        return np.full(arr2d.shape[0], np.nan)
    i_lo = int(lo * n)
    i_hi = max(int(hi * n), i_lo + 1)
    return arr2d[:, i_lo:i_hi].mean(axis=1)


def _load_array(traces: pl.DataFrame, col: str) -> np.ndarray:
    """Cast list-typed col to Array dtype + numpy 2D array (n_cells, n_steps)."""
    list_len = traces[col].list.len()[0]
    casted = traces.select(pl.col(col).cast(pl.Array(pl.Float64, list_len)))
    return casted[col].to_numpy()


def compute_per_step_measurables(
    runs: pl.DataFrame,
) -> dict[str, np.ndarray]:
    """Compute per-step measurables directly in numpy. Loads each
    relevant trace column once, casts to Array, gets 2D numpy."""
    print('  loading online_max_q_per_step + target_max_q_per_step...', flush=True)
    t = time.monotonic()
    traces = pl.read_parquet(
        CORPUS_DIR / 'traces.parquet',
        columns=['id', 'online_max_q_per_step', 'target_max_q_per_step'],
    )
    # Re-order to match runs order for column alignment with scalar cell_ids
    traces = runs.select('id').join(traces, on='id', how='inner')
    online_max = _load_array(traces, 'online_max_q_per_step')
    target_max = _load_array(traces, 'target_max_q_per_step')
    print(f'    [{time.monotonic()-t:.1f}s] arrays: {online_max.shape}', flush=True)

    out: dict[str, np.ndarray] = {}

    # q_max_growth: late_quarter / max(|early_quarter|, 1e-9)
    early_q = _mean_window(online_max, 0.0, 0.25)
    late_q = _mean_window(online_max, 0.75, 1.0)
    out['q_max_growth'] = late_q / np.maximum(np.abs(early_q), 1e-9)

    # target_staleness_late: mean(|online-target| / max(|online|,|target|,1e-6) over late 50%)
    n_steps = online_max.shape[1]
    lo, hi = int(0.5 * n_steps), n_steps
    abs_gap = np.abs(online_max[:, lo:hi] - target_max[:, lo:hi])
    denom = np.maximum.reduce([
        np.abs(online_max[:, lo:hi]),
        np.abs(target_max[:, lo:hi]),
        np.full_like(online_max[:, lo:hi], 1e-6),
    ])
    out['target_staleness_late'] = (abs_gap / denom).mean(axis=1)

    # target_staleness_early: mean over early 25%
    lo, hi = 0, int(0.25 * n_steps)
    abs_gap = np.abs(online_max[:, lo:hi] - target_max[:, lo:hi])
    denom = np.maximum.reduce([
        np.abs(online_max[:, lo:hi]),
        np.abs(target_max[:, lo:hi]),
        np.full_like(online_max[:, lo:hi], 1e-6),
    ])
    out['target_staleness_early'] = (abs_gap / denom).mean(axis=1)
    del online_max, target_max, abs_gap, denom, traces

    # v_vs_max_delta_late: mean late 50% (online_max - online_mean)
    print('  loading online_max + online_mean...', flush=True)
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
    print(f'    [{time.monotonic()-t:.1f}s] done', flush=True)

    # td_residual_late: mean late 50% |td_error|
    print('  loading td_error...', flush=True)
    t = time.monotonic()
    traces = pl.read_parquet(
        CORPUS_DIR / 'traces.parquet', columns=['id', 'td_error'],
    )
    traces = runs.select('id').join(traces, on='id', how='inner')
    td = _load_array(traces, 'td_error')
    out['td_residual_late'] = _mean_window(np.abs(td), 0.5, 1.0)
    del td, traces
    print(f'    [{time.monotonic()-t:.1f}s] done', flush=True)

    # greedy_match_late: mean late 50% (online_argmax == target_argmax)
    print('  loading argmax cols...', flush=True)
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
    print(f'    [{time.monotonic()-t:.1f}s] done', flush=True)

    return out


def compute_short_list_measurables(
    runs: pl.DataFrame,
) -> dict[str, list[float]]:
    """Compute short-list-col measurables via iter_rows. Tractable
    because predicted_q_at_start, mc_return are only ~50 floats/cell."""
    print('  loading short trace cols...', flush=True)
    traces = pl.read_parquet(CORPUS_DIR / 'traces.parquet', columns=SHORT_TRACE_COLS)
    joined = runs.join(traces, on='id', how='inner')

    all_names = SHORT_LIST_CANDIDATES + SCALAR_TARGETS
    # de-dupe: jensen_gap appears only once
    seen: set[str] = set()
    cands: list[str] = []
    for n in all_names:
        if n not in seen:
            cands.append(n); seen.add(n)

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

    # Read runs (HPs only), preserve order so we can align numpy arrays.
    runs = pl.read_parquet(CORPUS_DIR / 'runs.parquet')
    runs = runs.select([c for c in dict.fromkeys([
        'id', 'arm_key',
    ] + list(PAIR_BY)) if c in runs.columns])
    print(f'[{time.monotonic()-t0:.1f}s] runs: {len(runs)} rows', flush=True)

    # Pass 1: per-step measurables in numpy
    print('=== Pass 1: per-step measurables (numpy) ===', flush=True)
    per_step = compute_per_step_measurables(runs)
    for k, v in per_step.items():
        n_finite = int(np.isfinite(v).sum())
        print(f'  {k}: {n_finite}/{len(v)} finite', flush=True)

    # Pass 2: short-list measurables via iter_rows
    print('=== Pass 2: short-list measurables (iter_rows) ===', flush=True)
    short = compute_short_list_measurables(runs)
    for k, v in short.items():
        n_finite = sum(1 for x in v if isinstance(x, float) and not math.isnan(x))
        print(f'  {k}: {n_finite}/{len(v)} finite', flush=True)

    # Build small frame
    extra_cols: list[pl.Series] = []
    for k, v in per_step.items():
        extra_cols.append(pl.Series(name=k, values=v.tolist()))
    for k, v in short.items():
        extra_cols.append(pl.Series(name=k, values=v))
    small = runs.with_columns(extra_cols)
    print(f'[{time.monotonic()-t0:.1f}s] small frame: {small.shape}', flush=True)

    cells = small.filter(pl.col('arm_key').is_in(['baseline', DDQN])).to_dicts()
    print(f'cells (baseline+DDQN): {len(cells)}', flush=True)

    # Audit
    print()
    print('=== Mediator audit on FourRooms (mech-HELD: Δ_jens<0) ===\n', flush=True)
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

    out = Path('experiments/findings/sync_curve_breakout/mediator_audit_fourrooms.json')
    out.write_text(json.dumps(results, indent=2))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
