"""Exploration script — truncation-aware mediators on the 200k
corpus.

Question: do per-env per-cell truncation strategies change the
mediator-vs-outcome correlation picture?

The current `_late` mediators (in `runs_with_mediators.parquet`)
average over the last 50% of each cell's trajectory. But cells'
peak-eval-burst-step varies per env (Acrobot ~60–70%, Freeway
~35–45%); a fixed late-50% window contaminates with post-peak
trajectory for early-peaking envs.

This script computes mediators on-the-fly under three strategies:

1. `no_trunc` — late 50% of full trajectory (matches existing
   `_late` measurables; sanity-check baseline).
2. `peak_truncated` — truncate trajectory at peak step, then
   compute late 50% of the truncated trace.
3. `peak_centered` — window of width 25% of trajectory centered
   at peak step.

For each (env, arm, strategy, mediator), compute:
- mean across cells (per-arm summary)
- per-env Pearson r vs `outcome.eval_best_burst_mean`

Output: per-env table comparing within-env Pearson under each
strategy. Lets us see whether truncation reveals stronger or
weaker mediator-vs-outcome signals.

200k-only because mc_return for 50k cells lives in zarr (would
need a separate read path); 200k cells have it as a parquet
nested-list column.

Run: `uv run python experiments/explore_truncation.py`."""
from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import scipy.stats as ss


_DATA_DIR = Path(__file__).parent / 'data' / 'ddqn'
_RUNS_PATH = _DATA_DIR / 'runs_with_mediators.parquet'
_TRACES_PATH = _DATA_DIR / 'traces.parquet'


# ============ Truncation strategies ============

def _trunc_indices(
    n: int, peak_idx: int, strategy: str,
) -> tuple[int, int]:
    """Return (lo, hi) slice indices into a length-`n` array
    under each strategy. `peak_idx` is in the same units as `n`
    (per-step trajectory: peak training step; per-burst: peak
    burst index).

    All strategies guarantee `lo < hi <= n` and `hi > 0` so the
    caller can safely slice. NaN-degenerate cases return
    `(0, 0)` and the caller emits NaN for the mediator."""
    if n <= 0 or peak_idx < 0 or peak_idx > n:
        return 0, 0
    if strategy == 'no_trunc':
        # Late 50% of full trajectory.
        return n // 2, n
    if strategy == 'peak_truncated':
        # Truncate at peak; late 50% of truncated.
        if peak_idx < 4:
            return 0, 0
        return peak_idx // 2, peak_idx
    if strategy == 'peak_centered':
        # Width 25% of full trajectory, centered at peak.
        half_w = max(1, n // 8)  # 12.5% on each side
        lo = max(0, peak_idx - half_w)
        hi = min(n, peak_idx + half_w)
        if hi - lo < 4:
            return 0, 0
        return lo, hi
    raise ValueError(f'unknown strategy {strategy!r}')


# ============ On-the-fly mediator computation ============

def _q_gap_truncated(
    online_max: Sequence[float],
    online_min: Sequence[float],
    peak_step: int,
    total_steps: int,
    strategy: str,
) -> float:
    """Mean of (online_max - online_min) over the truncation
    window, in q-gap units. Maps `peak_step` (training step) to
    array index by `peak_idx = peak_step * n / total_steps`."""
    arr_max = np.asarray(online_max, dtype=np.float64)
    arr_min = np.asarray(online_min, dtype=np.float64)
    n = len(arr_max)
    peak_idx = int(peak_step * n / max(1, total_steps))
    lo, hi = _trunc_indices(n, peak_idx, strategy)
    if lo >= hi:
        return float('nan')
    return float(np.mean(arr_max[lo:hi] - arr_min[lo:hi]))


def _td_residual_truncated(
    td_error: Sequence[float],
    peak_step: int,
    total_steps: int,
    strategy: str,
) -> float:
    """Mean of |td_error| over the truncation window."""
    arr = np.asarray(td_error, dtype=np.float64)
    n = len(arr)
    peak_idx = int(peak_step * n / max(1, total_steps))
    lo, hi = _trunc_indices(n, peak_idx, strategy)
    if lo >= hi:
        return float('nan')
    return float(np.mean(np.abs(arr[lo:hi])))


def _learning_curve_auc_truncated(
    mc_return: Sequence[Sequence[float]],
    peak_step: int,
    total_steps: int,
    strategy: str,
) -> float:
    """Trapezoidal AUC of per-burst mean return over the
    truncation window. `mc_return` is `(n_bursts, K)`; peak burst
    index inferred from `peak_step / eval_every` where
    `eval_every = total_steps / n_bursts`."""
    if not mc_return:
        return float('nan')
    burst_means = np.asarray(
        [float(np.mean(np.asarray(b, dtype=np.float64))) for b in mc_return],
        dtype=np.float64,
    )
    n_bursts = len(burst_means)
    if n_bursts < 2:
        return float('nan')
    eval_every = max(1, total_steps // n_bursts)
    peak_burst_idx = int(min(n_bursts - 1, peak_step // eval_every))
    lo, hi = _trunc_indices(n_bursts, peak_burst_idx, strategy)
    if hi - lo < 2:
        return float('nan')
    sub = burst_means[lo:hi]
    return float(np.trapezoid(sub) / (sub.size - 1))


# ============ Pearson per env ============

def _pearson_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 4:
        return float('nan'), float('nan')
    xs = x[finite]
    ys = y[finite]
    if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return float('nan'), float('nan')
    r, p = ss.pearsonr(xs, ys)  # pyright: ignore[reportUnknownMemberType]
    return float(r), float(p)  # pyright: ignore[reportArgumentType]


# ============ Driver ============

_STRATEGIES: tuple[str, ...] = (
    'no_trunc', 'peak_truncated', 'peak_centered',
)
_OUTCOME = 'outcome.eval_best_burst_mean'


def main() -> None:
    if not _RUNS_PATH.exists() or not _TRACES_PATH.exists():
        print(f'corpus missing under {_DATA_DIR}', file=sys.stderr)
        sys.exit(1)

    runs = pl.read_parquet(_RUNS_PATH).filter(
        pl.col('total_steps') == 200000,
    ).select(
        'id', 'env_name', 'arm_key', 'total_steps',
        'outcome.eval_best_burst_step',
        _OUTCOME,
    )
    print(f'runs (200k): {len(runs)} cells, '
          f'{runs["env_name"].n_unique()} envs')

    traces = pl.scan_parquet(_TRACES_PATH).filter(
        pl.col('total_steps') == 200000,
    ).select(
        'id',
        'online_max_q_per_step', 'online_min_q_per_step',
        'td_error', 'mc_return',
    ).collect()
    print(f'traces (200k): {len(traces)} rows')

    df = traces.join(runs, on='id')
    print(f'joined: {len(df)} rows')

    # Compute truncated mediators per cell × strategy, in plain
    # numpy. Polars' map_elements is heavy here; iterate rows.
    rows: list[dict[str, object]] = []
    for r in df.iter_rows(named=True):
        env: str = r['env_name']  # pyright: ignore[reportAny]
        arm: str = r['arm_key']  # pyright: ignore[reportAny]
        peak_step = int(r['outcome.eval_best_burst_step'])  # pyright: ignore[reportAny]
        total_steps = int(r['total_steps'])  # pyright: ignore[reportAny]
        outcome = float(r[_OUTCOME])  # pyright: ignore[reportAny]
        for strat in _STRATEGIES:
            q_gap = _q_gap_truncated(
                r['online_max_q_per_step'],  # pyright: ignore[reportAny]
                r['online_min_q_per_step'],  # pyright: ignore[reportAny]
                peak_step, total_steps, strat,
            )
            td_res = _td_residual_truncated(
                r['td_error'],  # pyright: ignore[reportAny]
                peak_step, total_steps, strat,
            )
            auc = _learning_curve_auc_truncated(
                r['mc_return'],  # pyright: ignore[reportAny]
                peak_step, total_steps, strat,
            )
            rows.append({
                'env_name': env, 'arm_key': arm,
                'strategy': strat,
                'q_gap_trunc': q_gap,
                'td_residual_trunc': td_res,
                'learning_curve_auc_trunc': auc,
                'outcome': outcome,
            })
    out_df = pl.DataFrame(rows)
    print(f'computed mediators × strategies: {len(out_df)} rows')

    # Per-env Pearson r per strategy per mediator.
    print()
    print('=' * 110)
    print(f'Per-env Pearson r vs {_OUTCOME} (200k corpus, '
          f'pooled across arms)')
    print('=' * 110)
    print(
        f'{"env":<26} {"strategy":<16} '
        f'{"q_gap r":>10} {"td_resid r":>12} {"lc_auc r":>10}'
    )
    envs: list[str] = sorted(set(out_df['env_name'].to_list()))
    for env in envs:
        env_df = out_df.filter(pl.col('env_name') == env)
        if env_df.height < 6:
            continue
        for strat in _STRATEGIES:
            sub = env_df.filter(pl.col('strategy') == strat)
            if sub.height < 6:
                continue
            outcome_arr = np.asarray(
                sub['outcome'].to_list(), dtype=np.float64,
            )
            r_q, _ = _pearson_safe(
                np.asarray(sub['q_gap_trunc'].to_list(), dtype=np.float64),
                outcome_arr,
            )
            r_td, _ = _pearson_safe(
                np.asarray(sub['td_residual_trunc'].to_list(), dtype=np.float64),
                outcome_arr,
            )
            r_auc, _ = _pearson_safe(
                np.asarray(sub['learning_curve_auc_trunc'].to_list(), dtype=np.float64),
                outcome_arr,
            )

            def _fmt(r: float) -> str:
                if math.isnan(r):
                    return f'{"nan":>10}'
                marker = '*' if abs(r) > 0.5 else ' '
                return f'{r:>+9.3f}{marker}'
            print(
                f'{env:<26} {strat:<16} '
                f'{_fmt(r_q)} {_fmt(r_td):>12} {_fmt(r_auc)}'
            )

    # Cross-env summary: mean |r| per strategy per mediator.
    print()
    print('=' * 110)
    print('Cross-env summary: mean |r| per strategy (higher = stronger '
          'mediator-vs-outcome signal)')
    print('=' * 110)
    print(
        f'  {"strategy":<20} {"mean|q_gap r|":>14} '
        f'{"mean|td_res r|":>16} {"mean|lc_auc r|":>16}'
    )
    for strat in _STRATEGIES:
        rs_q: list[float] = []
        rs_td: list[float] = []
        rs_auc: list[float] = []
        for env in envs:
            env_df = out_df.filter(pl.col('env_name') == env)
            sub = env_df.filter(pl.col('strategy') == strat)
            if sub.height < 6:
                continue
            outcome_arr = np.asarray(
                sub['outcome'].to_list(), dtype=np.float64,
            )
            for arr_col, dest in (
                ('q_gap_trunc', rs_q),
                ('td_residual_trunc', rs_td),
                ('learning_curve_auc_trunc', rs_auc),
            ):
                arr = np.asarray(
                    sub[arr_col].to_list(), dtype=np.float64,
                )
                r, _ = _pearson_safe(arr, outcome_arr)
                if not math.isnan(r):
                    dest.append(abs(r))

        def _mean(xs: list[float]) -> str:
            return f'{(sum(xs)/len(xs)):>14.3f}' if xs else f'{"nan":>14}'
        print(
            f'  {strat:<20} {_mean(rs_q)} '
            f'{_mean(rs_td):>16} {_mean(rs_auc):>16}'
        )


if __name__ == '__main__':
    main()
    sys.exit(0)
