"""Extract per-cell features from any sweep's traces.parquet.

Reusable: takes a sweep directory, reads traces.parquet, writes
TWO small files adjacent to runs.parquet:

  - `per_burst_scalars.parquet` — early/late/peak reductions
    consumed by `build_universal_ddqn_dataset.py`.
  - `per_burst_arrays.parquet` — full per-burst mc/bias
    trajectories consumed by `build_universal_per_burst_dataset.py`.

Sequence after this runs:
  1. extract_per_burst_scalars <sweep_dir>
  2. archive + purge raw traces (small derived files stay local)

Usage: uv run python scripts/extract_per_burst_scalars.py <sweep_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl


_NEEDED: tuple[str, ...] = (
    'id', 'mc_return', 'predicted_q_at_start',
    'online_std_q_per_step',
    'online_max_q_per_step', 'online_min_q_per_step',
)


def _sigma_late_proxy(
    max_arr: np.ndarray, min_arr: np.ndarray,
) -> float:
    if max_arr.size < 2 or min_arr.size != max_arr.size:
        return float('nan')
    half = (max_arr - min_arr) / 2.0
    half = half[~np.isnan(half)]
    if half.size < 2:
        return float('nan')
    late = half[half.size // 2:]
    return float(np.mean(late))


def _per_cell(row: dict[str, object]) -> dict[str, float] | None:
    cell_id = row.get('id')
    if not isinstance(cell_id, str):
        return None
    mc = row.get('mc_return')
    pq = row.get('predicted_q_at_start')
    if mc is None or pq is None:
        return None
    mc_arr = np.asarray(mc, dtype=np.float64)
    pq_arr = np.asarray(pq, dtype=np.float64)
    if mc_arr.ndim != 2 or pq_arr.ndim != 2 or mc_arr.shape != pq_arr.shape:
        return None
    n_bursts = mc_arr.shape[0]
    if n_bursts < 4:
        return None
    bias_per_burst = (pq_arr - mc_arr).mean(axis=1)
    mc_per_burst = mc_arr.mean(axis=1)
    q = max(1, n_bursts // 4)
    out: dict[str, float] = {
        'id': cell_id,  # type: ignore[dict-item]
        'bias_early': float(np.mean(bias_per_burst[:q])),
        'bias_late': float(np.mean(bias_per_burst[-q:])),
        'mc_early': float(np.mean(mc_per_burst[:q])),
        'mc_late': float(np.mean(mc_per_burst[-q:])),
        'mc_peak_burst': float(np.argmax(mc_per_burst)),
        'mc_range': float(np.max(mc_per_burst) - np.min(mc_per_burst)),
    }
    std = row.get('online_std_q_per_step')
    if std is not None:
        std_arr = np.asarray(std, dtype=np.float64)
        std_arr = std_arr[~np.isnan(std_arr)]
        if std_arr.size >= 2:
            late = std_arr[std_arr.size // 2:]
            out['sigma_late'] = float(np.mean(late))
            return out
    mx = row.get('online_max_q_per_step')
    mn = row.get('online_min_q_per_step')
    if mx is not None and mn is not None:
        out['sigma_late'] = _sigma_late_proxy(
            np.asarray(mx, dtype=np.float64),
            np.asarray(mn, dtype=np.float64),
        )
        return out
    out['sigma_late'] = float('nan')
    return out


def _per_cell_arrays(row: dict[str, object]) -> dict[str, object] | None:
    cell_id = row.get('id')
    mc = row.get('mc_return')
    pq = row.get('predicted_q_at_start')
    if not isinstance(cell_id, str) or mc is None or pq is None:
        return None
    mc_arr = np.asarray(mc, dtype=np.float64)
    pq_arr = np.asarray(pq, dtype=np.float64)
    if mc_arr.ndim != 2 or pq_arr.ndim != 2 or mc_arr.shape != pq_arr.shape:
        return None
    if mc_arr.shape[0] < 2:
        return None
    out: dict[str, object] = {
        'id': cell_id,
        'mc_per_burst': mc_arr.mean(axis=1).tolist(),
        'mc_var_per_burst': mc_arr.var(axis=1).tolist(),
        'bias_per_burst': (pq_arr - mc_arr).mean(axis=1).tolist(),
    }
    # Per-burst σ_Q proxy — Hasselt-floor signal at burst granularity.
    # Real σ_Q is variance over actions of online Q-values; we average
    # the (max−min)/2 spread across the per-step series within each
    # burst window. Step-series length per burst = total_steps / n_bursts.
    std = row.get('online_std_q_per_step')
    mx = row.get('online_max_q_per_step')
    mn = row.get('online_min_q_per_step')
    sigma_arr: np.ndarray | None = None
    if std is not None:
        s = np.asarray(std, dtype=np.float64)
        if s.size >= mc_arr.shape[0]:
            sigma_arr = s
    elif mx is not None and mn is not None:
        m1 = np.asarray(mx, dtype=np.float64)
        m2 = np.asarray(mn, dtype=np.float64)
        if m1.size == m2.size and m1.size >= mc_arr.shape[0]:
            sigma_arr = (m1 - m2) / 2.0
    if sigma_arr is not None:
        n_bursts = mc_arr.shape[0]
        # Bin the per-step series into n_bursts windows; average σ_Q per
        # window. Truncated end if step-count not divisible.
        steps_per_burst = sigma_arr.size // n_bursts
        if steps_per_burst > 0:
            trimmed = sigma_arr[: steps_per_burst * n_bursts]
            reshaped = trimmed.reshape(n_bursts, steps_per_burst)
            with np.errstate(invalid='ignore'):
                per_burst_sigma = np.nanmean(reshaped, axis=1)
            out['sigma_per_burst'] = per_burst_sigma.tolist()
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: extract_per_burst_scalars.py <sweep_dir>')
        sys.exit(1)
    sweep_dir = Path(sys.argv[1])
    traces_path = sweep_dir / 'traces.parquet'
    scalars_path = sweep_dir / 'per_burst_scalars.parquet'
    arrays_path = sweep_dir / 'per_burst_arrays.parquet'

    if not traces_path.exists():
        print(f'error: {traces_path} not found')
        sys.exit(1)

    available = pl.read_parquet(traces_path, n_rows=1).columns
    cols = [c for c in _NEEDED if c in available]
    print(f'reading {traces_path} (columns: {cols})')
    df = pl.read_parquet(str(traces_path), columns=cols)
    print(f'  {len(df)} rows')

    scalars: list[dict[str, float]] = []
    arrays: list[dict[str, object]] = []
    for r in df.iter_rows(named=True):
        sc = _per_cell(r)
        if sc is not None:
            scalars.append(sc)
        ar = _per_cell_arrays(r)
        if ar is not None:
            arrays.append(ar)
    print(f'extracted {len(scalars)} scalar rows, {len(arrays)} array rows')

    pl.DataFrame(scalars).write_parquet(str(scalars_path))
    pl.DataFrame(arrays).write_parquet(str(arrays_path))
    print(f'wrote {scalars_path} '
          f'({scalars_path.stat().st_size / 1024:.1f} KB)')
    print(f'wrote {arrays_path} '
          f'({arrays_path.stat().st_size / 1024:.1f} KB)')


if __name__ == '__main__':
    main()
