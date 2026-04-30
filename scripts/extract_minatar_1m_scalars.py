"""Extract per-cell scalar features from MinAtar 1M traces
without consolidating into a single traces.parquet.

Disk-economical: processes each arm-traces file (or restores
from R2) one at a time, computes the per-cell scalars the
universal dataset assembler consumes (sigma_late, bias_early,
bias_late, mc_early, mc_late, mc_peak_burst, mc_range), and
writes to a small parquet at
`experiments/data/minatar_1M/per_burst_scalars.parquet`.

Sources processed in this order:
  1. arm005__Freeway-MinAtar__ddqn (local from previous restore)
  2. minatar_1M_spaceinvaders/{runs,traces}.parquet (local; both
     arms split out)
  3. Re-restore the 5 deleted arm-traces from R2 one at a time,
     process, delete.

After this runs, the universal dataset assembler can be pointed
at minatar_1M/runs.parquet + this per-cell-scalars file."""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import numpy as np
import polars as pl


_MAIN = Path('experiments/data/minatar_1M')
_SI = Path('experiments/data/minatar_1M_spaceinvaders')
_OUT = _MAIN / 'per_burst_scalars.parquet'


_REMOTE_TRACES_TO_RESTORE: tuple[str, ...] = (
    'tmp/arm000__Asterix-MinAtar__vanilla_dqn__traces.parquet',
    'tmp/arm001__Asterix-MinAtar__ddqn__traces.parquet',
    'tmp/arm002__Breakout-MinAtar__vanilla_dqn__traces.parquet',
    'tmp/arm003__Breakout-MinAtar__ddqn__traces.parquet',
    'tmp/arm004__Freeway-MinAtar__vanilla_dqn__traces.parquet',
)


def _sigma_late_proxy(
    max_arr: np.ndarray, min_arr: np.ndarray,
) -> float:
    if max_arr.size < 2 or min_arr.size != max_arr.size:
        return float('nan')
    half_range = (max_arr - min_arr) / 2.0
    half_range = half_range[~np.isnan(half_range)]
    if half_range.size < 2:
        return float('nan')
    late = half_range[half_range.size // 2:]
    return float(np.mean(late))


def _per_cell_scalars(
    row: dict[str, object],
) -> dict[str, float] | None:
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
    # σ_Q proxy from std/max/min if available.
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


_NEEDED_TRACE_COLS: tuple[str, ...] = (
    'id', 'mc_return', 'predicted_q_at_start',
    'online_std_q_per_step',
    'online_max_q_per_step', 'online_min_q_per_step',
)


def _process_traces_file(
    traces_path: Path, accum: list[dict[str, object]],
) -> None:
    print(f'  reading {traces_path.name} ...', flush=True)
    available = pl.read_parquet(traces_path, n_rows=1).columns
    cols = [c for c in _NEEDED_TRACE_COLS if c in available]
    df = pl.read_parquet(str(traces_path), columns=cols)
    n_before = len(accum)
    for row in df.iter_rows(named=True):
        sc = _per_cell_scalars(row)
        if sc is not None:
            accum.append(sc)
    print(f'    +{len(accum) - n_before} cells', flush=True)
    del df


def _restore_one(remote_relpath: str) -> Path:
    cmd = [
        'uv', 'run', 'python', '-m', 'corroborate', 'restore',
        str(_MAIN), '--files', remote_relpath, '--overwrite',
    ]
    print(f'  restore {remote_relpath} ...')
    subprocess.run(cmd, check=True, capture_output=True)
    return _MAIN / remote_relpath


def main() -> None:
    accum: list[dict[str, object]] = []

    # 1. Local arm traces in tmp/.
    local_arm_traces = sorted((_MAIN / 'tmp').glob('arm*__*__traces.parquet'))
    for p in local_arm_traces:
        _process_traces_file(p, accum)
        p.unlink()

    # 2. SpaceInvaders top-level traces.
    si_top = _SI / 'traces.parquet'
    if si_top.exists():
        _process_traces_file(si_top, accum)
        # leave SpaceInvaders top-level alone; user may want
        # the consolidated SpaceInvaders corpus separately.

    # 3. Re-restore + process the 5 deleted arm traces.
    for relpath in _REMOTE_TRACES_TO_RESTORE:
        local = _MAIN / relpath
        if local.exists():
            _process_traces_file(local, accum)
            local.unlink()
            continue
        try:
            local = _restore_one(relpath)
        except subprocess.CalledProcessError as e:
            print(f'  restore failed: {e}')
            continue
        if local.exists():
            _process_traces_file(local, accum)
            local.unlink()

    print(f'\ntotal cells with scalars: {len(accum)}')
    out_df = pl.DataFrame(accum, strict=False)
    out_df.write_parquet(str(_OUT))
    print(f'wrote {_OUT} ({_OUT.stat().st_size / 1024:.1f} KB)')


if __name__ == '__main__':
    # Ensure AWS env vars are set for the CLI restore subprocess.
    if 'AWS_ACCESS_KEY_ID' not in os.environ:
        env_path = Path('.env')
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
    main()
