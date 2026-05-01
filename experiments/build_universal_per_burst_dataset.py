"""Universal per-burst paired-delta dataset.

Long-format pivot of the universal DDQN dataset: one row per
`(corpus, env, hp_signature, seed, burst)` — both arms paired —
with per-burst delta columns:

  - `delta_mc[burst]` = mc_per_burst_ddqn[k] − mc_per_burst_vanilla[k]
  - `delta_bias[burst]` = bias_per_burst_ddqn[k] − bias_per_burst_vanilla[k]
  - `mc_vanilla[burst]`, `mc_ddqn[burst]`,
    `bias_vanilla[burst]`, `bias_ddqn[burst]` — components

Plus the constant per-cell features (env, n_actions,
log_obs_dim, total_steps, seed) repeated across each burst row.

The cell-mean dataset (`paired_delta_cells.parquet`) collapses
within-cell temporal dynamics; this long-format dataset exposes
them. Storage is small (~MB scale): 1710 cells × ~50 bursts ×
~10 cols ≈ 1MB.

Per-burst arrays come from:
  - For corpora where traces are local (action_dim_sweep,
    ddqn 200k, expectile_3way, ddqn_effective_cohort): read
    on-the-fly and pivot.
  - For minatar_1M (traces deleted to save disk): read from
    `per_burst_arrays.parquet` precomputed by
    `extract_minatar_1m_scalars.py`.

Output: `experiments/data/ddqn_universal/paired_delta_per_burst.parquet`.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

from corroborate.rl.env_catalogue import get as _get_env_spec


_DATA = Path('experiments/data')
_OUT = _DATA / 'ddqn_universal' / 'paired_delta_per_burst.parquet'


_CORPORA: tuple[str, ...] = (
    'action_dim_sweep',
    'ddqn',
    'expectile_3way',
    'ddqn_effective_cohort',
    'minatar_1M',
    'fourrooms_1m',
)


_HP_KEYS: tuple[str, ...] = (
    'replay.capacity', 'replay.batch_size',
    'optimizer.inner.lr', 'sync_period',
    'total_steps', 'reward_scale',
)
# `reward_scale` is part of the cell identity; without it, cells
# from the same seed at different scales clobber each other in
# the pairing dict. Legacy corpora that don't write the column
# return `None` for it via `row.get(...)` — the None hashes
# consistently, so legacy pairing is unchanged.


def _env_features(env: str) -> dict[str, float | int]:
    try:
        spec = _get_env_spec(env)
    except KeyError:
        return {}
    obs = spec.observation_shape
    obs_n = int(np.prod(np.asarray(obs))) if obs else 1
    horizon = float(spec.horizon) if spec.horizon else 1000.0
    return {
        'n_actions': int(spec.n_actions),
        'log_obs_dim': math.log(max(obs_n, 1)),
        'log_horizon': math.log(max(horizon, 1.0)),
    }


def _per_burst_arrays_from_traces(
    traces_path: Path,
) -> dict[str, tuple[list[float], list[float]]]:
    """`{cell_id → (mc_per_burst, bias_per_burst)}` from a
    traces.parquet file. Only `id`, `mc_return`,
    `predicted_q_at_start` columns read."""
    available = pl.read_parquet(traces_path, n_rows=1).columns
    cols = [
        c for c in ('id', 'mc_return', 'predicted_q_at_start')
        if c in available
    ]
    if 'mc_return' not in cols or 'predicted_q_at_start' not in cols:
        return {}
    df = pl.read_parquet(str(traces_path), columns=cols)
    out: dict[str, tuple[list[float], list[float]]] = {}
    for row in df.iter_rows(named=True):
        cell_id = row.get('id')
        mc = row.get('mc_return')
        pq = row.get('predicted_q_at_start')
        if not isinstance(cell_id, str) or mc is None or pq is None:
            continue
        mc_arr = np.asarray(mc, dtype=np.float64)
        pq_arr = np.asarray(pq, dtype=np.float64)
        if mc_arr.ndim != 2 or pq_arr.ndim != 2 or mc_arr.shape != pq_arr.shape:
            continue
        if mc_arr.shape[0] < 2:
            continue
        out[cell_id] = (
            mc_arr.mean(axis=1).tolist(),
            (pq_arr - mc_arr).mean(axis=1).tolist(),
        )
    return out


def _per_burst_arrays_from_precomputed(
    path: Path,
) -> dict[str, tuple[list[float], list[float]]]:
    """`{cell_id → (mc_per_burst, bias_per_burst)}` from a
    precomputed `per_burst_arrays.parquet` (e.g. minatar_1M)."""
    df = pl.read_parquet(path)
    out: dict[str, tuple[list[float], list[float]]] = {}
    for row in df.iter_rows(named=True):
        cell_id = row.get('id')
        mc = row.get('mc_per_burst')
        bias = row.get('bias_per_burst')
        if not isinstance(cell_id, str) or mc is None or bias is None:
            continue
        out[cell_id] = (list(mc), list(bias))
    return out


def _build_corpus_per_burst(corpus: str) -> list[dict[str, object]]:
    base = _DATA / corpus
    runs_path = base / 'runs.parquet'
    if not runs_path.exists():
        print(f'  skip {corpus} — no runs.parquet')
        return []
    print(f'  loading {corpus} ...')
    runs_df = pl.read_parquet(runs_path)
    if 'intervention_name' not in runs_df.columns:
        return []
    arms = set(runs_df['intervention_name'].unique().to_list())
    if 'ddqn' not in arms or 'vanilla_dqn' not in arms:
        return []

    # Per-burst arrays: precomputed has priority over traces.
    precomp = base / 'per_burst_arrays.parquet'
    if precomp.exists():
        per_burst = _per_burst_arrays_from_precomputed(precomp)
    else:
        traces_path = base / 'traces.parquet'
        per_burst = (
            _per_burst_arrays_from_traces(traces_path)
            if traces_path.exists() else {}
        )

    # Bin cells by (env, hp_sig, seed, arm).
    by_key: dict[tuple, dict[str, dict[str, object]]] = {}
    for row in runs_df.iter_rows(named=True):
        env = row.get('env_name')
        arm = row.get('intervention_name')
        seed = row.get('seed')
        if (
            not isinstance(env, str)
            or arm not in {'ddqn', 'vanilla_dqn'}
            or not isinstance(seed, int)
        ):
            continue
        hp_sig = tuple(row.get(k) for k in _HP_KEYS)
        slot = by_key.setdefault((env, hp_sig, seed), {})
        slot[str(arm)] = dict(row)

    out: list[dict[str, object]] = []
    n_paired = 0
    for (env, hp_sig, seed), arms_dict in by_key.items():
        if 'ddqn' not in arms_dict or 'vanilla_dqn' not in arms_dict:
            continue
        d_cell = arms_dict['ddqn']
        v_cell = arms_dict['vanilla_dqn']
        d_id = d_cell.get('id')
        v_id = v_cell.get('id')
        if not (isinstance(d_id, str) and isinstance(v_id, str)):
            continue
        d_arr = per_burst.get(d_id)
        v_arr = per_burst.get(v_id)
        if d_arr is None or v_arr is None:
            continue
        d_mc, d_bias = d_arr
        v_mc, v_bias = v_arr
        n_b = min(len(d_mc), len(v_mc), len(d_bias), len(v_bias))
        if n_b < 2:
            continue
        ef = _env_features(env)
        if not ef:
            continue
        ts = int(d_cell.get('total_steps', 0)) or int(v_cell.get('total_steps', 0))
        n_paired += 1
        for k in range(n_b):
            out.append({
                'corpus': corpus,
                'env_name': env,
                'total_steps': ts,
                'seed': seed,
                'burst_index': k,
                'burst_frac': k / max(n_b - 1, 1),
                'n_bursts': n_b,
                **{kk: hp_sig[i] for i, kk in enumerate(_HP_KEYS) if hp_sig[i] is not None},
                'n_actions': int(ef['n_actions']),
                'log_obs_dim': float(ef['log_obs_dim']),
                'log_horizon': float(ef['log_horizon']),
                'mc_vanilla': float(v_mc[k]),
                'mc_ddqn': float(d_mc[k]),
                'delta_mc': float(d_mc[k] - v_mc[k]),
                'bias_vanilla': float(v_bias[k]),
                'bias_ddqn': float(d_bias[k]),
                'delta_bias': float(d_bias[k] - v_bias[k]),
            })
    print(f'    {corpus}: {n_paired} paired cells, {len(out)} burst rows')
    return out


def main() -> None:
    print(f'Building per-burst universal dataset; corpora={_CORPORA}')
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for corpus in _CORPORA:
        rows.extend(_build_corpus_per_burst(corpus))
    if not rows:
        print('no rows')
        return
    df = pl.DataFrame(rows, strict=False)

    # mc_progress: vanilla's per-burst mc_return normalized to
    # the env-specific [floor, asymptote] range. Captures "where
    # is vanilla in its learning trajectory at burst k" — the
    # strongest single predictor of Δmc per the within-env
    # analysis.
    asymptote = df.group_by('env_name').agg(
        pl.col('mc_vanilla').max().alias('mc_vanilla_asymptote'),
        pl.col('mc_vanilla').min().alias('mc_vanilla_floor'),
    )
    df = df.join(asymptote, on='env_name')
    df = df.with_columns(
        mc_progress=(
            (pl.col('mc_vanilla') - pl.col('mc_vanilla_floor'))
            / (pl.col('mc_vanilla_asymptote')
               - pl.col('mc_vanilla_floor') + 1e-9)
        ).clip(0.0, 1.0),
    )

    df.write_parquet(str(_OUT))
    print(f'\nrows: {len(df)}, file: {_OUT.stat().st_size / 1024:.1f} KB')
    print()
    print('per-corpus burst counts:')
    print(df.group_by('corpus').agg(
        pl.len().alias('rows'),
        pl.col('seed').n_unique().alias('seeds'),
        pl.col('env_name').n_unique().alias('envs'),
        pl.col('n_bursts').max().alias('max_bursts'),
    ).sort('corpus'))


if __name__ == '__main__':
    main()
