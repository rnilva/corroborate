"""Universal DDQN paired-delta dataset.

Unifies every locally-available corpus with both `ddqn` and
`vanilla_dqn` arms. For each (corpus, env, hp_signature, seed)
where both arms exist, produces ONE differential cell with:

  - `delta_jensen_gap` = jensen_gap_ddqn − jensen_gap_vanilla
  - `delta_outcome_best_burst` = best_burst_ddqn − best_burst_vanilla
  - `delta_outcome_final` = final_mean_ddqn − final_mean_vanilla
  - `dormancy_gap_avg` = mean(dormancy_gap_ddqn,
                             dormancy_gap_vanilla)
    (per-cell using the alias path: σ_Q from
    online_std_q_per_step + n_actions from env catalogue)
  - env-feature columns: `n_actions`, `log_obs_dim`,
    `log_horizon`
  - `convergence_class_vanilla` ∈ {solved, partial, unsolved,
    absent} — derived per (env, total_steps) on the vanilla
    arm via `corroborate.rl.convergence.classify_envs`.
  - `corpus`, `env_name`, `total_steps`, `seed`,
    `replay.capacity`, `replay.batch_size`,
    `optimizer.inner.lr`, `sync_period`

Output: `experiments/data/ddqn_universal/paired_delta_cells.parquet`.
Small (kB-MB scale); the trace files stay where they are.

Goal: this is the input to scope discovery. The bridges-as-zoo
approach is corpus-by-corpus; here we collapse into one row per
paired comparison and ask which conditions on this row predict
`delta_outcome_best_burst > 0` (where DDQN actually helps).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

from corroborate.persistence import read_runrows
from corroborate.rl.convergence import classify_envs
from corroborate.rl.env_catalogue import get as _get_env_spec


_DATA = Path('experiments/data')
_OUT_DIR = _DATA / 'ddqn_universal'
_OUT_FILE = _OUT_DIR / 'paired_delta_cells.parquet'


# Corpora with both ddqn and vanilla_dqn arms locally available.
# `nstep_intervention*` arms are ddqn_1step / ddqn_3step (no
# plain ddqn or vanilla_dqn) so they're excluded from the
# universal join. cartpole_hp* corpora are vanilla-only.
# `minatar_1M` doesn't have a consolidated traces.parquet
# (disk-economical: per-burst scalars precomputed in
# `per_burst_scalars.parquet` via
# `scripts/extract_minatar_1m_scalars.py`). The assembler reads
# the precomputed scalars when traces are absent.
_CORPORA: tuple[str, ...] = (
    'action_dim_sweep',
    'ddqn',
    'expectile_3way',
    'ddqn_better_hp',
    'ddqn_effective_cohort',
    'minatar_1M',
    'fourrooms_1m',
)


_PRECOMPUTED_SCALARS_NAME: str = 'per_burst_scalars.parquet'


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


def _per_burst_features_per_id(
    traces_path: Path,
) -> dict[str, dict[str, float]]:
    """Read per-cell (n_bursts, K)-shaped `mc_return` and
    `predicted_q_at_start`; reduce to early/late/range scalars
    that capture temporal structure (rev 9: mechanism operates
    early on FourRooms; outcome stable across bursts).

    Returns `{cell_id → {feature → value}}` for the joining step.
    Features are computed per-cell, NOT per-arm — the caller
    pairs them at the (env, hp_sig, seed) level after."""
    available = pl.read_parquet(traces_path, n_rows=1).columns
    needed = ['id', 'mc_return', 'predicted_q_at_start']
    if not all(c in available for c in needed):
        return {}
    df = pl.read_parquet(traces_path, columns=needed)
    out: dict[str, dict[str, float]] = {}
    for row in df.iter_rows(named=True):
        cell_id = row['id']
        mc = row['mc_return']
        pq = row['predicted_q_at_start']
        if not isinstance(cell_id, str) or mc is None or pq is None:
            continue
        mc_arr = np.asarray(mc, dtype=np.float64)
        pq_arr = np.asarray(pq, dtype=np.float64)
        if mc_arr.ndim != 2 or pq_arr.ndim != 2 or mc_arr.shape != pq_arr.shape:
            continue
        n_bursts = mc_arr.shape[0]
        if n_bursts < 4:
            continue
        bias_per_burst = (pq_arr - mc_arr).mean(axis=1)  # (n_bursts,)
        mc_per_burst = mc_arr.mean(axis=1)               # (n_bursts,)
        # Early = first quarter, late = last quarter.
        q = max(1, n_bursts // 4)
        out[cell_id] = {
            'bias_early': float(np.mean(bias_per_burst[:q])),
            'bias_late': float(np.mean(bias_per_burst[-q:])),
            'bias_peak_burst': float(np.argmax(np.abs(bias_per_burst))),
            'mc_early': float(np.mean(mc_per_burst[:q])),
            'mc_late': float(np.mean(mc_per_burst[-q:])),
            'mc_peak_burst': float(np.argmax(mc_per_burst)),
            'mc_range': float(np.max(mc_per_burst) - np.min(mc_per_burst)),
        }
    return out


def _late_sigma_q_per_id(traces_path: Path) -> dict[str, float]:
    """σ_Q late-window mean per cell — read traces ONCE per
    corpus, project to cell-id scalar.

    Primary source: `online_std_q_per_step` (true per-step σ
    across actions, from `Q_TRACE_REDUCTIONS`).

    Fallback for older corpora (e.g. ddqn 200k pre-dating that
    reduction): half-range `(max - min) / 2` as a σ proxy.
    Biased upward by a factor of √(action_dim/2) approximately
    (the proxy overstates true σ for |A|>2), but rank-preserving
    for the dormancy ordering. The downstream dormancy threshold
    `gap > 0` may slightly shift; the relative classification
    across cells of the same env should hold."""
    available = pl.read_parquet(traces_path, n_rows=1).columns
    if 'online_std_q_per_step' in available:
        df = pl.read_parquet(
            traces_path,
            columns=['id', 'online_std_q_per_step'],
        )
        out: dict[str, float] = {}
        for row in df.iter_rows(named=True):
            cell_id = row['id']
            arr = row['online_std_q_per_step']
            if not isinstance(cell_id, str) or arr is None:
                continue
            v = np.asarray(arr, dtype=np.float64)
            v = v[~np.isnan(v)]
            if v.size < 2:
                out[cell_id] = float('nan')
                continue
            late = v[v.size // 2:]
            out[cell_id] = float(np.mean(late))
        return out
    if {'online_max_q_per_step', 'online_min_q_per_step'} <= set(available):
        df = pl.read_parquet(
            traces_path,
            columns=['id', 'online_max_q_per_step', 'online_min_q_per_step'],
        )
        out_proxy: dict[str, float] = {}
        for row in df.iter_rows(named=True):
            cell_id = row['id']
            mx = row['online_max_q_per_step']
            mn = row['online_min_q_per_step']
            if not isinstance(cell_id, str) or mx is None or mn is None:
                continue
            mx_arr = np.asarray(mx, dtype=np.float64)
            mn_arr = np.asarray(mn, dtype=np.float64)
            if mx_arr.size < 2 or mn_arr.size != mx_arr.size:
                out_proxy[cell_id] = float('nan')
                continue
            half_range = (mx_arr - mn_arr) / 2.0
            half_range = half_range[~np.isnan(half_range)]
            if half_range.size < 2:
                out_proxy[cell_id] = float('nan')
                continue
            late = half_range[half_range.size // 2:]
            out_proxy[cell_id] = float(np.mean(late))
        return out_proxy
    return {}


def _dormancy_gap(
    sigma_late: float, n_actions: int, observed_bias: float | None,
) -> float:
    if (
        n_actions < 2
        or math.isnan(sigma_late)
        or observed_bias is None
        or math.isnan(float(observed_bias))
    ):
        return float('nan')
    floor = sigma_late * math.sqrt(2.0 * math.log(n_actions))
    return max(0.0, floor - max(0.0, float(observed_bias)))


def _convergence_class_per_env_step(
    runs_path: Path,
) -> dict[tuple[str, int], str]:
    """Map (env, total_steps) → convergence_class on the vanilla
    arm. `absent` covers envs without a defensible solve threshold
    OR envs with no vanilla cells at this total_steps."""
    runs_obj = read_runrows(str(runs_path))
    by_step: dict[int, list[object]] = {}
    for r in runs_obj:
        ts = r.measurements.get('total_steps')
        arm = r.measurements.get('intervention_name')
        if not isinstance(ts, int) or arm != 'vanilla_dqn':
            continue
        by_step.setdefault(ts, []).append(r)
    out: dict[tuple[str, int], str] = {}
    for ts, rs in by_step.items():
        classes = classify_envs(rs)  # type: ignore[arg-type]
        for env, ec in classes.items():
            out[(env, ts)] = ec.classification
    return out


def _build_corpus_cells(corpus: str) -> list[dict[str, object]] | None:
    base = _DATA / corpus
    runs_path = base / 'runs.parquet'
    traces_path = base / 'traces.parquet'
    if not runs_path.exists():
        print(f'  skip {corpus} — no runs.parquet')
        return None
    print(f'  loading {corpus} ...')
    runs_df = pl.read_parquet(runs_path)
    if 'intervention_name' not in runs_df.columns:
        return None
    arms = set(runs_df['intervention_name'].unique().to_list())
    if 'ddqn' not in arms or 'vanilla_dqn' not in arms:
        print(f'    skip {corpus} — no ddqn+vanilla pair')
        return None

    sigma_by_id: dict[str, float] = {}
    burst_by_id: dict[str, dict[str, float]] = {}
    precomputed_path = base / _PRECOMPUTED_SCALARS_NAME
    if precomputed_path.exists():
        # Precomputed per-cell scalars (e.g. minatar_1M, where
        # per-arm traces were extracted to a small parquet to
        # avoid consolidating multi-GB traces).
        scalars_df = pl.read_parquet(precomputed_path)
        for row in scalars_df.iter_rows(named=True):
            cell_id = row.get('id')
            if not isinstance(cell_id, str):
                continue
            burst_by_id[cell_id] = {
                k: float(row[k])
                for k in ('bias_early', 'bias_late',
                          'mc_early', 'mc_late',
                          'mc_peak_burst', 'mc_range')
                if k in row and isinstance(row[k], (int, float))
            }
            sigma_v = row.get('sigma_late')
            if isinstance(sigma_v, (int, float)):
                sigma_by_id[cell_id] = float(sigma_v)
    elif traces_path.exists():
        sigma_by_id = _late_sigma_q_per_id(traces_path)
        burst_by_id = _per_burst_features_per_id(traces_path)

    conv_class = _convergence_class_per_env_step(runs_path)

    # Bin all cells by (env, hp_signature, seed, arm).
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
        key = (env, hp_sig, seed)
        slot = by_key.setdefault(key, {})
        slot[str(arm)] = dict(row)

    cells_out: list[dict[str, object]] = []
    for (env, hp_sig, seed), arms_dict in by_key.items():
        if 'ddqn' not in arms_dict or 'vanilla_dqn' not in arms_dict:
            continue
        d_cell = arms_dict['ddqn']
        v_cell = arms_dict['vanilla_dqn']
        ef = _env_features(env)
        if not ef:
            continue
        n_actions = int(ef['n_actions'])
        ts = int(d_cell.get('total_steps', 0)) or int(v_cell.get('total_steps', 0))
        d_jensen = d_cell.get('mechanism.jensen_gap')
        v_jensen = v_cell.get('mechanism.jensen_gap')
        if not isinstance(d_jensen, (int, float)) or not isinstance(v_jensen, (int, float)):
            continue
        d_best = d_cell.get('outcome.eval_best_burst_mean')
        v_best = v_cell.get('outcome.eval_best_burst_mean')
        d_final = d_cell.get('outcome.eval_final_mean')
        v_final = v_cell.get('outcome.eval_final_mean')
        d_dorm = _dormancy_gap(
            sigma_by_id.get(d_cell['id'], float('nan')) if isinstance(d_cell.get('id'), str) else float('nan'),
            n_actions,
            v_jensen,  # observed bias on the BASELINE arm — the floor's denominator
        )
        v_dorm = _dormancy_gap(
            sigma_by_id.get(v_cell['id'], float('nan')) if isinstance(v_cell.get('id'), str) else float('nan'),
            n_actions,
            v_jensen,
        )
        d_id = d_cell.get('id') if isinstance(d_cell.get('id'), str) else None
        v_id = v_cell.get('id') if isinstance(v_cell.get('id'), str) else None
        d_burst = burst_by_id.get(d_id, {}) if d_id else {}
        v_burst = burst_by_id.get(v_id, {}) if v_id else {}

        def _bf(d: dict[str, float], k: str) -> float:
            return float(d.get(k, float('nan')))

        cells_out.append({
            'corpus': corpus,
            'env_name': env,
            'total_steps': ts,
            'seed': seed,
            **{k: hp_sig[i] for i, k in enumerate(_HP_KEYS) if hp_sig[i] is not None},
            'n_actions': n_actions,
            'log_obs_dim': float(ef['log_obs_dim']),
            'log_horizon': float(ef['log_horizon']),
            'jensen_gap_vanilla': float(v_jensen),
            'jensen_gap_ddqn': float(d_jensen),
            'delta_jensen_gap': float(d_jensen) - float(v_jensen),
            # Per-burst time-sliced features (vanilla baseline).
            'bias_early_vanilla': _bf(v_burst, 'bias_early'),
            'bias_late_vanilla': _bf(v_burst, 'bias_late'),
            'mc_early_vanilla': _bf(v_burst, 'mc_early'),
            'mc_late_vanilla': _bf(v_burst, 'mc_late'),
            'mc_peak_burst_vanilla': _bf(v_burst, 'mc_peak_burst'),
            'mc_range_vanilla': _bf(v_burst, 'mc_range'),
            # Per-burst time-sliced features (ddqn arm).
            'bias_early_ddqn': _bf(d_burst, 'bias_early'),
            'bias_late_ddqn': _bf(d_burst, 'bias_late'),
            'mc_early_ddqn': _bf(d_burst, 'mc_early'),
            'mc_late_ddqn': _bf(d_burst, 'mc_late'),
            # Time-sliced deltas (treatment − baseline).
            'delta_bias_early': (
                _bf(d_burst, 'bias_early') - _bf(v_burst, 'bias_early')
            ),
            'delta_bias_late': (
                _bf(d_burst, 'bias_late') - _bf(v_burst, 'bias_late')
            ),
            'delta_mc_early': (
                _bf(d_burst, 'mc_early') - _bf(v_burst, 'mc_early')
            ),
            'delta_mc_late': (
                _bf(d_burst, 'mc_late') - _bf(v_burst, 'mc_late')
            ),
            'outcome_best_vanilla': float(v_best) if isinstance(v_best, (int, float)) else float('nan'),
            'outcome_best_ddqn': float(d_best) if isinstance(d_best, (int, float)) else float('nan'),
            'delta_outcome_best': (
                (float(d_best) - float(v_best))
                if isinstance(v_best, (int, float)) and isinstance(d_best, (int, float))
                else float('nan')
            ),
            'outcome_final_vanilla': float(v_final) if isinstance(v_final, (int, float)) else float('nan'),
            'outcome_final_ddqn': float(d_final) if isinstance(d_final, (int, float)) else float('nan'),
            'delta_outcome_final': (
                (float(d_final) - float(v_final))
                if isinstance(v_final, (int, float)) and isinstance(d_final, (int, float))
                else float('nan')
            ),
            'sigma_late_vanilla': sigma_by_id.get(
                v_cell['id'], float('nan')
            ) if isinstance(v_cell.get('id'), str) else float('nan'),
            'sigma_late_ddqn': sigma_by_id.get(
                d_cell['id'], float('nan')
            ) if isinstance(d_cell.get('id'), str) else float('nan'),
            'dormancy_gap_vanilla': v_dorm,
            'dormancy_gap_ddqn': d_dorm,
            'dormancy_gap_avg': (
                (v_dorm + d_dorm) / 2.0
                if not (math.isnan(v_dorm) or math.isnan(d_dorm))
                else float('nan')
            ),
            'convergence_class_vanilla': conv_class.get(
                (env, ts), 'absent',
            ),
        })
    print(f'    {corpus}: {len(cells_out)} paired cells')
    return cells_out


def main() -> None:
    print('Building universal DDQN paired-delta dataset...')
    print(f'corpora: {_CORPORA}')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_cells: list[dict[str, object]] = []
    for corpus in _CORPORA:
        cells = _build_corpus_cells(corpus)
        if cells:
            all_cells.extend(cells)
    print(f'\ntotal paired cells: {len(all_cells)}')
    if not all_cells:
        print('no cells found — check corpus availability.')
        return
    df = pl.DataFrame(all_cells, strict=False)
    df.write_parquet(str(_OUT_FILE))
    print(f'wrote {_OUT_FILE} ({_OUT_FILE.stat().st_size / 1024:.1f} KB)')

    # Quick summary slice for the operator's eye.
    print()
    print('per (corpus, env) cell counts:')
    summary = df.group_by('corpus', 'env_name').len().sort('corpus', 'env_name')
    print(summary)


if __name__ == '__main__':
    main()
