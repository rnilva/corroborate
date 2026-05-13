"""Search for what mediates the Q-channel beyond bg.

At cross-corpus per-burst scale (`findings_two_channel_cross_corpus.md`),
partial ρ(q_per_burst, mc_per_burst | bg_per_burst) = +0.342 — the
Q-channel survives conditioning on the clip-magnitude channel at
80% of its marginal. What explains this residual?

Aggregate per-burst to per-cell, then per env compute:
  baseline: ρ(q, mc | bg)
  with candidate m: ρ(q, mc | bg, m)  for each candidate m

Whichever candidate most reduces the partial ρ is the dominant
mediator of the Q-channel.

Uses Spearman-rank ordinary least squares for multivariate
partial correlation:
  rank everything → regress mc_rank on (q_rank, bg_rank, m_rank)
  → coefficient of q_rank, partialled out of bg and m, is the
  partial Spearman ρ(q, mc | bg, m)."""
from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
from scipy import stats


CACHE = 'experiments/data/cache/ddqn.parquet'

# Candidates: original (no-backfill) + newly-backfilled via the
# REQUIRED_MEASURABLES hatch
CANDIDATES = [
    # Previously-cached
    'argmax_entropy_late',
    'q_autocorr_late',
    'target_staleness_late',
    'jensen_dormancy_gap',
    'effective_horizon',
    # Newly backfilled (Q-quality candidates)
    'q_action_std_late',
    'q_argmax_margin_late',
    'argmax_persistence_late',
    'q_max_temporal_cv_late',
    'q_mc_calibration_pearson',
]


def _rank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x)


def _partial_spearman_multi(
    y: np.ndarray, x: np.ndarray, controls: np.ndarray,
) -> tuple[float, int]:
    """ρ(y, x | controls) via rank-OLS.

    Returns (partial_rho, n_used)."""
    mask = np.isfinite(y) & np.isfinite(x)
    for j in range(controls.shape[1]):
        mask &= np.isfinite(controls[:, j])
    y, x, controls = y[mask], x[mask], controls[mask]
    n = len(y)
    if n < 10:
        return float('nan'), n
    y_r = _rank(y)
    x_r = _rank(x)
    c_r = np.column_stack([_rank(controls[:, j]) for j in range(controls.shape[1])])
    # Residualize y, x on c_r
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            y_res = y_r - c_r @ np.linalg.lstsq(c_r, y_r, rcond=None)[0]
            x_res = x_r - c_r @ np.linalg.lstsq(c_r, x_r, rcond=None)[0]
        except np.linalg.LinAlgError:
            return float('nan'), n
    if y_res.std() == 0 or x_res.std() == 0:
        return float('nan'), n
    rho = np.corrcoef(y_res, x_res)[0, 1]
    return float(rho), n


def _fz_pool(rhos: list[float], ns: list[int]) -> float:
    zs, ws = [], []
    for r, n in zip(rhos, ns, strict=True):
        if not math.isfinite(r) or n < 4 or abs(r) >= 1.0:
            continue
        zs.append(0.5 * math.log((1+r)/(1-r)) * (n-3))
        ws.append(n-3)
    if not zs or sum(ws) == 0:
        return float('nan')
    return math.tanh(sum(zs) / sum(ws))


def load_per_cell_panel() -> pl.DataFrame:
    """Aggregate per-burst arrays to per-cell means; join with
    candidate cell-level measurables."""
    select_cols = [
        'env_name', 'arm_key', 'seed',
        'bootstrap_gap_magnitude_per_burst',
        'q_per_burst',
        'mc_return_raw__mean_axis_-1',
        *CANDIDATES,
    ]
    df = pl.scan_parquet(CACHE).select(select_cols).filter(
        pl.col('bootstrap_gap_magnitude_per_burst').is_not_null()
        & pl.col('q_per_burst').is_not_null()
        & pl.col('mc_return_raw__mean_axis_-1').is_not_null()
    ).collect()

    rows = []
    for cell in df.iter_rows(named=True):
        bg, q, mc = (
            cell['bootstrap_gap_magnitude_per_burst'],
            cell['q_per_burst'],
            cell['mc_return_raw__mean_axis_-1'],
        )
        if not bg or not q or not mc:
            continue
        bg_arr = np.asarray(bg, dtype=np.float64)
        q_arr = np.asarray(q, dtype=np.float64)
        mc_arr = np.asarray(mc, dtype=np.float64)
        n_b = min(bg_arr.size, q_arr.size, mc_arr.size)
        if n_b == 0:
            continue
        bg_arr, q_arr, mc_arr = bg_arr[:n_b], q_arr[:n_b], mc_arr[:n_b]
        valid = np.isfinite(bg_arr) & np.isfinite(q_arr) & np.isfinite(mc_arr)
        if not valid.any():
            continue
        row = {
            'env_name': cell['env_name'],
            'bg': float(bg_arr[valid].mean()),
            'q': float(q_arr[valid].mean()),
            'mc': float(mc_arr[valid].mean()),
        }
        for c in CANDIDATES:
            v = cell.get(c)
            row[c] = float(v) if v is not None else float('nan')
        rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    panel = load_per_cell_panel()
    print(f'Panel: {panel.height} cells × {panel.select("env_name").n_unique()} envs')
    print()

    envs = sorted(panel.get_column('env_name').unique().to_list())

    # Baseline: ρ(q, mc | bg)
    per_env_baseline = {}
    for env in envs:
        sub = panel.filter(pl.col('env_name') == env)
        if sub.height < 20:
            continue
        rho, n = _partial_spearman_multi(
            sub.get_column('mc').to_numpy(),
            sub.get_column('q').to_numpy(),
            sub.select('bg').to_numpy(),
        )
        per_env_baseline[env] = (rho, n)

    rhos = [r for r, _ in per_env_baseline.values()]
    ns = [n for _, n in per_env_baseline.values()]
    pool_baseline = _fz_pool(rhos, ns)
    print(f'Baseline pool: partial ρ(q, mc | bg) = {pool_baseline:+.3f}')
    print()

    # For each candidate: ρ(q, mc | bg, candidate)
    print(f'{"candidate":<40s} | {"pool ρ":>10s} | {"Δ":>9s} | n_envs')
    print('-' * 80)
    def _test(candidate_set: list[str]) -> tuple[float, int]:
        per_env_cand = {}
        ctrl_cols = ['bg'] + candidate_set
        for env in envs:
            sub = panel.filter(pl.col('env_name') == env)
            if sub.height < 20:
                continue
            valid = sub
            for c in candidate_set:
                valid = valid.filter(pl.col(c).is_finite())
            if valid.height < 20:
                continue
            ctrl = valid.select(ctrl_cols).to_numpy()
            rho, n = _partial_spearman_multi(
                valid.get_column('mc').to_numpy(),
                valid.get_column('q').to_numpy(),
                ctrl,
            )
            per_env_cand[env] = (rho, n)
        rhos = [r for r, _ in per_env_cand.values()]
        ns = [n for _, n in per_env_cand.values()]
        return _fz_pool(rhos, ns), len(per_env_cand)

    # Single-mediator tests
    for cand in CANDIDATES:
        pool_c, n_envs = _test([cand])
        delta = pool_c - pool_baseline
        print(f'{cand:<40s} | {pool_c:>+10.3f} | {delta:>+9.3f} | {n_envs}')

    print()
    print('Joint conditioning (top candidates simultaneously):')
    print('-' * 80)
    joint_sets = [
        ['q_argmax_margin_late', 'q_action_std_late'],
        ['q_argmax_margin_late', 'jensen_dormancy_gap'],
        ['q_argmax_margin_late', 'q_action_std_late', 'jensen_dormancy_gap'],
        ['q_argmax_margin_late', 'q_action_std_late', 'jensen_dormancy_gap', 'effective_horizon'],
    ]
    for js in joint_sets:
        label = ' + '.join(c.replace('_late', '').replace('jensen_', 'jens_').replace('effective_horizon', 'eff_h') for c in js)
        pool_j, n_envs = _test(js)
        delta = pool_j - pool_baseline
        print(f'  {label:<60s} | {pool_j:>+10.3f} | {delta:>+9.3f} | {n_envs}')


if __name__ == '__main__':
    main()
