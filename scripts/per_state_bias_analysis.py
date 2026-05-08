"""Per-state cumulative bias vs bias-at-start: which predicts
Δ_outcome better when DDQN's mechanism is held?

Two analyses per env:
1. Within-cell late-window: scalar Δ_jens vs scalar
   Δ_per_state_bias_late, paired across seeds.
2. Per-burst panel: per-(seed, burst) Δs at the eval-burst
   resolution; the framework-canonical form.

Reads per_state_bias_probe sweep outputs. Adds new envs by
appending their sweep dir to ENVS below.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src/corroborate_rl'))

import polars as pl
import numpy as np
import scipy.stats as ss
import numpy.linalg as la

import corroborate.analyses  # noqa: F401
import corroborate_rl.dqn.measurables  # noqa: F401
from corroborate_rl.dqn.measurables import (
    jensen_gap, mean_per_state_cumulative_bias_late,
)

DDQN_KEY = (
    'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
)
BASELINE_KEY = 'baseline'

ENVS = [
    ('FourRooms γ=0.99', REPO / 'experiments/data/per_state_bias_probe_fr'),
    ('Acrobot γ=0.999',
     REPO / 'experiments/data/per_state_bias_probe_acrobot'),
    ('MetaMaze γ=0.999',
     REPO / 'experiments/data/per_state_bias_probe_metamaze'),
]


def cell_scalar_bias(traces_path: Path) -> dict[str, dict[str, float]]:
    traces = pl.read_parquet(traces_path)
    out: dict[str, dict[str, float]] = {}
    for r in traces.iter_rows(named=True):
        rec = {
            'predicted_q_per_step': np.asarray(
                r['predicted_q_per_step'], dtype=np.float64,
            ),
            'mc_return_from_step': np.asarray(
                r['mc_return_from_step'], dtype=np.float64,
            ),
            'active_per_step': np.asarray(
                r['active_per_step'], dtype=np.float64,
            ),
            'predicted_q_at_start': np.asarray(
                r['predicted_q_at_start'], dtype=np.float64,
            ),
            'mc_return': np.asarray(r['mc_return'], dtype=np.float64),
        }
        out[r['id']] = {
            'jensen_gap': jensen_gap.fn(rec),
            'mean_per_state_bias_late': (
                mean_per_state_cumulative_bias_late.fn(rec)
            ),
        }
    return out


def cell_per_burst_bias(
    traces_path: Path,
) -> dict[str, dict[str, np.ndarray]]:
    """Per cell, return arrays of shape (n_bursts,) for jens_pb,
    per_state_pb, outcome_pb."""
    traces = pl.read_parquet(traces_path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for r in traces.iter_rows(named=True):
        pq0 = np.asarray(r['predicted_q_at_start'], dtype=np.float64)
        mc = np.asarray(r['mc_return'], dtype=np.float64)
        pq_step = np.asarray(r['predicted_q_per_step'], dtype=np.float64)
        mc_step = np.asarray(r['mc_return_from_step'], dtype=np.float64)
        act = np.asarray(r['active_per_step'], dtype=np.float64)
        bias = pq_step - mc_step
        weighted = bias * act
        num = weighted.sum(axis=(1, 2))
        den = act.sum(axis=(1, 2))
        per_state_pb = np.where(den > 0, num / den, np.nan)
        out[r['id']] = {
            'jens_pb': (pq0 - mc).mean(axis=1),
            'per_state_pb': per_state_pb,
            'outcome_pb': mc.mean(axis=1),
        }
    return out


def _ols_with_se(X, y):
    beta, *_ = la.lstsq(X, y, rcond=None)
    yhat = X @ beta
    n_obs = len(y)
    k = X.shape[1]
    sse = ((y - yhat) ** 2).sum() / max(n_obs - k, 1)
    cov = sse * la.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p_vals = [2 * (1 - ss.t.cdf(abs(tt), n_obs - k)) for tt in t]
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, se, t, p_vals, r2


def partial_rho(y, x, z):
    rxy = ss.spearmanr(y, x).statistic
    rxz = ss.spearmanr(x, z).statistic
    ryz = ss.spearmanr(y, z).statistic
    d = ((1 - rxz ** 2) * (1 - ryz ** 2)) ** 0.5
    if d < 1e-12:
        return float('nan'), float('nan')
    rho = (rxy - rxz * ryz) / d
    n = len(y)
    if abs(rho) >= 1:
        return rho, float('nan')
    t = rho * ((n - 3) / (1 - rho ** 2)) ** 0.5
    return rho, float(2 * (1 - ss.t.cdf(abs(t), n - 3)))


def analyze_within_cell(label: str, sweep_dir: Path):
    runs = pl.read_parquet(sweep_dir / 'runs.parquet').select(
        ['id', 'seed', 'arm_key', 'eval_best_burst_mean'],
    )
    bias_by_id = cell_scalar_bias(sweep_dir / 'traces.parquet')
    bias_df = pl.DataFrame(
        [{'id': i, **vals} for i, vals in bias_by_id.items()],
    )
    df = runs.join(bias_df, on='id')
    van = df.filter(pl.col('arm_key') == BASELINE_KEY).sort('seed')
    dd = df.filter(pl.col('arm_key') == DDQN_KEY).sort('seed')
    paired = van.join(dd, on='seed', suffix='_dd')

    dY = (
        paired['eval_best_burst_mean_dd'] - paired['eval_best_burst_mean']
    ).to_numpy()
    dJens = (paired['jensen_gap_dd'] - paired['jensen_gap']).to_numpy()
    dPerState = (
        paired['mean_per_state_bias_late_dd']
        - paired['mean_per_state_bias_late']
    ).to_numpy()
    mech_held = dJens < 0
    finite = (
        np.isfinite(dY) & np.isfinite(dJens) & np.isfinite(dPerState)
        & mech_held
    )
    if finite.sum() < 5:
        print(f'\n=== {label} within-cell: insufficient pairs ({finite.sum()}) ===')
        return
    dY, dJens, dPerState = dY[finite], dJens[finite], dPerState[finite]
    n = len(dY)
    print(f'\n=== {label} WITHIN-CELL (n={n} mech-HELD) ===')
    rj, pj = ss.pearsonr(dY, dJens)
    rp_, pp_ = ss.pearsonr(dY, dPerState)
    print(f'  marginal r(Δy, Δjens)      = {rj:+.3f} (p={pj:.4f})')
    print(f'  marginal r(Δy, Δpstate)    = {rp_:+.3f} (p={pp_:.4f})')
    X = np.column_stack([np.ones(n), dJens, dPerState])
    beta, se, t, pv, r2 = _ols_with_se(X, dY)
    print(f'  joint OLS R²={r2:.3f}: β_jens={beta[1]:+.4f} (p={pv[1]:.4f}), '
          f'β_pstate={beta[2]:+.4f} (p={pv[2]:.4f})')


def analyze_per_burst(label: str, sweep_dir: Path):
    runs = pl.read_parquet(sweep_dir / 'runs.parquet').select(
        ['id', 'seed', 'arm_key'],
    )
    pb = cell_per_burst_bias(sweep_dir / 'traces.parquet')
    n_bursts = next(iter(pb.values()))['jens_pb'].shape[0]

    seeds_in_van: dict[int, str] = {}
    seeds_in_dd: dict[int, str] = {}
    for r in runs.iter_rows(named=True):
        if r['arm_key'] == BASELINE_KEY:
            seeds_in_van[r['seed']] = r['id']
        elif r['arm_key'] == DDQN_KEY:
            seeds_in_dd[r['seed']] = r['id']
    paired_seeds = sorted(set(seeds_in_van) & set(seeds_in_dd))

    all_dY: list[float] = []
    all_dJ: list[float] = []
    all_dP: list[float] = []
    for s in paired_seeds:
        van = pb[seeds_in_van[s]]
        dd = pb[seeds_in_dd[s]]
        for b in range(n_bursts):
            dY = float(dd['outcome_pb'][b] - van['outcome_pb'][b])
            dJ = float(dd['jens_pb'][b] - van['jens_pb'][b])
            dP = float(dd['per_state_pb'][b] - van['per_state_pb'][b])
            if np.isfinite(dY) and np.isfinite(dJ) and np.isfinite(dP) and dJ < 0:
                all_dY.append(dY)
                all_dJ.append(dJ)
                all_dP.append(dP)

    dY = np.array(all_dY)
    dJ = np.array(all_dJ)
    dP = np.array(all_dP)
    n = len(dY)
    print(f'\n=== {label} PER-BURST POOLED (n={n} mech-HELD seed×burst) ===')
    if n < 5:
        print('  insufficient')
        return
    rj, pj = ss.pearsonr(dY, dJ)
    rp_, pp_ = ss.pearsonr(dY, dP)
    print(f'  marginal r(Δy, Δjens)      = {rj:+.3f} (p={pj:.4f})')
    print(f'  marginal r(Δy, Δpstate)    = {rp_:+.3f} (p={pp_:.4f})')
    X = np.column_stack([np.ones(n), dJ, dP])
    beta, se, t, pv, r2 = _ols_with_se(X, dY)
    print(f'  joint OLS R²={r2:.3f}: β_jens={beta[1]:+.4f} (p={pv[1]:.4f}), '
          f'β_pstate={beta[2]:+.4f} (p={pv[2]:.4f})')
    rho_j, p_j = partial_rho(dY, dJ, dP)
    rho_p, p_p = partial_rho(dY, dP, dJ)
    print(f'  partial ρ(Δy, Δjens|Δpstate)   = {rho_j:+.3f} (p={p_j:.4f})')
    print(f'  partial ρ(Δy, Δpstate|Δjens)   = {rho_p:+.3f} (p={p_p:.4f})')


def main():
    for label, sweep_dir in ENVS:
        if not (sweep_dir / 'traces.parquet').exists():
            print(f'\n(skipping {label} — no traces.parquet)')
            continue
        analyze_within_cell(label, sweep_dir)
        analyze_per_burst(label, sweep_dir)


if __name__ == '__main__':
    main()
