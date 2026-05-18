"""Two-channel decomposition restricted to env-canonical configs.

The original cross-corpus analysis (`scripts/analyze_per_burst_two_channels.py`)
pooled across HP variants within each env (γ, sync_period, capacity, lr,
network depth, action_duplicate_k, reward_scale, wrappers). Those HPs
are common causes of (Q-level, mc_return), confounding the apparent
Q-channel. The MountainCar dig showed cross-cell ρ(q, mc | bg) =
+0.375 → +0.024 when conditioning on (lr, capacity).

This script restricts each env to its canonical config and reruns
the partial-Spearman analysis. The canonical excludes all HP-sweep
cells; what's left is seed variance + bursts within the standard
configuration.

Env-canonical configs:
- MLP envs (Acrobot, CartPole, FourRooms, MountainCar, MetaMaze,
  bsuite): hidden=(64,64), channels=null
- MinAtar (Asterix, Breakout, Freeway, SpaceInvaders): hidden=(128),
  channels=(16,32)
- jumanji (PacMan, SlidingTile, Snake): hidden=(64), channels=(8,16)

All other axes pinned: γ=0.99, sync=100, n_step=1, capacity=50000,
lr=1e-4, no action_duplicate, no reward_scale, no Polyak, no
wrappers, total_steps=200000."""
from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
from scipy import stats


CACHE = 'experiments/data/cache/ddqn.parquet'

# Env-canonical network architecture by env-class
MLP_ENVS = {
    'Acrobot-v1', 'CartPole-v1', 'FourRooms-misc', 'MountainCar-v0',
    'MetaMaze-misc',
    'BernoulliBandit-misc', 'Catch-bsuite', 'DeepSea-bsuite',
    'DiscountingChain-bsuite', 'UmbrellaChain-bsuite',
}
MINATAR_ENVS = {
    'Asterix-MinAtar', 'Breakout-MinAtar', 'Freeway-MinAtar',
    'SpaceInvaders-MinAtar',
}
JUMANJI_ENVS = {
    'PacMan-jumanji', 'SlidingTilePuzzle-jumanji', 'Snake-jumanji',
}


def canonical_scope() -> pl.Expr:
    """Pl.Expr selecting only canonical-config cells for each env,
    AND respecting the ddqn hypothesis's MODULE_SCOPE (excludes
    `-bsuite` diagnostic envs)."""
    base = (
        # MODULE_SCOPE: ddqn hypothesis excludes bsuite diagnostic envs
        ~pl.col('env_name').str.ends_with('-bsuite')
        & (pl.col('gamma') == 0.99)
        & (pl.col('sync_period') == 100)
        & (pl.col('replay.capacity') == 50000)
        & (pl.col('optimizer.inner.lr') == 0.0001)
        & ((pl.col('n_step') == 1) | pl.col('n_step').is_null())
        & pl.col('action_duplicate_k').is_null()
        & (pl.col('reward_scale').is_null() | (pl.col('reward_scale') == 1.0))
        & pl.col('target_sync.tau').is_null()
        & (pl.col('total_steps') == 200000)
        & (pl.col('wrappers') == '()')
    )
    # Env-class-specific architecture pinning
    mlp = pl.col('env_name').is_in(list(MLP_ENVS)) & (pl.col('q_network.hidden') == '(64,64)')
    minatar = pl.col('env_name').is_in(list(MINATAR_ENVS)) & (pl.col('q_network.hidden') == '(128)') & (pl.col('q_network.channels') == '(16,32)')
    jumanji = pl.col('env_name').is_in(list(JUMANJI_ENVS)) & (pl.col('q_network.hidden') == '(64)') & (pl.col('q_network.channels') == '(8,16)')
    return base & (mlp | minatar | jumanji)


def _rank(x):
    return stats.rankdata(x)


def _partial(y, x, controls):
    mask = np.isfinite(y) & np.isfinite(x)
    for j in range(controls.shape[1]):
        mask &= np.isfinite(controls[:, j])
    y, x, controls = y[mask], x[mask], controls[mask]
    if len(y) < 10:
        return float('nan'), len(y)
    y_r, x_r = _rank(y), _rank(x)
    c_r = np.column_stack([_rank(controls[:, j]) for j in range(controls.shape[1])])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            y_res = y_r - c_r @ np.linalg.lstsq(c_r, y_r, rcond=None)[0]
            x_res = x_r - c_r @ np.linalg.lstsq(c_r, x_r, rcond=None)[0]
        except np.linalg.LinAlgError:
            return float('nan'), len(y)
    if y_res.std() == 0 or x_res.std() == 0:
        return float('nan'), len(y)
    return float(np.corrcoef(y_res, x_res)[0, 1]), len(y)


def _fz_pool(rhos, ns):
    zs, ws = [], []
    for r, n in zip(rhos, ns, strict=True):
        if not math.isfinite(r) or n < 4 or abs(r) >= 1.0:
            continue
        zs.append(0.5 * math.log((1+r)/(1-r)) * (n-3))
        ws.append(n-3)
    if not zs or sum(ws) == 0:
        return float('nan')
    return math.tanh(sum(zs) / sum(ws))


def main() -> None:
    df = pl.scan_parquet(CACHE).filter(canonical_scope()).select([
        'env_name', 'arm_key',
        'bootstrap_gap_magnitude_per_burst', 'q_per_burst',
        'mc_return_raw__mean_axis_-1',
        'q_argmax_margin_late', 'jensen_dormancy_gap', 'argmax_persistence_late',
    ]).filter(
        pl.col('bootstrap_gap_magnitude_per_burst').is_not_null()
        & pl.col('q_per_burst').is_not_null()
        & pl.col('mc_return_raw__mean_axis_-1').is_not_null()
    ).collect()

    print(f'Canonical-scoped cells: {df.height} across {df.select("env_name").n_unique()} envs')
    print()
    print('Per-env cells:')
    print(df.group_by('env_name').agg(pl.len()).sort('env_name'))
    print()

    # Unfold per-burst
    rows = []
    for cell in df.iter_rows(named=True):
        bg = np.asarray(cell['bootstrap_gap_magnitude_per_burst'] or [], dtype=np.float64)
        q = np.asarray(cell['q_per_burst'] or [], dtype=np.float64)
        mc = np.asarray(cell['mc_return_raw__mean_axis_-1'] or [], dtype=np.float64)
        n_b = min(bg.size, q.size, mc.size)
        for j in range(n_b):
            if not (np.isfinite(bg[j]) and np.isfinite(q[j]) and np.isfinite(mc[j])): continue
            rows.append({
                'env_name': cell['env_name'],
                'bg': bg[j], 'q': q[j], 'mc': mc[j],
                'margin': cell.get('q_argmax_margin_late') or float('nan'),
                'dormancy': cell.get('jensen_dormancy_gap') or float('nan'),
                'persistence': cell.get('argmax_persistence_late') or float('nan'),
            })
    panel = pl.DataFrame(rows)
    print(f'Per-burst panel: {panel.height} rows')
    print()

    # Per-env partial Spearman with various conditioning
    print(f'{"env":<28s} | {"n":>5} | {"ρ(q,mc|bg)":>11s} | {"+margin+dorm":>13s} | {"+persistence":>13s}')
    print('-' * 95)
    rhos_baseline, rhos_md, rhos_mdp, ns = [], [], [], []
    for env in sorted(panel.get_column('env_name').unique().to_list()):
        sub = panel.filter(pl.col('env_name') == env)
        if sub.height < 30: continue
        r1, n = _partial(
            sub.get_column('mc').to_numpy(), sub.get_column('q').to_numpy(),
            sub.select('bg').to_numpy(),
        )
        valid = sub.filter(
            pl.col('margin').is_finite() & pl.col('dormancy').is_finite()
        )
        if valid.height >= 20:
            r2, _ = _partial(
                valid.get_column('mc').to_numpy(),
                valid.get_column('q').to_numpy(),
                valid.select(['bg', 'margin', 'dormancy']).to_numpy(),
            )
        else:
            r2 = float('nan')
        valid_p = sub.filter(
            pl.col('margin').is_finite() & pl.col('dormancy').is_finite() & pl.col('persistence').is_finite()
        )
        if valid_p.height >= 20:
            r3, _ = _partial(
                valid_p.get_column('mc').to_numpy(),
                valid_p.get_column('q').to_numpy(),
                valid_p.select(['bg', 'margin', 'dormancy', 'persistence']).to_numpy(),
            )
        else:
            r3 = float('nan')
        rhos_baseline.append(r1); rhos_md.append(r2); rhos_mdp.append(r3); ns.append(n)
        print(f'{env:<28s} | {sub.height:>5} | {r1:>+11.3f} | {r2:>+13.3f} | {r3:>+13.3f}')

    print()
    print('Cross-env Fisher-z pool:')
    print(f'  partial ρ(q, mc | bg)                        = {_fz_pool(rhos_baseline, ns):+.3f}')
    print(f'  partial ρ(q, mc | bg, margin, dormancy)      = {_fz_pool(rhos_md, ns):+.3f}')
    print(f'  partial ρ(q, mc | bg, margin, dorm, persist) = {_fz_pool(rhos_mdp, ns):+.3f}')


if __name__ == '__main__':
    main()
