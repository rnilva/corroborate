"""Cross-env per-burst two-channel decomposition.

Tests whether the two-channel finding from the original per-burst
PC (Acrobot + FourRooms only — limited by per-burst measurable
availability at the time) holds at scale now that `q_per_burst`
is in the cache across all 17 envs.

The two channels:
  C1 (clip-magnitude): bg_per_burst → mc_per_burst
  C2 (Q-magnitude):    q_per_burst → mc_per_burst, independent of C1

Per-env partial Spearman with the other channel as the conditioning
variable. If both channels survive conditioning across envs, the
two-channel structure is corroborated cross-corpus. Cross-env
Fisher-z pool gives the overall verdict.

Also stratifies by env polarity (positive-Q SURVIVE vs negative-Q
REACH) to test whether the q-channel's polarity-asymmetry visible
in cell-level |Q| (`findings_ddqn_reward_sign_conditional.md`)
manifests at per-burst level too."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy import stats

from corroborate.stats import fisher_z_pool


CACHE_PATH = 'experiments/data/cache/ddqn.parquet'

PER_BURST_BG = 'bootstrap_gap_magnitude_per_burst'
PER_BURST_Q = 'q_per_burst'
PER_BURST_MC = 'mc_return_raw__mean_axis_-1'


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Partial Spearman ρ(x, y | z) via the three-correlation identity.
    Returns (rho, n) where n is the sample size."""
    rho_xy = stats.spearmanr(x, y).statistic
    rho_xz = stats.spearmanr(x, z).statistic
    rho_yz = stats.spearmanr(y, z).statistic
    den = np.sqrt(max(0.0, 1.0 - rho_xz**2) * max(0.0, 1.0 - rho_yz**2))
    if den == 0.0:
        return float('nan'), len(x)
    return float((rho_xy - rho_xz * rho_yz) / den), len(x)


def load_panel() -> pl.DataFrame:
    """Load cache, filter to cells with all three per-burst arrays,
    unfold into (cell, burst) rows."""
    df = pl.scan_parquet(CACHE_PATH).select([
        'env_name', 'arm_key', 'seed', 'env_reward_polarity',
        PER_BURST_BG, PER_BURST_Q, PER_BURST_MC,
    ]).filter(
        pl.col(PER_BURST_BG).is_not_null()
        & pl.col(PER_BURST_Q).is_not_null()
        & pl.col(PER_BURST_MC).is_not_null()
    ).collect()

    rows = []
    for cell in df.iter_rows(named=True):
        bg, q, mc = cell[PER_BURST_BG], cell[PER_BURST_Q], cell[PER_BURST_MC]
        if not bg or not q or not mc:
            continue
        n = min(len(bg), len(q), len(mc))
        for i in range(n):
            if any(math.isnan(v) for v in (bg[i], q[i], mc[i])):
                continue
            rows.append({
                'env_name': cell['env_name'],
                'arm_key': cell['arm_key'],
                'polarity': cell.get('env_reward_polarity'),
                'burst': i,
                'bg': bg[i],
                'q': q[i],
                'mc': mc[i],
            })
    panel = pl.DataFrame(rows)
    return panel


def per_env_analysis(panel: pl.DataFrame, min_rows: int = 30) -> pl.DataFrame:
    rows = []
    for env in sorted(panel.get_column('env_name').unique().to_list()):
        sub = panel.filter(pl.col('env_name') == env)
        if sub.height < min_rows:
            continue
        bg = sub.get_column('bg').to_numpy()
        q = sub.get_column('q').to_numpy()
        mc = sub.get_column('mc').to_numpy()

        rho_bg_mc = stats.spearmanr(bg, mc).statistic
        rho_q_mc = stats.spearmanr(q, mc).statistic
        rho_bg_q = stats.spearmanr(bg, q).statistic
        partial_bg_mc_q, _ = _partial_spearman(bg, mc, q)
        partial_q_mc_bg, _ = _partial_spearman(q, mc, bg)

        # Survival ratio: how much of each marginal correlation
        # survives after conditioning on the other channel
        surv_bg = (
            partial_bg_mc_q / rho_bg_mc
            if rho_bg_mc != 0 else float('nan')
        )
        surv_q = (
            partial_q_mc_bg / rho_q_mc
            if rho_q_mc != 0 else float('nan')
        )

        # Median polarity (per-env property)
        pol = sub.get_column('polarity').drop_nulls()
        polarity = float(pol.mean()) if pol.len() else float('nan')

        rows.append({
            'env': env, 'n': sub.height, 'polarity': polarity,
            'rho_bg_mc': float(rho_bg_mc),
            'rho_q_mc': float(rho_q_mc),
            'rho_bg_q': float(rho_bg_q),
            'partial_bg_mc_q': partial_bg_mc_q,
            'partial_q_mc_bg': partial_q_mc_bg,
            'surv_bg': float(surv_bg) if math.isfinite(surv_bg) else float('nan'),
            'surv_q': float(surv_q) if math.isfinite(surv_q) else float('nan'),
        })
    return pl.DataFrame(rows).sort('polarity', nulls_last=True)


def fisher_z_summary(per_env: pl.DataFrame) -> dict[str, tuple[float, float]]:
    cols = ['rho_bg_mc', 'rho_q_mc', 'partial_bg_mc_q', 'partial_q_mc_bg']
    out: dict[str, tuple[float, float]] = {}
    ns = per_env.get_column('n').to_list()
    for c in cols:
        rhos = per_env.get_column(c).to_list()
        rho, p = fisher_z_pool(tuple(rhos), tuple(ns))
        out[c] = (rho, p)
    return out


def main() -> None:
    panel = load_panel()
    print(f'panel: {panel.height} (cell, burst) rows across {panel.select("env_name").n_unique()} envs')
    print()

    per_env = per_env_analysis(panel)
    print('Per-env: ρ(bg,mc), ρ(q,mc), partial ρ(bg,mc|q), partial ρ(q,mc|bg)')
    print('=' * 110)
    with pl.Config(tbl_rows=20, tbl_cols=11, fmt_str_lengths=30, tbl_width_chars=180):
        print(per_env)

    pool = fisher_z_summary(per_env)
    print()
    print('Cross-env Fisher-z pool (sample-size weighted):')
    for k, (rho, p) in pool.items():
        print(f'  {k:25s} = {rho:+.3f}   p = {p:.4g}')

    # Polarity stratification by sign of mean q_per_burst per env
    # (env_reward_polarity is NaN on many envs; sign(q) is the
    # downstream consequence and is always defined).
    q_sign_per_env = (
        panel.group_by('env_name')
        .agg(pl.col('q').mean().alias('q_mean'))
        .with_columns(
            pl.when(pl.col('q_mean') > 0).then(pl.lit('+'))
            .when(pl.col('q_mean') < 0).then(pl.lit('-'))
            .otherwise(pl.lit('?')).alias('q_sign')
        )
    )
    per_env = per_env.join(
        q_sign_per_env.select(['env_name', 'q_sign', 'q_mean']),
        left_on='env', right_on='env_name', how='left',
    )

    print()
    print('Polarity stratification by sign(mean q_per_burst):')
    for sign_label, name in (('+', 'positive-Q'), ('-', 'negative-Q')):
        sub = per_env.filter(pl.col('q_sign') == sign_label)
        envs = sub.get_column('env').to_list()
        print(f'  {name} (n_envs={sub.height}): {envs}')
        if sub.height:
            pool = fisher_z_summary(sub)
            for k, (rho, p) in pool.items():
                print(f'    {k:25s} = {rho:+.3f}  p={p:.4g}')


if __name__ == '__main__':
    main()
