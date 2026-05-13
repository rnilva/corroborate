"""Empirical proof: DDQN's clip is reward-sign-conditional.

Structural claim (`findings_ddqn_reward_sign_conditional.md`):
DDQN's bootstrap target value is `target_q[argmax_online] ≤ max_a target_q`
by construction — a ONE-SIDED downward clip. The downstream effect on
the trained Q's magnitude depends on the sign of the env's optimal Q:
- Positive-Q envs: downward clip reduces upward overestimation → |Q| ↓
- Negative-Q envs: downward clip pushes Q more negative → |Q| ↑

Empirical tests in this script:
1. Per-env paired Δ|Q| (DDQN vs vanilla): predicted sign matches sign(Q_van)
2. Cross-env regression: Δ|Q| vs Q_van should have negative slope (clip
   intensity scales with positive-Q amplitudes)
3. α dose-response: for envs with the α-sweep, Δ|Q| should be monotone in α
4. Structural Δ Q < 0 universally (the clip is always downward, regardless
   of env Q sign)

Run: `uv run python scripts/verify_clip_asymmetry.py`

Output corroborates `findings/ddqn/finding_polarity_conditional_chain.py`
and `findings_ddqn_reward_sign_conditional.md` memory."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats


CACHE_PATH = 'experiments/data/cache/ddqn.parquet'
PAIR_KEYS = [
    'env_name', 'seed', 'gamma', 'n_step',
    'total_steps', 'replay.capacity', 'sync_period',
]


def _tag_arm(s: str) -> tuple[str, float]:
    """Map arm_key string to (label, alpha) where alpha is the
    double-greedification damping."""
    if s == 'baseline':
        return ('vanilla', 0.0)
    if 'Claim:double_greedify' in s and 'dampened' not in s:
        return ('ddqn', 1.0)
    if 'alpha=0.75' in s:
        return ('ddqn_a075', 0.75)
    if 'alpha=0.5' in s:
        return ('ddqn_a050', 0.5)
    if 'alpha=0.25' in s:
        return ('ddqn_a025', 0.25)
    return ('other', float('nan'))


def _load() -> pl.DataFrame:
    df = pl.scan_parquet(CACHE_PATH).select(
        ['env_name', 'arm_key', 'q_late_mean', 'seed', 'gamma',
         'n_step', 'total_steps', 'replay.capacity', 'sync_period']
    ).collect()
    tagged = [_tag_arm(k) for k in df.get_column('arm_key').to_list()]
    return (
        df
        .with_columns([
            pl.Series('arm', [a for a, _ in tagged]),
            pl.Series('alpha', [v for _, v in tagged]),
        ])
        .filter(pl.col('q_late_mean').is_not_null() & pl.col('q_late_mean').is_finite())
        .filter(pl.col('arm') != 'other')
    )


def test_1_per_env_paired_sign() -> pl.DataFrame:
    """Per-env paired Δ|Q| test."""
    df = _load().filter(pl.col('arm').is_in(['vanilla', 'ddqn']))
    pivoted = (
        df.pivot(values='q_late_mean', index=PAIR_KEYS, on='arm', aggregate_function='mean')
        .filter(pl.col('vanilla').is_not_null() & pl.col('ddqn').is_not_null())
        .with_columns([
            (pl.col('ddqn') - pl.col('vanilla')).alias('dQ'),
            (pl.col('ddqn').abs() - pl.col('vanilla').abs()).alias('dAbsQ'),
        ])
    )

    rows = []
    for env in sorted(pivoted.select('env_name').unique().to_series().to_list()):
        sub = pivoted.filter(pl.col('env_name') == env)
        v = sub.get_column('vanilla').to_numpy()
        dq = sub.get_column('dQ').to_numpy()
        da = sub.get_column('dAbsQ').to_numpy()
        n = len(dq)
        if n < 5:
            continue
        rows.append({
            'env': env,
            'n': n,
            'Q_van': float(v.mean()),
            'sign_Q_van': '+' if v.mean() > 0 else '-',
            'dQ_mean': float(dq.mean()),
            'dAbsQ_mean': float(da.mean()),
            'd_dAbsQ': float(da.mean() / da.std(ddof=1)) if da.std(ddof=1) > 0 else float('nan'),
            'p_dAbsQ': float(stats.ttest_1samp(da, 0.0).pvalue),
            'predicted_sign': '-' if v.mean() > 0 else '+',
            'observed_sign': '-' if da.mean() < 0 else '+',
        })

    out = pl.DataFrame(rows).with_columns(
        (pl.col('predicted_sign') == pl.col('observed_sign')).alias('match')
    ).sort('Q_van')
    return out


def test_2_cross_env_regression(per_env: pl.DataFrame) -> dict[str, float]:
    """Cross-env regression of Δ|Q| on Q_van.

    Predicted: negative slope (Δ|Q| more negative as Q_van increases).
    Includes a second pass excluding non-convergent envs (Snake,
    MetaMaze, PacMan), which dominate the regression with their
    instability."""
    Q = per_env.get_column('Q_van').to_numpy()
    dA = per_env.get_column('dAbsQ_mean').to_numpy()

    full = stats.linregress(Q, dA)
    # PacMan dominates as outlier; exclude unstable envs
    unstable = {'Snake-jumanji', 'MetaMaze-misc', 'PacMan-jumanji'}
    mask = ~per_env.get_column('env').is_in(list(unstable)).to_numpy()
    clean = stats.linregress(Q[mask], dA[mask])

    return {
        'full_r': float(full.rvalue),
        'full_p': float(full.pvalue),
        'full_slope': float(full.slope),
        'full_n': int(len(Q)),
        'clean_r': float(clean.rvalue),
        'clean_p': float(clean.pvalue),
        'clean_slope': float(clean.slope),
        'clean_n': int(mask.sum()),
    }


def test_3_alpha_dose_response() -> pl.DataFrame:
    """Per-env α dose-response on Δ|Q|.

    Predicted: |Δ|Q|| monotone in α (more decoupling = more clip).
    Only Breakout and SpaceInvaders have the full α sweep on this corpus."""
    df = _load()
    rows = []
    for env in sorted(df.select('env_name').unique().to_series().to_list()):
        sub = df.filter(pl.col('env_name') == env)
        pivot_df = sub.pivot(values='q_late_mean', index=PAIR_KEYS, on='arm', aggregate_function='mean')

        results: dict[float, float] = {}
        v_sign = '?'
        for arm, alpha in [('ddqn_a025', 0.25), ('ddqn_a050', 0.5), ('ddqn_a075', 0.75), ('ddqn', 1.0)]:
            if arm not in pivot_df.columns:
                continue
            paired = pivot_df.filter(pl.col('vanilla').is_not_null() & pl.col(arm).is_not_null())
            if paired.height < 5:
                continue
            v = paired.get_column('vanilla').to_numpy()
            d = paired.get_column(arm).to_numpy()
            results[alpha] = float((np.abs(d) - np.abs(v)).mean())
            v_sign = '+' if float(v.mean()) > 0 else '-'

        if len(results) >= 3:
            a_vals = list(results.keys())
            d_vals = list(results.values())
            rho, p = stats.spearmanr(a_vals, d_vals)
            rows.append({
                'env': env,
                'sign_Q_van': v_sign,
                'a025': results.get(0.25, None),
                'a050': results.get(0.5, None),
                'a075': results.get(0.75, None),
                'a100': results.get(1.0, None),
                'rho_alpha': float(rho),
                'p_alpha': float(p),
            })

    return pl.DataFrame(rows)


def test_5_within_env_clip_predicts_dAbsQ() -> pl.DataFrame:
    """Within-env Spearman ρ(bg_van, Δ|Q|).

    The CLEANEST mechanism test: does the magnitude of the AVAILABLE
    clip (`bootstrap_gap_magnitude` on vanilla) predict how much |Q|
    changed under DDQN, within env?

    Predicted: ρ flips sign with env Q-sign.
    - Positive-Q envs: more clip → bigger |Q| decrease (ρ < 0)
    - Negative-Q envs: more clip → bigger |Q| increase (ρ > 0)

    Cross-env summary: mean ρ over positive-Q envs should be NEGATIVE,
    mean ρ over negative-Q envs should be POSITIVE."""
    df = pl.scan_parquet(CACHE_PATH).select(
        ['env_name', 'arm_key', 'q_late_mean', 'bootstrap_gap_magnitude',
         'seed', 'gamma', 'n_step', 'total_steps', 'replay.capacity', 'sync_period']
    ).collect()
    df = df.with_columns(
        pl.when(pl.col('arm_key') == 'baseline').then(pl.lit('vanilla'))
        .when(pl.col('arm_key').str.contains('Claim:double_greedify')).then(pl.lit('ddqn'))
        .otherwise(pl.lit('other')).alias('arm')
    )
    df = df.filter(pl.col('q_late_mean').is_finite() & pl.col('bootstrap_gap_magnitude').is_finite())
    df = df.filter(pl.col('arm').is_in(['vanilla', 'ddqn']))

    q = df.pivot(values='q_late_mean', index=PAIR_KEYS, on='arm', aggregate_function='mean').rename({'vanilla': 'q_van', 'ddqn': 'q_ddqn'})
    bg = df.pivot(values='bootstrap_gap_magnitude', index=PAIR_KEYS, on='arm', aggregate_function='mean').rename({'vanilla': 'bg_van', 'ddqn': 'bg_ddqn'})
    joined = q.join(bg, on=PAIR_KEYS, how='inner').filter(
        pl.col('q_van').is_not_null() & pl.col('q_ddqn').is_not_null()
        & pl.col('bg_van').is_not_null() & pl.col('bg_ddqn').is_not_null()
    ).with_columns(
        (pl.col('q_ddqn').abs() - pl.col('q_van').abs()).alias('dAbsQ'),
    )

    rows = []
    for env in sorted(joined.select('env_name').unique().to_series().to_list()):
        sub = joined.filter(pl.col('env_name') == env)
        n = sub.height
        if n < 10:
            continue
        bg_van = sub.get_column('bg_van').to_numpy()
        dabsq = sub.get_column('dAbsQ').to_numpy()
        qvan_mean = float(sub.get_column('q_van').mean())
        sign = '+' if qvan_mean > 0 else '-'
        rho, p = stats.spearmanr(bg_van, dabsq)
        rows.append({
            'env': env,
            'n': n,
            'sign_Q_van': sign,
            'rho_bg_dAbsQ': float(rho),
            'p_rho': float(p),
        })
    return pl.DataFrame(rows)


def test_4_signed_clip_universal(per_env: pl.DataFrame) -> dict[str, object]:
    """Structural test: Δ Q should be universally negative
    (the bootstrap target is always lower under DDQN)."""
    dq = per_env.get_column('dQ_mean').to_numpy()
    n_neg = int((dq < 0).sum())
    n_total = len(dq)
    binom = stats.binomtest(n_neg, n_total, p=0.5, alternative='greater')
    return {
        'n_neg': n_neg,
        'n_total': n_total,
        'binom_p': float(binom.pvalue),
        'mean_dQ_pooled': float(dq.mean()),
    }


def main() -> None:
    print('=' * 80)
    print('Test 1: Per-env paired sign-match (Δ|Q| sign predicts from sign(Q_van))')
    print('=' * 80)
    per_env = test_1_per_env_paired_sign()
    with pl.Config(tbl_rows=20, tbl_cols=15, fmt_str_lengths=30, tbl_width_chars=160):
        print(per_env)

    n_match = int(per_env.get_column('match').sum())
    n_total = per_env.height
    binom_p = stats.binomtest(n_match, n_total, p=0.5, alternative='greater').pvalue
    print(f'\nMatched: {n_match}/{n_total} envs (binomial one-sided p={binom_p:.4f})')

    print()
    print('=' * 80)
    print('Test 2: Cross-env regression — Δ|Q| ~ Q_van (predicted negative slope)')
    print('=' * 80)
    reg = test_2_cross_env_regression(per_env)
    print(f"  All envs (n={reg['full_n']}): r={reg['full_r']:+.3f} p={reg['full_p']:.4f} slope={reg['full_slope']:+.4f}")
    print(f"  Excluding unstable (n={reg['clean_n']}): r={reg['clean_r']:+.3f} p={reg['clean_p']:.4f} slope={reg['clean_slope']:+.4f}")

    print()
    print('=' * 80)
    print('Test 3: α dose-response (predicted: |Δ|Q|| monotone in α)')
    print('=' * 80)
    dose = test_3_alpha_dose_response()
    if dose.height > 0:
        with pl.Config(tbl_rows=10, tbl_cols=10, fmt_str_lengths=30, tbl_width_chars=160):
            print(dose)

    print()
    print('=' * 80)
    print('Test 4: Structural prediction — Δ Q < 0 universally (bootstrap clip is downward)')
    print('=' * 80)
    sig = test_4_signed_clip_universal(per_env)
    print(f"  Envs with mean ΔQ < 0: {sig['n_neg']}/{sig['n_total']}")
    print(f"  Pooled mean ΔQ across envs: {sig['mean_dQ_pooled']:+.4f}")
    print(f"  Binomial one-sided p>0.5: p={sig['binom_p']:.4f}")

    print()
    print('=' * 80)
    print('Test 5: Within-env ρ(bg_van, Δ|Q|) — clip-magnitude predicts |Q| change,')
    print('         sign of ρ flips with sign of Q_van (the cleanest mechanism test)')
    print('=' * 80)
    within = test_5_within_env_clip_predicts_dAbsQ()
    with pl.Config(tbl_rows=20, tbl_cols=10, fmt_str_lengths=30, tbl_width_chars=160):
        print(within.sort('rho_bg_dAbsQ'))

    by_sign = within.group_by('sign_Q_van').agg([
        pl.col('rho_bg_dAbsQ').mean().alias('mean_rho'),
        pl.col('rho_bg_dAbsQ').median().alias('median_rho'),
        pl.len().alias('n_envs'),
    ])
    print()
    print('Mean ρ(bg_van, Δ|Q|) by env Q-sign (predicted opposite signs):')
    print(by_sign)
    # Sign-flip test: how many envs have ρ in predicted direction?
    pred_match = within.with_columns(
        pl.when(
            (pl.col('sign_Q_van') == '+') & (pl.col('rho_bg_dAbsQ') < 0)
        ).then(True)
        .when(
            (pl.col('sign_Q_van') == '-') & (pl.col('rho_bg_dAbsQ') > 0)
        ).then(True)
        .otherwise(False)
        .alias('match')
    )
    n_match = int(pred_match.get_column('match').sum())
    n_total = pred_match.height
    binom_p = stats.binomtest(n_match, n_total, p=0.5, alternative='greater').pvalue
    print(f'Envs with ρ in predicted direction: {n_match}/{n_total} (binomial one-sided p={binom_p:.4f})')


if __name__ == '__main__':
    main()
