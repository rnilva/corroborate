"""Q1 — Defend staleness causality vs inverse causality.

The user's concern: paired-Δ analysis showed `Δ_target_staleness_late`
mediates ~27-65% of `Δ_outcome` (mech-HELD conditioning). But Δ is an
observation across paired (vanilla, DDQN) cells; the link could be
inverse-causal — successful seeds end up with low staleness rather
than low staleness causing success.

Three causal-inference checks:

1. **Partial Spearman** ρ(Δ_stale, Δ_o | Δ_jens) per env.
   If Δ_stale predicts Δ_o ONLY through Δ_jens, partial ρ → 0 after
   conditioning on the mech step. If staleness is a separate channel,
   partial ρ stays significant.

2. **Stratified partial Spearman** ρ(Δ_stale, Δ_o | Δ_jens) pooled
   across mech-HELD strata (env_idx). JCI-style: env-confound treated
   as backdoor, mech step as conditioning set.

3. **DoWhy backdoor adjustment** with explicit DAG:

       do(DDQN) → Δ_jens → Δ_stale → Δ_outcome
                                     ↑
                                  seed (random)

   Estimate: ATE(Δ_stale → Δ_outcome) using linear-regression
   backdoor adjustment, refute with placebo-treatment and
   random-common-cause.

The proper Pearl-rung-2 test (do(τ) directly varies staleness at
fixed sync_period) is the running Polyak-τ sweep — these checks are
the rung-1.5 evidence we have NOW from observational paired-Δ data.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from pathlib import Path

import numpy as np
import polars as pl
from corroborate.graph.discovery import (
    partial_spearman_rho, stratified_partial_spearman_rho,
)

CORPUS_DIR = Path('experiments/data/ddqn')
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'

# Envs with non-degenerate target_staleness panel.
SKIP_ENVS = {
    'BernoulliBandit-misc', 'GaussianBandit-misc', 'MNISTBandit-bsuite',
    'Catch-bsuite', 'DeepSea-bsuite', 'DiscountingChain-bsuite',
    'Freeway-MinAtar', 'MemoryChain-bsuite', 'UmbrellaChain-bsuite',
}


def _pair_arms(df: pl.DataFrame, env: str) -> pl.DataFrame:
    sub = df.filter(pl.col('env_name') == env)
    cols = ['seed', 'jensen_gap', 'target_staleness_late', 'eval_best_burst_mean']
    v = sub.filter(pl.col('arm_key') == 'baseline').select(cols)
    d = sub.filter(pl.col('arm_key') == DDQN).select(cols)
    if len(v) == 0 or len(d) == 0:
        return pl.DataFrame()
    v = v.rename({c: f'{c}_v' for c in v.columns if c != 'seed'})
    d = d.rename({c: f'{c}_d' for c in d.columns if c != 'seed'})
    j = v.join(d, on='seed', how='inner').filter(
        pl.col('eval_best_burst_mean_v').is_finite()
        & pl.col('eval_best_burst_mean_d').is_finite()
        & pl.col('jensen_gap_v').is_finite()
        & pl.col('jensen_gap_d').is_finite()
        & pl.col('target_staleness_late_v').is_finite()
        & pl.col('target_staleness_late_d').is_finite()
    )
    return j


def main() -> None:
    runs = pl.read_parquet(CORPUS_DIR / 'runs.parquet', columns=['id', 'env_name', 'arm_key', 'seed'])
    ms = pl.read_parquet(CORPUS_DIR / 'measurements.parquet')
    df = runs.join(ms, on='id', how='inner')
    print(f'ddqn corpus: {len(df)} cells across {df["env_name"].n_unique()} envs', flush=True)

    # ============================================================
    # CHECK 1 — Per-env partial Spearman ρ(Δ_stale, Δ_o | Δ_jens).
    # ============================================================
    print()
    print('=== CHECK 1: ρ(Δ_stale, Δ_o | Δ_jens) per env (mech-HELD) ===\n')
    print(f'{"env":<24} {"n":>4} {"ρ_marg":>8} {"p_marg":>9} {"ρ_part":>8} {"p_part":>9} {"reading":>30}')
    print('-' * 100)

    rows = []
    pooled_d_o = []
    pooled_d_stale = []
    pooled_d_jens = []
    pooled_strata = []

    for env in sorted(df['env_name'].unique()):
        if env in SKIP_ENVS:
            continue
        j = _pair_arms(df, env)
        if len(j) == 0:
            continue
        d_o = (j['eval_best_burst_mean_d'] - j['eval_best_burst_mean_v']).to_numpy()
        d_stale = (j['target_staleness_late_d'] - j['target_staleness_late_v']).to_numpy()
        d_jens = (j['jensen_gap_d'] - j['jensen_gap_v']).to_numpy()

        # Mech-HELD conditioning: only pairs where Δ_jens < 0 (DDQN
        # actually reduced bias). This is the upstream filter the
        # bridges use.
        mask = d_jens < 0
        if mask.sum() < 5:
            continue
        d_o_h, d_stale_h, d_jens_h = d_o[mask], d_stale[mask], d_jens[mask]

        # Marginal ρ(Δ_stale, Δ_o)
        from scipy.stats import spearmanr
        rho_marg, p_marg = spearmanr(d_stale_h, d_o_h)
        # Partial ρ(Δ_stale, Δ_o | Δ_jens)
        rho_part, p_part = partial_spearman_rho(d_stale_h, d_o_h, d_jens_h)

        # Reading: marginal sig + partial sig → independent staleness channel.
        # Marginal sig + partial ns → staleness explained by Δ_jens.
        # Marginal ns → no detectable signal at this n.
        if abs(rho_marg) >= 0.3 and p_marg < 0.05:
            if abs(rho_part) >= 0.2 and p_part < 0.10:
                reading = 'INDEP CHANNEL (survives)'
            else:
                reading = 'mediated by Δ_jens'
        else:
            reading = 'underpowered/no signal'

        print(
            f'{env:<24} {int(mask.sum()):>4d} '
            f'{rho_marg:>+8.3f} {p_marg:>9.3g} '
            f'{rho_part:>+8.3f} {p_part:>9.3g} '
            f'{reading:>30}',
            flush=True,
        )
        rows.append({
            'env': env, 'n_pairs_mech_held': int(mask.sum()),
            'rho_marginal': float(rho_marg), 'p_marginal': float(p_marg),
            'rho_partial': float(rho_part), 'p_partial': float(p_part),
            'reading': reading,
        })

        pooled_d_o.append(d_o_h)
        pooled_d_stale.append(d_stale_h)
        pooled_d_jens.append(d_jens_h)
        pooled_strata.extend([env] * int(mask.sum()))

    # ============================================================
    # CHECK 2 — Pooled stratified partial Spearman.
    # ============================================================
    if pooled_d_o:
        po = np.concatenate(pooled_d_o)
        ps = np.concatenate(pooled_d_stale)
        pj = np.concatenate(pooled_d_jens)

        print()
        print('=== CHECK 2: stratified partial ρ pooled across envs ===\n')
        rho_strat, p_strat = stratified_partial_spearman_rho(
            ps, po, pj, pooled_strata, min_stratum_size=5,
        )
        print(f'  ρ_strat(Δ_stale, Δ_o | Δ_jens, strata=env) = {rho_strat:+.4f}', flush=True)
        print(f'  p = {p_strat:.4g}', flush=True)
        print(f'  n_pooled = {len(po)} across {len(set(pooled_strata))} envs', flush=True)
        if not np.isnan(rho_strat) and abs(rho_strat) >= 0.15 and p_strat < 0.05:
            print(f'  reading: STALENESS IS AN INDEPENDENT CHANNEL (survives env-stratification + mech adjustment)', flush=True)
        else:
            print(f'  reading: staleness signal absorbed by env + Δ_jens', flush=True)

    # ============================================================
    # CHECK 3 — Within-env-standardized backdoor adjustment.
    # Standardize Δ_o, Δ_stale, Δ_jens within each env BEFORE pooling
    # so the regression coefficient is on a unit scale (env-FE
    # absorbed by the standardization). Avoids the scale-disparity
    # blow-up where outcome ranges differ 100× across envs.
    # ============================================================
    if pooled_d_o:
        print()
        print('=== CHECK 3: within-env-standardized ATE(Δ_stale → Δ_o) ===\n')

        # Stack within-env-standardized Δ vectors.
        zo, zs, zj = [], [], []
        env_idx = {e: i for i, e in enumerate(sorted(set(pooled_strata)))}
        for e in env_idx:
            mask = np.array([s == e for s in pooled_strata])
            o_e = po[mask]
            s_e = ps[mask]
            j_e = pj[mask]
            if o_e.std() == 0 or s_e.std() == 0 or j_e.std() == 0:
                continue
            zo.append((o_e - o_e.mean()) / o_e.std())
            zs.append((s_e - s_e.mean()) / s_e.std())
            zj.append((j_e - j_e.mean()) / j_e.std())
        zo = np.concatenate(zo)
        zs = np.concatenate(zs)
        zj = np.concatenate(zj)

        # OLS: zo ~ α + β_stale·zs + β_jens·zj
        n = len(zo)
        X = np.column_stack([np.ones(n), zs, zj])
        beta, _, rank, _ = np.linalg.lstsq(X, zo, rcond=None)
        y_pred = X @ beta
        resid = zo - y_pred
        df = n - rank
        sigma2 = float((resid ** 2).sum() / df) if df > 0 else float('nan')
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))

        beta_stale, se_stale = float(beta[1]), float(se[1])
        beta_jens, se_jens = float(beta[2]), float(se[2])
        z_stale = beta_stale / se_stale if se_stale > 0 else float('nan')
        z_jens = beta_jens / se_jens if se_jens > 0 else float('nan')

        print(f'  Standardized within-env (z-score), then pooled across {len(env_idx)} envs (n={n})', flush=True)
        print(f'  β(Δ_stale → Δ_o | Δ_jens) = {beta_stale:+.4f} ± {se_stale:.4f}  z={z_stale:+.2f}', flush=True)
        print(f'  β(Δ_jens  → Δ_o | Δ_stale) = {beta_jens:+.4f} ± {se_jens:.4f}  z={z_jens:+.2f}', flush=True)

        # Refutation 1: placebo treatment (shuffle Δ_stale within env).
        rng = np.random.default_rng(42)
        zs_placebo = np.empty_like(zs)
        offset = 0
        for e in env_idx:
            mask = np.array([s == e for s in pooled_strata])
            o_e = po[mask]
            if o_e.std() == 0:
                continue
            n_e = mask.sum()
            zs_placebo[offset:offset + n_e] = rng.permutation(zs[offset:offset + n_e])
            offset += n_e
        X_placebo = X.copy()
        X_placebo[:, 1] = zs_placebo
        beta_placebo, _, _, _ = np.linalg.lstsq(X_placebo, zo, rcond=None)
        print(f'  PLACEBO β(within-env shuffled Δ_stale) = {float(beta_placebo[1]):+.4f}  (should be ~0)', flush=True)

        # Refutation 2: random common cause (additive noise to Δ_jens).
        zj_noise = zj + rng.normal(0, 1.0, size=n)
        X_rcc = X.copy()
        X_rcc[:, 2] = zj_noise
        beta_rcc, _, _, _ = np.linalg.lstsq(X_rcc, zo, rcond=None)
        print(f'  RCC     β(Δ_stale, with z_jens noised) = {float(beta_rcc[1]):+.4f}  (drift = {float(beta_rcc[1]) - beta_stale:+.4f})', flush=True)

        # Refutation 3 — INVERSE-CAUSAL test. If outcome→staleness, then
        # regressing zs ~ zo + zj should give β_o > 0, but the asymmetry
        # is observable as: forward β_stale = ρ_part / sd ratio. We
        # already have ρ_part = +0.081. Run the reverse: zs ~ α + β_o·zo + β_j·zj.
        X_rev = np.column_stack([np.ones(n), zo, zj])
        beta_rev, _, _, _ = np.linalg.lstsq(X_rev, zs, rcond=None)
        y_pred_rev = X_rev @ beta_rev
        resid_rev = zs - y_pred_rev
        df_rev = n - 3
        sigma2_rev = float((resid_rev ** 2).sum() / df_rev)
        cov_rev = sigma2_rev * np.linalg.pinv(X_rev.T @ X_rev)
        se_rev = np.sqrt(np.diag(cov_rev))
        beta_rev_o = float(beta_rev[1])
        se_rev_o = float(se_rev[1])
        z_rev_o = beta_rev_o / se_rev_o if se_rev_o > 0 else float('nan')
        print()
        print('  REVERSE: zs ~ α + β·zo + β·zj  (inverse-causal probe)', flush=True)
        print(f'    β(Δ_o → Δ_stale | Δ_jens) = {beta_rev_o:+.4f} ± {se_rev_o:.4f}  z={z_rev_o:+.2f}', flush=True)
        if abs(z_stale) > abs(z_rev_o) and abs(z_stale) >= 2.0:
            forward_strength = 'forward stronger than reverse'
        elif abs(z_rev_o) > abs(z_stale):
            forward_strength = 'REVERSE STRONGER — possible inverse causality'
        else:
            forward_strength = 'forward ≈ reverse — direction undetermined from observational data'
        print(f'    reading: {forward_strength}', flush=True)

        rows_pooled = {
            'rho_strat_partial': float(rho_strat) if not np.isnan(rho_strat) else None,
            'p_strat_partial': float(p_strat) if not np.isnan(p_strat) else None,
            'beta_stale_adjusted': beta_stale,
            'se_stale_adjusted': se_stale,
            'z_stale_adjusted': z_stale,
            'beta_jens_adjusted': beta_jens,
            'se_jens_adjusted': se_jens,
            'z_jens_adjusted': z_jens,
            'placebo_beta_stale': float(beta_placebo[1]),
            'rcc_beta_stale': float(beta_rcc[1]),
            'reverse_beta_o': beta_rev_o,
            'reverse_z_o': z_rev_o,
            'direction_reading': forward_strength,
        }
    else:
        rows_pooled = {}

    print()
    print('=== Verdict on staleness causality ===\n')

    indep_envs = [r for r in rows if r['reading'] == 'INDEP CHANNEL (survives)']
    print(f'  per-env INDEP-channel verdicts: {len(indep_envs)}/{len(rows)}', flush=True)
    if indep_envs:
        for r in indep_envs:
            print(f'    {r["env"]}: ρ_part={r["rho_partial"]:+.3f}, p={r["p_partial"]:.3g}', flush=True)

    if rows_pooled:
        z = rows_pooled.get('z_stale_adjusted', float('nan'))
        if isinstance(z, float) and abs(z) >= 2.0:
            print(f'  pooled OLS β(Δ_stale → Δ_o | Δ_jens, env-FE) z={z:+.2f} — survives mech adjustment', flush=True)
        elif isinstance(z, float):
            print(f'  pooled OLS β z={z:+.2f} — does NOT survive', flush=True)
        z2 = rows_pooled.get('rho_strat_partial', float('nan'))
        if isinstance(z2, float) and not np.isnan(z2):
            print(f'  stratified partial ρ = {z2:+.3f} (p={rows_pooled["p_strat_partial"]:.3g})', flush=True)

    print()
    print('  Important caveat: NONE of these probes are Pearl-rung-2.', flush=True)
    print('  The proper test is do(τ) at fixed sync_period — Polyak-τ', flush=True)
    print('  intervention sweep currently running on GPU (1440 cells).', flush=True)

    out_dir = Path('experiments/findings/sync_curve_breakout')
    out = out_dir / 'staleness_causal_inference.json'
    out.write_text(json.dumps({'per_env': rows, 'pooled': rows_pooled}, indent=2, default=str))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
