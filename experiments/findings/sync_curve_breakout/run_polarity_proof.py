"""Formal proof: env reward polarity (goal vs survival) flips the
sign of `r(Δeff_h, Δoutcome)` per seed.

Hypothesis: Goal envs (shorter trajectory = better) have NEGATIVE
per-seed coupling between Δ_eff_h and Δ_outcome (more chain-shortening
→ more outcome benefit). Survival envs (longer trajectory = better)
have POSITIVE coupling. Cross-env meta-regressions wash this out
because the signs cancel.

Formal tests:
1. Per-env Pearson r(Δeh, Δmc) with Fisher-z 95% CI
2. Binomial sign test: do signs match polarity prediction across envs?
3. Stratified Fisher-z pooling per polarity class — test whether
   each pool is significantly non-zero AND has predicted sign
4. Pooled-rho difference test: are the two polarity-pool rhos
   significantly different from each other (interaction test)?
5. Z-test on each pool's rho against zero
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import binomtest, norm, pearsonr

from corroborate.graph.discovery import partial_spearman_rho

# Polarity coding from env semantics. GOAL: shorter trajectory = better
# (negative-step penalty until terminal reward, or terminal-only reward).
# SURVIVAL: longer trajectory = better (per-step positive reward, or
# accumulated rewards that require staying alive).
POLARITY: dict[str, str] = {
    'Acrobot-v1':              'goal',     # swing-up to height; -1/step
    'FourRooms-misc':          'goal',     # reach exit; reward at exit
    'MountainCar-v0':          'goal',     # reach flag; -1/step
    'DiscountingChain-bsuite': 'goal',     # reach end of chain
    'DeepSea-bsuite':          'goal',     # reach treasure
    'MemoryChain-bsuite':      'goal',     # reach end with correct action
    'UmbrellaChain-bsuite':    'goal',     # accumulate to terminal reward
    'MetaMaze-misc':           'goal',     # navigation env
    'CartPole-v1':             'survival', # +1/step while pole upright
    'Breakout-MinAtar':        'survival', # break bricks while ball alive
    'SpaceInvaders-MinAtar':   'survival', # kill aliens, avoid death
    'Asterix-MinAtar':         'survival', # collect rewards, avoid enemies
    'Pong-misc':               'survival', # game accumulating points
    # Excluded — fixed-length / single-step (no policy-driven eff_h variation):
    # 'Catch-bsuite':            'episodic',
    # 'Freeway-MinAtar':         'episodic_fixed_length',
    # 'BernoulliBandit-misc':    'single_step',
    # 'GaussianBandit-misc':     'single_step',
    # 'MNISTBandit-bsuite':      'single_step',
}


def main() -> None:
    df = pl.read_parquet('experiments/data/cache/ddqn_universe.parquet')
    treatment_substr = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
    baseline = 'baseline'

    df = df.with_columns(
        eff_h_new = pl.when(pl.col('gamma') < 1.0).then(
            1.0 / (1.0 - pl.col('gamma') * pl.col('bootstrap_fraction'))
        ).otherwise(float('nan'))
    )

    # Drop only the Q-explosion regime (q_div >= 1000). Dormant cells
    # (q_div < 0.02) and cells without q_div are KEPT — dormancy doesn't
    # directly break the polarity prediction (Δeff_h still tracks env
    # type via policy effect on episode length, regardless of whether
    # vanilla overestimates Q). Q-explosion DOES break it because Q
    # magnitude swamps the policy-quality channel.
    NOT_Q_EXPLODED = (
        pl.col('q_divergence_score').is_nan()
        | (pl.col('q_divergence_score') < 1000.0)
    )

    # Per-env scope refinements: some envs have multiple contrastive
    # corpora at different HP regimes. The polarity prediction operates
    # in the link-active regime per env.
    # MetaMaze: needs γ ≥ 0.995 per CLAIM 5b — the chain-depth-amplifier
    # regime where DDQN benefit emerges. Lower γ has no DDQN benefit
    # and pooling across γ creates cross-regime aggregation artifacts.
    PER_ENV_SCOPE: dict[str, pl.Expr] = {
        'MetaMaze-misc': pl.col('gamma') >= 0.995,
    }

    rows = []
    for env, polarity in POLARITY.items():
        env_filter = pl.col('env_name') == env
        env_scope = PER_ENV_SCOPE.get(env)
        if env_scope is not None:
            env_filter = env_filter & env_scope
        sub = df.filter(env_filter & NOT_Q_EXPLODED)
        arms = sorted(sub['arm_key'].unique())
        treatment = next((a for a in arms if treatment_substr in a), None)
        if treatment is None or baseline not in arms:
            continue
        # Pair by (corpus, gamma, total_steps, sync_period, seed) to
        # prevent cross-HP pairing within the same corpus. Several
        # corpora vary one HP across cells:
        #   gamma_sweep_metamaze*: γ ∈ {0.9, 0.95, 0.99, 0.995, 0.999}
        #   asterix_q_stability:   sync_period ∈ {100, 10000}
        #   ddqn:                  total_steps ∈ {50k, 200k}
        # Pairing without these keys cross-pairs cells under different
        # HPs and induces Simpson-style spurious correlations.
        pair_keys = ['corpus', 'gamma', 'total_steps', 'sync_period', 'seed']
        v = sub.filter(pl.col('arm_key') == baseline).select(pair_keys + ['eff_h_new', 'eval_best_burst_mean']).rename({'eff_h_new':'eh_v','eval_best_burst_mean':'ov'})
        d = sub.filter(pl.col('arm_key') == treatment).select(pair_keys + ['eff_h_new', 'eval_best_burst_mean']).rename({'eff_h_new':'eh_d','eval_best_burst_mean':'od'})
        j = v.join(d, on=pair_keys, how='inner').filter(
            pl.col('eh_v').is_not_nan() & pl.col('eh_d').is_not_nan()
            & pl.col('ov').is_not_nan() & pl.col('od').is_not_nan()
        )
        arr = j.to_pandas()
        n = len(arr)
        if n < 3: continue
        d_eh = (arr['eh_d'] - arr['eh_v']).to_numpy()
        d_mc = (arr['od'] - arr['ov']).to_numpy()
        if d_eh.std() == 0 or d_mc.std() == 0:
            r, p = float('nan'), float('nan')
        else:
            r, p = pearsonr(d_eh, d_mc)
        # Fisher-z transform for CI
        if abs(r) < 1.0 and not math.isnan(r) and n > 3:
            z = 0.5 * math.log((1 + r) / (1 - r))
            se = 1.0 / math.sqrt(n - 3)
            z_lo = z - 1.96 * se
            z_hi = z + 1.96 * se
            r_lo = math.tanh(z_lo)
            r_hi = math.tanh(z_hi)
        else:
            r_lo = r_hi = float('nan')
        predicted_sign = -1 if polarity == 'goal' else +1
        match = bool(r * predicted_sign > 0) if not math.isnan(r) else None
        rows.append({
            'env': env, 'polarity': polarity, 'n_pairs': n,
            'r': r, 'p': p, 'r_lo': r_lo, 'r_hi': r_hi,
            'predicted_sign': predicted_sign, 'sign_matches': match,
            'mean_d_eh': float(d_eh.mean()), 'mean_d_mc': float(d_mc.mean()),
        })

    print('=' * 105)
    print('Per-env per-seed r(Δeff_h, Δoutcome) with Fisher-z 95% CI')
    print('=' * 105)
    print(f'{"env":<25} {"polarity":<10} {"n":>4} {"r":>8} {"95% CI":>22} {"p":>10} {"pred":>6} {"match":>6}')
    for row in sorted(rows, key=lambda r: (r['polarity'], r['env'])):
        ci = f'[{row["r_lo"]:>+.3f}, {row["r_hi"]:>+.3f}]' if not math.isnan(row['r_lo']) else 'n/a'
        match_str = '✓' if row['sign_matches'] else ('✗' if row['sign_matches'] is False else '-')
        print(f'{row["env"]:<25} {row["polarity"]:<10} {row["n_pairs"]:>4} {row["r"]:>+8.3f} {ci:>22} {row["p"]:>10.3g} {("−" if row["predicted_sign"]<0 else "+"):>6} {match_str:>6}')

    print()
    print('=' * 105)
    print('Test 1: Binomial sign test — across all polarity-coded envs, do signs match prediction?')
    print('=' * 105)
    matches = sum(1 for r in rows if r['sign_matches'] is True)
    n_total = sum(1 for r in rows if r['sign_matches'] in (True, False))
    bt = binomtest(matches, n_total, p=0.5, alternative='greater')
    print(f'  Sign matches: {matches}/{n_total} envs')
    print(f'  Binomial p (one-sided H1: matches > 0.5) = {bt.pvalue:.4f}')
    print(f'  Binomial 95% CI on match rate = {bt.proportion_ci(method="exact").low:.3f} .. {bt.proportion_ci(method="exact").high:.3f}')

    print()
    print('=' * 105)
    print('Test 2: Within-polarity Fisher-z pool — is each pool rho significantly non-zero with predicted sign?')
    print('=' * 105)
    for polarity in ('goal', 'survival'):
        in_pol = [r for r in rows if r['polarity'] == polarity and not math.isnan(r['r'])]
        z_vals = []
        ws = []
        for r in in_pol:
            r_clamp = max(-0.999999, min(0.999999, r['r']))
            z = 0.5 * math.log((1 + r_clamp) / (1 - r_clamp))
            n = r['n_pairs']
            z_vals.append(z)
            ws.append(n - 3)  # weight by df
        total_w = sum(ws)
        if total_w == 0:
            continue
        z_pool = sum(w * z for w, z in zip(ws, z_vals)) / total_w
        rho_pool = math.tanh(z_pool)
        z_stat = z_pool * math.sqrt(total_w)
        p = 2 * (1 - float(norm.cdf(abs(z_stat))))
        # 95% CI
        z_lo = z_pool - 1.96 / math.sqrt(total_w)
        z_hi = z_pool + 1.96 / math.sqrt(total_w)
        rho_lo = math.tanh(z_lo)
        rho_hi = math.tanh(z_hi)
        sign_predicted = '−' if polarity == 'goal' else '+'
        sign_match = (rho_pool < 0 and polarity == 'goal') or (rho_pool > 0 and polarity == 'survival')
        print(f'  {polarity.upper():<10} pool: ρ_pooled = {rho_pool:>+.3f}, 95% CI [{rho_lo:>+.3f}, {rho_hi:>+.3f}], '
              f'p = {p:.3g}, n_envs = {len(in_pol)}, total_w = {total_w}, predicted sign = {sign_predicted}, match = {"✓" if sign_match else "✗"}')

    print()
    print('=' * 105)
    print('Test 3: Cross-polarity difference — is goal-pool ρ significantly different from survival-pool ρ?')
    print('=' * 105)
    goal_zs = []
    goal_ws = []
    surv_zs = []
    surv_ws = []
    for r in rows:
        if math.isnan(r['r']): continue
        r_clamp = max(-0.999999, min(0.999999, r['r']))
        z = 0.5 * math.log((1 + r_clamp) / (1 - r_clamp))
        n = r['n_pairs']
        if r['polarity'] == 'goal':
            goal_zs.append(z); goal_ws.append(n - 3)
        else:
            surv_zs.append(z); surv_ws.append(n - 3)
    z_g = sum(w * z for w, z in zip(goal_ws, goal_zs)) / sum(goal_ws)
    z_s = sum(w * z for w, z in zip(surv_ws, surv_zs)) / sum(surv_ws)
    se_diff = math.sqrt(1.0/sum(goal_ws) + 1.0/sum(surv_ws))
    z_diff = (z_s - z_g) / se_diff  # H1: survival > goal
    p_diff = 2 * (1 - float(norm.cdf(abs(z_diff))))
    print(f'  z_goal = {z_g:+.3f} (ρ = {math.tanh(z_g):+.3f})')
    print(f'  z_survival = {z_s:+.3f} (ρ = {math.tanh(z_s):+.3f})')
    print(f'  z_survival − z_goal = {z_s - z_g:+.3f}, SE = {se_diff:.3f}, z_stat = {z_diff:.3f}, p_two_sided = {p_diff:.3g}')

    out = Path('experiments/findings/sync_curve_breakout/polarity_proof.json')
    out.write_text(json.dumps({
        'per_env': rows,
        'binomial_sign_test': {'matches': matches, 'n_total': n_total, 'p': bt.pvalue},
    }, indent=2))
    print()
    print(f'wrote: {out}')


if __name__ == '__main__':
    main()
