"""Per-env REACH-cohort decomposition.

`finding_mediation_polarity_conditional` Bridge 1 (marginal
ρ(bg, outcome) on REACH cells) pools to −0.18 — below the +0.2
substantive threshold. The pool was Fisher-z over per-env ρs;
this script reports the per-env ρ + sample size + key auxiliary
diagnostics so we can SEE which envs drive the pool down and
why, BEFORE generalising to "marginal bg→outcome is null".

Reports per env:

  rho_bg_outcome       marginal ρ(bg, outcome)
  rho_bg_outcome_pj    partial ρ(bg, outcome | jens)
  rho_jens_outcome     marginal ρ(jens, outcome)   — direct path
  d_outcome            (DDQN − vanilla) mean diff on raw outcome
  d_bg                 mech magnitude (paired diff on bg)
  d_jens               mech magnitude (paired diff on jens)
  outcome_sd_vanilla   outcome SD on vanilla — saturation diag
  outcome_sd_ddqn      outcome SD on ddqn — saturation diag

Reads `experiments/data/cache/ddqn.parquet`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src/corroborate_rl'))


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial Spearman ρ(x, y | z) via rank-regression residuals."""
    if x.size < 4:
        return float('nan')
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    # OLS residuals of rx and ry on rz
    Z = np.column_stack([np.ones_like(rz), rz])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex = rx - Z @ bx
    ey = ry - Z @ by
    if ex.std() == 0 or ey.std() == 0:
        return float('nan')
    return float(np.corrcoef(ex, ey)[0, 1])


def main() -> int:
    cache = REPO / 'experiments/data/cache/ddqn.parquet'
    df = pl.read_parquet(cache)
    # MODULE_SCOPE approx: canonical, drop bsuite; require finite
    # bg/jens/outcome + polarity.
    df = df.filter(
        ~pl.col('env_name').str.contains('bsuite')
        & pl.col('bootstrap_gap_magnitude').is_finite()
        & pl.col('jensen_gap').is_finite()
        & pl.col('eval_best_burst_raw_mean').is_finite()
        & pl.col('env_reward_polarity').is_finite()
    )
    reach = df.filter(pl.col('env_reward_polarity') < -0.3)

    print(f'{"env":<20} {"n":>4} {"rho_bgY":>8} {"pj":>8} {"rho_jY":>8} '
          f'{"d_out":>8} {"d_bg":>8} {"d_jens":>8} {"sd_v":>7} {"sd_d":>7}')
    print('-' * 100)

    # Determine which arm is vanilla / DDQN per env. The arm_key
    # 'baseline' is vanilla; any 'bootstrap=…double_greedify' is DDQN.
    rows: list[tuple[float, ...]] = []
    for env in reach.select(pl.col('env_name').unique()).to_series().to_list():
        sub = reach.filter(pl.col('env_name') == env)
        arms = sub.select(pl.col('arm_key').unique()).to_series().to_list()
        vanilla_arms = [a for a in arms if 'baseline' in a or a == 'vanilla']
        ddqn_arms = [a for a in arms if 'double_greedify' in a or 'ddqn' in a.lower()]
        if not vanilla_arms or not ddqn_arms:
            print(f'{env:<20}  skip: arms={arms}')
            continue

        bg = sub.get_column('bootstrap_gap_magnitude').to_numpy()
        jens = sub.get_column('jensen_gap').to_numpy()
        out = sub.get_column('eval_best_burst_raw_mean').to_numpy()
        n = bg.size

        # marginals + partial (across all cells, ignoring arm)
        if bg.std() > 0 and out.std() > 0:
            rho_bg_out = float(stats.spearmanr(bg, out).statistic)
        else:
            rho_bg_out = float('nan')
        if jens.std() > 0 and out.std() > 0:
            rho_jens_out = float(stats.spearmanr(jens, out).statistic)
        else:
            rho_jens_out = float('nan')
        pj = partial_spearman(bg, out, jens)

        # arm means
        v = sub.filter(pl.col('arm_key').is_in(vanilla_arms))
        d = sub.filter(pl.col('arm_key').is_in(ddqn_arms))
        d_out = float(d['eval_best_burst_raw_mean'].mean()) - \
            float(v['eval_best_burst_raw_mean'].mean())
        d_bg = float(d['bootstrap_gap_magnitude'].mean()) - \
            float(v['bootstrap_gap_magnitude'].mean())
        d_jens = float(d['jensen_gap'].mean()) - \
            float(v['jensen_gap'].mean())
        sd_v = float(v['eval_best_burst_raw_mean'].std() or 0)
        sd_d = float(d['eval_best_burst_raw_mean'].std() or 0)

        print(
            f'{env:<20} {n:>4} {rho_bg_out:>8.3f} {pj:>8.3f} '
            f'{rho_jens_out:>8.3f} {d_out:>8.2f} {d_bg:>8.3f} '
            f'{d_jens:>8.3f} {sd_v:>7.2f} {sd_d:>7.2f}'
        )
        rows.append((rho_bg_out, pj, rho_jens_out, d_out, d_bg, d_jens, n))

    print()
    # Fisher-z pool for sanity-check against the Finding's −0.18.
    rhos = [r[0] for r in rows if not math.isnan(r[0])]
    ns = [r[6] for r in rows if not math.isnan(r[0])]
    if rhos:
        zs = [math.atanh(max(min(r, 0.9999), -0.9999)) for r in rhos]
        ws = [n - 3 for n in ns]
        wm = sum(z * w for z, w in zip(zs, ws)) / sum(ws)
        pooled = math.tanh(wm)
        print(f'Fisher-z pooled ρ(bg, outcome): {pooled:.4f}  '
              f'(n_strata={len(rhos)}, n_total={sum(ns)})')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
