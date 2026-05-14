"""Per-env DDQN-vs-vanilla decomposition across the full canonical cohort.

The user's "per-env before generalisation" discipline: don't
report cross-env aggregates without first checking each env's
individual story.

Per env, reports:
  n_v / n_d            cell counts per arm
  pol / align          env-property scopes (REACH/SURVIVE/saturated)
  Δ_jens               mech-canonical (Q − MC_disc); has Q-MC algebra
  Δ_bg_frac            MC-free; rate of online/target argmax disagreement
  Δ_bg_q99             MC-free; tail wedge magnitude
  Δ_q_late             Q-magnitude channel (Bellman-entangled but
                       independent of MC)
  Δ_out_raw            γ-invariant outcome; the canonical reporting metric
  Δ_out_disc           discounted outcome; algebraically tied to Q
  d_out_raw            unpaired Cohen's d on raw outcome (effect size)
  d_jens               unpaired Cohen's d on jens (mech effect size)

Reads `experiments/data/cache/ddqn.parquet`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src/corroborate_rl'))


def _arm_diff(d: pl.DataFrame, v: pl.DataFrame, col: str) -> float:
    if col not in d.columns or col not in v.columns:
        return float('nan')
    return float(d[col].mean() or float('nan')) - float(v[col].mean() or float('nan'))


def _cohens_d_unpaired(d: pl.DataFrame, v: pl.DataFrame, col: str) -> float:
    if col not in d.columns or col not in v.columns:
        return float('nan')
    xd = d[col].drop_nulls().to_numpy()
    xv = v[col].drop_nulls().to_numpy()
    if xd.size < 3 or xv.size < 3:
        return float('nan')
    pooled_sd = np.sqrt(0.5 * (xd.var(ddof=1) + xv.var(ddof=1)))
    if pooled_sd == 0:
        return float('nan')
    return float((xd.mean() - xv.mean()) / pooled_sd)


def main() -> int:
    cache = REPO / 'experiments/data/cache/ddqn.parquet'
    df = pl.read_parquet(cache).filter(
        ~pl.col('env_name').str.contains('bsuite')
    )
    print(f'cache: {df.shape[0]} cells; {df["env_name"].n_unique()} envs')
    print()
    print(
        f'{"env":<24} {"n_v":>4} {"n_d":>4} {"pol":>5} {"align":>6} '
        f'{"Δ_jens":>8} {"Δ_bg_frac":>11} {"Δ_bg_q99":>10} '
        f'{"Δ_q_late":>10} {"Δ_out_raw":>10} {"Δ_out_disc":>11} '
        f'{"d_out_raw":>10} {"d_jens":>8}'
    )
    print('-' * 145)
    for env in df.select(pl.col('env_name').unique()).to_series().sort().to_list():
        sub = df.filter(pl.col('env_name') == env)
        v = sub.filter(pl.col('arm_key').str.contains('baseline'))
        d = sub.filter(~pl.col('arm_key').str.contains('baseline'))
        if v.is_empty() or d.is_empty():
            continue
        pol = float(sub['env_reward_polarity'].mean() or float('nan'))
        align = float(sub['env_disc_raw_alignment'].mean() or float('nan')) \
            if 'env_disc_raw_alignment' in sub.columns else float('nan')
        print(
            f'{env:<24} {v.shape[0]:>4} {d.shape[0]:>4} '
            f'{pol:>+5.2f} {align:>+6.2f} '
            f'{_arm_diff(d, v, "jensen_gap"):>+8.2f} '
            f'{_arm_diff(d, v, "bootstrap_gap_frac_active"):>+11.4f} '
            f'{_arm_diff(d, v, "bootstrap_gap_q99"):>+10.4f} '
            f'{_arm_diff(d, v, "q_late_mean"):>+10.2f} '
            f'{_arm_diff(d, v, "eval_best_burst_raw_mean"):>+10.2f} '
            f'{_arm_diff(d, v, "eval_best_burst_mean"):>+11.3f} '
            f'{_cohens_d_unpaired(d, v, "eval_best_burst_raw_mean"):>+10.2f} '
            f'{_cohens_d_unpaired(d, v, "jensen_gap"):>+8.2f}'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
