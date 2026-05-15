"""Diagnostic for the three "mech-without-translation" envs at
canonical 1M (Asterix-MinAtar, SpaceInvaders-MinAtar, MetaMaze-misc).

These envs are the ones where the canonical-cache DoWhy mediation
through `jens` over-absorbs to 400-977% (memory
`findings_ddqn_mediator_heterogeneity`): DDQN reduces jens as
expected, jens should mediate to outcome, but observed Δ_outcome
≈ 0 — implying a strong COMPETING direct effect on outcome with
opposite sign.

This script does NOT attempt to mediate. It just looks at per-env
arm-mean deltas (DDQN − vanilla) across candidate measurables and
contrasts the non-translating cohort against the translating
cohort (Freeway/Acrobot/Breakout/PacMan/MountainCar). Whatever
moves differently between the two cohorts is the candidate
competing channel — to be then verified with a designed
intervention sweep, not committed to as a finding.

Independent-samples form per memory `feedback_paired_g_in_rl`:
no per-seed pairing across arms.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src/corroborate_rl'))

CACHE = REPO / 'experiments' / 'data' / 'cache' / 'ddqn.parquet'

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASELINE = 'baseline'

NON_TRANSLATING = ('Asterix-MinAtar', 'SpaceInvaders-MinAtar', 'MetaMaze-misc')
TRANSLATING = (
    'Freeway-MinAtar', 'Acrobot-v1', 'Breakout-MinAtar',
    'PacMan-jumanji', 'MountainCar-v0',
)

# Candidate competing-mediator measurables. Anything that
# (a) DDQN modulates and (b) could plausibly drive outcome.
CANDIDATES: tuple[str, ...] = (
    'eval_best_burst_mean',       # outcome (paper convention; γ-discounted)
    'eval_best_burst_raw_mean',   # outcome (γ-invariant)
    'jensen_gap',                 # mediator under test
    # Q-shape channel
    'q_late_mean',
    'q_action_std_late',
    'q_argmax_margin_late',
    'q_max_temporal_cv_late',
    'q_divergence_score',
    # Policy-statistics channel
    'argmax_entropy_late',
    'argmax_persistence_late',
    # Bootstrap-gap variants (per memory bg-not-causally-manipulated)
    'bootstrap_gap_magnitude',
    'bootstrap_gap_frac_active',
    'bootstrap_gap_q99',
    # Calibration / staleness
    'q_mc_calibration_pearson',
    'target_staleness_late',
    'jensen_dormancy_gap',
)


def main() -> None:
    df = pl.read_parquet(str(CACHE))
    df = df.filter(pl.col('total_steps') == 1000000)

    rows: list[dict[str, str | float | int]] = []
    for env in NON_TRANSLATING + TRANSLATING:
        sub = df.filter(pl.col('env_name') == env)
        ddqn_seeds = sub.filter(pl.col('arm_key') == DDQN)
        base_seeds = sub.filter(pl.col('arm_key') == BASELINE)
        for m in CANDIDATES:
            if m not in df.columns:
                continue
            ddqn_vals = ddqn_seeds[m].drop_nulls().drop_nans().to_numpy()
            base_vals = base_seeds[m].drop_nulls().drop_nans().to_numpy()
            if ddqn_vals.size < 3 or base_vals.size < 3:
                continue
            mu_d, mu_b = float(np.mean(ddqn_vals)), float(np.mean(base_vals))
            # Pooled SD for an honest Cohen's-d arm-diff (independent samples).
            sd = math.sqrt(
                (np.var(ddqn_vals, ddof=1) + np.var(base_vals, ddof=1)) / 2
            )
            d_cohen = (mu_d - mu_b) / sd if sd > 0 else float('nan')
            rows.append({
                'env': env,
                'cohort': 'NON-TRANS' if env in NON_TRANSLATING else 'TRANS',
                'measurable': m,
                'ddqn_mean': mu_d,
                'base_mean': mu_b,
                'delta': mu_d - mu_b,
                'cohen_d': d_cohen,
                'n_d': int(ddqn_vals.size),
                'n_b': int(base_vals.size),
            })

    out = pl.DataFrame(rows)
    # For each measurable, contrast median Cohen's-d in each cohort.
    print('\n=== per-measurable cohort median Cohen\'s-d (DDQN − vanilla) ===')
    print('Sign-differential measurables are the candidate competing channel.\n')
    summary = (
        out.group_by(['measurable', 'cohort'])
        .agg(pl.col('cohen_d').median().alias('median_d'))
        .pivot(on='cohort', index='measurable', values='median_d')
    )
    # Reorder columns
    summary = summary.with_columns([
        (pl.col('NON-TRANS') - pl.col('TRANS')).alias('non_minus_trans'),
    ]).sort(pl.col('non_minus_trans').abs(), descending=True)
    print(summary)

    print('\n=== per-env detail: non-translating cohort ===')
    for env in NON_TRANSLATING:
        env_rows = out.filter(pl.col('env') == env)
        print(f'\n--- {env} ---')
        print(
            env_rows.select([
                'measurable', 'ddqn_mean', 'base_mean', 'delta', 'cohen_d',
            ]).sort(pl.col('cohen_d').abs(), descending=True),
        )


if __name__ == '__main__':
    main()
