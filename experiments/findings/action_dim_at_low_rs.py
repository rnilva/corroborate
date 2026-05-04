"""Hasselt √(log|A|) test on the `action_dim_at_low_rs` sweep.

Pearl-rung-2 corollary of CLAIM 7 (under-learning rescue at
rs=0.1 on FourRooms): the do-effect mean_diff(ddqn − vanilla) in
native units should scale with √(log|A|) at the rescue-regime
reward scale, per Hasselt-2010 ε = σ_Q · √(2 log|A|). Larger |A|
→ larger Jensen bias for vanilla → bigger gap for DDQN to fill →
bigger interventional contrast.

This script:

1. Loads the four sparse-reward env paired_g cells at rs=0.1.
2. Computes mean_diff (native units) per (env, rs) using the
   registered `outcome_native` measurable.
3. Tabulates observed vs predicted using FourRooms (|A|=4, gap
   ≈ 0.49 from `reward_scale_low_fourrooms`) as the anchor:
     env-rescue-gap_predicted = 0.49 × √(log|A| / log 4)
4. Reports Pearson r between predicted and observed gap across
   the 4 envs at rs=0.1.
5. Repeats at rs=1.0 (control: should be small/null everywhere
   except whatever sweet-spotting pre-exists).

The MNISTBandit row is a NULL test: it has |A|=10 but is NOT a
sparse-reward env (every step gets a digit-conditioned reward).
If MNISTBandit's gap matches the √(log 10 / log 4) prediction,
the mechanism extends beyond sparsity. If it falls flat, the
sparsity-conditioning is the binding constraint — the
√(log|A|) story attaches to the joint scope (sparse AND large
|A|), not to |A| alone.

Usage:
    uv run python -m experiments.findings.action_dim_at_low_rs

Reads: experiments/data/action_dim_at_low_rs/runs.parquet
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl

import corroborate_rl.dqn.measurables as _m  # registers outcome_native
from corroborate.analyses.paired_g import paired_g

assert _m  # keep registration import live for type checker
from corroborate_rl import env_catalogue as _ec


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'action_dim_at_low_rs'
    / 'runs.parquet'
)

# FourRooms anchor: established gap at rs=0.1 from
# reward_scale_low_fourrooms (mean_diff ≈ +0.49 native).
FOURROOMS_ANCHOR_GAP = 0.49
FOURROOMS_LOG_A = math.log(4.0)


def _predicted_gap(n_actions: int) -> float:
    """ε ∝ √(log|A|), so gap ∝ √(log|A|). Anchor at FourRooms."""
    return FOURROOMS_ANCHOR_GAP * math.sqrt(
        math.log(max(n_actions, 2)) / FOURROOMS_LOG_A,
    )


def _format_row(
    env: str, n_a: int, rs: float,
    observed: float, observed_se: float,
    predicted: float | None, p_value: float, n_pairs: int,
) -> str:
    pred_str = f'{predicted:+.3f}' if predicted is not None else '   .  '
    return (
        f'  {env:<28} |A|={n_a:<3} rs={rs:<5}'
        f'  obs={observed:+.3f}±{observed_se:.3f}  pred={pred_str}'
        f'  p={p_value:.4f}  n={n_pairs}'
    )


def main() -> None:
    if not RUNS.exists():
        print(f'(skip — {RUNS} missing)')
        return
    df = pl.read_parquet(RUNS)
    cells = list(df.iter_rows(named=True))
    print(
        f'# action_dim_at_low_rs Hasselt √(log|A|) test '
        f'({len(cells)} cells)',
    )
    print('=' * 90)

    envs = sorted(set(df['env_name'].unique().to_list()))
    rss = sorted(set(df['reward_scale'].unique().to_list()))
    print(f'envs: {envs}')
    print(f'reward scales: {rss}')
    print()

    n_actions: dict[str, int] = {
        env: int(_ec.get(env).n_actions) for env in envs
    }
    print('|A|:', n_actions)
    print()

    for rs in rss:
        if rs == 0.1:
            label = 'rescue-regime (rs=0.1)'
        elif rs == 1.0:
            label = 'control (rs=1.0)'
        else:
            label = f'rs={rs}'
        print(f'## {label}')
        print('-' * 90)

        observed_pairs: list[tuple[str, int, float, float, float]] = []
        for env in sorted(envs, key=lambda e: n_actions[e]):
            n_a = n_actions[env]
            scoped = [
                c for c in cells
                if c.get('env_name') == env and c.get('reward_scale') == rs
            ]
            r = paired_g.fn(
                scoped,
                treatment_arm='ddqn',
                baseline_arm='vanilla_dqn',
                pair_by=('seed',),
                source='outcome_native',
            )
            pred = _predicted_gap(n_a) if rs == 0.1 else None
            print(_format_row(
                env, n_a, rs,
                r.mean_diff, r.mean_diff_se,
                pred, r.mean_diff_p_value, r.n_pairs,
            ))
            observed_pairs.append((env, n_a, rs, r.mean_diff, pred or 0.0))

        if rs == 0.1 and len(observed_pairs) >= 2:
            obs = [p[3] for p in observed_pairs if not math.isnan(p[3])]
            pre = [p[4] for p in observed_pairs if not math.isnan(p[3])]
            envs_used = [p[0] for p in observed_pairs if not math.isnan(p[3])]
            if len(obs) >= 2:
                n = float(len(obs))
                obs_mean = sum(obs) / n
                pre_mean = sum(pre) / n
                cov = sum(
                    (o - obs_mean) * (p - pre_mean) for o, p in zip(obs, pre)
                )
                obs_var = sum((o - obs_mean) ** 2 for o in obs)
                pre_var = sum((p - pre_mean) ** 2 for p in pre)
                if obs_var > 0.0 and pre_var > 0.0:
                    r_pearson = cov / math.sqrt(obs_var * pre_var)
                else:
                    r_pearson = float('nan')
                print()
                print(
                    f'  Pearson r(predicted, observed) across '
                    f'{len(obs)} envs = {r_pearson:+.3f}',
                )
                print(f'  envs used: {envs_used}')
        print()


if __name__ == '__main__':
    main()
