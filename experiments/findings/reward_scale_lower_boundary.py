"""Lower-boundary mapping of the FourRooms rescue regime.

Combines two FourRooms reward-scale sweeps into one 7-point
response curve:

  reward_scale_low_fourrooms:    rs ∈ {0.01, 0.03, 0.1, 0.3, 1.0}
  reward_scale_lower_boundary:   rs ∈ {0.001, 0.003, 0.01, 0.03}
                                  (extends 1 OOM below the
                                  earlier rs=0.01 floor)

Both sweeps use identical HPs (200k steps, capacity=50k, 64×64
MLP, lr=1e-4, gamma=0.99, sync_period=100, n_seeds=30 per cell).
The rs=0.01 and rs=0.03 cells appear in both sweeps — used to
verify within-noise reproducibility before merging.

Pearl-rung-2 prediction (from the Hasselt-floor reading):
  - At very small rs (≤0.003): both arms below their gradient-
    starvation floor → mean_diff ≈ 0.
  - At rs ∈ [0.01, 0.3]: vanilla under floor, DDQN above →
    mean_diff > 0 (rescue regime).
  - At rs ≥ 1.0: both above floor → mean_diff small.

The lower-boundary sweep maps where DDQN itself fails. If
mean_diff(0.001) ≈ mean_diff(1.0) ≈ 0 but mean_diff(0.03) ≈
+0.5, that's the inverted-U: rescue is bounded both above (no
need) and below (no signal).

Usage:
    uv run python -m experiments.findings.reward_scale_lower_boundary
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl

import corroborate.rl.dqn.measurables as _m  # registers outcome_native
from corroborate.analyses.paired_g import paired_g

assert _m  # keep registration import live for type checker


REPO_ROOT = Path(__file__).resolve().parents[2]
LOW_FOURROOMS = (
    REPO_ROOT / 'experiments' / 'data' / 'reward_scale_low_fourrooms'
    / 'runs.parquet'
)
LOWER_BOUNDARY = (
    REPO_ROOT / 'experiments' / 'data' / 'reward_scale_lower_boundary'
    / 'runs.parquet'
)


def _format_row(
    rs: float,
    mean_diff: float, mean_diff_se: float,
    p_value: float, n_pairs: int,
    vanilla_native: float | None, ddqn_native: float | None,
) -> str:
    v = f'{vanilla_native:+.3f}' if vanilla_native is not None else '   .  '
    d = f'{ddqn_native:+.3f}' if ddqn_native is not None else '   .  '
    return (
        f'  rs={rs:<7}  vanilla_native={v}  ddqn_native={d}'
        f'  Δ={mean_diff:+.3f}±{mean_diff_se:.3f}'
        f'  p={p_value:.4f}  n={n_pairs}'
    )


def _arm_mean(cells: list[dict[str, object]], arm: str, rs: float) -> float:
    """Mean native-outcome for an arm at a given rs."""
    vals: list[float] = []
    for c in cells:
        if c.get('intervention_name') != arm:
            continue
        rs_v = c.get('reward_scale', 1.0)
        if not isinstance(rs_v, (int, float)) or float(rs_v) != rs:
            continue
        outcome = c.get('outcome.eval_best_burst_mean')
        if not isinstance(outcome, (int, float)):
            continue
        vals.append(float(outcome) / float(rs_v))
    return sum(vals) / len(vals) if vals else float('nan')


def main() -> None:
    if not LOW_FOURROOMS.exists() and not LOWER_BOUNDARY.exists():
        print('(skip — neither parquet present)')
        return

    frames: list[pl.DataFrame] = []
    if LOW_FOURROOMS.exists():
        frames.append(pl.read_parquet(LOW_FOURROOMS))
    if LOWER_BOUNDARY.exists():
        frames.append(pl.read_parquet(LOWER_BOUNDARY))

    # Column intersection — drop schema-divergent columns before concat.
    common_cols: set[str] = set(frames[0].columns)
    for f in frames[1:]:
        common_cols &= set(f.columns)
    df = pl.concat([f.select(sorted(common_cols)) for f in frames])

    print('# FourRooms reward-scale lower-boundary mapping')
    print('=' * 100)
    print(f'merged corpus: {df.shape[0]} cells from {len(frames)} sweeps')

    rss = sorted(set(df['reward_scale'].unique().to_list()))
    print(f'reward scales: {rss}')
    print()

    cells = list(df.iter_rows(named=True))
    for rs in rss:
        r = paired_g.fn(
            cells,
            treatment_arm='ddqn',
            baseline_arm='vanilla_dqn',
            pair_by=('seed',),
            source='outcome_native',
            env_name='FourRooms-misc',
            extra_filters={'reward_scale': rs},
        )
        v_mean = _arm_mean(cells, 'vanilla_dqn', rs)
        d_mean = _arm_mean(cells, 'ddqn', rs)
        print(_format_row(
            rs, r.mean_diff, r.mean_diff_se,
            r.mean_diff_p_value, r.n_pairs,
            v_mean if not math.isnan(v_mean) else None,
            d_mean if not math.isnan(d_mean) else None,
        ))


if __name__ == '__main__':
    main()
