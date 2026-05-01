"""Pearl-rung-2 mechanism test: does DDQN's per-burst attenuation
on SpaceInvaders depend on the negative-reward (-1 hit) stochasticity?

Sweep `spaceinvaders_no_hit_penalty.yaml` runs two conditions:
  default: stock SpaceInvaders rewards (-1 per hit, +1 per kill)
  clipped: reward_clip_min=0 strips the -1 (positive-only)

Hypothesis (CLAIM 8 mechanism reading): the late-burst crossover
(DDQN < vanilla after burst ≈ 6) is driven by neg-reward
stochasticity. Predict: in the clipped condition, the crossover
DISAPPEARS (DDQN's late-mean g goes from ≤ -0.3 to ~0).

Two competing hypotheses (PAPER §5):
  - Pessimism: vanilla's overestimation acts as
    optimism-under-uncertainty. Stripping -1 removes the
    differential pessimism source → crossover disappears.
  - Q-explosion: late training, Q diverges in both arms (bias↔
    burst ρ=+0.95). DDQN's noise-reduction makes mc track the
    explosion → curve drops faster. Stripping the hit doesn't
    fix divergence → crossover persists.

This script:

1. Joins runs.parquet × traces.parquet to enable per-burst
   computation (needs `mc_return` trajectory).
2. Filters cells into the two conditions:
     default = `reward_clip_min` absent or NaN
     clipped = `reward_clip_min == 0.0`
3. Runs `paired_g_per_burst` independently per condition.
4. Reports per-burst g for both conditions side by side, plus
   the pessimism-vs-Q-explosion verdict.

Usage:
    uv run python -m experiments.findings.spaceinvaders_no_hit_penalty
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from corroborate.analyses.paired_g_per_burst import (
    PerBurstResult, paired_g_per_burst,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'spaceinvaders_no_hit_penalty'
    / 'runs.parquet'
)
TRACES = (
    REPO_ROOT / 'experiments' / 'data' / 'spaceinvaders_no_hit_penalty'
    / 'traces.parquet'
)


def _has_clip(cell: dict[str, object]) -> bool:
    """True if the cell has the reward_clip wrapper applied."""
    v = cell.get('reward_clip_min')
    return isinstance(v, (int, float)) and not math.isnan(float(v)) \
        and float(v) == 0.0


def _print_strata(label: str, result: PerBurstResult) -> None:
    print(f'## {label}  (n_strata={len(result.strata)})')
    print('-' * 70)
    if not result.strata:
        print('  (no strata — likely missing arm or pair-key disjoint)')
        print()
        return
    for s in sorted(result.strata, key=lambda x: x.burst_index):
        sig = '*' if s.se > 0 and abs(s.g / s.se) > 1.96 else ' '
        print(
            f'  burst {s.burst_index:>2}  '
            f'g={s.g:+.3f}±{s.se:.3f}{sig}  '
            f'n={s.n_pairs}',
        )
    late_strata = [s for s in result.strata if s.burst_index >= 6]
    if late_strata:
        late_mean = sum(s.g for s in late_strata) / len(late_strata)
        print(f'  late-mean (burst ≥ 6) g = {late_mean:+.3f}')
    print()


def main() -> None:
    if not RUNS.exists() or not TRACES.exists():
        print(f'(skip — {RUNS.parent} parquets missing)')
        return

    runs = pl.read_parquet(
        RUNS,
        columns=[
            'id', 'intervention_name', 'env_name', 'seed',
            'reward_clip_min',
        ],
    )
    traces = pl.read_parquet(
        TRACES, columns=['id', 'mc_return'],
    )
    joined = list(
        runs.join(traces, on='id', how='inner').iter_rows(named=True),
    )

    print('# spaceinvaders_no_hit_penalty: Pearl-rung-2 mechanism test')
    print('=' * 70)
    print(f'joined rows: {len(joined)}')

    default = [c for c in joined if not _has_clip(c)]
    clipped = [c for c in joined if _has_clip(c)]
    print(f'default cells: {len(default)}, clipped cells: {len(clipped)}')
    print()

    default_r = paired_g_per_burst.fn(
        default,
        treatment_arm='ddqn', baseline_arm='vanilla_dqn',
        pair_by=('seed',), source='mc_return', reduction='mean',
        env_name='SpaceInvaders-MinAtar',
    )
    clipped_r = paired_g_per_burst.fn(
        clipped,
        treatment_arm='ddqn', baseline_arm='vanilla_dqn',
        pair_by=('seed',), source='mc_return', reduction='mean',
        env_name='SpaceInvaders-MinAtar',
    )
    _print_strata('default (with -1 hit penalty)', default_r)
    _print_strata('clipped (clip_min=0, no negative reward)', clipped_r)

    # Verdict on the mechanism hypothesis.
    def_late = [s for s in default_r.strata if s.burst_index >= 6]
    clp_late = [s for s in clipped_r.strata if s.burst_index >= 6]
    if def_late and clp_late:
        d = sum(s.g for s in def_late) / len(def_late)
        c = sum(s.g for s in clp_late) / len(clp_late)
        print(f'late-mean default = {d:+.3f}, clipped = {c:+.3f}')
        if d <= -0.3 and c > -0.1:
            print('  → MECHANISM VERDICT: HELD')
            print('     neg-reward stochasticity is the binding factor;')
            print('     stripping it removes the late-burst attenuation.')
        elif d <= -0.3 and c <= -0.3:
            print('  → MECHANISM VERDICT: REFUTED')
            print('     attenuation persists under reward clipping; the')
            print('     mechanism is NOT neg-reward stochasticity. Likely')
            print('     candidate: late-training Q-explosion (universal).')
        else:
            print('  → MECHANISM VERDICT: AMBIGUOUS')
            print('     default condition did not reproduce the')
            print('     late-burst attenuation; sweep underpowered.')


if __name__ == '__main__':
    main()
