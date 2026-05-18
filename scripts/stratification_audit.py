"""Empirical: does flat-stratification (env × HP) artificially
tighten SE vs env-only stratification?

The substrate's existing DoWhy + arm_diff_pooled bridges use
`stratify_by=('env_name', 'sync_period', 'gamma', 'total_steps',
'action_duplicate_k')` — a CARTESIAN product of env and HP axes.
Each (env, HP-config) is its own stratum. With multiple HP
configs per env, this can produce 2-3× more strata than
env-only.

The question: when HP replicates within an env are similar
(low HP-driven variance), does the flat-stratification's higher
n_strata artificially tighten the pooled-d's SE and inflate
significance?

Comparison runs the SAME stratified_arm_diff_pooled primitive on
the SAME data with two `stratify_by` choices:
  (a) flat:   ('env_name', 'sync_period', 'gamma', ...)
  (b) env-only: ('env_name',)

For one bridge (bootstrap_gap_magnitude) on the ddqn_sweeps
cache (which has the multi-config cells canonical excludes).

Run: `uv run python scripts/stratification_audit.py`."""
from __future__ import annotations

import polars as pl

from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    stratified_arm_diff_pooled,
)
from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM


CACHES = [
    ('ddqn (canonical 1M)', 'experiments/data/cache/ddqn.parquet'),
    ('ddqn_sweeps (HP-mixed)', 'experiments/data/cache/ddqn_sweeps.parquet'),
]


def main() -> None:
    for label, cache in CACHES:
        df = pl.scan_parquet(cache).collect()
        print(f'\n=== {label}: {df.height} cells × {df["env_name"].n_unique()} envs ===')
        _run_one(df)


def _run_one(df: pl.DataFrame) -> None:

    # Drop cells where the columns we'll stratify by are missing
    # (some legacy corpora don't have target_sync.tau, etc.)
    cells_iter = df.to_dicts()

    # Source measurable for comparison
    source = 'jensen_gap'

    print(f'Bridge: DDQN reduces {source} (vanilla vs full DDQN).')
    print()
    print(f'{"stratify_by":<60s} | n_strata | pooled_d | pooled_se')
    print('-' * 100)

    stratify_choices: list[tuple[str, tuple[str, ...]]] = [
        ('env-only', ('env_name',)),
        ('flat: env + sync', ('env_name', 'sync_period')),
        ('flat: env + sync + gamma', ('env_name', 'sync_period', 'gamma')),
        ('flat: env + sync + gamma + n_step', ('env_name', 'sync_period', 'gamma', 'n_step')),
        ('flat: env + sync + gamma + n_step + rs', ('env_name', 'sync_period', 'gamma', 'n_step', 'reward_scale')),
        ('flat: env + sync + gamma + n_step + rs + adk', ('env_name', 'sync_period', 'gamma', 'n_step', 'reward_scale', 'action_duplicate_k')),
    ]
    for label, stratify_by in stratify_choices:
        try:
            result = stratified_arm_diff_pooled.fn(
                cells_iter,
                source=source,
                treatment_arm=DDQN_ARM,
                baseline_arm=VANILLA_ARM,
                stratify_by=stratify_by,
                min_seeds_per_arm=2,
                min_vanilla_predictor=-1e9,
            )
            print(f'{label:<60s} | {result.n_strata:>8} | {result.pooled_d:>+8.3f} | {result.pooled_se:>9.4f}')
        except Exception as e:
            print(f'{label:<60s} | ERROR: {e}')


if __name__ == '__main__':
    main()
