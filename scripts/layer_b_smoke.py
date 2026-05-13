"""Smoke: cross_config_paired_slope on canonical for the dose-
response form of bias_correction.

Predictor: bootstrap_gap_magnitude (paired Δ)
Target:    eval_best_burst_raw_mean (paired Δ)
Configs:   env_name (one per env at canonical → n=12)
"""
from __future__ import annotations

import polars as pl

from corroborate.analyses.cross_config_paired_slope import (
    cross_config_paired_slope,
)
from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM


CACHE = 'experiments/data/cache/ddqn.parquet'


def main() -> None:
    df = pl.scan_parquet(CACHE).collect()
    cells = df.to_dicts()
    print(f'cells: {len(cells)}, envs: {df["env_name"].n_unique()}')

    result = cross_config_paired_slope.fn(
        cells,
        treatment_arm=DDQN_ARM,
        baseline_arm=VANILLA_ARM,
        predictor='bootstrap_gap_magnitude',
        target='eval_best_burst_raw_mean',
        config_keys=('env_name',),
        pair_by=('seed',),
        min_pairs_per_config=5,
        min_configs=4,
    )
    print()
    print(f'n_configs   = {result.n_configs}')
    print(f'rho         = {result.rho:+.4f}')
    print(f'p_value     = {result.p_value:.4f}')

    # Per-env labeling so we can identify saturating-outcome envs.
    env_summary = (
        df.group_by('env_name')
        .agg(
            pl.col('eval_best_burst_raw_mean').mean().alias('mean_outcome'),
            pl.col('eval_best_burst_raw_mean').min().alias('min_outcome'),
            pl.col('eval_best_burst_raw_mean').max().alias('max_outcome'),
            pl.col('eval_best_burst_raw_mean').std().alias('std_outcome'),
        )
        .sort('env_name')
    )
    print()
    print('per-env outcome distribution (identify saturating envs):')
    print(env_summary)


if __name__ == '__main__':
    main()
