"""Cross-env DDQN-2010 Panel: mechanism (jensen_gap→0) is scope-
invariant, OUTCOME is scope-dependent (free lunch ↔ harm).

Builds one Panel over 4 MinAtar envs × {vanilla, ddqn2016, paired} and
carries the framework's typed columns (`id`, `program`, `arm_key`) plus
the bridges' measurables:
  jensen_gap                          (mechanism: overestimation bias)
  mean_per_state_cumulative_bias_late (unclipped bias)
  eval_late_burst_mean                (GREEDY late-eval return — the
      registered measurable: mean of mc_return over the last 30% of
      bursts; n_episodes-unbiased. NOT the training-return
      `late_window_mean`, which reads ep_return.)

Canonical corpora pre-date the typed `RunRow.program` column, so their
cells are stamped `program='dqn'` (they ARE single-net DQN). The
treatment cells carry `program='paired_dqn'` natively. `eval_late_burst_
mean` is derived from the per-burst means `mc_return__mean_axis_-1`
(== the registered measurable: mean over last-30% bursts); canonical
carries that in measurements.parquet, paired computes it from its local
traces — so no canonical-trace download is needed.

Run / promote:
    JAX_PLATFORMS=cpu uv run --package corroborate_rl \\
        python3 -m experiments.findings.ddqn2010_xenv._exploration
    PROMOTE=1 … (writes ddqn2010_xenv.parquet + sidecars)
"""
from __future__ import annotations

import os

import polars as pl

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate registry

from corroborate.data import Panel
from experiments.findings.ddqn2010_xenv import _scope as S

# All study corpora run total_steps/eval_every = 1e6/2e4 = 50 bursts;
# eval_late_burst_mean is the mean over the last 30% → last 15 bursts.
_N_BURSTS = 50
_LATE_TAIL = (_N_BURSTS + 2) // 3  # == registered eval_late_burst_mean: ceil(n/3)


def build_cells() -> pl.DataFrame:
    """Pure-polars (Any-free). Preserves `id`/`program`/`arm_key`;
    stamps `program='dqn'` on canonical cells (pre-program-column)."""
    frames: list[pl.DataFrame] = []
    for env in S.ENVS:
        canon = Panel.from_corpus(S.CANONICAL[env]).cells
        paired = Panel.from_corpus(
            S.PAIRED[env], join_traces=True,
        ).with_measurables([
            'mc_return__mean_axis_-1', 'jensen_gap',
            'mean_per_state_cumulative_bias_late',
        ]).cells
        for df in (canon, paired):
            if df.height == 0 or 'mc_return__mean_axis_-1' not in df.columns:
                continue
            program = (
                pl.col('program') if 'program' in df.columns
                else pl.lit(S.BASELINE_PROGRAM)
            ).fill_null(S.BASELINE_PROGRAM)
            keep = df.select([
                'id',  # RunRow UUID — extent_hash = hash of admitted ids.
                'env_name', 'seed', 'gamma', 'corpus', 'arm_key',
                'jensen_gap', 'mean_per_state_cumulative_bias_late',
                program.alias('program'),
                # == registered `eval_late_burst_mean` (mean over last-30%
                # bursts of per-burst mean mc_return).
                pl.col('mc_return__mean_axis_-1')
                .list.tail(_LATE_TAIL).list.mean().alias('eval_late_burst_mean'),
            ])
            frames.append(keep)
    return pl.concat(frames, how='diagonal_relaxed')


_CKPT = '/tmp/_ddqn2010_xenv_cells.parquet'

if __name__ == '__main__':
    if os.environ.get('CKPT') != 'fresh' and os.path.exists(_CKPT):
        cells = pl.read_parquet(_CKPT)
        print(f'(loaded cached cells from {_CKPT})')
    else:
        cells = build_cells()
        cells.write_parquet(_CKPT)
    print('panel:', cells.height, 'cells')
    # Per-(env, arm) jensen_gap + greedy late-eval. paired jens → 0 at
    # every env (scope-invariant mechanism); the per-env paired-vs-vanilla
    # eval gap is the scope-dependent outcome (authoritative Cohen's d +
    # verdict from `corroborate hypothesis experiments.findings.ddqn2010_xenv`).
    print(
        cells.with_columns(S.display_arm().alias('arm'))
        .group_by(['env_name', 'arm']).agg(
            pl.len().alias('n'),
            pl.col('jensen_gap').mean().round(2).alias('jens'),
            pl.col('eval_late_burst_mean').mean().round(2).alias('eval_greedy'),
        ).sort(['env_name', 'arm'])
    )

    if os.environ.get('PROMOTE'):
        panel = Panel.from_dataframe(cells, stratify_by=('env_name', 'program'))
        path = panel.to_cache('experiments.findings.ddqn2010_xenv')
        print('\nPROMOTED → cache:', path)
