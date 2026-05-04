"""Per-cell solve-status classification — substrate utility for
post-sweep stratified analysis.

Lifted from the deprecated `experiments/analyze_link_by_solved.py`
(deleted in Phase 6). The 4-class taxonomy:

- `saturated` — `best_burst` within `ceiling_eps` of the
  per-env corpus-max. Both arms reach the same outcome; paired
  Hedges' g is structurally null regardless of policy quality.
  Excluding saturated cells before link analysis prevents this
  measurement degeneracy from washing out signal.
- `solved` — `is_solved(env, best) is True` AND not saturated.
  Cell crossed the env's canonical threshold (per
  `env_solve_thresholds.SOLVE_THRESHOLDS`) with headroom.
- `unsolved` — `is_solved(env, best) is False`.
- `no_threshold` — env has no canonical solve threshold.

The "saturated" detection is the load-bearing contribution: it
catches cells where the outcome metric has been clipped by an
env-side ceiling (e.g. discounted-return saturation on
short-horizon envs), distinct from genuine policy convergence.

Substrate-coupled to:
- `eval_best_burst_mean` outcome key (the substrate's canonical
  best-window-mean reduction).
- `SOLVE_THRESHOLDS` per-env threshold table.
- `is_solved` predicate."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import polars as pl

from corroborate_rl.env_solve_thresholds import SOLVE_THRESHOLDS, is_solved


type CellClass = Literal[
    'saturated', 'solved', 'unsolved', 'no_threshold', 'missing',
]


def classify_cell(
    env_name: str,
    best_burst: float,
    corpus_max_per_env: Mapping[str, float],
    *,
    ceiling_eps: float = 1e-3,
) -> CellClass:
    """Classify one cell by solve-status. `best_burst` is the
    cell's best-window-mean outcome (typically
    `eval_best_burst_mean`); `corpus_max_per_env` is the per-env
    maximum across the WHOLE corpus (so saturation is detected
    against the achievable outcome ceiling, not a per-cell value).

    Saturation check fires first: if `best_burst` is within
    `ceiling_eps` of the corpus-max, the cell is `'saturated'`
    regardless of any threshold. This catches the case where
    treatment and baseline both hit the same hard outcome ceiling
    — paired g would be structurally null, masking any policy
    quality difference.

    `'solved'` requires both (a) `is_solved(env, best) is True`
    and (b) not saturated — otherwise `is_solved` plus saturation
    would double-count corpus-max cells as both solved AND
    saturated."""
    corpus_max = corpus_max_per_env.get(env_name, float('-inf'))
    if abs(best_burst - corpus_max) <= ceiling_eps:
        return 'saturated'
    spec = SOLVE_THRESHOLDS.get(env_name)
    if spec is None or spec.threshold is None:
        return 'no_threshold'
    is_s = is_solved(env_name, best_burst)
    if is_s is True:
        return 'solved'
    if is_s is False:
        return 'unsolved'
    return 'no_threshold'


def with_cell_class(
    df: pl.DataFrame,
    *,
    outcome_path: str = 'eval_best_burst_mean',
    cell_class_column: str = '_cell_class',
    ceiling_eps: float = 1e-3,
) -> pl.DataFrame:
    """Augment a corpus DataFrame with a `cell_class_column`
    carrying each cell's `CellClass` label.

    The corpus-max is computed per-env across all rows (both arms
    contribute) so saturation reflects the corpus's achievable
    ceiling — matching the analysis-time semantics where you ask
    "is this cell at the ceiling that ANYONE in this corpus
    reached on this env?".

    Cells with non-numeric or NaN outcome are labelled
    `'missing'` rather than failing — analysis layers can filter
    them out."""
    corpus_max: dict[str, float] = {
        env: float(
            df.filter(pl.col('env_name') == env)[outcome_path].max()
            or float('-inf')
        )
        for env in df['env_name'].unique()
    }
    classes: list[CellClass] = []
    for row in df.iter_rows(named=True):
        env = row['env_name']
        best = row.get(outcome_path)
        if not isinstance(best, (int, float)) or math.isnan(float(best)):
            classes.append('missing')
            continue
        classes.append(
            classify_cell(
                str(env), float(best), corpus_max,
                ceiling_eps=ceiling_eps,
            ),
        )
    return df.with_columns(pl.Series(cell_class_column, classes))


__all__ = ['CellClass', 'classify_cell', 'with_cell_class']
