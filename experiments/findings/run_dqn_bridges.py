"""Run the authored DDQN bridges against the action_dim_sweep
corpus and print every verdict.

This is the file-protocol artifact in action: the bridges in
`dqn_bridges.py` are authored once; this script loads a corpus,
evaluates every bridge against it, and prints the typed
verdicts. Falsification = run against a different corpus, see
which verdicts change.

Usage:
  uv run python -m experiments.findings.run_dqn_bridges
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from corroborate.claim_bridge import evaluate
from experiments.findings.dqn_bridges import ACTION_DIM_BRIDGES


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_DIM_PARQUET = (
    REPO_ROOT / 'experiments' / 'data' / 'action_dim_sweep'
    / 'runs.parquet'
)


def _format_paired_g(result: object) -> str:
    """Compact one-liner for a PairedGResult."""
    from corroborate.analyses.paired_g import PairedGResult
    if not isinstance(result, PairedGResult):
        return str(result)
    return (
        f'g={result.g:+.3f}, p={result.p_value:.4f}, '
        f'n={result.n_pairs}'
    )


def _format_meta_regression(result: object) -> str:
    from corroborate.meta_regression import MetaRegressionResult
    if not isinstance(result, MetaRegressionResult):
        return str(result)
    return (
        f'n_strata={result.n_strata}, '
        f'coefs={[(c.name, round(c.coefficient, 3)) for c in result.coefficients]}'
    )


def _format_analysis_result(result: object) -> str:
    """Dispatch by registered analysis return type."""
    from corroborate.analyses.paired_g import PairedGResult
    from corroborate.meta_regression import MetaRegressionResult
    if isinstance(result, PairedGResult):
        return _format_paired_g(result)
    if isinstance(result, MetaRegressionResult):
        return _format_meta_regression(result)
    return str(result)


def main() -> None:
    if not ACTION_DIM_PARQUET.exists():
        raise FileNotFoundError(
            f'corpus missing at {ACTION_DIM_PARQUET}',
        )

    print(f'corpus: {ACTION_DIM_PARQUET}')
    df = pl.read_parquet(ACTION_DIM_PARQUET)
    cells = list(df.iter_rows(named=True))
    print(f'  {len(cells)} cells, '
          f'{df["env_name"].n_unique()} envs, '
          f'{df["intervention_name"].n_unique()} arms')
    print()
    print(f'{"BRIDGE":<55} {"VERDICT":<22} STATS')
    print('-' * 110)
    for bridge in ACTION_DIM_BRIDGES:
        out = evaluate(bridge, cells)
        for analysis_name, result in out.analysis_results.items():
            stats = _format_analysis_result(result)
            print(
                f'{bridge.name:<55} {out.verdict.value:<22} '
                f'[{analysis_name}] {stats}',
            )


if __name__ == '__main__':
    main()
