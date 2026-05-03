"""CLI thin-wrapper around `corroborate.runner.run_module`.

Run any bridges-module-as-hypothesis (`experiments/findings/<X>.py`
exporting `BRIDGES`) on a data input, with the per-module cache:

    python scripts/run_hypothesis.py experiments.findings.ddqn_universe \\
        --data experiments/data/

Library code lives in `corroborate.runner`; this file is purely the
argparse + verdict-printing surface."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from corroborate.claim_bridge import BridgeEvaluation
from corroborate.runner import run_module


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog='run_hypothesis',
        description='Run a hypothesis-module on a data input, with cache.',
    )
    parser.add_argument(
        'module',
        help='dotted module path, e.g. experiments.findings.ddqn_universe',
    )
    parser.add_argument(
        '--data', type=Path, default=None,
        help='parquet file or directory of corpora to ingest',
    )
    parser.add_argument(
        '--cache-path', type=Path, default=None,
        help='explicit cache path; defaults to '
             'experiments/data/cache/<module-leaf>.parquet',
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='compute fresh, no cache read or write',
    )
    parser.add_argument(
        '--no-write-cache', action='store_true',
        help='read cache for speedup but don\'t persist updates',
    )
    parser.add_argument(
        '--rebuild', action='store_true',
        help='invalidate the per-module cache before running',
    )
    parser.add_argument(
        '--no-restore', action='store_true',
        help='don\'t restore archived corpora from cloud on miss',
    )
    args = parser.parse_args(argv)

    results = run_module(
        cast(str, args.module),
        data=cast(Path | None, args.data),
        cache_path=cast(Path | None, args.cache_path),
        use_cache=not cast(bool, args.no_cache),
        write_cache=not cast(bool, args.no_write_cache),
        rebuild=cast(bool, args.rebuild),
        restore_from_cloud=not cast(bool, args.no_restore),
    )
    _print_verdicts(results)


def _print_verdicts(results: dict[str, BridgeEvaluation]) -> None:
    counts: dict[str, int] = {}
    for name, ev in results.items():
        v = ev.verdict.value
        counts[v] = counts.get(v, 0) + 1
        bits: list[str] = []
        for ar in ev.analysis_results.values():
            bits.append(_summarize(ar))
        suffix = ' | '.join(bits) if bits else ''
        print(f'{name:60s}  {v:24s}  {suffix}')
    print()
    print('verdict counts:')
    for k in sorted(counts):
        print(f'  {k:24s}  {counts[k]}')


def _summarize(result: object) -> str:
    type_name = type(result).__name__
    parts: list[str] = [type_name]
    for attr in ('g', 'se', 'mean_diff', 'p_value', 'n_pairs'):
        v = getattr(result, attr, None)
        if isinstance(v, (int, float)):
            parts.append(f'{attr}={v:.3g}')
    return ' '.join(parts)


if __name__ == '__main__':
    main()
