"""Evaluate DDQN_UNIVERSE_BRIDGES against
`experiments/data/universal_evidence.parquet` and print every
bridge's verdict + a one-line analysis-result summary. Used to
inspect verdict distribution after framework / bridge edits."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import polars as pl  # noqa: E402

from corroborate.claim_bridge import evaluate  # noqa: E402
import corroborate.rl.dqn.measurables  # noqa: F401, E402  # populate registry
from experiments.findings.ddqn_universe import (  # noqa: E402
    DDQN_UNIVERSE_BRIDGES,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
UNIVERSAL = REPO_ROOT / 'experiments' / 'data' / 'universal_evidence.parquet'


def _summarize(result: object) -> str:
    name = type(result).__name__
    parts: list[str] = [name]
    for attr in ('g', 'se', 'mean_diff', 'p_value', 'n_pairs'):
        v = getattr(result, attr, None)
        if isinstance(v, (int, float)):
            parts.append(f'{attr}={v:.3g}')
    return ' '.join(parts)


def _to_cells(df: pl.DataFrame) -> list[Mapping[str, object]]:
    return [dict(r) for r in df.iter_rows(named=True)]


def main() -> None:
    if not UNIVERSAL.exists():
        raise SystemExit(f'no universal cache at {UNIVERSAL}')
    df = pl.read_parquet(UNIVERSAL)
    cells = _to_cells(df)
    print(f'loaded {len(cells)} cells from {UNIVERSAL.name}\n')

    counts: dict[str, int] = {}
    for bridge in DDQN_UNIVERSE_BRIDGES:
        try:
            outcome = evaluate(bridge, cells)
        except Exception as e:  # noqa: BLE001
            counts['ERROR'] = counts.get('ERROR', 0) + 1
            print(f'{bridge.name:60s}  ERROR  {e!r}')
            continue
        v = outcome.verdict
        counts[v.value] = counts.get(v.value, 0) + 1
        bits = [_summarize(ar) for ar in outcome.analysis_results.values()]
        print(f'{bridge.name:60s}  {v.value:24s}  {" | ".join(bits)}')

    print('\nverdict counts:')
    for k, c in sorted(counts.items()):
        print(f'  {k:24s}  {c}')


if __name__ == '__main__':
    main()
