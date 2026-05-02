"""CLI for `corroborate.evidence_cache`.

Two modes:

  Per-corpus:
    uv run python -m experiments.build_bridge_cache \\
        --module experiments.findings.ddqn_universe \\
        --corpus experiments/data/ddqn

  Universal (auto-discover every corpus under `--data-root`):
    uv run python -m experiments.build_bridge_cache \\
        --module experiments.findings.ddqn_universe \\
        --universal \\
        --out experiments/data/universal_evidence.parquet

`--module` must export a `Sequence[Bridge]` named via
`--bridges-attr` (default: `DDQN_UNIVERSE_BRIDGES`).

The discovered measurable set is the union over `bridge.source`,
`bridge.target`, and any `bridge.params[*]` string values that
resolve in the @measurable registry, transitively closed via the
@measurable graph.
"""
from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import TypeIs, cast

from corroborate._argparse_boundary import to_mapping
from corroborate._narrow import optional_str, require_str
from corroborate.claim_bridge import Bridge
from corroborate.evidence_cache import build_cache, build_universal_cache


def _is_bridge_sequence(obj: object) -> TypeIs[Sequence[Bridge]]:
    """Narrow `getattr(module, attr)`'s `object` value to
    `Sequence[Bridge]` — runtime invariant the framework can
    express where typeshed's `getattr` returns `Any`."""
    if not isinstance(obj, Sequence):
        return False
    # `isinstance(_, Sequence)` narrows to `Sequence[Unknown]`;
    # re-bind through `Sequence[object]` so the loop var is typed.
    seq: Sequence[object] = cast(Sequence[object], obj)
    for b in seq:
        if not isinstance(b, Bridge):
            return False
    return True


def _bridges_from(module_name: str, attr: str) -> Sequence[Bridge]:
    mod = importlib.import_module(module_name)
    obj: object = cast(object, getattr(mod, attr))
    if not _is_bridge_sequence(obj):
        raise TypeError(
            f'{module_name}.{attr} is not Sequence[Bridge]; '
            f'got {type(obj).__name__}',
        )
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True)
    parser.add_argument('--bridges-attr', default='DDQN_UNIVERSE_BRIDGES')
    parser.add_argument('--corpus', default=None)
    parser.add_argument('--universal', action='store_true')
    parser.add_argument('--data-root', default='experiments/data')
    parser.add_argument('--out', default=None)
    parser.add_argument('--out-name', default='runs_with_bridge_cache.parquet')
    parser.add_argument('--force', action='store_true')

    raw = to_mapping(parser.parse_args())
    module_name = require_str(raw, 'module')
    bridges_attr = require_str(raw, 'bridges_attr')
    out_name = require_str(raw, 'out_name')
    data_root_s = require_str(raw, 'data_root')
    corpus_s = optional_str(raw, 'corpus')
    out_s = optional_str(raw, 'out')
    universal = bool(raw.get('universal'))
    force = bool(raw.get('force'))

    bridges = _bridges_from(module_name, bridges_attr)

    if universal:
        out_path = Path(out_s) if out_s is not None else (
            Path(data_root_s) / 'universal_evidence.parquet'
        )
        build_universal_cache(
            bridges,
            data_root=Path(data_root_s),
            out_path=out_path,
            out_name=out_name,
            skip_up_to_date=not force,
        )
        return

    if corpus_s is None:
        raise SystemExit(
            'must pass either --corpus <dir> or --universal',
        )
    corpus = Path(corpus_s)
    runs_path = corpus / 'runs.parquet'
    traces_path = corpus / 'traces.parquet'
    out_path = Path(out_s) if out_s is not None else (corpus / out_name)
    if not runs_path.exists():
        raise SystemExit(f'no runs.parquet at {runs_path}')
    build_cache(bridges, runs_path, traces_path, out_path)


if __name__ == '__main__':
    main()
