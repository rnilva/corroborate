"""Importable substrate-shaped fixture for the parallel-ingest
fork-safety tests (`test_runner_ingest_fork_safety.py`).

This module MUST be importable by dotted name from a fresh
interpreter: the `forkserver` / `spawn` workers re-import it via
`runner._reestablish_registry` to re-establish the measurable
registry (a forkserver worker starts with an EMPTY registry — no
copy-on-write inheritance). Registering the measurable inline in
a test function wouldn't be re-importable, so it lives here at
module scope where importing the module re-runs the `@measurable`
decorator.

The measurable `trace_doubled_sum` reads a per-cell trace column
(`signal`) and reduces it — exactly the shape that silently
null-pads when the worker's registry is missing the substrate
(the bug the equivalence test guards against)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from corroborate.measurables import measurable

#: Trace column the synthetic corpora carry per cell. The
#: measurable reduces it; if the worker registry lacks the
#: measurable, this column is joined-then-dropped and the
#: measurable's output column never appears.
TRACE_COLUMN = 'signal'

#: Name of the measurable's output column the equivalence test
#: asserts on. Bare name (registered-measurable namespace), per
#: the persistence-shape path-keyed convention.
MEASURABLE_NAME = 'trace_doubled_sum'


@measurable(name=MEASURABLE_NAME, reads=(TRACE_COLUMN,))
def trace_doubled_sum(record: Mapping[str, object]) -> float:
    """Closed-form reduction of the per-cell `signal` trace:
    `2 * sum(signal)`. Deliberately trivial — the test asserts
    the parallel and sequential ingest paths agree on this value,
    so the reduction only needs to be deterministic and
    trace-dependent (forcing the trace join + registry lookup
    the bug short-circuits)."""
    raw = record[TRACE_COLUMN]
    if not isinstance(raw, Sequence):
        raise TypeError(f'{TRACE_COLUMN} not a sequence: {type(raw)!r}')
    total = 0.0
    for v in raw:
        if not isinstance(v, (int, float)):
            raise TypeError(f'{TRACE_COLUMN} element not numeric: {v!r}')
        total += float(v)
    return 2.0 * total


def expected_value(signal: Sequence[float]) -> float:
    """Closed form the test compares the framework's output
    against (no reimplementation of framework logic — this is the
    analytical answer `2 * sum(signal)`)."""
    return 2.0 * sum(signal)


def write_corpus(
    corpus_dir: Path,
    *,
    env_name: str,
    cell_signals: Mapping[str, Sequence[float]],
) -> None:
    """Materialise a minimal two-store corpus under `corpus_dir`:

    - `runs.parquet` — `id` + provenance columns the ingest path
      stamps / dedups on (`env_name`, `arm_key`, `seed`).
    - `traces.parquet` — `id` + the `signal` per-cell trace column
      the measurable reads.

    `cell_signals` maps each cell `id` to its per-cell signal
    array. Distinct `env_name` per corpus keeps CI4 content-dedup
    from collapsing cells across corpora."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    ids = list(cell_signals)
    runs = pl.DataFrame({
        'id': ids,
        'env_name': [env_name] * len(ids),
        'arm_key': ['baseline'] * len(ids),
        'seed': list(range(len(ids))),
    })
    runs.write_parquet(corpus_dir / 'runs.parquet')
    traces = pl.DataFrame({
        'id': ids,
        TRACE_COLUMN: [list(cell_signals[i]) for i in ids],
    })
    traces.write_parquet(corpus_dir / 'traces.parquet')
