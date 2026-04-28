"""Persistence — parquet round-trip for schema rows.

Every row type writes as a flat columnar parquet via `pl.DataFrame
([r.as_dict() for r in rows]).write_parquet(path)`. No JSON
wrapping, no struct columns: each typed-provenance field becomes
its own typed column, and each `measurements` entry becomes its
own typed column at top level. Querying HPs at the dataframe
level just works (`df.filter(pl.col('optimizer.inner.lr') < 1e-3)`).

Heterogeneous measurement keys across rows: parquet requires every
column to have one type per file, but different cells/arms can
carry different paths. Polars handles missing columns by null-
padding when constructing the DataFrame from a list-of-dicts —
rows that don't carry a path get null in that column, which
`from_row_dict` skips on read.

All ten functions (5 row types × {write, read}) are top-level to
keep `schema.py` polars-free."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from corroborate._polars_boundary import to_dicts as _to_dicts
from corroborate.schema import (
    ArmRow,
    ComparisonRow,
    CorpusRow,
    RunRow,
    TraceRow,
)


# ============ RunRow ============

def write_runrows(rows: Iterable[RunRow], path: Path) -> None:
    """Write RunRows to a flat columnar parquet. Round-trip pair:
    `read_runrows`."""
    records = [r.as_dict() for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_runrows(path: Path) -> list[RunRow]:
    df = pl.read_parquet(path)
    return [RunRow.from_row_dict(d) for d in _to_dicts(df)]


# ============ ArmRow ============

def write_armrows(rows: Iterable[ArmRow], path: Path) -> None:
    records = [r.as_dict() for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_armrows(path: Path) -> list[ArmRow]:
    df = pl.read_parquet(path)
    return [ArmRow.from_row_dict(d) for d in _to_dicts(df)]


# ============ ComparisonRow ============

def write_comparisonrows(rows: Iterable[ComparisonRow], path: Path) -> None:
    records = [r.as_dict() for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_comparisonrows(path: Path) -> list[ComparisonRow]:
    df = pl.read_parquet(path)
    return [ComparisonRow.from_row_dict(d) for d in _to_dicts(df)]


# ============ CorpusRow ============

def write_corpusrows(rows: Iterable[CorpusRow], path: Path) -> None:
    records = [r.as_dict() for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_corpusrows(path: Path) -> list[CorpusRow]:
    df = pl.read_parquet(path)
    return [CorpusRow.from_row_dict(d) for d in _to_dicts(df)]


# ============ TraceRow ============

# Trace persistence is the v9-`traces.parquet` analog: raw per-
# cell observations as a flat columnar parquet. Same shape as
# the four row stores above; the distinguishing feature is that
# trace leaves can be 1-D lists (per-step trajectories) in
# addition to scalars.

def write_tracerows(rows: Iterable[TraceRow], path: Path) -> None:
    """Write TraceRows to a flat columnar parquet. Round-trip
    pair: `read_tracerows`."""
    records = [r.as_dict() for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_tracerows(path: Path) -> list[TraceRow]:
    df = pl.read_parquet(path)
    return [TraceRow.from_row_dict(d) for d in _to_dicts(df)]
