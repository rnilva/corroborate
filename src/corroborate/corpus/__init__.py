"""Corpus — typed persistence + cross-arm aggregation.

The corpus is the framework's per-cell evidence store + the
typed schemas that flow through it. Five sub-modules:

- `corpus.schema` — `RunRow`, `TraceRow`, `HypothesisComparisonRow`,
  `GroupStats`, the leaf-type aliases.
- `corpus.persistence` — parquet read/write, dtype tightening,
  trace-record streaming, polars-expr trace reductions.
- `corpus.cloud` — fsspec-backed archive / restore for remote
  parquet stores; `RemoteManifest` typed accessor.
- `corpus.aggregate` — per-arm aggregation and the
  `hypothesis_comparison_from_cells` builder; `leaf_signature`
  configurational fingerprint.
- `corpus.stratum` — `StratumG[K]` typed effect-size container.

Consumers `from corroborate.corpus import X`."""
from corroborate.corpus.aggregate import (
    hypothesis_comparison_from_cells,
    leaf_signature,
)
from corroborate.corpus.cloud import (
    RemoteFile,
    RemoteManifest,
    archive,
    archived_uri,
    is_archived,
    ls,
    purge,
    restore,
)
from corroborate.corpus.persistence import (
    apply_trace_reductions,
    iter_trace_records,
    read_runrows,
    read_tracerows,
    stream_concat_parquets,
    tighten_trace_dtypes,
    write_runrows,
    write_tracerows,
)
from corroborate.corpus.schema import (
    GroupStats,
    HypothesisComparisonRow,
    MeasurementLeaf,
    RunRow,
    TraceLeaf,
    TraceRow,
)
from corroborate.corpus.stratum import (
    StratumG,
)

__all__ = [
    'GroupStats',
    'HypothesisComparisonRow',
    'MeasurementLeaf',
    'RemoteFile',
    'RemoteManifest',
    'RunRow',
    'StratumG',
    'TraceLeaf',
    'TraceRow',
    'apply_trace_reductions',
    'archive',
    'archived_uri',
    'hypothesis_comparison_from_cells',
    'is_archived',
    'iter_trace_records',
    'leaf_signature',
    'ls',
    'purge',
    'read_runrows',
    'read_tracerows',
    'restore',
    'stream_concat_parquets',
    'tighten_trace_dtypes',
    'write_runrows',
    'write_tracerows',
]
