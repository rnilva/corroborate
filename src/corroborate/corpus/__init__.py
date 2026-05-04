"""Corpus — typed persistence + cross-arm aggregation.

Four sub-modules:

- `corpus.schema` — typed records (`RunRow`, `TraceRow`,
  `HypothesisComparisonRow`, `GroupStats`, `StratumG[K]`,
  leaf-type aliases).
- `corpus.persistence` — parquet read/write, dtype tightening,
  trace-record streaming, polars-expr trace reductions.
- `corpus.cloud` — fsspec-backed archive / restore for remote
  parquet stores; `RemoteManifest` typed accessor.
- `corpus.aggregate` — per-arm aggregation +
  `hypothesis_comparison_from_cells` builder; `leaf_signature`
  configurational fingerprint.

Public surface re-exported here: typed schemas, the corpus
builder, the fingerprint, leaf-type aliases. Parquet I/O
helpers, cloud archive primitives, and `RemoteManifest`
accessors live on the submodule path
(`corroborate.corpus.persistence.X`,
`corroborate.corpus.cloud.X`)."""
from corroborate.corpus.aggregate import (
    hypothesis_comparison_from_cells,
    leaf_signature,
)
from corroborate.corpus.schema import (
    GroupStats,
    HypothesisComparisonRow,
    MeasurementLeaf,
    RunRow,
    StratumG,
    TraceLeaf,
    TraceRow,
)

__all__ = [
    'GroupStats',
    'HypothesisComparisonRow',
    'MeasurementLeaf',
    'RunRow',
    'StratumG',
    'TraceLeaf',
    'TraceRow',
    'hypothesis_comparison_from_cells',
    'leaf_signature',
]
