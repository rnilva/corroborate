"""Corpus — typed persistence + cross-arm aggregation.

Five sub-modules:

- `corpus.schema` — `RunRow`, `TraceRow`,
  `HypothesisComparisonRow`, `GroupStats`, leaf-type aliases.
- `corpus.persistence` — parquet read/write, dtype tightening,
  trace-record streaming, polars-expr trace reductions.
- `corpus.cloud` — fsspec-backed archive / restore for remote
  parquet stores; `RemoteManifest` typed accessor.
- `corpus.aggregate` — per-arm aggregation +
  `hypothesis_comparison_from_cells` builder; `leaf_signature`
  configurational fingerprint.
- `corpus.stratum` — `StratumG[K]` typed effect-size container.

Public surface re-exported here: typed schemas (`RunRow`,
`TraceRow`, `HypothesisComparisonRow`), the corpus builder, the
fingerprint, and the leaf-type aliases. Parquet I/O helpers,
cloud archive primitives, and `RemoteManifest` accessors live
on the submodule path (`corroborate.corpus.persistence.X`,
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
    'RunRow',
    'StratumG',
    'TraceLeaf',
    'TraceRow',
    'hypothesis_comparison_from_cells',
    'leaf_signature',
]
