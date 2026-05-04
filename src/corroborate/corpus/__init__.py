"""Corpus — typed persistence + leaf-signature fingerprint.

Three sub-modules:

- `corpus.schema` — typed records (`RunRow`, `TraceRow`,
  `StratumG[K]`, leaf-type aliases).
- `corpus.persistence` — parquet read/write, dtype tightening,
  trace-record streaming, polars-expr trace reductions.
- `corpus.cloud` — fsspec-backed archive / restore for remote
  parquet stores; `RemoteManifest` typed accessor.
- `corpus.leaf_signature` — configurational fingerprint helper.

The cross-arm paired-comparison surface lives in
`corroborate.analyses.paired_comparison` — it's an analysis
result, not a persisted row.

Public surface re-exported here: typed schemas, the leaf-signature
fingerprint, leaf-type aliases. Parquet I/O helpers, cloud
archive primitives, and `RemoteManifest` accessors live on the
submodule path (`corroborate.corpus.persistence.X`,
`corroborate.corpus.cloud.X`)."""
from corroborate.corpus.leaf_signature import leaf_signature
from corroborate.corpus.schema import (
    MeasurementLeaf,
    RunRow,
    StratumG,
    TraceLeaf,
    TraceRow,
)

__all__ = [
    'MeasurementLeaf',
    'RunRow',
    'StratumG',
    'TraceLeaf',
    'TraceRow',
    'leaf_signature',
]
