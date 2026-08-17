"""Exploration data surface.

`Panel` is the substrate-author's pre-bridge-authoring entry
point: load cells from a corpus / cache directly (no ingest
dance), probe per-stratum diagnostics, narrow scope, derive
per-stratum aggregates, then call any registered @analysis on
`panel.cells` (a polars DataFrame — the canonical analysis
input).

Bridges themselves consume @analysis results by typed parameter
name (pytest-fixture style); Panel doesn't enter the bridge-
resolution path. The bridge author's exploratory probe and
their production bridge converge through the SAME @analysis
primitives, not through a shared Panel type.

Externally-produced runs enter through `load_runs`
(`corroborate.data.loader`): a directory of plain producer files
becomes a DataFrame of one row per run, ready for polars
exploration, `Panel.from_dataframe`, or direct bridge
evaluation. The loader is a reader, not a gatekeeper — study-
design checks live on the claim being evaluated, as admission
gates.

See `corroborate/data/panel.py` for the load-bearing types."""
from __future__ import annotations

from corroborate.data.loader import config_columns, load_runs
from corroborate.data.panel import (
    CorpusSource,
    DerivedSpec,
    MeasurableAvailability,
    Panel,
    PanelDiagnostics,
)

__all__ = [
    'CorpusSource',
    'DerivedSpec',
    'MeasurableAvailability',
    'Panel',
    'PanelDiagnostics',
    'config_columns',
    'load_runs',
]
