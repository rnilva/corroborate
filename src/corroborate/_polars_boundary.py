"""Polars boundary — typed wrapper around `DataFrame.to_dicts()`.

`polars.DataFrame.to_dicts()` is typed as `list[dict[str, Any]]`
in polars stubs (polars uses `Any` for heterogeneous-value cells,
which the framework's `Mapping[str, object]` discipline forbids
casual use of). This module narrows the result to
`list[Mapping[str, object]]` at the boundary; downstream
isinstance/TypeIs predicates do per-field narrowing.

Same rationale as `_json_boundary`. Two scoped
`pyright: ignore[reportAny]` comments — the polars equivalent of
the json laundering. Without this boundary, every `read_*rows`
in `persistence.py` would silently convert `Any` to `object`
through implicit parameter-type compatibility, and basedpyright
would never catch a stat field that genuinely needs narrowing.

Module name is underscore-prefixed to signal **internal use
only**. External users should import polars directly."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl


def to_dicts(df: pl.DataFrame) -> Sequence[Mapping[str, object]]:
    """Decode a polars DataFrame to a `Sequence[Mapping[str,
    object]]`. `Sequence` (covariant) and `Mapping` (covariant in
    value) widen polars' `list[dict[str, Any]]` cleanly without
    requiring `pyright: ignore` — the framework's value-side
    narrowing happens via TypeIs predicates downstream."""
    return df.to_dicts()
