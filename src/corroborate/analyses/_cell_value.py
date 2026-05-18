"""Shared cell-value helpers — resolve a scalar / per-burst
array from a corpus row, and build a hashable key for paired /
panel grouping.

These are infrastructure for analyses that consume `Mapping[str,
object]` rows. Extracted from `paired/paired_g.py` and
`paired/paired_g_per_burst.py` so subpackages outside `paired/`
(`panel/`, `spearman/`, `link/`, `dowhy/`) don't have to import
through `paired/` for these reads — keeps the subpackage
dependency graph cleaner.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from corroborate.measurables.measurable import Measurable


def resolve_value(record: Mapping[str, object], source: str) -> float:
    """Resolve `source` from a cell record. Tries the persisted
    field-path read first; falls back to the measurable registry.

    Record-first matches the "persisted columns are authoritative"
    discipline: when cell_runner has already persisted a scalar at
    `source`, downstream analyses use that value rather than
    recomputing via the registered measurable (which might fail
    on a corpus row that doesn't carry the source-side raw arrays).
    The measurable fallback covers analyses that request a derived
    quantity by name on a record where only the raw inputs were
    persisted (e.g. ad-hoc reductions over the trace store).

    A *present* key with a None / NaN / non-numeric value is
    treated as a cached miss (returns NaN) — DO NOT fall through
    to the measurable. The bridge cache writes None for cells
    where the measurable couldn't resolve at build time (e.g.
    corpus without traces); recomputing here would re-trigger the
    same failure with no new information AND mask the universal-
    merge schema heterogeneity. Only an *absent* key falls through
    to the registry.

    Raises `KeyError` if `source` isn't in the record AND no
    measurable is registered under that name."""
    if source in record:
        v = record[source]
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)):
            return float(v)
        return float('nan')
    from corroborate.measurables import get_registered as _get_m
    m = _get_m(source)
    if m is not None:
        computed: object = m(record)
        if isinstance(computed, (int, float)):
            return float(computed)
        raise TypeError(
            f'measurable {source!r} returned non-scalar '
            f'{type(computed).__name__}; paired-g source must be scalar',
        )
    raise KeyError(
        f'no scalar at path {source!r} in record and no measurable '
        f'named {source!r}',
    )


def key_tuple(
    record: Mapping[str, object], pair_by: tuple[str, ...],
) -> tuple[object, ...]:
    """Build a hashable key from `record` for the columns named
    in `pair_by`. Used by paired-analyses to group cells into
    matched (treatment, baseline) sets and by stratum analyses
    to group cells into per-stratum buckets."""
    return tuple(record[k] for k in pair_by)


def evaluate_per_burst_source(
    source: Measurable[Mapping[str, object], npt.NDArray[np.floating]],
    cell: Mapping[str, object],
) -> np.ndarray:
    """Per-burst value extraction with cache-first dispatch.

    1. **Cache hit**: if `source.name` is a column on `cell` (the
       per-module cache materialised the composed Measurable as a
       list column at build time), read the pre-computed array.
       Cells flagged None for that column (universal-merge corpora
       lacking traces for some arms) are skipped to an empty array.
    2. **Fallback**: evaluate `source(cell)` from the raw record.
       Used when no cache (synthetic test cells, ad-hoc analyses,
       on-the-fly bridge invocation against runs.parquet).

    The cache-first path is what lets the runner DROP the raw
    trace columns after `_compute_measurables` — once the
    per-burst array is materialised under `source.name`, no
    consumer needs the 2D `(n_bursts, n_episodes)` source again.
    Returns shape `(n_bursts,)` on success or `()` on missing /
    malformed input so downstream filtering naturally excludes
    the cell."""
    cached = cell.get(source.name)
    if cached is not None:
        try:
            arr = np.asarray(cached, dtype=np.float64)
        except (TypeError, ValueError):
            arr = None
        else:
            if arr.ndim == 1:
                return arr
    try:
        arr = np.asarray(source(cell), dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return np.array([], dtype=np.float64)
    if arr.ndim != 1:
        return np.array([], dtype=np.float64)
    return arr
