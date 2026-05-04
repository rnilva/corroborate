"""Helper for per-burst analyses' duplicate-key error reporting.

When two cells in the same `(env, arm, pair_by)` bucket land in
the analysis, they're either:
- replicates (differ only on provenance tags) — author can opt
  into `dedupe_strategy='mean'`,
- regime-distinct (differ on substrate-level columns like
  `total_steps`, `eval_every`, sweep-specific HPs) — author needs
  to add those columns to `pair_by` so each regime is its own
  stratum, OR scope-filter to one regime.

`_distinguishing_columns` mechanically computes which path applies:
the columns whose values differ across the duplicate cells. The
analyses use it to build error messages that point the author at
the offending columns rather than just saying "duplicate".

Domain-agnostic: the helper doesn't know what RL or any substrate
considers a 'regime'. It just lists columns. The substrate's own
notion of regime emerges from the report — e.g. RL sweeps will
typically surface `total_steps` and `eval_every`."""
from __future__ import annotations

from collections.abc import Iterable, Mapping


# Cell-id columns whose values are expected to differ even between
# byte-equivalent replicates. These are filtered out of the diff
# report so authors aren't told "the timestamps differ".
_PROVENANCE_TAGS: frozenset[str] = frozenset({
    'id', 'parent_id', 'cycle_id', 'timestamp', 'corpus',
})


def _distinguishing_columns(
    cells: Iterable[Mapping[str, object]],
    *,
    skip: frozenset[str] = frozenset(),
) -> dict[str, list[object]]:
    """Return columns whose values differ across `cells`, excluding
    framework-provenance tags and any names in `skip`. Values that
    can't be hashed (lists / arrays / dicts) are compared by their
    `str` form so heterogeneous-shape cells still surface a diff.

    Returned dict: column name → sorted-by-string list of distinct
    values observed (deduped). Empty dict means cells differ only
    on provenance — the author wanted dedupe_strategy='mean'."""
    cell_list = list(cells)
    if len(cell_list) < 2:
        return {}
    all_columns: set[str] = set()
    for cell in cell_list:
        all_columns |= set(cell.keys())
    candidate = all_columns - _PROVENANCE_TAGS - skip

    out: dict[str, list[object]] = {}
    for col in candidate:
        seen: dict[str, object] = {}
        for cell in cell_list:
            v = cell.get(col)
            key = repr(v)
            if key not in seen:
                seen[key] = v
        if len(seen) >= 2:
            out[col] = sorted(seen.values(), key=repr)
    return out


def format_diff(diff: Mapping[str, list[object]]) -> str:
    """Pretty-print a `_distinguishing_columns` result as a single
    line for inclusion in error messages. Truncates each column's
    value list to 5 entries."""
    if not diff:
        return '<replicates — only provenance tags differ>'
    parts: list[str] = []
    for col in sorted(diff):
        vals = diff[col]
        head = vals[:5]
        suffix = '' if len(vals) <= 5 else f', …({len(vals)} total)'
        parts.append(
            f"{col}={[repr(v) for v in head]}{suffix}",
        )
    return '; '.join(parts)


__all__ = ['_distinguishing_columns', 'format_diff']
