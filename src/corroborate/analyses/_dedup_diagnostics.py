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

**None-as-wildcard convention.** When a cell column has `None`
alongside an explicit value (e.g. `[None, 1.0]`), the None is
treated as "field not set, default applied" — the column does
NOT distinguish the cells. This handles the cross-sweep cache
shape where some sub-sweeps wrote a column explicitly while
others left it absent (polars null-pads on read). Pairs with the
canonical_str default-elision (`_internals/canonical.py`): both
layers treat default-equal-or-implicit as equivalent, so
regime-mismatch reports only fire on substantive HP differences.

Domain-agnostic: the helper doesn't know what RL or any substrate
considers a 'regime'. It just lists columns. The substrate's own
notion of regime emerges from the report — e.g. RL sweeps will
typically surface `total_steps` and `eval_every`."""
from __future__ import annotations

from collections.abc import Iterable, Mapping


# Cell-id columns whose values are expected to differ even between
# byte-equivalent replicates. These are filtered out of the diff
# report so authors aren't told "the timestamps differ".
#
# `intervention_name` is in the legacy-author-label position:
# substrate stopped emitting it in Phase 6 (cells carry typed
# `arm_key = canonical_str(intervention)` instead), but old cache
# parquets still have the column with sub-sweep aliases like
# `'ddqn'` / `'ddqn_n1'` / `'ddqn_g099'` — all canonicalising to
# the same `arm_key`. Skipping it here equates legitimate cross-
# sub-sweep replicates instead of false-positive flagging them.
#
# `claim_graph_signature` is the substrate-composition fingerprint
# but was inconsistently populated across cache builds (None vs
# value drift). The arm_key already encodes the canonical
# composition; the signature column adds nothing per duplicate.
_PROVENANCE_TAGS: frozenset[str] = frozenset({
    'id', 'parent_id', 'cycle_id', 'timestamp', 'corpus',
    'intervention_name', 'claim_graph_signature',
    # Per-cell framework-stamped verdict (RunRow.verdict). Computed
    # from per-cell invariant evaluation; may legitimately drift
    # across cache builds when invariant thresholds are retuned.
    # Not regime-defining.
    'verdict',
})


def _distinguishing_columns(
    cells: Iterable[Mapping[str, object]],
    *,
    skip: frozenset[str] = frozenset(),
) -> dict[str, list[object]]:
    """Return columns whose values differ across `cells`, excluding
    framework-provenance tags, registered-measurable names, and any
    names in `skip`. Values that can't be hashed (lists / arrays /
    dicts) are compared by their `str` form so heterogeneous-shape
    cells still surface a diff.

    `None` in a cell value means "field not set, default applied"
    — the value is treated as a wildcard against the explicit
    values in other cells. A column whose distinct values are
    `{None, X}` for a single explicit `X` is NOT distinguishing
    (the None cell would have produced X if the field had been
    materialised). A column with `{None, X, Y}` for two distinct
    explicit values IS distinguishing (None is consistent with at
    most one of X or Y, so the regime ambiguity resolves to X-vs-Y
    at minimum).

    **Registered measurables are skipped.** A registered
    `@measurable`'s value is DERIVED from the cell's raw record;
    two cells from byte-equivalent training runs may have slightly
    different derived values when the measurable was computed
    against different cache-build code (e.g., a refactored
    reduction). Distinguishing-cols flags the difference, but
    semantically the cells ARE the same training run — the
    difference is in the cache-build provenance, not the regime.
    Mirrors `leaf_signature`'s exclusion of `registered_names()`
    from the configurational fingerprint.

    Returned dict: column name → sorted-by-string list of distinct
    NON-NULL values observed (deduped). Empty dict means cells
    differ only on provenance / derived-measurable drift — the
    author wanted dedupe_strategy='mean'."""
    cell_list = list(cells)
    if len(cell_list) < 2:
        return {}
    # Lazy import: `corroborate.measurables` would cycle if imported
    # at module top (analyses.* → measurables → analyses).
    from corroborate.measurables import registered_names
    derived = frozenset(registered_names())
    all_columns: set[str] = set()
    for cell in cell_list:
        all_columns |= set(cell.keys())
    candidate = all_columns - _PROVENANCE_TAGS - derived - skip

    out: dict[str, list[object]] = {}
    for col in candidate:
        seen: dict[str, object] = {}
        for cell in cell_list:
            v = cell.get(col)
            if v is None:
                # None is the wildcard — "field not set, default
                # applied". Skip from distinct-value tally.
                continue
            key = repr(v)
            if key not in seen:
                seen[key] = v
        # Column is distinguishing iff there are at least 2 distinct
        # non-null values. {None, X} reduces to {X} which is
        # non-distinguishing; {None, X, Y} reduces to {X, Y} which
        # is distinguishing.
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
