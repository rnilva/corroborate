"""Per-stratum panel — the framework's shared "group-then-analyze"
primitive.

`per_stratum_panel(cells, stratify_by, analysis)` groups cells by
the key extracted by `stratify_by` and runs `analysis` on each
group's cell-set. Returns a panel of `(key, result)` pairs in
deterministic key order.

The shape every per-env / per-stratum analysis wants:

  cells  →  group-by stratifier  →  per-group analysis  →  panel

Bridge-side analyses (`per_env_paired_g_panel`, etc.) consume this
helper so the per-stratum loop lives in exactly one place. The
analysis fn handles all per-group statistics (paired_g, regression,
etc.); this helper handles ONLY the grouping.

Cells whose stratifier returns None are skipped — letting the
caller treat "no key" cells (missing env_name, etc.) as outside
the panel rather than coercing them into a nonsense bucket.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol, Self


class _SupportsLessThan(Protocol):
    """Stratum keys must support `<` so the panel iterates in
    deterministic order. str / int / tuple-of-str all qualify."""
    def __lt__(self, other: Self, /) -> bool: ...


def per_stratum_panel[K: _SupportsLessThan, R](
    cells: Iterable[Mapping[str, object]],
    *,
    stratify_by: Callable[[Mapping[str, object]], K | None],
    analysis: Callable[[Sequence[Mapping[str, object]]], R],
    min_cells_per_stratum: int = 1,
    key_filter: Callable[[K], bool] | None = None,
) -> tuple[tuple[K, R], ...]:
    """Group `cells` by `stratify_by`; run `analysis` on each
    surviving group; return a panel of `(key, result)` pairs.

    `stratify_by(cell)` returns the stratum key (e.g. env_name)
    or `None` to skip the cell.

    `analysis(cells_in_stratum)` runs once per surviving group and
    returns whatever per-group result the caller needs.

    `min_cells_per_stratum` drops groups smaller than the minimum
    BEFORE invoking `analysis`. Default 1 = include any non-empty
    group; raise to require at least N cells (e.g. 2 for paired
    contrasts).

    `key_filter`, when supplied, excludes keys for which
    `key_filter(key)` returns False. The wrapper for env-scoped
    panels passes this to enforce a positive `env_filter`
    intersected with envs actually present in `cells`.

    Result order is `sorted(keys)` for deterministic iteration —
    callers that need a specific order should sort downstream.
    Keys must be comparable; if they aren't, sort fails loudly
    (better than silent non-determinism)."""
    by_stratum: dict[K, list[Mapping[str, object]]] = {}
    for cell in cells:
        key = stratify_by(cell)
        if key is None:
            continue
        if key_filter is not None and not key_filter(key):
            continue
        by_stratum.setdefault(key, []).append(cell)

    panel: list[tuple[K, R]] = []
    for key in sorted(by_stratum):
        subset = by_stratum[key]
        if len(subset) < min_cells_per_stratum:
            continue
        panel.append((key, analysis(subset)))
    return tuple(panel)


__all__ = ['per_stratum_panel']
