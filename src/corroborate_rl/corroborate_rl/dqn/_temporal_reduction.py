"""`@temporal_reduction` — pair-registers `<base>_late` + `<base>_per_burst`
measurables from a single window-reduction function.

The implementation has two complementary aggregations of any per-step trace
column:

- **`<name>_late`**: scalar from the late-50% slice of the per-step
  array. Used by per-cell bridges that summarise convergence.
- **`<name>_per_burst`**: 1-D NDArray, one value per eval burst, sliced
  by `eval_step_index`. Used by per-burst panel bridges that show
  training-time dynamics.

For window-invariant reductions (mean, max, std, n_unique, entropy),
the two share an identical reduction kernel — only the windowing
strategy differs. Authoring them as separate `@measurable` functions
duplicates the windowing boilerplate and lets the sibling pair drift
(e.g., one renamed, the other not). `@temporal_reduction` registers
both from a single declaration.

For window-variant reductions (`policy_churn_late`'s consecutive-state-
revisit flip-rate is the in-tree example), set
`window_invariant=False` so only `<name>_late` is registered — the
per-burst version must be authored explicitly because the reduction's
semantics change with window size (within a short burst the policy
hasn't drifted enough between revisits for the flip-rate to track its
late-window interpretation)."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import overload

import numpy as np
import numpy.typing as npt

from corroborate.measurables.measurable import Measurable, register


type WindowReduction = Callable[[npt.NDArray[np.float64]], float]


def temporal_reduction(
    *,
    reads: tuple[str, ...],
    late_name: str | None = None,
    per_burst_name: str | None = None,
    window_invariant: bool = True,
) -> Callable[[WindowReduction], WindowReduction]:
    """Decorator that registers up to two measurables — `<late_name>`
    (scalar, late-50%) and `<per_burst_name>` (NDArray, per-burst) —
    from a single window-reduction function `(window) -> float`.

    At least one of `late_name` / `per_burst_name` must be supplied.
    The common case sets both (auto-paired siblings). When only
    `per_burst_name` is set, the decorator adds a per-burst sibling
    to an existing `@measurable`-registered `_late` counterpart
    without re-registering the late version. When only `late_name`
    is set, the late-only path is registered (useful when there is
    no meaningful per-burst sibling — e.g., n_episodes_late).

    `reads`: exactly one per-step trace column. The decorator injects
    `eval_step_index` into the per-burst measurable's reads contract
    automatically (callers don't have to remember it).

    `window_invariant`: when False, supplying `per_burst_name` raises.
    The reduction's interpretation depends on window size — pairing
    would silently rename a quantitatively-different statistic. Author
    the per-burst sibling explicitly with the renamed semantics in its
    docstring.

    Returns the original reduction function (so it can be referenced /
    tested directly); the registry-side effect happens on decoration."""
    if late_name is None and per_burst_name is None:
        raise ValueError(
            '@temporal_reduction requires at least one of late_name '
            '/ per_burst_name.',
        )
    if len(reads) != 1:
        raise ValueError(
            f'@temporal_reduction expects exactly one per-step trace '
            f'column in `reads`; got {reads!r}. Multi-column reductions '
            f'(e.g. policy_churn over argmax + state_hash) need '
            f'@measurable directly.',
        )
    if not window_invariant and per_burst_name is not None:
        raise ValueError(
            f'window_invariant=False but per_burst_name={per_burst_name!r} '
            f'supplied. The reduction\'s interpretation depends on '
            f'window size; auto-pairing would conflate two distinct '
            f'statistics. Author the per-burst sibling explicitly.',
        )

    trace_col = reads[0]

    def decorator(fn: WindowReduction) -> WindowReduction:
        # ---- Late-half scalar measurable ----
        if late_name is not None:
            late_name_local = late_name

            def _late_fn(record: Mapping[str, object]) -> float:
                arr_obj = record.get(trace_col)
                if arr_obj is None:
                    return float('nan')
                arr = np.asarray(arr_obj, dtype=np.float64).flatten()
                if arr.size < 2:
                    return float('nan')
                return float(fn(arr[arr.size // 2:]))

            register(Measurable(
                fn=_late_fn, name=late_name_local, reads=reads,
            ))

        # ---- Per-burst NDArray measurable ----
        if per_burst_name is not None:
            per_burst_reads = reads + ('eval_step_index',)

            def _per_burst_fn(
                record: Mapping[str, object],
            ) -> npt.NDArray[np.float64]:
                arr_obj = record.get(trace_col)
                eval_idx_obj = record.get('eval_step_index')
                if arr_obj is None or eval_idx_obj is None:
                    return np.zeros((0,), dtype=np.float64)
                arr = np.asarray(arr_obj, dtype=np.float64).flatten()
                eval_idx = np.asarray(eval_idx_obj).flatten()
                n = arr.size
                n_bursts = eval_idx.size
                if n == 0 or n_bursts == 0:
                    return np.zeros((0,), dtype=np.float64)
                edges = np.linspace(
                    0, n, n_bursts + 1, dtype=np.int64,
                )
                return np.array(
                    [
                        float(fn(arr[edges[i]:edges[i+1]]))
                        for i in range(n_bursts)
                    ],
                    dtype=np.float64,
                )

            register(Measurable(
                fn=_per_burst_fn, name=per_burst_name, reads=per_burst_reads,
            ))

        return fn

    return decorator


__all__ = ['temporal_reduction', 'WindowReduction']
