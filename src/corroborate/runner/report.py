"""Per-`runner.run()` JSON audit report.

Every invocation of `runner.run()` (with `write_report=True`)
serializes a structured report to disk capturing: per-bridge verdict
+ every typed analysis-result dataclass (fields AND `@property`
accessors) + admission-gate outcomes + cell sample sizes + provenance
(git commit, timestamp, the existing measurable-signature manifest).

The report is the load-bearing audit artifact:

- Reviewers diff `experiments/findings/<short>.run.json` to see
  verdict-landscape drift across PRs.
- A snapshot pytest re-runs each committed report's hypothesis and
  asserts verdict identity — sentinel against accidental bridge edits.
- Memory entries that name effect sizes can be cross-checked against
  the report rather than trusting hand-typed numbers.

The cache parquet (`experiments/data/cache/<short>.parquet`) remains
a pure speedup — separate decision whether to commit.

Generic over Result types: a single `_coerce_value` walks any
`@dataclass(frozen=True, slots=True)` Result via `dataclasses.fields`
+ `@property` introspection. Substrate authors who add a new analysis
Result class get JSON serialization for free; no per-class
`as_dict()` boilerplate needed.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import math
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl

from corroborate.bridge.bridge import Bridge, BridgeEvaluation, _filter_with_missing_cols


SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ErroredBridgeEntry:
    """A bridge whose `evaluate()` raised during the run. Captured
    so the audit trail surfaces authoring bugs that would otherwise
    vanish into stderr (the historical behavior)."""
    bridge_name: str
    error_type: str
    error_message: str
    traceback_repr: str


@dataclass(frozen=True, slots=True)
class BridgeReportEntry:
    """One bridge's outcome: structural metadata + verdict + every
    analysis result (with property accessors expanded) + admission
    gates + sample-size diagnostics. JSON-friendly via `_coerce_value`."""
    bridge_name: str
    source_name: str
    target_name: str
    direction: str
    tier: str
    pair_by: tuple[str, ...]
    predicted_direction: str | None
    scope_repr: str | None
    params: Mapping[str, object]
    n_cells_pre_scope: int
    n_cells_in_scope: int
    verdict: str
    analysis_results: Mapping[str, Mapping[str, object]]
    warnings: tuple[Mapping[str, object], ...]
    blocked_by: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class RunReport:
    """Top-level run audit. All fields JSON-serializable via
    `_coerce_value`."""
    schema_version: int
    hypothesis_module: str
    timestamp_utc: str
    git_commit: str | None
    n_cells_total: int
    cache_path: str | None
    measurable_signatures: Mapping[str, str]
    bridges: tuple[BridgeReportEntry, ...]
    errored_bridges: tuple[ErroredBridgeEntry, ...]


# ============ Serializer ============


_COERCE_WARNINGS_EMITTED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _COERCE_WARNINGS_EMITTED:
        return
    _COERCE_WARNINGS_EMITTED.add(key)
    print(f'runner/report: {message}', file=sys.stderr)


_SKIP = object()  # sentinel: caller should drop this key entirely


def _coerce_value(v: object) -> object:
    """Recursive coercion of any value to a JSON-friendly Python
    primitive (or container of primitives).

    Rules (checked in order):
      - None / bool / str / int → as-is (bool checked before int
        because `isinstance(True, int)` is True)
      - float: NaN → "NaN", +inf → "Infinity", -inf → "-Infinity";
        else as-is. String sentinels preserve "computed and degenerate"
        vs "not measured" (null) — diffing on null would conflate
      - Enum → .value (idiomatic per RunRow.as_dict at schema.py:250
        and codebase-wide convention for Verdict / Direction / Tier /
        GateLevel / PredictedDirection)
      - numpy.generic (np.float64, np.int64, np.bool_) → .item() then
        re-coerce
      - numpy.ndarray → .tolist() then re-coerce element-wise
      - Mapping → {str(k): _coerce_value(v) for k,v in m.items()}
      - tuple/list → [_coerce_value(x) for x in v]
      - frozenset/set → sorted-by-repr list of coerced elements
      - dataclasses.is_dataclass(v) and not isinstance(v, type) →
        walks both `dataclasses.fields()` AND `@property` accessors
        on the class. This is load-bearing: many Result types (e.g.
        PairedGResult.p_value) expose headline numbers as properties,
        not fields. Skipping properties would lose the audit-relevant
        numbers
      - Measurable (duck-typed: has `.name` and `.signature` callable)
        → .name string
      - Callable (functions, methods, partials) → skip (return the
        `_SKIP` sentinel) — captured upstream for `Bridge.params`
        filtering
      - pl.Expr → str(v) (rare; defensive)
      - Fallback: str(v) with one-time stderr warning

    Returns either a JSON-friendly value or `_SKIP` to signal the
    caller (a Mapping coercer) to drop the key. `_SKIP` is private.
    """
    # bool BEFORE int (isinstance(True, int) is True; we want JSON true/false)
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):  # not bool (handled above)
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return 'NaN'
        if math.isinf(v):
            return 'Infinity' if v > 0 else '-Infinity'
        return v
    if isinstance(v, Enum):
        # Enum.value may itself need coercion (e.g. int-valued enum)
        return _coerce_value(v.value)
    if isinstance(v, np.generic):
        # `np.generic.item()` returns the Python primitive (`float`,
        # `int`, `bool`); re-coerce so NaN-floats hit the float branch.
        return _coerce_value(v.item())
    if isinstance(v, np.ndarray):
        return [_coerce_value(x) for x in v.tolist()]
    if isinstance(v, Mapping):
        out: dict[str, object] = {}
        for k, item in v.items():
            coerced = _coerce_value(item)
            if coerced is _SKIP:
                continue
            out[str(k)] = coerced
        return out
    if isinstance(v, (tuple, list)):
        return [
            _coerce_value(x) for x in v
            if _coerce_value(x) is not _SKIP
        ]
    if isinstance(v, (frozenset, set)):
        items = [_coerce_value(x) for x in v]
        items = [x for x in items if x is not _SKIP]
        return sorted(items, key=repr)
    # Measurable (duck-typed). Avoid hard import to keep this module
    # decoupled from the measurables registry.
    if _is_measurable(v):
        return getattr(v, 'name')
    # Dataclass instance — walk fields + @property accessors.
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _coerce_dataclass(v)
    if isinstance(v, pl.Expr):
        return str(v)
    if callable(v):
        return _SKIP
    _warn_once(
        f'fallback:{type(v).__name__}',
        f'no coercion rule for type {type(v).__name__!r}; '
        f'falling back to str(value).',
    )
    return str(v)


def _is_measurable(v: object) -> bool:
    """Duck-typed Measurable check: has `.name: str` and a callable
    `.signature`. Used by `_coerce_value` to map Measurable → name,
    and by `_coerce_bridge_params` to bypass the callable filter
    (Measurable instances are themselves callable)."""
    return (
        hasattr(v, 'name') and isinstance(getattr(v, 'name'), str)
        and callable(getattr(v, 'signature', None))
    )


def _coerce_dataclass(v: object) -> dict[str, object]:
    """Walk a dataclass instance: emit each field PLUS every
    `@property` descriptor on the class. Properties that raise
    (e.g., `p_value` when SE is zero — well-defined NaN but the
    underlying scipy call may also crash) are recorded as `"NaN"` so
    the report shape is stable but the failure is visible.
    """
    out: dict[str, object] = {}
    for f in dataclasses.fields(v):
        coerced = _coerce_value(getattr(v, f.name))
        if coerced is _SKIP:
            continue
        out[f.name] = coerced
    cls = type(v)
    # Properties live on the class, not instances. `inspect.getattr_static`
    # avoids invoking the descriptor (so we can detect it as a property
    # instead of getting its value).
    for name in dir(cls):
        if name.startswith('_') or name in out:
            continue
        try:
            descriptor = inspect.getattr_static(cls, name)
        except AttributeError:
            continue
        if not isinstance(descriptor, property):
            continue
        try:
            value = getattr(v, name)
        except Exception:  # noqa: BLE001  # property body may legitimately raise
            out[name] = 'NaN'
            continue
        coerced = _coerce_value(value)
        if coerced is _SKIP:
            continue
        out[name] = coerced
    return out


# ============ Build / write ============


def _git_short_sha(repo_root: Path) -> str | None:
    """`git -C <root> rev-parse --short=12 HEAD`, 2-second timeout.
    Returns None on any failure (CalledProcessError, FileNotFoundError,
    timeout). No precedent in the codebase for git-info helpers; first
    use here. If a future module needs git provenance too, promote to
    `corroborate._internals.git`."""
    try:
        proc = subprocess.run(
            ['git', '-C', str(repo_root), 'rev-parse', '--short=12', 'HEAD'],
            capture_output=True, text=True, check=True, timeout=2.0,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    sha = proc.stdout.strip()
    return sha or None


def _coerce_bridge_params(
    bridge_name: str, params: Mapping[str, object],
) -> dict[str, object]:
    """Coerce `Bridge.params` to JSON-friendly dict. Filters out
    plain callables (rare but possible — a substrate-author kwarg
    that's a closure). Measurable instances are NOT filtered (they're
    callable but carry typed `.name` semantics — `_coerce_value`
    serializes them as `.name`). Emits a one-time stderr warning per
    omitted key."""
    out: dict[str, object] = {}
    for k, v in params.items():
        # Measurable check FIRST — Measurables are callable by design
        # but carry meaningful identity (the registered name).
        if _is_measurable(v):
            out[str(k)] = _coerce_value(v)
            continue
        if isinstance(v, Callable) and not isinstance(v, type):  # type: ignore[arg-type]
            # A class is technically Callable but represents a typed
            # value; serialize via str(cls). Plain functions/methods
            # get filtered.
            _warn_once(
                f'param-callable:{bridge_name}.{k}',
                f'bridge {bridge_name!r} param {k!r} is callable '
                f'({type(v).__name__}); omitted from report.',
            )
            continue
        coerced = _coerce_value(v)
        if coerced is _SKIP:
            continue
        out[str(k)] = coerced
    return out


def _build_bridge_entry(
    bridge: Bridge,
    evaluation: BridgeEvaluation,
    cells: pl.DataFrame,
) -> BridgeReportEntry:
    """One bridge → its report entry. Re-applies `bridge.scope` to
    compute `n_cells_in_scope` (cheap; the same filter `evaluate()`
    already did). Doesn't touch `BridgeEvaluation` shape."""
    n_pre = cells.height
    if bridge.scope is None or n_pre == 0:
        n_in_scope = n_pre
    else:
        n_in_scope = _filter_with_missing_cols(cells, bridge.scope).height
    scope_repr = str(bridge.scope) if bridge.scope is not None else None
    analysis_results_dict: dict[str, Mapping[str, object]] = {}
    for fixture_name, result in evaluation.analysis_results.items():
        coerced = _coerce_value(result)
        if not isinstance(coerced, Mapping):
            # Result wasn't a dataclass / Mapping after coercion.
            # Wrap in a single-key dict so the report shape stays
            # `fixture → object` instead of mixing shapes.
            coerced = {'value': coerced}
        analysis_results_dict[fixture_name] = coerced
    warnings_list: list[Mapping[str, object]] = []
    for w in evaluation.warnings:
        coerced_w = _coerce_value(w)
        if isinstance(coerced_w, Mapping):
            warnings_list.append(coerced_w)
    blocked_by_dict: Mapping[str, object] | None = None
    if evaluation.blocked_by is not None:
        coerced_b = _coerce_value(evaluation.blocked_by)
        if isinstance(coerced_b, Mapping):
            blocked_by_dict = coerced_b
    # Direction.value is already a string ('direct', 'at_most', etc.).
    # Tier is an IntEnum so .value is an int; use .name.lower() to
    # match the readable convention in graph/causal.py:192's __repr__
    # ('invariant'/'associational'/'interventional').
    return BridgeReportEntry(
        bridge_name=bridge.name,
        source_name=bridge.source_name,
        target_name=bridge.target_name,
        direction=bridge.direction.value,
        tier=bridge.tier.name.lower(),
        pair_by=tuple(bridge.pair_by),
        predicted_direction=bridge.predicted_direction,
        scope_repr=scope_repr,
        params=MappingProxyType(_coerce_bridge_params(bridge.name, bridge.params)),
        n_cells_pre_scope=n_pre,
        n_cells_in_scope=n_in_scope,
        verdict=evaluation.verdict.value,
        analysis_results=MappingProxyType(analysis_results_dict),
        warnings=tuple(warnings_list),
        blocked_by=blocked_by_dict,
    )


def _build_errored_entry(bridge_name: str, exc: BaseException) -> ErroredBridgeEntry:
    return ErroredBridgeEntry(
        bridge_name=bridge_name,
        error_type=type(exc).__name__,
        error_message=str(exc),
        traceback_repr=''.join(
            traceback.format_exception(type(exc), exc, exc.__traceback__),
        ),
    )


def build_report(
    *,
    hypothesis_module_name: str,
    bridges: Sequence[Bridge],
    results: Mapping[str, BridgeEvaluation],
    errors: Mapping[str, BaseException],
    cells: pl.DataFrame,
    cache_path: Path | None,
    measurable_signatures: Mapping[str, str],
    repo_root: Path,
) -> RunReport:
    """Assemble the structured RunReport from runner state.

    `bridges` is the full bridge tuple from the hypothesis;
    `results` maps bridge_name → BridgeEvaluation for bridges that
    completed successfully; `errors` maps bridge_name → exception
    for bridges that raised. Bridges in `bridges` but absent from
    both dicts are silently skipped (shouldn't happen in practice —
    `runner.run()` populates one or the other for every bridge).
    """
    bridge_entries: list[BridgeReportEntry] = []
    errored_entries: list[ErroredBridgeEntry] = []
    for b in bridges:
        if b.name in results:
            bridge_entries.append(_build_bridge_entry(b, results[b.name], cells))
        elif b.name in errors:
            errored_entries.append(_build_errored_entry(b.name, errors[b.name]))
    cache_path_str: str | None
    if cache_path is None:
        cache_path_str = None
    else:
        try:
            cache_path_str = str(cache_path.relative_to(repo_root).as_posix())
        except ValueError:
            cache_path_str = str(cache_path.as_posix())
    return RunReport(
        schema_version=SCHEMA_VERSION,
        hypothesis_module=hypothesis_module_name,
        timestamp_utc=datetime.now(tz=timezone.utc).isoformat(timespec='seconds'),
        git_commit=_git_short_sha(repo_root),
        n_cells_total=cells.height,
        cache_path=cache_path_str,
        measurable_signatures=MappingProxyType(dict(measurable_signatures)),
        bridges=tuple(bridge_entries),
        errored_bridges=tuple(errored_entries),
    )


def write_report(report: RunReport, path: Path) -> None:
    """Atomic JSON write. Reuses the cloud.py:156 `_save_manifest`
    idiom verbatim: write to `path.with_suffix(path.suffix + '.tmp')`,
    then `tmp.replace(path)` for posix-atomic publish.

    `sort_keys=True`, `indent=2`, trailing newline → byte-deterministic
    diffable output. `allow_nan=False` is safe because `_coerce_value`
    has already converted NaN/inf to the string sentinels `"NaN"` /
    `"Infinity"` / `"-Infinity"`.
    """
    payload = _coerce_value(report)
    text = json.dumps(
        payload, indent=2, sort_keys=True,
        ensure_ascii=False, allow_nan=False,
    )
    tmp = path.with_suffix(path.suffix + '.tmp')
    _ = tmp.write_text(text + '\n')
    tmp.replace(path)


__all__ = [
    'BridgeReportEntry',
    'ErroredBridgeEntry',
    'RunReport',
    'SCHEMA_VERSION',
    'build_report',
    'write_report',
]
