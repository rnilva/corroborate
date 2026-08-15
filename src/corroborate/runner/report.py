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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl

from corroborate.bridge.bridge import Bridge, BridgeEvaluation
from corroborate.measurables.measurable import Measurable


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
    assumption_violations: tuple[str, ...] = ()
    extent_hash: int = 0


@dataclass(frozen=True, slots=True)
class RunReport:
    """Top-level run audit. All fields JSON-serializable via
    `_coerce_value`."""
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


def reset_warnings() -> None:
    """Clear the warn-once dedupe set. Test fixtures call this so
    test order doesn't determine which warnings fire (a warning
    emitted by an earlier test would silence the same path in a
    later test, breaking warning-coverage assertions)."""
    _COERCE_WARNINGS_EMITTED.clear()


_SKIP = object()  # sentinel: caller should drop this key entirely


def _coerce_value(v: object) -> object:
    """Recursive coercion of any value to a JSON-friendly Python
    primitive (or container of primitives).

    Rules (checked in order):
      - None / bool / str / int → as-is (bool checked before int
        because `isinstance(True, int)` is True)
      - float: NaN / +inf / -inf → None (JSON null). Both
        "computed-degenerate" (e.g. `p_value` when SE=0) and
        "not measured" map to null. The audit user gets a
        typed-clean Float64 column from `polars.read_json`
        instead of a String hosting "NaN" sentinels (which would
        contradict CLAUDE.md §Persistence "Hard rule: no JSON-
        wrapped struct columns" — `df.filter(pl.col('g') > 0)`
        works).
      - Enum → .value (idiomatic per RunRow.as_dict at schema.py:250
        and codebase-wide convention for Verdict / Direction / Tier /
        GateLevel / PredictedDirection)
      - numpy.generic (np.float64, np.int64, np.bool_) → .item() then
        re-coerce
      - numpy.ndarray → .tolist() then re-coerce element-wise
      - Mapping → {str(k): _coerce_value(v) for k,v in m.items()}
      - tuple / list → single-pass coerce + filter `_SKIP` (the
        previous double-call comprehension was a real bug — every
        property descriptor on a list element fired twice, doubling
        compute and risking divergent output for properties with
        side effects)
      - frozenset / set → sorted-by-repr list of coerced elements
      - Measurable (typed isinstance against the registry's class).
        Was duck-typed (`.name: str` + callable `.signature`) until
        the reviewer pointed out that any class growing the same
        shape — `Bridge` itself in the planned bridge-graph work —
        would silently match.
      - dataclass instance → walks both `dataclasses.fields()` AND
        `@property` accessors on the class (load-bearing: many
        Result types — `PairedGResult.p_value`,
        `VerdictCounts.held_fraction` — expose headline numbers as
        properties, not fields)
      - pl.Expr → str(v) (rare; defensive)
      - Callable (functions, methods, partials) → skip (return the
        `_SKIP` sentinel) — captured upstream for `Bridge.params`
        filtering
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
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, Enum):
        # Enum.value may itself need coercion (e.g. int-valued enum)
        return _coerce_value(v.value)
    if isinstance(v, np.generic):
        # `np.generic.item()` returns the Python primitive (`float`,
        # `int`, `bool`); re-coerce so NaN-floats hit the float branch.
        return _coerce_value(v.item())
    if isinstance(v, np.ndarray):
        out_arr: list[object] = []
        for x in v.tolist():
            coerced = _coerce_value(x)
            if coerced is _SKIP:
                continue
            out_arr.append(coerced)
        return out_arr
    if isinstance(v, Mapping):
        out_map: dict[str, object] = {}
        for k, item in v.items():
            coerced = _coerce_value(item)
            if coerced is _SKIP:
                continue
            out_map[str(k)] = coerced
        return out_map
    if isinstance(v, (tuple, list)):
        out_seq: list[object] = []
        for x in v:
            coerced = _coerce_value(x)
            if coerced is _SKIP:
                continue
            out_seq.append(coerced)
        return out_seq
    if isinstance(v, (frozenset, set)):
        items: list[object] = []
        for x in v:
            coerced = _coerce_value(x)
            if coerced is _SKIP:
                continue
            items.append(coerced)
        return sorted(items, key=repr)
    # Typed Measurable check (no duck-typing — see docstring above).
    if isinstance(v, Measurable):
        return v.name
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


def _coerce_dataclass(v: object) -> dict[str, object]:
    """Walk a dataclass instance: emit each field PLUS every
    `@property` descriptor on the class. Properties that raise
    emit a one-time stderr warning + record `null` (the failure
    is the warning, not a sentinel string in the data — string
    sentinels would break typed downstream readers, see
    `_coerce_value` float-branch docstring).
    """
    # Runtime invariant: callers dispatch here only after an
    # `is_dataclass` check on the instance; assert re-narrows for
    # the type checker without changing the contract.
    assert dataclasses.is_dataclass(v) and not isinstance(v, type)
    out: dict[str, object] = {}
    cls = type(v)
    for f in dataclasses.fields(v):
        coerced = _coerce_value(getattr(v, f.name))
        if coerced is _SKIP:
            continue
        out[f.name] = coerced
    # Properties live on the class, not instances. `inspect.getattr_static`
    # avoids invoking the descriptor (so we can detect it as a property
    # without invoking it).
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
        except Exception as e:  # noqa: BLE001  # property body may legitimately raise
            _warn_once(
                f'property-raised:{cls.__module__}.{cls.__name__}.{name}',
                f'property {cls.__name__}.{name} raised '
                f'({type(e).__name__}: {e}); recorded as null.',
            )
            out[name] = None
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
    that's a closure). Measurable instances are NOT filtered (they
    are callable by design but carry typed `.name` semantics —
    `_coerce_value` serializes them as `.name`). Emits a one-time
    stderr warning per omitted key."""
    out: dict[str, object] = {}
    for k, v in params.items():
        # Measurable check FIRST — Measurables are callable by design
        # but carry meaningful identity (the registered name).
        if isinstance(v, Measurable):
            out[str(k)] = v.name
            continue
        if callable(v) and not isinstance(v, type):
            # A class is technically callable but represents a typed
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
    n_cells_total: int,
) -> BridgeReportEntry:
    """One bridge → its report entry. Reads `n_cells_in_scope`
    directly from `evaluation` (populated by `evaluate()`); does
    NOT recompute the scope filter."""
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
        else:
            _warn_once(
                f'warning-non-mapping:{type(w).__name__}',
                f'bridge {bridge.name!r} warning of type '
                f'{type(w).__name__} did not coerce to Mapping; dropped.',
            )
    blocked_by_dict: Mapping[str, object] | None = None
    if evaluation.blocked_by is not None:
        coerced_b = _coerce_value(evaluation.blocked_by)
        if isinstance(coerced_b, Mapping):
            blocked_by_dict = coerced_b
        else:
            _warn_once(
                f'blocked-by-non-mapping:{type(evaluation.blocked_by).__name__}',
                f'bridge {bridge.name!r} blocked_by of type '
                f'{type(evaluation.blocked_by).__name__} did not '
                f'coerce to Mapping; dropped.',
            )
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
        n_cells_pre_scope=n_cells_total,
        n_cells_in_scope=evaluation.n_cells_in_scope,
        verdict=evaluation.verdict.value,
        analysis_results=MappingProxyType(analysis_results_dict),
        warnings=tuple(warnings_list),
        blocked_by=blocked_by_dict,
        assumption_violations=evaluation.assumption_violations,
        extent_hash=evaluation.extent_hash,
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


def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` looking for a `.git` directory or file.
    Returns the parent containing it, or `start` if none found.
    Used to make `git_commit` and `cache_path` relative paths
    deterministic regardless of the cwd from which `runner.run()`
    is invoked (running from a notebook one directory above the
    repo would otherwise silently lose `git_commit` and emit absolute
    `cache_path`)."""
    for p in (start, *start.parents):
        if (p / '.git').exists():
            return p
    return start


def build_report(
    *,
    hypothesis_module_name: str,
    bridges: Sequence[Bridge],
    results: Mapping[str, BridgeEvaluation],
    errors: Mapping[str, BaseException],
    n_cells_total: int,
    cache_path: Path | None,
    measurable_signatures: Mapping[str, str],
    repo_root: Path | None = None,
) -> RunReport:
    """Assemble the structured RunReport from runner state.

    `bridges` is the full bridge tuple from the hypothesis;
    `results` maps bridge_name → BridgeEvaluation for bridges that
    completed successfully; `errors` maps bridge_name → exception
    for bridges that raised. Bridges in `bridges` but absent from
    both dicts are silently skipped (shouldn't happen in practice —
    `runner.run()` populates one or the other for every bridge).

    `n_cells_total` is supplied separately rather than passed as a
    DataFrame because the report doesn't need the data — only the
    height for diagnostics. `BridgeEvaluation.n_cells_in_scope`
    already carries the per-bridge size populated by `evaluate()`.

    `repo_root` defaults to walking up from the current file's
    directory looking for `.git`.
    """
    if repo_root is None:
        repo_root = _find_repo_root(Path(__file__).resolve().parent)
    bridge_entries: list[BridgeReportEntry] = []
    errored_entries: list[ErroredBridgeEntry] = []
    for b in bridges:
        if b.name in results:
            bridge_entries.append(_build_bridge_entry(b, results[b.name], n_cells_total))
        elif b.name in errors:
            errored_entries.append(_build_errored_entry(b.name, errors[b.name]))
    cache_path_str: str | None
    if cache_path is None:
        cache_path_str = None
    else:
        try:
            cache_path_str = str(cache_path.resolve().relative_to(repo_root).as_posix())
        except ValueError:
            cache_path_str = str(cache_path.as_posix())
    return RunReport(
        hypothesis_module=hypothesis_module_name,
        timestamp_utc=datetime.now(tz=timezone.utc).isoformat(timespec='seconds'),
        git_commit=_git_short_sha(repo_root),
        n_cells_total=n_cells_total,
        cache_path=cache_path_str,
        measurable_signatures=MappingProxyType(dict(measurable_signatures)),
        bridges=tuple(bridge_entries),
        errored_bridges=tuple(errored_entries),
    )


_FLOAT_SIG_FIGS = 6


def _round_floats(payload: object) -> object:
    """Round every float in the payload to `_FLOAT_SIG_FIGS` significant
    figures. Walks dict/list recursively. Done at JSON-write time only —
    in-memory `_coerce_value` output keeps full precision so callers
    can choose between the two views.

    Last-digit scipy / statsmodels noise (`g: 0.0987955046061797` vs
    `g: 0.09879550460617969` on otherwise-identical runs) drowns the
    audit signal in a `git diff` of the report. 6 sig figs is plenty
    for a reviewer to spot a real verdict-magnitude shift while
    suppressing rerun-noise diffs."""
    if isinstance(payload, float):
        if math.isnan(payload) or math.isinf(payload):
            return None
        # `format(v, '.6g')` collapses 0.09879550460617969 → '0.0987955';
        # round-trip via float() yields a stable canonical representation.
        return float(format(payload, f'.{_FLOAT_SIG_FIGS}g'))
    if isinstance(payload, dict):
        return {k: _round_floats(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_round_floats(x) for x in payload]
    return payload


def write_report(report: RunReport, path: Path) -> None:
    """Atomic JSON write. Reuses the cloud.py:156 `_save_manifest`
    idiom verbatim: write to `path.with_suffix(path.suffix + '.tmp')`,
    then `tmp.replace(path)` for posix-atomic publish.

    Floats are rounded to 6 sig figs (`_round_floats`) at write time
    only, so on-disk diffs are stable while in-memory consumers of
    `RunReport` keep full precision.

    `sort_keys=True`, `indent=2`, trailing newline → diff-friendly
    output. `allow_nan=False` is safe because `_coerce_value` has
    already converted NaN/inf to JSON null."""
    payload = _round_floats(_coerce_value(report))
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
    'build_report',
    'write_report',
]
