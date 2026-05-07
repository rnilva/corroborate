"""Defensive primitives at the corpus boundary
(CORPUS_INTEGRITY.md).

Each invariant CI1-CI8 has a check function that surfaces
violations as a typed exception. Called at the boundaries
where corpora enter the framework: `_load_directory` for
ingest, `cloud.archive` for outbound writes, trace restore
inside `_load_one_corpus`.

Currently implemented:
- CI1 nested-corpus detection (`assert_no_nested_corpora`).
- CI3 cloud-root collision (`assert_unique_remote_root`).
- CI4 dedup-volatile-strip (`_volatile_object_repr_columns`
  in `runner.py`).
- CI5 archive precondition (`assert_archive_eligible`).
- CI6 row-level orphan eviction (in
  `corpus/measurements.py:build_measurements`).
- CI8 trace-id subset check (`assert_traces_subset_of_runs`).

CI2 (per-corpus measurements id uniqueness) lives in
`corpus/measurements.py` because it needs the polars frame
post-load. CI7 (broader trace eviction) is Phase 3, deferred.

Each check produces a list of violators (not just the first)
so a single `find/fix` pass surfaces everything in one error.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl


# ============ Sentinel — sweep-in-progress marker ============

IN_PROGRESS_SENTINEL = '.in_progress'
"""Filename of the sentinel a sweep dispatcher drops at corpus
root to signal "this corpus is mid-flight, the runner / CI1
should ignore it." Removed on successful merge.

Behavior when present at `<corpus_dir>/.in_progress`:
- `assert_no_nested_corpora`: skips the corpus's subtree audit.
- `runner._load_directory`: skips the corpus entirely (no
  ingest, no SKIPPED-no-runs-parquet noise).

Hidden filename so it doesn't clutter `ls`. The dispatcher is
expected to:
  1. Create the sentinel before writing per-arm sub-dirs.
  2. Remove the sentinel after the merge step finalizes.
A killed-mid-merge sweep leaves the sentinel in place — next
ingest skips the corpus, user investigates + fixes."""


def is_in_progress(corpus_dir: Path) -> bool:
    """Cheap sentinel check — used by both CI1 and the runner."""
    return (corpus_dir / IN_PROGRESS_SENTINEL).exists()


# ============ CI1 — corpora are leaves ============


@dataclass(frozen=True, slots=True)
class NestedCorpusViolation:
    """One CI1 violation: a corpus directory contains a
    sub-directory which itself contains `runs.parquet`."""
    parent: Path
    nested: Path


class NestedCorpusError(RuntimeError):
    """Raised by `assert_no_nested_corpora` when at least one
    parent corpus has a nested sub-corpus.

    Carries the full list of violators so the user can fix
    everything in one pass rather than discovering them one at
    a time."""

    def __init__(
        self, violations: Sequence[NestedCorpusViolation],
    ) -> None:
        self.violations = tuple(violations)
        super().__init__(self._format())

    def _format(self) -> str:
        lines = [
            f'CORPUS_INTEGRITY.md CI1 — {len(self.violations)} '
            f'corpus dir(s) contain nested sub-corpora; the runner '
            f'walks one level deep and would silently skip the '
            f'inner ones. Flatten the layout (move nested dirs to '
            f'siblings) and retry.',
        ]
        for v in self.violations:
            lines.append(f'  - {v.parent}/ contains {v.nested}')
        return '\n'.join(lines)


def _check_corpus_no_nested(
    sub: Path,
) -> list[NestedCorpusViolation]:
    """Return any nested-corpus violations within `sub`. Returns
    `[]` when sentinel'd (sweep mid-flight) or genuinely clean.
    Shared between root-level walk and named-corpora ingest paths
    so both produce the same shape of error."""
    if not sub.is_dir():
        return []
    if (sub / IN_PROGRESS_SENTINEL).exists():
        return []
    out: list[NestedCorpusViolation] = []
    for p in sub.rglob('runs.parquet'):
        if p.parent == sub:
            continue
        try:
            rel_parts = p.parent.relative_to(sub).parts
        except ValueError:
            rel_parts = ()
        if 'tmp' in rel_parts:
            continue
        out.append(NestedCorpusViolation(
            parent=sub, nested=p.parent,
        ))
    return out


def assert_named_corpora_no_nested(
    corpus_dirs: Sequence[Path],
) -> None:
    """CI1 check for the named-corpora ingest path. Same semantics
    as `assert_no_nested_corpora` but operates on an explicit list
    of corpus dirs rather than walking a root."""
    violations: list[NestedCorpusViolation] = []
    for sub in corpus_dirs:
        violations.extend(_check_corpus_no_nested(sub))
    if violations:
        raise NestedCorpusError(violations)


def assert_no_nested_corpora(root: Path) -> None:
    """Scan `root` for CI1 violations: a top-level corpus dir
    (one whose subtree contains `runs.parquet` at any depth)
    that has more than one such file. Two shapes count:

    1. **Hybrid**: parent has its own `runs.parquet` AND a
       sub-directory with another `runs.parquet`. The runner
       ingests the parent, silently drops the inner.
    2. **Pure nested**: parent has no `runs.parquet` of its
       own but contains one or more sub-directories that do.
       The runner sees an empty corpus, SKIPS it, silently
       dropping all inner corpora.

    Both shapes get flagged. The user fixes by either flattening
    (`mv parent/sub/ parent_sub/`) or by removing the parent
    entirely if the inner data has been promoted elsewhere.

    **Sentinel escape hatch**: a corpus dir carrying a
    `.in_progress` marker file (typically dropped by a sweep
    dispatcher mid-execution and removed on successful merge)
    is skipped by both this audit AND the runner's directory
    walk. The "nested per-arm subdirs during sweep" pattern is
    legitimate and shouldn't trigger CI1. Once the sweep
    completes and removes the sentinel, the next ingest
    enforces CI1 normally.

    `root` is `experiments/data/` (or whatever the analysis is
    pointing at). The check is one-shot at ingest entry.
    """
    if not root.is_dir():
        return  # caller's normal not-a-dir path handles this

    violations: list[NestedCorpusViolation] = []
    for sub in sorted(root.iterdir()):
        violations.extend(_check_corpus_no_nested(sub))

    if violations:
        raise NestedCorpusError(violations)


# ============ CI5 — archive refuses trivial files ============


class ArchivePrecondition(RuntimeError):
    """Raised by `assert_archive_eligible` when a local file is
    too small or otherwise lacks the integrity signal that
    `archive()` requires before pushing.

    The canonical example from CORPUS_INTEGRITY.md was the
    0-byte `traces.parquet` placeholders that
    `archive_unarchived.py` blindly pushed for `action_dim_wide`
    and `reward_scale_sweep` after their sweep merges were
    interrupted. Refusing upfront keeps cloud state clean."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(
            f'CORPUS_INTEGRITY.md CI5 — {path} fails archive '
            f'precondition: {reason}. A trivial / corrupt file '
            f'should not be pushed to the cloud archive.'
        )


def assert_archive_eligible(
    path: Path, *, min_size: int = 1024,
) -> None:
    """Validate that `path` is safe to archive: file exists,
    is at least `min_size` bytes (default 1 KiB), and (when
    `.parquet`) ends with the `PAR1` magic footer.

    Mirrors `runner._file_present`'s checks; lives here so
    `cloud.archive()` can call it without importing from the
    runner. Raises `ArchivePrecondition` rather than returning
    False — `archive()` callers want a loud error when the
    file is broken, not a silent skip."""
    if not path.exists():
        raise ArchivePrecondition(path, 'file does not exist')
    try:
        size = path.stat().st_size
    except OSError as e:
        raise ArchivePrecondition(
            path, f'stat failed: {e}',
        ) from e
    if size < min_size:
        raise ArchivePrecondition(
            path,
            f'size {size} bytes < {min_size} minimum '
            f'(probably an empty placeholder from an interrupted '
            f'sweep merge)',
        )
    if path.suffix == '.parquet':
        try:
            with path.open('rb') as fh:
                _ = fh.seek(-4, 2)
                magic = fh.read(4)
        except OSError as e:
            raise ArchivePrecondition(
                path, f'PAR1 footer read failed: {e}',
            ) from e
        if magic != b'PAR1':
            raise ArchivePrecondition(
                path,
                f'parquet missing PAR1 magic footer (got {magic!r}); '
                f'truncated or corrupt write',
            )


# ============ CI3 — cloud-root uniqueness ============


class RemoteRootCollision(RuntimeError):
    """Raised when two local corpora claim the same `remote_root`.

    Two corpora pushing to the same s3 prefix silently overwrite
    each other on every archive — only one's data survives. The
    canonical example from CORPUS_INTEGRITY.md was the
    `minatar_sync_curve/{ddqn_sync1k, ddqn_sync3k, vanilla_sync1k,
    vanilla_sync3k}` quartet sharing
    `s3://corroborate-archive/minatar_sync_curve`."""

    def __init__(
        self,
        remote_root: str,
        claiming_dir: Path,
        existing_dir: Path,
    ) -> None:
        self.remote_root = remote_root
        self.claiming_dir = claiming_dir
        self.existing_dir = existing_dir
        super().__init__(
            f'CORPUS_INTEGRITY.md CI3 — remote_root '
            f'{remote_root!r} is already claimed by '
            f'{existing_dir}; {claiming_dir} cannot also archive '
            f'to it (silent overwrite would lose data). Use a '
            f'distinct cloud prefix per corpus.'
        )


def assert_unique_remote_root(
    sweep_dir: Path, remote_root: str,
) -> None:
    """Scan `sweep_dir.parent` for sibling corpora's `_remote.json`
    files; raise `RemoteRootCollision` if any sibling claims the
    same `remote_root` as this archive call.

    Call at the entry of `cloud.archive()` so the check fires
    BEFORE any cloud I/O. The sweep_dir's own existing manifest
    (if any) is excluded from the comparison — re-archiving the
    same dir to the same root is fine; collision is only between
    DIFFERENT local corpora."""
    parent = sweep_dir.parent
    if not parent.is_dir():
        return

    # Lazy import to avoid circular dep with cloud.py
    from corroborate.corpus.cloud import MANIFEST_NAME, _load_manifest

    for sibling in sorted(parent.iterdir()):
        if not sibling.is_dir():
            continue
        if sibling.resolve() == sweep_dir.resolve():
            continue
        if not (sibling / MANIFEST_NAME).exists():
            continue
        m = _load_manifest(sibling)
        if m is None:
            continue
        if m.remote_root == remote_root:
            raise RemoteRootCollision(
                remote_root=remote_root,
                claiming_dir=sweep_dir,
                existing_dir=sibling,
            )


# ============ CI8 — traces.id ⊆ runs.id ============


@dataclass(frozen=True, slots=True)
class TraceContaminationStats:
    """Summary of CI8 audit on a single corpus."""
    corpus_dir: Path
    runs_count: int
    traces_count: int
    spurious_count: int
    overlap_count: int

    @property
    def is_contaminated(self) -> bool:
        """traces.parquet has at least one id absent from runs."""
        return self.spurious_count > 0


class TraceContaminationError(RuntimeError):
    """Raised when `traces.parquet` carries ids not present in
    `runs.parquet` — the cloud-collision residue pattern from
    CORPUS_INTEGRITY.md.

    Two corpora pushed to the same cloud root pre-CI3 silently
    overwrite each other; the survivor's traces.parquet contains
    a different sweep's UUIDs while the local runs.parquet still
    records the original sweep. The runner's left-join would
    produce all-null trace columns for every cell, and
    trace-dependent measurables fail per-cell with AxisError.

    Refusing the contaminated traces upfront — with the runner
    falling back to "no cloud traces" — produces honest null
    measurables for cells without traces, rather than silently
    null-everywhere from a join collision."""

    def __init__(self, stats: TraceContaminationStats) -> None:
        self.stats = stats
        super().__init__(
            f'CORPUS_INTEGRITY.md CI8 — '
            f'{stats.corpus_dir.name}/traces.parquet contains '
            f'{stats.spurious_count} id(s) absent from '
            f'runs.parquet (traces={stats.traces_count} ids, '
            f'runs={stats.runs_count} cells, overlap='
            f'{stats.overlap_count}). The cloud archive is from a '
            f'different sweep (cloud-collision residue). Either '
            f'remove `traces.parquet` from the manifest, or '
            f'restore from a clean source.'
        )


def audit_trace_contamination(
    corpus_dir: Path,
) -> TraceContaminationStats | None:
    """Scan `corpus_dir/runs.parquet` and `corpus_dir/traces.parquet`
    (local copies — no cloud read), compute the id-set relation.

    Returns `None` when either file is absent (caller decides
    whether absence is OK). Returns a stats record otherwise;
    the caller can check `stats.is_contaminated` for CI8 violation.
    """
    runs_p = corpus_dir / 'runs.parquet'
    traces_p = corpus_dir / 'traces.parquet'
    if not runs_p.exists() or not traces_p.exists():
        return None
    try:
        runs_ids = set(
            pl.read_parquet(runs_p, columns=['id'])['id'].to_list(),
        )
        traces_ids = set(
            pl.read_parquet(traces_p, columns=['id'])['id'].to_list(),
        )
    except (pl.exceptions.ComputeError, OSError, ValueError):
        # Caller's other integrity paths (`_file_present`, etc.)
        # handle parse errors; CI8 just doesn't audit when the
        # files are unreadable.
        return None
    spurious = traces_ids - runs_ids
    overlap = traces_ids & runs_ids
    return TraceContaminationStats(
        corpus_dir=corpus_dir,
        runs_count=len(runs_ids),
        traces_count=len(traces_ids),
        spurious_count=len(spurious),
        overlap_count=len(overlap),
    )


def assert_traces_subset_of_runs(corpus_dir: Path) -> None:
    """Run `audit_trace_contamination` and raise
    `TraceContaminationError` when `traces.parquet` has spurious
    ids. Caller decides where to surface this — typically inside
    the runner's per-corpus restore path, after a fresh download
    and BEFORE the join with runs.parquet."""
    stats = audit_trace_contamination(corpus_dir)
    if stats is not None and stats.is_contaminated:
        raise TraceContaminationError(stats)


__all__ = [
    'IN_PROGRESS_SENTINEL',
    'ArchivePrecondition',
    'NestedCorpusError',
    'NestedCorpusViolation',
    'RemoteRootCollision',
    'TraceContaminationError',
    'TraceContaminationStats',
    'assert_archive_eligible',
    'assert_named_corpora_no_nested',
    'assert_no_nested_corpora',
    'assert_traces_subset_of_runs',
    'assert_unique_remote_root',
    'audit_trace_contamination',
    'is_in_progress',
]
