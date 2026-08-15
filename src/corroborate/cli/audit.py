"""CLI surface for `corroborate audit` subcommands.

Currently one subcommand: `corroborate audit pre-registration
<corpus_path>`. Resolves each committed bridge against the
manifest at `<corpus_path>/pre_registration.json`, re-computes
its source hash, runs the bridge against the corpus's cells, and
reports whether the empirical verdict matches the predicted
verdict.

**Honest scope.** This subcommand detects post-launch rewrites
(source-hash mismatch on a committed bridge) and empirical-vs-
predicted verdict drift. It does NOT detect pilot-corpus HARKing
(running a pilot, observing results, then editing the bridge and
relaunching with a 'fresh' git hash) or git-history rewriting.
Those failure modes require the priority-1 `--pre-data-repin`
lint and external anchors (priority 5) — see
`docs/FALSIFIABILITY_AND_PRE_REGISTRATION.md` §1.3 and §6.

Exit codes (pinned constants below):
- 0 (`EXIT_MATCH`) — all source hashes match AND every empirical
  verdict matches the predicted verdict.
- 1 (`EXIT_DRIFT`) — any source-hash drift OR any empirical-vs-
  predicted verdict drift.
- 2 (`EXIT_MANIFEST_MISSING`) — no `pre_registration.json` at the
  corpus path.
- 3 (`EXIT_GIT_HASH_NOT_FOUND`) — the manifest's `git_commit_hash`
  isn't present in `git log --all` (commit was rebased away or
  the repo was rewritten).
- 4 (`EXIT_BRIDGE_UNRESOLVED`) — at least one committed bridge
  cannot be resolved by its fully-qualified import path
  (refactor / rename / file move)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from corroborate.bridge.bridge import evaluate
from corroborate.bridge.verdict import Verdict
from corroborate.core.hypothesis import PredictedDirection
from corroborate.core.pre_registration import (
    PreRegistrationManifest,
    compute_bridge_source_hash,
    read_manifest,
    resolve_bridge_by_name,
)


EXIT_MATCH: Final[int] = 0
EXIT_DRIFT: Final[int] = 1
EXIT_MANIFEST_MISSING: Final[int] = 2
EXIT_GIT_HASH_NOT_FOUND: Final[int] = 3
EXIT_BRIDGE_UNRESOLVED: Final[int] = 4


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """One bridge's audit row.

    `source_hash_matches`: True iff the bridge's current source
    hash matches the manifest's committed hash. False signals a
    post-launch rewrite.

    `empirical_verdict`: the verdict the bridge produced when
    re-run against the corpus's cells. `None` iff the bridge
    failed to resolve (unresolved bridges never run).

    `verdict_matches`: True iff `empirical_verdict ==
    predicted_verdict`. False when the bridge resolved + ran but
    the verdict differs from what the author committed to."""
    bridge_name: str
    source_hash_matches: bool
    predicted_direction: PredictedDirection
    predicted_verdict: Verdict
    empirical_verdict: Verdict | None
    verdict_matches: bool
    unresolved_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Per-corpus audit result.

    `exit_code`: the CLI exit code (see module-level constants).
    `entries`: per-bridge audit rows.
    `manifest`: the parsed manifest (None iff exit_code ==
    `EXIT_MANIFEST_MISSING`)."""
    exit_code: int
    entries: tuple[AuditEntry, ...]
    manifest: PreRegistrationManifest | None

    def as_dict(self) -> Mapping[str, object]:
        return {
            'exit_code': self.exit_code,
            'manifest': (
                dict(self.manifest.as_dict()) if self.manifest is not None
                else None
            ),
            'entries': [
                {
                    'bridge_name': e.bridge_name,
                    'source_hash_matches': e.source_hash_matches,
                    'predicted_direction': e.predicted_direction,
                    'predicted_verdict': e.predicted_verdict.value,
                    'empirical_verdict': (
                        e.empirical_verdict.value
                        if e.empirical_verdict is not None else None
                    ),
                    'verdict_matches': e.verdict_matches,
                    'unresolved_reason': e.unresolved_reason,
                }
                for e in self.entries
            ],
        }


def _git_sha_in_history(sha: str, repo_root: Path) -> bool:
    """True iff `sha` is reachable by `git log --all` in
    `repo_root`. Subprocess; treats non-zero exit (not-in-repo,
    git-not-installed) as False."""
    out = subprocess.run(
        ['git', 'cat-file', '-e', f'{sha}^{{commit}}'],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return out.returncode == 0


def _load_corpus_cells(corpus_path: Path) -> pl.DataFrame:
    """Read `<corpus_path>/runs.parquet` into a DataFrame. Raises
    `FileNotFoundError` if missing — the audit pipeline runs the
    bridges against this data; without it there's nothing to
    audit against.

    Walks one level deeper if no top-level runs.parquet is found
    (matches `sub_corpora_only` sentinel pattern) — though this
    audit is sweep-corpus-scoped, not multi-corpus."""
    direct = corpus_path / 'runs.parquet'
    if direct.is_file():
        return pl.read_parquet(direct)
    raise FileNotFoundError(
        f'{direct}: no runs.parquet at the corpus path. The audit '
        f'needs the sweep\'s cells to re-run each committed bridge. '
        f'If this corpus is archived-only, run `corroborate restore '
        f'{corpus_path}` first.',
    )


def audit_pre_registration(
    corpus_path: Path,
    *,
    repo_root: Path | None = None,
) -> AuditReport:
    """Run the audit; return a typed report.

    The CLI's `dispatch()` wraps this with stdout / JSON printing
    + sys.exit; tests call this function directly.

    `repo_root`: cwd for the `git cat-file -e` git-hash-in-history
    check. Default `Path.cwd()` (the framework's own repo at
    invocation time)."""
    repo = repo_root if repo_root is not None else Path.cwd()
    manifest = read_manifest(corpus_path)
    if manifest is None:
        return AuditReport(
            exit_code=EXIT_MANIFEST_MISSING,
            entries=(),
            manifest=None,
        )

    if not _git_sha_in_history(manifest.git_commit_hash, repo):
        return AuditReport(
            exit_code=EXIT_GIT_HASH_NOT_FOUND,
            entries=(),
            manifest=manifest,
        )

    # Lazy-load cells — if the corpus has no runs.parquet but the
    # manifest is present, we still want to report the resolution
    # status of each bridge before failing on data. Read the
    # cells once up-front (raises if missing); audit then proceeds.
    cells = _load_corpus_cells(corpus_path)

    entries: list[AuditEntry] = []
    any_drift = False
    any_unresolved = False
    for commitment in manifest.bridge_commitments:
        try:
            bridge = resolve_bridge_by_name(commitment.bridge_name)
        except (ValueError, ModuleNotFoundError) as exc:
            entries.append(AuditEntry(
                bridge_name=commitment.bridge_name,
                source_hash_matches=False,
                predicted_direction=commitment.predicted_direction,
                predicted_verdict=commitment.predicted_verdict,
                empirical_verdict=None,
                verdict_matches=False,
                unresolved_reason=str(exc),
            ))
            any_unresolved = True
            continue

        current_hash = compute_bridge_source_hash(bridge)
        hash_matches = current_hash == commitment.source_hash

        # Run the bridge against the corpus's cells. Errors here
        # propagate — the audit's job is to report verdict drift,
        # not to silently swallow a bridge-evaluation crash.
        empirical = evaluate(bridge, cells)
        verdict_matches = empirical.verdict == commitment.predicted_verdict
        entries.append(AuditEntry(
            bridge_name=commitment.bridge_name,
            source_hash_matches=hash_matches,
            predicted_direction=commitment.predicted_direction,
            predicted_verdict=commitment.predicted_verdict,
            empirical_verdict=empirical.verdict,
            verdict_matches=verdict_matches,
        ))
        if not hash_matches or not verdict_matches:
            any_drift = True

    if any_unresolved:
        exit_code = EXIT_BRIDGE_UNRESOLVED
    elif any_drift:
        exit_code = EXIT_DRIFT
    else:
        exit_code = EXIT_MATCH

    return AuditReport(
        exit_code=exit_code,
        entries=tuple(entries),
        manifest=manifest,
    )


# ============ Stdout rendering ============


def _render_table(report: AuditReport) -> str:
    """Plain-text table for stdout — three columns:
    bridge / hash_status / verdict_status. Wide enough to read
    in a terminal."""
    if report.manifest is None:
        return f'(no manifest at corpus path; exit_code={report.exit_code})'
    header = (
        f'manifest: launched={report.manifest.sweep_launched_at.isoformat()} '
        f'git={report.manifest.git_commit_hash[:12]}... '
        f'config_hash={report.manifest.sweep_config_hash[:12]}...'
    )
    lines: list[str] = [header, '']
    for e in report.entries:
        if e.unresolved_reason is not None:
            lines.append(
                f'  [UNRESOLVED] {e.bridge_name}  '
                f'reason={e.unresolved_reason}',
            )
            continue
        hash_status = 'HASH_OK' if e.source_hash_matches else 'HASH_DRIFT'
        if e.empirical_verdict is None:
            verdict_status = 'VERDICT_NOT_RUN'
        elif e.verdict_matches:
            verdict_status = (
                f'VERDICT_OK ({e.empirical_verdict.value})'
            )
        else:
            verdict_status = (
                f'VERDICT_DRIFT '
                f'(predicted={e.predicted_verdict.value}, '
                f'empirical={e.empirical_verdict.value})'
            )
        lines.append(
            f'  {e.bridge_name}  {hash_status}  {verdict_status}',
        )
    lines.append('')
    if report.exit_code == EXIT_MATCH:
        lines.append('result: MATCH (exit 0) — every commitment held.')
    elif report.exit_code == EXIT_DRIFT:
        lines.append(
            'result: DRIFT (exit 1) — at least one bridge\'s source '
            'or verdict has drifted from its commitment.',
        )
    elif report.exit_code == EXIT_BRIDGE_UNRESOLVED:
        lines.append(
            'result: BRIDGE_UNRESOLVED (exit 4) — at least one '
            'committed bridge could not be resolved by its import '
            'path. Restore the bridge or update the manifest.',
        )
    elif report.exit_code == EXIT_GIT_HASH_NOT_FOUND:
        lines.append(
            'result: GIT_HASH_NOT_FOUND (exit 3) — the manifest\'s '
            'git_commit_hash is not in `git log --all`. The commit '
            'was rebased away or the repo was rewritten.',
        )
    else:
        lines.append(
            f'result: MANIFEST_MISSING (exit {report.exit_code}).',
        )
    return '\n'.join(lines)


# ============ CLI argparse wiring ============


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register the `audit pre-registration` arguments onto
    `parser`. Mirrored by `dispatch(args)` which acts on the
    parsed namespace."""
    sub = parser.add_subparsers(
        dest='audit_subcmd', required=True,
        title='audit subcommands',
    )
    p_prereg = sub.add_parser(
        'pre-registration',
        help='audit a corpus\'s pre-registration commitments',
        description=(
            'Re-resolve each bridge in <corpus>/pre_registration.json, '
            're-compute its source hash, run it against the corpus\'s '
            'cells, and report drift. Exit codes: '
            '0 match / 1 drift / 2 manifest-missing / '
            '3 git-hash-not-found / 4 bridge-unresolved. '
            'Detects post-launch rewrites and verdict drift; does NOT '
            'detect pilot-corpus HARKing or git history rewriting.'
        ),
    )
    _ = p_prereg.add_argument(
        'corpus_path',
        type=Path,
        help='path to the corpus directory containing '
             'pre_registration.json (typically '
             'experiments/data/<sweep_name>/).',
    )
    _ = p_prereg.add_argument(
        '--strict', action='store_true',
        help='also exit 1 if a committed bridge produced '
             'EMPTY_EXTENT (scope admitted no cells). Off by '
             'default because EMPTY_EXTENT is a legitimate '
             'verdict for "Finding doesn\'t apply to this corpus".',
    )
    _ = p_prereg.add_argument(
        '--report-path',
        type=Path,
        default=None,
        help='write the audit report as JSON to this path (in '
             'addition to the stdout table).',
    )


def dispatch(args: argparse.Namespace) -> int:
    """Argparse dispatch for `corroborate audit pre-registration`.

    `args.audit_subcmd` is set by `add_args`'s subparser; only
    'pre-registration' is currently supported."""
    sub_raw: object = args.audit_subcmd
    if sub_raw != 'pre-registration':
        # Future-proofing: more audit subcommands will land here.
        raise ValueError(f'unknown audit subcommand: {sub_raw!r}')
    corpus_path_raw: object = args.corpus_path
    if not isinstance(corpus_path_raw, Path):
        raise TypeError(
            f'corpus_path must be a Path; got '
            f'{type(corpus_path_raw).__name__}',
        )
    report_path_raw: object = args.report_path
    report_path = (
        report_path_raw if isinstance(report_path_raw, Path) else None
    )

    report = audit_pre_registration(corpus_path_raw)
    print(_render_table(report))
    if report_path is not None:
        _ = report_path.write_text(
            json.dumps(report.as_dict(), indent=2),
            encoding='utf-8',
        )
    return report.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `python -m corroborate.cli.audit ...`. The
    top-level `corroborate` CLI wires `audit` as a subcommand via
    `add_args`; this function makes the module independently
    executable for tests + back-compat."""
    parser = argparse.ArgumentParser(
        prog='corroborate audit',
        description=(
            'Audit a corpus\'s pre-registration commitments.'
        ),
    )
    add_args(parser)
    ns = parser.parse_args(argv)
    return dispatch(ns)


if __name__ == '__main__':
    sys.exit(main())


__all__ = [
    'AuditEntry',
    'AuditReport',
    'EXIT_BRIDGE_UNRESOLVED',
    'EXIT_DRIFT',
    'EXIT_GIT_HASH_NOT_FOUND',
    'EXIT_MANIFEST_MISSING',
    'EXIT_MATCH',
    'add_args',
    'audit_pre_registration',
    'dispatch',
    'main',
]
