"""Pre-registration manifest — write/read/audit round-trips.

Five tests, in two groups:

1-4: pure write/read + source-hash round-trips. Don't touch the
runner or the CLI; live in the core module's contract.

5: end-to-end audit on a tiny LG-SCM corpus. Drives the runner's
sweep launch (with `pre_registered_bridges`) and the
`corroborate audit pre-registration` CLI surface."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Importing analyses populates the registry.
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.bridge.bridge import (
    Direction, Tier, claim_bridge,
)
from corroborate.bridge.verdict import Verdict
from corroborate.core.intervention import DoEffect, Intervention
from corroborate.core.claim import claim
from corroborate.core.pre_registration import (
    BridgeCommitment,
    PreRegistrationManifest,
    SCHEMA_VERSION,
    compute_bridge_source_hash,
    read_manifest,
    resolve_bridge_by_name,
    write_manifest,
)


# ============ Shared fixtures for tests 1-4 ============


@claim
def _t_op_pre_reg(x: int) -> int:
    return x


@claim
def _b_op_pre_reg(x: int) -> int:
    return x


_INTERVENTION = DoEffect(arms=(
    (Intervention(slot_path='op', replacement=_b_op_pre_reg),),
    (Intervention(slot_path='op', replacement=_t_op_pre_reg),),
))


# Two bridges differ only in a numeric literal in the body — the
# hash must flip between them. Both live at module scope so they
# have a fully-qualified import path (`tests.test_pre_registration.
# bridge_floor_a` / `..._b`); `resolve_bridge_by_name` finds them
# through standard `importlib.import_module` + `getattr`.

@claim_bridge(
    source=_INTERVENTION,
    target='outcome',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    predicted_direction='a_gt_b',
)
def bridge_floor_a(
    *,
    treatment_arm: str = '',
    baseline_arm: str = '',
) -> Verdict:
    """Synthetic bridge — body has a numeric literal `harm_floor=0.3`
    that test 3 mutates to `0.5` in a sibling bridge."""
    del treatment_arm, baseline_arm
    harm_floor = 0.3
    if harm_floor < 0.5:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source=_INTERVENTION,
    target='outcome',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    predicted_direction='a_gt_b',
)
def bridge_floor_b(
    *,
    treatment_arm: str = '',
    baseline_arm: str = '',
) -> Verdict:
    """Sibling of `bridge_floor_a` differing only in `harm_floor`'s
    numeric literal (`0.5` vs `0.3`) — source hash must flip."""
    del treatment_arm, baseline_arm
    harm_floor = 0.5
    if harm_floor < 0.5:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# ============ Test 1: round trip ============


def test_round_trip_manifest() -> None:
    """Build a 1-bridge manifest, write to tmp, read back, assert
    dataclass equality."""
    bridge = resolve_bridge_by_name(
        'tests.test_pre_registration.bridge_floor_a',
    )
    h = compute_bridge_source_hash(bridge)
    manifest = PreRegistrationManifest(
        sweep_launched_at=datetime(2026, 5, 18, tzinfo=UTC),
        git_commit_hash='0' * 40,
        sweep_config_hash='a' * 64,
        bridge_commitments=(
            BridgeCommitment(
                bridge_name='tests.test_pre_registration.bridge_floor_a',
                source_hash=h,
                predicted_direction='a_gt_b',
                predicted_verdict=Verdict.HELD,
            ),
        ),
    )

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = Path(tmp)
        written = write_manifest(corpus_dir, manifest)
        assert written.exists()

        # The on-disk JSON carries the canonical schema_version.
        raw = json.loads(written.read_text(encoding='utf-8'))
        assert isinstance(raw, dict)
        assert raw.get('schema_version') == SCHEMA_VERSION

        read_back = read_manifest(corpus_dir)
        assert read_back == manifest


def test_round_trip_refuses_overwrite() -> None:
    """Manifest is immutable per spec §5 — a second `write_manifest`
    call at the same path raises FileExistsError rather than
    overwriting."""
    bridge = resolve_bridge_by_name(
        'tests.test_pre_registration.bridge_floor_a',
    )
    manifest = PreRegistrationManifest(
        sweep_launched_at=datetime(2026, 5, 18, tzinfo=UTC),
        git_commit_hash='0' * 40,
        sweep_config_hash='a' * 64,
        bridge_commitments=(
            BridgeCommitment(
                bridge_name='tests.test_pre_registration.bridge_floor_a',
                source_hash=compute_bridge_source_hash(bridge),
                predicted_direction='a_gt_b',
                predicted_verdict=Verdict.HELD,
            ),
        ),
    )
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = Path(tmp)
        _ = write_manifest(corpus_dir, manifest)
        with pytest.raises(FileExistsError, match='immutable'):
            _ = write_manifest(corpus_dir, manifest)


# ============ Test 2: source_hash idempotent on unchanged source ============


def test_source_hash_unchanged_is_idempotent() -> None:
    """Compute the hash twice on the same bridge — must match.

    Captures the simplest invariant of any hash function plus the
    AST-canonicalisation: the second-call hash is computed via the
    same path as the first (same `inspect.getsource` →
    `ast.parse` → `ast.dump` → sha256), so any path-dependence
    bug surfaces here before semantic drift tests come into
    play."""
    bridge = resolve_bridge_by_name(
        'tests.test_pre_registration.bridge_floor_a',
    )
    h1 = compute_bridge_source_hash(bridge)
    h2 = compute_bridge_source_hash(bridge)
    assert h1 == h2


# ============ Test 3: source_hash flips on a semantic change ============


def test_source_hash_flips_on_semantic_change() -> None:
    """`bridge_floor_a` and `bridge_floor_b` differ only in a
    numeric literal (`harm_floor=0.3` vs `0.5`). Hashes must
    differ — the AST-of-source path catches the literal change."""
    bridge_a = resolve_bridge_by_name(
        'tests.test_pre_registration.bridge_floor_a',
    )
    bridge_b = resolve_bridge_by_name(
        'tests.test_pre_registration.bridge_floor_b',
    )
    h_a = compute_bridge_source_hash(bridge_a)
    h_b = compute_bridge_source_hash(bridge_b)
    assert h_a != h_b, (
        f'expected distinct hashes for bridges with different '
        f'`harm_floor` literals; got {h_a[:12]}... == {h_b[:12]}...'
    )


# ============ Test 4: source_hash stable across whitespace / comments ============


def test_source_hash_stable_across_cosmetic_changes() -> None:
    """Two bridges semantically identical but differing in
    whitespace, comments, and docstring text must hash
    identically.

    The contract of the algorithm (spec §3) is precisely that
    black/ruff reformats AND docstring edits don't bust the hash
    — only semantic body changes do. The AST-of-source step
    strips whitespace + comments; the framework's
    `_strip_docstrings` walk drops the leading docstring node
    from every function/class/module in the parsed tree."""
    import ast
    import hashlib
    import json
    from corroborate.core.pre_registration import (
        _strip_docstrings,  # pyright: ignore[reportPrivateUsage]
    )

    # Construct two source strings whose bodies are semantically
    # identical but differ in: leading blank line, docstring
    # text, comment placement, trailing whitespace. The semantic
    # body (`x = 1; return x`) is byte-identical.
    src_a = (
        'def fn():\n'
        '    """doc string a"""\n'
        '    x = 1\n'
        '    return x\n'
    )
    src_b = (
        '\n'
        'def fn():\n'
        '\n'
        '    """doc string b — completely different prose."""\n'
        '    # explanatory comment\n'
        '    x = 1  # trailing comment\n'
        '\n'
        '    return x\n'
    )

    def _hash(src: str) -> str:
        tree = ast.parse(src)
        _strip_docstrings(tree)
        ast_repr = ast.dump(
            tree, annotate_fields=True, include_attributes=False,
        )
        payload = ast_repr + '\n' + json.dumps({}, sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    assert _hash(src_a) == _hash(src_b), (
        'expected docstring + whitespace + comment edits to be '
        'canonicalised away by ast.dump+_strip_docstrings; '
        'they were not. This is the empirical contract of '
        '`compute_bridge_source_hash`.'
    )

    # Sanity check the contract's other half: a SEMANTIC change
    # (different literal) must flip the hash via the same path.
    src_c = (
        'def fn():\n'
        '    """doc string a"""\n'
        '    x = 2\n'  # ← literal changed from 1 to 2
        '    return x\n'
    )
    assert _hash(src_a) != _hash(src_c), (
        'expected a literal change (1 → 2) to flip the hash; '
        'it did not — the AST canonicalisation is too aggressive.'
    )


# ============ Test 5: end-to-end audit on a tiny corpus ============


def _make_audit_corpus(corpus_dir: Path) -> None:
    """Build a tiny corpus (synthetic cells in runs.parquet) that
    `bridge_floor_a` can run against. Per spec §8 test 5: we want
    the audit's exit code 0 path, with the bridge producing the
    predicted verdict (HELD)."""
    import polars as pl
    treatment_key = _INTERVENTION.arm_keys()[1]
    baseline_key = _INTERVENTION.arm_keys()[0]
    cells: list[dict[str, object]] = []
    for s in range(30):
        cells.append({
            'id': f'cell-t-{s}',
            'arm_key': treatment_key,
            'seed': s,
            'env_name': 'LGSCM',
            'outcome': 1.0 + 0.01 * s,
        })
        cells.append({
            'id': f'cell-b-{s}',
            'arm_key': baseline_key,
            'seed': s,
            'env_name': 'LGSCM',
            'outcome': 0.0 + 0.01 * s,
        })
    pl.DataFrame(cells).write_parquet(corpus_dir / 'runs.parquet')


def test_audit_end_to_end_fixture_corpus(tmp_path: Path) -> None:
    """End-to-end: build a fixture corpus, write the manifest via
    the public write_manifest API, run `audit_pre_registration`,
    assert exit code 0 + report shows verdict matches.

    Walks the runner-commit's write helper and the CLI-commit's
    audit verifier in one pass. The synthetic corpus has 30
    seeds × 2 arms = 60 cells; the bridge body returns HELD
    unconditionally on this fixture (its scope is implicitly the
    full corpus). The author committed `predicted_verdict=HELD`,
    so the audit must exit 0."""
    import subprocess
    from corroborate.cli.audit import (
        EXIT_BRIDGE_UNRESOLVED,
        EXIT_DRIFT,
        EXIT_GIT_HASH_NOT_FOUND,
        EXIT_MANIFEST_MISSING,
        EXIT_MATCH,
        audit_pre_registration,
    )
    from corroborate.core.pre_registration import (
        BridgeCommitmentInput, build_commitments,
    )

    # Pin all 5 exit codes — a future refactor that silently
    # renumbers them would break the contract that scripts /
    # operators depend on. The values are part of the CLI's
    # public surface.
    assert EXIT_MATCH == 0
    assert EXIT_DRIFT == 1
    assert EXIT_MANIFEST_MISSING == 2
    assert EXIT_GIT_HASH_NOT_FOUND == 3
    assert EXIT_BRIDGE_UNRESOLVED == 4

    corpus_dir = tmp_path / 'fixture_corpus'
    corpus_dir.mkdir()
    _make_audit_corpus(corpus_dir)

    commitments = build_commitments((
        BridgeCommitmentInput(
            bridge_name='tests.test_pre_registration.bridge_floor_a',
            predicted_direction='a_gt_b',
            predicted_verdict=Verdict.HELD,
        ),
    ))
    head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    manifest = PreRegistrationManifest(
        sweep_launched_at=datetime(2026, 5, 18, tzinfo=UTC),
        git_commit_hash=head,
        sweep_config_hash='deadbeef' * 8,
        bridge_commitments=commitments,
    )
    _ = write_manifest(corpus_dir, manifest)

    report = audit_pre_registration(corpus_dir)
    assert report.exit_code == EXIT_MATCH, (
        f'expected MATCH (0); got {report.exit_code}; '
        f'entries: {report.entries!r}'
    )
    assert len(report.entries) == 1
    entry = report.entries[0]
    assert entry.bridge_name == (
        'tests.test_pre_registration.bridge_floor_a'
    )
    assert entry.source_hash_matches is True
    assert entry.empirical_verdict == Verdict.HELD
    assert entry.predicted_verdict == Verdict.HELD
    assert entry.verdict_matches is True

    # Adjacent: missing-manifest path → exit code 2.
    other_corpus = tmp_path / 'empty_corpus'
    other_corpus.mkdir()
    _make_audit_corpus(other_corpus)
    report_missing = audit_pre_registration(other_corpus)
    assert report_missing.exit_code == EXIT_MANIFEST_MISSING


def test_audit_drift_on_source_hash_change(tmp_path: Path) -> None:
    """Source-hash drift → exit code 1. Manually commit a hash
    that doesn't match the current source; the audit must flag
    the mismatch."""
    from corroborate.cli.audit import (
        EXIT_BRIDGE_UNRESOLVED, EXIT_DRIFT, audit_pre_registration,
    )

    corpus_dir = tmp_path / 'drifting_corpus'
    corpus_dir.mkdir()
    _make_audit_corpus(corpus_dir)

    # Fabricate a manifest with a deliberately-wrong source_hash.
    head = (
        __import__('subprocess')
        .run(['git', 'rev-parse', 'HEAD'],
             check=True, capture_output=True, text=True)
        .stdout.strip()
    )
    manifest = PreRegistrationManifest(
        sweep_launched_at=datetime(2026, 5, 18, tzinfo=UTC),
        git_commit_hash=head,
        sweep_config_hash='deadbeef' * 8,
        bridge_commitments=(
            BridgeCommitment(
                bridge_name='tests.test_pre_registration.bridge_floor_a',
                # Wrong hash — the audit must flag the mismatch.
                source_hash='not-the-real-hash-' + 'x' * 46,
                predicted_direction='a_gt_b',
                predicted_verdict=Verdict.HELD,
            ),
        ),
    )
    _ = write_manifest(corpus_dir, manifest)

    report = audit_pre_registration(corpus_dir)
    # source-hash drift always lands on EXIT_DRIFT (1), not
    # EXIT_BRIDGE_UNRESOLVED (4) — the bridge resolved fine; only
    # the hash differs.
    assert report.exit_code == EXIT_DRIFT, (
        f'expected DRIFT (1); got {report.exit_code}; '
        f'entries: {report.entries!r}'
    )
    assert report.exit_code != EXIT_BRIDGE_UNRESOLVED
    entry = report.entries[0]
    assert entry.source_hash_matches is False
    # The empirical verdict is still HELD (the body is unchanged),
    # so verdict_matches stays True — only the source hash drifted.
    assert entry.verdict_matches is True


def test_audit_drift_on_verdict_mismatch(tmp_path: Path) -> None:
    """Verdict drift → exit code 1. Commit a predicted_verdict
    that doesn't match what the bridge produces; the audit must
    flag the mismatch."""
    from corroborate.cli.audit import EXIT_DRIFT, audit_pre_registration
    from corroborate.core.pre_registration import build_commitments
    from corroborate.core.pre_registration import BridgeCommitmentInput

    corpus_dir = tmp_path / 'verdict_drift_corpus'
    corpus_dir.mkdir()
    _make_audit_corpus(corpus_dir)

    commitments = build_commitments((
        BridgeCommitmentInput(
            bridge_name='tests.test_pre_registration.bridge_floor_a',
            predicted_direction='a_gt_b',
            # bridge_floor_a always returns HELD on this corpus,
            # so committing NO_EFFECT guarantees a verdict drift.
            predicted_verdict=Verdict.NO_EFFECT,
        ),
    ))
    head = (
        __import__('subprocess')
        .run(['git', 'rev-parse', 'HEAD'],
             check=True, capture_output=True, text=True)
        .stdout.strip()
    )
    manifest = PreRegistrationManifest(
        sweep_launched_at=datetime(2026, 5, 18, tzinfo=UTC),
        git_commit_hash=head,
        sweep_config_hash='deadbeef' * 8,
        bridge_commitments=commitments,
    )
    _ = write_manifest(corpus_dir, manifest)

    report = audit_pre_registration(corpus_dir)
    assert report.exit_code == EXIT_DRIFT
    entry = report.entries[0]
    # Source hash IS correct (we ran through build_commitments
    # against the real bridge); only the verdict differs.
    assert entry.source_hash_matches is True
    assert entry.verdict_matches is False
    assert entry.empirical_verdict == Verdict.HELD
    assert entry.predicted_verdict == Verdict.NO_EFFECT
