"""Tests for the catalogue command — inventory of local + cloud corpora.

Uses fsspec's `file://` backend with per-test `tmp_path` so cloud
discovery is exercised without real network. Mirrors the structure
of `tests/test_cloud.py`.

Eight test shapes:
1. Classifier truth table over all 16 input cells.
2. Walk + classification of one fixture per reachable state.
3. `tmp/` pruning — per-arm shard dirs are not surfaced.
4. Local-only mode — no cloud queries fire, cloud is None.
5. Nested local corpus surfaces both parent and child.
6. Nested cloud orphan — round-trip via `cloud.archive`.
7. IN_PROGRESS_SCAFFOLD — sentinel-only dir surfaces with that status.
8. `runs_row_count` populates from a real `runs.parquet`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import polars as pl
import pytest

from corroborate.corpus import catalogue, cloud
from corroborate.corpus.integrity import IN_PROGRESS_SENTINEL


# ============ Fixtures ============

def _write_real_parquet(p: Path, n_rows: int = 4) -> None:
    """A polars-readable parquet with an `id` column. Padded with a
    string column so the file clears `cloud.archive`'s CI5 1KiB
    minimum even at small row counts."""
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        'id': [f'cell-{i}' for i in range(n_rows)],
        'x': list(range(n_rows)),
        # pad column — ensures size >= 1 KiB so cloud.archive accepts.
        'pad': ['x' * 256 for _ in range(max(n_rows, 4))][:n_rows],
    })
    df.write_parquet(p)


def _write_parquet_no_id(p: Path, n_rows: int = 3) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({'x': list(range(n_rows))})
    df.write_parquet(p)


def _write_local_manifest(corpus_dir: Path, remote_root: str) -> None:
    """Hand-write a `_remote.json` with no actual files. Used when
    we need a corpus to look "manifested" without doing a full
    archive round-trip."""
    import json
    payload = {
        'remote_root': remote_root,
        'files': [],
    }
    _ = (corpus_dir / cloud.MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2),
    )


# ============ 1. Classifier truth table ============

_ALL_NAMED: Final = (
    'CLOUD_AND_LOCAL', 'CLOUD_EVICTED', 'STALE_MANIFEST',
    'LOCAL_ONLY', 'CLOUD_ORPHAN', 'LINKAGE_LOST',
    'IN_PROGRESS_SCAFFOLD',
)


@pytest.mark.parametrize(
    ('local_dir', 'local_manifest', 'parquets', 'cloud_manifest', 'expected'),
    [
        # local_dir=F: cells where local_manifest or parquets exist are
        # UNREACHABLE (filesystem ontology); cell with neither is
        # UPSTREAM-FILTERED. Only (F,F,F,T) is reachable → CLOUD_ORPHAN.
        (False, False, False, False, 'RAISE'),
        (False, False, False, True,  'CLOUD_ORPHAN'),
        (False, False, True,  False, 'RAISE'),
        (False, False, True,  True,  'RAISE'),
        (False, True,  False, False, 'RAISE'),
        (False, True,  False, True,  'RAISE'),
        (False, True,  True,  False, 'RAISE'),
        (False, True,  True,  True,  'RAISE'),
        # local_dir=T: 8 reachable cells.
        (True,  False, False, False, 'IN_PROGRESS_SCAFFOLD'),
        (True,  False, False, True,  'CLOUD_ORPHAN'),
        (True,  False, True,  False, 'LOCAL_ONLY'),
        (True,  False, True,  True,  'LINKAGE_LOST'),
        (True,  True,  False, False, 'STALE_MANIFEST'),
        (True,  True,  False, True,  'CLOUD_EVICTED'),
        (True,  True,  True,  False, 'STALE_MANIFEST'),
        (True,  True,  True,  True,  'CLOUD_AND_LOCAL'),
    ],
)
def test_classifier_truth_table(
    local_dir: bool,
    local_manifest: bool,
    parquets: bool,
    cloud_manifest: bool,
    expected: str,
) -> None:
    if expected == 'RAISE':
        with pytest.raises(AssertionError):
            _ = catalogue._classify(  # pyright: ignore[reportPrivateUsage]
                has_local_dir=local_dir,
                has_local_manifest=local_manifest,
                has_local_parquets=parquets,
                has_cloud_manifest=cloud_manifest,
            )
    else:
        assert expected in _ALL_NAMED  # sanity on the parametrize itself
        got = catalogue._classify(  # pyright: ignore[reportPrivateUsage]
            has_local_dir=local_dir,
            has_local_manifest=local_manifest,
            has_local_parquets=parquets,
            has_cloud_manifest=cloud_manifest,
        )
        assert got == expected


# ============ 2. Walk + classification per reachable state ============

def test_walk_one_fixture_per_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / 'data'
    cloud_root = tmp_path / 'cloud'

    # CLOUD_AND_LOCAL: real archive round-trip then keep local parquets.
    cal_dir = data_root / 'cal_corpus'
    _write_real_parquet(cal_dir / 'runs.parquet', n_rows=5)
    cal_remote = f'file://{cloud_root / "cal_corpus"}'
    _ = cloud.archive(cal_dir, cal_remote)

    # CLOUD_EVICTED: archive then remove local parquets.
    cev_dir = data_root / 'cev_corpus'
    _write_real_parquet(cev_dir / 'runs.parquet', n_rows=7)
    cev_remote = f'file://{cloud_root / "cev_corpus"}'
    _ = cloud.archive(cev_dir, cev_remote)
    (cev_dir / 'runs.parquet').unlink()

    # STALE_MANIFEST: local manifest pointing at a cloud root that doesn't exist.
    sm_dir = data_root / 'sm_corpus'
    _write_real_parquet(sm_dir / 'runs.parquet', n_rows=3)
    _write_local_manifest(sm_dir, f'file://{cloud_root / "_nonexistent_sm"}')

    # LOCAL_ONLY: parquets, no manifest, no cloud match.
    lo_dir = data_root / 'lo_corpus'
    _write_real_parquet(lo_dir / 'runs.parquet', n_rows=2)

    # CLOUD_ORPHAN: archive then delete the local dir entirely.
    co_dir = data_root / 'co_corpus'
    _write_real_parquet(co_dir / 'runs.parquet', n_rows=4)
    co_remote = f'file://{cloud_root / "co_corpus"}'
    _ = cloud.archive(co_dir, co_remote)
    (co_dir / 'runs.parquet').unlink()
    (co_dir / cloud.MANIFEST_NAME).unlink()
    co_dir.rmdir()

    # LINKAGE_LOST: parquets on disk + cloud archive at the same name,
    # but no local manifest linking them.
    ll_dir = data_root / 'll_corpus'
    _write_real_parquet(ll_dir / 'runs.parquet', n_rows=6)
    _ = cloud.archive(ll_dir, f'file://{cloud_root / "ll_corpus"}')
    (ll_dir / cloud.MANIFEST_NAME).unlink()  # drop the manifest

    # IN_PROGRESS_SCAFFOLD: only the sentinel.
    ips_dir = data_root / 'ips_corpus'
    ips_dir.mkdir()
    _ = (ips_dir / IN_PROGRESS_SENTINEL).write_text('')

    rows = catalogue.catalogue(
        data_root,
        remote_prefix=f'file://{cloud_root}',
    )

    by_name = {r.name: r for r in rows}
    assert by_name['cal_corpus'].status == 'CLOUD_AND_LOCAL'
    assert by_name['cev_corpus'].status == 'CLOUD_EVICTED'
    assert by_name['sm_corpus'].status == 'STALE_MANIFEST'
    assert by_name['lo_corpus'].status == 'LOCAL_ONLY'
    assert by_name['co_corpus'].status == 'CLOUD_ORPHAN'
    assert by_name['ll_corpus'].status == 'LINKAGE_LOST'
    assert by_name['ips_corpus'].status == 'IN_PROGRESS_SCAFFOLD'
    assert by_name['ips_corpus'].in_progress is True


# ============ 3. tmp/ pruning ============

def test_tmp_dir_pruned(tmp_path: Path) -> None:
    data_root = tmp_path / 'data'
    corpus = data_root / 'a_corpus'
    _write_real_parquet(corpus / 'runs.parquet', n_rows=2)
    # per-arm shard inside tmp/<arm>/runs.parquet
    _write_real_parquet(corpus / 'tmp' / 'arm0' / 'runs.parquet', n_rows=1)

    rows = catalogue.catalogue(data_root, remote_prefix=None)
    names = {(r.parent, r.name) for r in rows}

    assert ('', 'a_corpus') in names
    assert all('tmp' not in (r.parent, r.name) for r in rows)
    assert all('arm0' not in (r.parent, r.name) for r in rows)


# ============ 4. Local-only mode ============

def test_local_only_mode_no_cloud_queries(tmp_path: Path) -> None:
    data_root = tmp_path / 'data'

    # Has manifest, parquets — STALE_MANIFEST in offline mode
    # (cloud not queried; manifest's mirror state unknown).
    a = data_root / 'a'
    _write_real_parquet(a / 'runs.parquet', n_rows=3)
    _write_local_manifest(a, 'file:///not/queried')

    # No manifest, parquets — LOCAL_ONLY.
    b = data_root / 'b'
    _write_real_parquet(b / 'runs.parquet', n_rows=4)

    # Only sentinel — IN_PROGRESS_SCAFFOLD.
    c = data_root / 'c'
    c.mkdir()
    _ = (c / IN_PROGRESS_SENTINEL).write_text('')

    rows = catalogue.catalogue(data_root, remote_prefix=None)

    for r in rows:
        assert r.cloud is None
        assert r.status in {'LOCAL_ONLY', 'STALE_MANIFEST',
                            'IN_PROGRESS_SCAFFOLD'}

    by_name = {r.name: r for r in rows}
    assert by_name['a'].status == 'STALE_MANIFEST'
    assert by_name['b'].status == 'LOCAL_ONLY'
    assert by_name['c'].status == 'IN_PROGRESS_SCAFFOLD'


# ============ 5. Nested local ============

def test_nested_local_surfaces_both(tmp_path: Path) -> None:
    data_root = tmp_path / 'data'
    parent = data_root / 'parent'
    child = parent / 'child'
    _write_real_parquet(parent / 'runs.parquet', n_rows=2)
    _write_real_parquet(child / 'runs.parquet', n_rows=3)

    rows = catalogue.catalogue(data_root, remote_prefix=None)
    addrs = {(r.parent, r.name) for r in rows}

    assert ('', 'parent') in addrs
    assert ('parent', 'child') in addrs


# ============ 6. Nested cloud orphan ============

def test_nested_cloud_orphan_surfaces(tmp_path: Path) -> None:
    data_root = tmp_path / 'data'
    cloud_root = tmp_path / 'cloud'

    # Archive a nested corpus, then delete the local dir.
    nested = data_root / 'parent' / 'child'
    _write_real_parquet(nested / 'runs.parquet', n_rows=2)
    remote = f'file://{cloud_root / "parent" / "child"}'
    _ = cloud.archive(nested, remote)

    # Tear down the nested local dir but keep the parent.
    (nested / 'runs.parquet').unlink()
    (nested / cloud.MANIFEST_NAME).unlink()
    nested.rmdir()
    # `parent` itself stays — possibly empty.

    rows = catalogue.catalogue(
        data_root,
        remote_prefix=f'file://{cloud_root}',
    )
    orphans = [r for r in rows
               if r.parent == 'parent' and r.name == 'child']
    assert len(orphans) == 1
    assert orphans[0].status == 'CLOUD_ORPHAN'


# ============ 7. IN_PROGRESS_SCAFFOLD ============

def test_in_progress_scaffold(tmp_path: Path) -> None:
    data_root = tmp_path / 'data'
    scaffold = data_root / 'crashed_sweep'
    scaffold.mkdir(parents=True)
    _ = (scaffold / IN_PROGRESS_SENTINEL).write_text('')

    rows = catalogue.catalogue(data_root, remote_prefix=None)
    by_name = {r.name: r for r in rows}

    assert 'crashed_sweep' in by_name
    r = by_name['crashed_sweep']
    assert r.status == 'IN_PROGRESS_SCAFFOLD'
    assert r.in_progress is True
    assert r.local is not None
    assert r.local.parquet_count == 0
    assert r.local.has_manifest is False
    assert r.local.runs_row_count is None


# ============ 9. Multi-root: shadowing ============

def test_multi_root_shadows_orphan(tmp_path: Path) -> None:
    """A corpus archived from root B but absent from root A:
    walking only A reports CLOUD_ORPHAN; walking both A + B
    shadows the orphan because B carries the local manifest."""
    root_a = tmp_path / 'data'
    root_b = tmp_path / 'probes'
    cloud_root = tmp_path / 'cloud'

    # Corpus lives under root_b, archived to cloud.
    corpus_b = root_b / 'pilot_foo'
    _write_real_parquet(corpus_b / 'runs.parquet', n_rows=3)
    _ = cloud.archive(
        corpus_b, f'file://{cloud_root / "pilot_foo"}',
    )

    # Walking only root_a → cloud orphan reported.
    rows_a = catalogue.catalogue(
        root_a, remote_prefix=f'file://{cloud_root}',
    )
    orphans_a = [r for r in rows_a if r.status == 'CLOUD_ORPHAN']
    assert any(r.name == 'pilot_foo' for r in orphans_a)

    # Walking [root_a, root_b] → corpus has local match in B,
    # so no orphan reported for `pilot_foo`.
    rows_both = catalogue.catalogue(
        [root_a, root_b], remote_prefix=f'file://{cloud_root}',
    )
    orphans_both = [r for r in rows_both if r.status == 'CLOUD_ORPHAN']
    assert not any(r.name == 'pilot_foo' for r in orphans_both)
    cal = [r for r in rows_both if r.name == 'pilot_foo']
    assert len(cal) == 1
    assert cal[0].status == 'CLOUD_AND_LOCAL'


# ============ 10. Multi-root: additive ============

def test_multi_root_additive(tmp_path: Path) -> None:
    """Distinct corpora under two roots both surface, once each."""
    root_a = tmp_path / 'data'
    root_b = tmp_path / 'probes'
    _write_real_parquet(root_a / 'cal_a' / 'runs.parquet', n_rows=2)
    _write_real_parquet(root_b / 'pilot_b' / 'runs.parquet', n_rows=2)

    rows = catalogue.catalogue([root_a, root_b], remote_prefix=None)
    names = [r.name for r in rows]
    assert names.count('cal_a') == 1
    assert names.count('pilot_b') == 1
    assert all(r.status == 'LOCAL_ONLY' for r in rows
               if r.name in {'cal_a', 'pilot_b'})


# ============ 8. runs_row_count populates ============

def test_runs_row_count_from_parquet(tmp_path: Path) -> None:
    data_root = tmp_path / 'data'

    # Has runs.parquet with `id` column → row count = 7.
    a = data_root / 'with_id'
    _write_real_parquet(a / 'runs.parquet', n_rows=7)

    # Has runs.parquet without `id` column → runs_row_count is None
    # (sniff_row_ids returns () for both absent column and zero rows;
    # the catalogue can't distinguish, mirrors sniff_row_ids semantics:
    # len(()) == 0).
    b = data_root / 'no_id_col'
    _write_parquet_no_id(b / 'runs.parquet', n_rows=3)

    # No runs.parquet (only traces.parquet) → runs_row_count is None
    # (file absent).
    c = data_root / 'no_runs_parquet'
    _write_real_parquet(c / 'traces.parquet', n_rows=5)

    rows = catalogue.catalogue(data_root, remote_prefix=None)
    by_name = {r.name: r for r in rows}

    assert by_name['with_id'].local is not None
    assert by_name['with_id'].local.runs_row_count == 7

    assert by_name['no_id_col'].local is not None
    assert by_name['no_id_col'].local.runs_row_count == 0  # column absent

    assert by_name['no_runs_parquet'].local is not None
    assert by_name['no_runs_parquet'].local.runs_row_count is None
