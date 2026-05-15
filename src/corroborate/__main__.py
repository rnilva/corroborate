"""Cloud archive CLI — `python -m corroborate <subcommand>`.

Subcommands:

  archive    upload sweep parquets to remote storage
  restore    download archived parquets from remote
  ls         show what is archived for a sweep
  purge      delete LOCAL copies of files in the manifest
  catalogue  inventory all corpora under a data root (local + cloud)

The first four operate on one sweep directory. The remote root
is pinned in the per-sweep manifest after the first `archive`;
later `restore`/`ls`/`purge` read it from there. `catalogue`
walks a data root and cross-references with a cloud prefix.

Python API mirror lives in `corroborate.cloud` and
`corroborate.corpus.catalogue`."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from corroborate.corpus import catalogue as _catalogue
from corroborate.corpus import cloud
from corroborate._internals.argparse import to_mapping
from corroborate._internals.narrow import (
    optional_str,
    optional_str_list,
    require_bool,
    require_str,
)


def _cmd_archive(args: Mapping[str, object]) -> int:
    sweep_dir = Path(require_str(args, 'sweep_dir'))
    remote = require_str(args, 'remote')
    files = optional_str_list(args, 'files')
    force = require_bool(args, 'force')
    purge_local = require_bool(args, 'purge_local')

    manifest = cloud.archive(
        sweep_dir, remote,
        files=files, force=force, purge_local=purge_local,
    )
    total = sum(f.size_bytes for f in manifest.files)
    print(
        f'archived {len(manifest.files)} files '
        f'({total / 1e9:.2f} GB) to {manifest.remote_root}',
    )
    if purge_local:
        print('  local copies deleted')
    return 0


def _cmd_restore(args: Mapping[str, object]) -> int:
    sweep_dir = Path(require_str(args, 'sweep_dir'))
    files = optional_str_list(args, 'files')
    overwrite = require_bool(args, 'overwrite')

    restored = cloud.restore(sweep_dir, files=files, overwrite=overwrite)
    print(f'restored {len(restored)} files to {sweep_dir}')
    for r in restored:
        print(f'  {r}')
    return 0


def _cmd_ls(args: Mapping[str, object]) -> int:
    sweep_dir = Path(require_str(args, 'sweep_dir'))
    m = cloud.ls(sweep_dir)
    print(f'remote: {m.remote_root}')
    print(f'files: {len(m.files)}')
    for f in m.files:
        gb = f.size_bytes / 1e9
        print(
            f'  {f.relpath}  {gb:.2f} GB  '
            f'sha256={f.sha256[:12]}…  pushed={f.pushed_at}',
        )
    total = sum(f.size_bytes for f in m.files)
    print(f'total: {total / 1e9:.2f} GB')
    return 0


def _cmd_purge(args: Mapping[str, object]) -> int:
    sweep_dir = Path(require_str(args, 'sweep_dir'))
    files = optional_str_list(args, 'files')
    deleted = cloud.purge(sweep_dir, files=files)
    print(f'deleted {len(deleted)} local files in {sweep_dir}')
    for r in deleted:
        print(f'  {r}')
    return 0


def _cmd_catalogue(args: Mapping[str, object]) -> int:
    data_root = Path(require_str(args, 'data_root')).resolve()
    cli_prefix = optional_str(args, 'remote_prefix')
    local_only = require_bool(args, 'local_only')
    include_misc = require_bool(args, 'include_misc')
    as_json = require_bool(args, 'json_output')

    if local_only:
        remote_prefix: str | None = None
    elif cli_prefix is not None:
        remote_prefix = cli_prefix
    else:
        env_prefix = os.environ.get('CORROBORATE_REMOTE_PREFIX')
        remote_prefix = env_prefix if env_prefix else None

    rows = _catalogue.catalogue(
        data_root,
        remote_prefix=remote_prefix,
        include_misc=include_misc,
    )
    if as_json:
        payload = [dataclasses.asdict(r) for r in rows]
        print(json.dumps(payload, default=str, indent=2))
    else:
        df = _catalogue.to_polars(rows)
        print(df)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m corroborate',
        description='cloud archive for sweep parquets',
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_archive = sub.add_parser(
        'archive',
        help='upload sweep parquets to remote storage',
    )
    _ = p_archive.add_argument('sweep_dir', help='path to sweep directory')
    _ = p_archive.add_argument(
        '--remote', required=True,
        help='fsspec URI prefix (e.g. s3://bucket/sweeps/ddqn). '
             'After the first archive this is pinned in the sweep '
             'manifest.',
    )
    _ = p_archive.add_argument(
        '--files', nargs='*', default=None,
        help='relpaths within sweep_dir to archive (default: '
             'top-level *.parquet, excludes tmp/ shards).',
    )
    _ = p_archive.add_argument(
        '--force', action='store_true',
        help='re-upload files already in the manifest with '
             'matching sha256.',
    )
    _ = p_archive.add_argument(
        '--purge-local', action='store_true',
        help='delete local copies after successful upload + size '
             'verification. Default is the safer two-step lifecycle '
             '(archive → verify → purge).',
    )

    p_restore = sub.add_parser(
        'restore',
        help='download archived parquets from remote',
    )
    _ = p_restore.add_argument('sweep_dir')
    _ = p_restore.add_argument(
        '--files', nargs='*', default=None,
        help='relpaths to restore (default: all in manifest).',
    )
    _ = p_restore.add_argument(
        '--overwrite', action='store_true',
        help='replace local files with mismatched sha256. Default '
             'is to raise instead, surfacing drift before clobbering.',
    )

    p_ls = sub.add_parser('ls', help='show what is archived for a sweep')
    _ = p_ls.add_argument('sweep_dir')

    p_purge = sub.add_parser(
        'purge',
        help='delete LOCAL copies of files in the manifest. Manifest '
             'is preserved so restore stays available.',
    )
    _ = p_purge.add_argument('sweep_dir')
    _ = p_purge.add_argument('--files', nargs='*', default=None)

    p_cat = sub.add_parser(
        'catalogue',
        help='inventory all corpora under a data root, cross-'
             'referenced against cloud archives under a remote prefix.',
    )
    _ = p_cat.add_argument(
        'data_root',
        help='directory containing per-sweep corpus dirs '
             '(e.g. experiments/data).',
    )
    cat_remote = p_cat.add_mutually_exclusive_group()
    _ = cat_remote.add_argument(
        '--remote-prefix', dest='remote_prefix', default=None,
        help='fsspec URI prefix for cloud discovery '
             '(e.g. s3://corroborate-archive/). Falls back to '
             '$CORROBORATE_REMOTE_PREFIX. Pass --local-only to '
             'explicitly skip cloud queries.',
    )
    _ = cat_remote.add_argument(
        '--local-only', dest='local_only', action='store_true',
        help='skip cloud discovery entirely (air-gapped / network-'
             'down mode). Statuses limited to LOCAL_ONLY / '
             'STALE_MANIFEST / IN_PROGRESS_SCAFFOLD; STALE_MANIFEST '
             'in this mode means "local manifest present, cloud '
             'unverified" rather than "cloud verified absent".',
    )
    _ = p_cat.add_argument(
        '--include-misc', dest='include_misc', action='store_true',
        help='include kind=misc rows (cache/, _old_logs/, ...). '
             'Default surfaces only kind=corpus rows.',
    )
    _ = p_cat.add_argument(
        '--json', dest='json_output', action='store_true',
        help='emit rows as a JSON array (nested local/cloud slices) '
             'instead of the default polars table.',
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    args = to_mapping(ns)
    cmd = require_str(args, 'cmd')
    if cmd == 'archive':
        return _cmd_archive(args)
    if cmd == 'restore':
        return _cmd_restore(args)
    if cmd == 'ls':
        return _cmd_ls(args)
    if cmd == 'purge':
        return _cmd_purge(args)
    if cmd == 'catalogue':
        return _cmd_catalogue(args)
    raise ValueError(f'unknown subcommand: {cmd}')


if __name__ == '__main__':
    sys.exit(main())
