"""Cloud archive CLI — `python -m corroborate <subcommand>`.

Subcommands:

  archive  upload sweep parquets to remote storage
  restore  download archived parquets from remote
  ls       show what is archived for a sweep
  purge    delete LOCAL copies of files in the manifest

Each subcommand operates on one sweep directory. The remote
root is pinned in the per-sweep manifest after the first
`archive`; later `restore`/`ls`/`purge` read it from there.

Python API mirror lives in `corroborate.cloud`."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from corroborate.corpus import cloud
from corroborate._internals.argparse import to_mapping
from corroborate._internals.narrow import (
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
    raise ValueError(f'unknown subcommand: {cmd}')


if __name__ == '__main__':
    sys.exit(main())
