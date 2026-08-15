"""Top-level CLI — `corroborate <subcommand>` (or `python -m corroborate ...`).

Subcommands:

  archive     upload sweep parquets to remote storage
  restore     download archived parquets from remote
  ls          show what is archived for a sweep
  purge       delete LOCAL copies of files in the manifest
  catalogue   inventory all corpora under a data root (local + cloud)
  hypothesis  run a bridges-module-as-hypothesis on a data input

`archive`/`restore`/`ls`/`purge` operate on one sweep directory.
The remote root is pinned in the per-sweep manifest after the
first `archive`; later `restore`/`ls`/`purge` read it from there.
`catalogue` walks one or more data roots and cross-references with
a cloud prefix. `hypothesis` is the framework's primary
research-loop entry point: load a `@claim`-driven hypothesis
module (e.g. `experiments.findings.ddqn`), ingest a corpus into
the per-hypothesis cache, and run the bridge evaluations.

Python API mirrors:
- `corroborate.cloud` (archive / restore / ls / purge)
- `corroborate.corpus.catalogue` (catalogue)
- `corroborate.runner.run` (hypothesis)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from corroborate.cli import audit as _audit
from corroborate.cli import hypothesis as _hypothesis
from corroborate.cli import sweep as _sweep
from corroborate.corpus import catalogue as _catalogue
from corroborate.corpus import cloud
from corroborate._internals.argparse import to_mapping
from corroborate._internals.cloud_auth import (
    CloudAuthError, preflight,
)
from corroborate._internals.narrow import (
    optional_str,
    optional_str_list,
    require_bool,
    require_str,
)


def _preflight_or_exit(
    remote_prefix: str, profile: str | None,
) -> int | None:
    """Run cloud-auth preflight; on failure, print the typed error
    + hint to stderr and return the CLI exit code. Returns None on
    success (caller continues). Returning the code (not raising)
    lets callers keep the early-return pattern used elsewhere.

    On preflight SUCCESS with `--profile <name>`, also export
    `AWS_PROFILE` to the process environment so the actual cloud
    op (which goes through `fsspec` / `s3fs`, NOT through the
    botocore.session we used for preflight) picks up the same
    profile. Without this, `--profile r2` would pass preflight
    but the upload/download would fall back to the default chain
    and likely fail with different creds."""
    try:
        preflight(remote_prefix, profile=profile)
    except CloudAuthError as e:
        sys.stderr.write(f'corroborate: cloud auth failed\n  {e}\n')
        return 1
    if profile is not None:
        os.environ['AWS_PROFILE'] = profile
    return None


def _cmd_archive(args: Mapping[str, object]) -> int:
    sweep_dir = Path(require_str(args, 'sweep_dir'))
    remote = require_str(args, 'remote')
    files = optional_str_list(args, 'files')
    force = require_bool(args, 'force')
    purge_local = require_bool(args, 'purge_local')
    profile = optional_str(args, 'profile')

    rc = _preflight_or_exit(remote, profile)
    if rc is not None:
        return rc

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
    profile = optional_str(args, 'profile')

    # Preflight needs the bucket — read it off the local manifest.
    manifest = cloud.load_manifest(sweep_dir)
    if manifest is None:
        sys.stderr.write(
            f'corroborate: no `_remote.json` at {sweep_dir} — '
            f'nothing to restore.\n',
        )
        return 1
    rc = _preflight_or_exit(manifest.remote_root, profile)
    if rc is not None:
        return rc

    restored = cloud.restore(sweep_dir, files=files, overwrite=overwrite)
    print(f'restored {len(restored)} files to {sweep_dir}')
    for r in restored:
        print(f'  {r}')
    return 0


def _cmd_ls(args: Mapping[str, object]) -> int:
    sweep_dir = Path(require_str(args, 'sweep_dir'))
    profile = optional_str(args, 'profile')

    manifest = cloud.load_manifest(sweep_dir)
    if manifest is None:
        sys.stderr.write(
            f'corroborate: no `_remote.json` at {sweep_dir}.\n',
        )
        return 1
    rc = _preflight_or_exit(manifest.remote_root, profile)
    if rc is not None:
        return rc

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
    cloud_fallback_prefix = optional_str(args, 'remote_prefix')
    deleted = cloud.purge(
        sweep_dir, files=files,
        cloud_fallback_prefix=cloud_fallback_prefix,
    )
    print(f'deleted {len(deleted)} local files in {sweep_dir}')
    for r in deleted:
        print(f'  {r}')
    return 0


def _cmd_catalogue(args: Mapping[str, object]) -> int:
    from corroborate._internals.narrow import require_str_list
    data_root_strs = require_str_list(args, 'data_root')
    data_roots = tuple(Path(s).resolve() for s in data_root_strs)
    cli_prefix = optional_str(args, 'remote_prefix')
    local_only = require_bool(args, 'local_only')
    include_misc = require_bool(args, 'include_misc')
    as_json = require_bool(args, 'json_output')
    leaves_mode = require_bool(args, 'leaves_mode')
    leaves_wide = require_bool(args, 'leaves_wide')
    profile = optional_str(args, 'profile')

    if leaves_mode:
        # Register implementation `@measurable` functions so
        # `registered_names()` filters them out of leaves.
        # Caller-provided implementation modules (CLI flag or env
        # var) take priority; the in-tree RL implementation is a
        # fallback if neither is specified.
        substrate_modules = optional_str_list(args, 'substrate_modules')
        if substrate_modules is None:
            env_mods = os.environ.get('CORROBORATE_SUBSTRATE_MODULES')
            substrate_modules = (
                [m.strip() for m in env_mods.split(',') if m.strip()]
                if env_mods else ['corroborate_rl.dqn.measurables']
            )
        import importlib
        for mod_name in substrate_modules:
            try:
                _ = importlib.import_module(mod_name)
            except ImportError:
                pass
        profiles = _catalogue.arm_leaves(data_roots)
        if as_json:
            payload = [dataclasses.asdict(p) for p in profiles]
            print(json.dumps(payload, default=str, indent=2))
        elif leaves_wide:
            print(_catalogue.arm_leaves_to_polars_wide(profiles))
        else:
            print(_catalogue.arm_leaves_to_polars_long(profiles))
        return 0

    if local_only:
        remote_prefix: str | None = None
    elif cli_prefix is not None:
        remote_prefix = cli_prefix
    else:
        env_prefix = os.environ.get('CORROBORATE_REMOTE_PREFIX')
        remote_prefix = env_prefix if env_prefix else None

    # Preflight only when we're actually going to touch cloud.
    if remote_prefix is not None:
        rc = _preflight_or_exit(remote_prefix, profile)
        if rc is not None:
            return rc

    rows = _catalogue.catalogue(
        data_roots,
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


def _build_parser(
    *, argv: Sequence[str] | None = None,
) -> argparse.ArgumentParser:
    """Build the top-level `corroborate` parser.

    `argv`, when provided, is threaded into the `sweep` subparser
    so its substrate-CLI peek + extension loading happens at
    parser-build time (substrate's `add_args(p_run)` registers
    implementation-specific options before `parser.parse_args(argv)`
    runs). When `argv` is None, the sweep subparser registers
    only framework args; implementation extensions are skipped."""
    parser = argparse.ArgumentParser(
        prog='corroborate',
        description='cloud archive for sweep parquets + corpus '
                    'catalogue (via `corroborate` console script '
                    'or `python -m corroborate`).',
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
    _ = p_archive.add_argument(
        '--profile', dest='profile', default=None,
        help='AWS profile name to use for credential resolution '
             '(forwarded to botocore via AWS_PROFILE). Falls back to '
             'the default chain (env vars → ~/.aws/credentials → '
             'IAM role).',
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
    _ = p_restore.add_argument(
        '--profile', dest='profile', default=None,
        help='AWS profile name (see archive --help).',
    )

    p_ls = sub.add_parser('ls', help='show what is archived for a sweep')
    _ = p_ls.add_argument('sweep_dir')
    _ = p_ls.add_argument(
        '--profile', dest='profile', default=None,
        help='AWS profile name (see archive --help).',
    )

    p_purge = sub.add_parser(
        'purge',
        help='delete LOCAL copies of files in the manifest. Manifest '
             'is preserved so restore stays available.',
    )
    _ = p_purge.add_argument('sweep_dir')
    _ = p_purge.add_argument('--files', nargs='*', default=None)
    _ = p_purge.add_argument(
        '--remote-prefix', dest='remote_prefix', default=None,
        help='fsspec URI prefix (e.g. s3://<your-bucket>/) for '
             'the cloud-fallback path. Used when the local '
             '_remote.json was lost (e.g., post-merge cleanup wiped '
             'the sub-corpus dir that held it). Discovers sub-archives '
             'at <prefix>/<sweep_name>/* and verifies each local '
             "parquet's row_ids are covered by the union of "
             'sub-archive row_ids before deletion.',
    )

    p_cat = sub.add_parser(
        'catalogue',
        help='inventory all corpora under a data root, cross-'
             'referenced against cloud archives under a remote prefix.',
    )
    _ = p_cat.add_argument(
        'data_root', nargs='+',
        help='one or more directories containing per-sweep corpus '
             'dirs. The project convention has two corpus-bearing '
             'roots: `experiments/data/` (canonical sweep output) '
             'and `experiments/probes/` (ad-hoc pilots). Pass both '
             'to avoid false-orphan reports.',
    )
    cat_remote = p_cat.add_mutually_exclusive_group()
    _ = cat_remote.add_argument(
        '--remote-prefix', dest='remote_prefix', default=None,
        help='fsspec URI prefix for cloud discovery '
             '(e.g. s3://<your-bucket>/). Falls back to '
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
    _ = p_cat.add_argument(
        '--leaves', dest='leaves_mode', action='store_true',
        help='switch output to per-(corpus, arm) leaf-signature view. '
             'Reads runs.parquet for every locally-present corpus and '
             'extracts the configurational leaves (composition-time '
             'kwargs; framework vocabulary — RL practice calls these '
             'hyperparameters). Default long format is `(corpus, arm, '
             'leaf_path, leaf_value)`; combine with --leaves-wide for '
             'a pivoted view.',
    )
    _ = p_cat.add_argument(
        '--leaves-wide', dest='leaves_wide', action='store_true',
        help='with --leaves, render in wide format (each leaf as a '
             'column; sweep arms collapse to `MULTI:[v1,v2,...]`).',
    )
    _ = p_cat.add_argument(
        '--substrate-module', dest='substrate_modules',
        nargs='*', default=None,
        help='with --leaves, import these modules to register '
             'their `@measurable` functions before leaf filtering. '
             'Falls back to $CORROBORATE_SUBSTRATE_MODULES '
             '(comma-separated). Default is '
             '`corroborate_rl.dqn.measurables` (the in-tree RL '
             'substrate).',
    )
    _ = p_cat.add_argument(
        '--profile', dest='profile', default=None,
        help='AWS profile name for the cloud preflight when '
             '--remote-prefix is used. See archive --help.',
    )

    p_hyp = sub.add_parser(
        'hypothesis',
        help='run a bridges-module-as-hypothesis on a data input '
             '(e.g. `corroborate hypothesis experiments.findings.ddqn '
             '--ingest-all experiments/data/`).',
    )
    _hypothesis.add_args(p_hyp)

    p_audit = sub.add_parser(
        'audit',
        help='audit corpus-level commitments (currently: '
             '`pre-registration` manifest written at sweep launch).',
    )
    _audit.add_args(p_audit)

    p_sweep = sub.add_parser(
        'sweep',
        help='run a YAML-configured sweep through a substrate '
             '(`corroborate sweep run <yaml>`).',
    )
    _sweep.add_args(p_sweep, argv=argv)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Thread argv through the parser so the `sweep` subparser can
    # peek for `--substrate <name>` and load that substrate's CLI
    # extensions BEFORE argparse runs (implementation-specific args
    # then get validated alongside framework args in one parse).
    effective_argv: Sequence[str] = (
        argv if argv is not None else sys.argv[1:]
    )
    parser = _build_parser(argv=effective_argv)
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
    if cmd == 'hypothesis':
        # `hypothesis.dispatch` consumes the parsed Namespace
        # directly (its arg names are richer + typed via `cast`s
        # rather than the `to_mapping` + `require_*` narrowing
        # pattern the other subcommands use). Pass `ns` through.
        return _hypothesis.dispatch(ns)
    if cmd == 'audit':
        return _audit.dispatch(ns)
    if cmd == 'sweep':
        return _sweep.dispatch(ns)
    raise ValueError(f'unknown subcommand: {cmd}')


if __name__ == '__main__':
    sys.exit(main())
