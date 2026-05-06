"""Re-merge `minatar_sync_curve_resume` hypothesis runs.parquet/traces.parquet
to include both Freeway and SpaceInvaders cells.

Background
----------
The original sweep dispatch wrote `<out_dir>/<cfg.name>/runs.parquet` per
`run_intervention` call. When `arms_shape: paired` produces multiple configs
sharing a hypothesis name (one per env × hypothesis), each call's merge step
overwrites the previous one — last writer wins. SpaceInvaders ran after
Freeway, so the local merged files contain only SpaceInvaders.

The cell-level shards (`tmp/cell*__<env>__*.parquet`) are env-distinct on S3
and intact in the manifest. This script reads the manifest, selects ALL tmp
shards (Freeway + SpaceInvaders), and streams the concat to fresh local
runs.parquet / traces.parquet — overwriting the SI-only versions.

Usage
-----
    set -a; . .env; set +a
    JAX_PLATFORMS=cpu uv run python scripts/remerge_minatar_sync_curve_resume.py
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from pathlib import Path

from corroborate.corpus.persistence import stream_concat_parquets


def _tmp_uris(manifest_path: Path, suffix: str) -> tuple[str, list[str]]:
    """Return (remote_root, sorted list of S3 URIs for tmp/cell*__<suffix>)."""
    m = json.loads(manifest_path.read_text())
    remote_root = m['remote_root'].rstrip('/')
    relpaths = sorted(
        f['relpath'] for f in m['files']
        if f['relpath'].startswith('tmp/') and f['relpath'].endswith(suffix)
    )
    return remote_root, [f'{remote_root}/{rp}' for rp in relpaths]


def remerge(hyp_dir: Path) -> None:
    manifest_path = hyp_dir / '_remote.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'{manifest_path}: missing')

    remote_root, runs_uris = _tmp_uris(manifest_path, '__runs.parquet')
    _, traces_uris = _tmp_uris(manifest_path, '__traces.parquet')

    print(f'[{hyp_dir.name}] remote_root = {remote_root}')
    print(f'  {len(runs_uris)} runs shards, {len(traces_uris)} traces shards')

    # Show a couple to confirm both envs are included.
    envs_seen = sorted({u.split('__')[1] for u in runs_uris})
    print(f'  envs in shards: {envs_seen}')

    runs_out = hyp_dir / 'runs.parquet'
    traces_out = hyp_dir / 'traces.parquet'

    print(f'  → re-merging {len(runs_uris)} runs shards into {runs_out}')
    stream_concat_parquets(runs_uris, runs_out)
    print(f'    {runs_out.stat().st_size / 1e3:.1f} KB')

    print(f'  → re-merging {len(traces_uris)} traces shards into {traces_out}')
    # chunk_size=2 — these are ~600-880 MB compressed each (~5GB decompressed),
    # local disk is tight; smaller chunks bound peak RAM and scratch.
    stream_concat_parquets(traces_uris, traces_out, chunk_size=2)
    print(f'    {traces_out.stat().st_size / 1e9:.2f} GB')


def main() -> None:
    root = Path('experiments/data/minatar_sync_curve_resume')
    for sub in ('ddqn_sync1k', 'ddqn_sync3k'):
        remerge(root / sub)
        print()


if __name__ == '__main__':
    main()
