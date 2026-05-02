"""Evidence cache — bridge-driven measurable cache materialisation.

A bridges file (e.g. `experiments/findings/ddqn_universe.py`)
self-declares its claim spec via `Sequence[Bridge]` exports. Each
Bridge names its `source` + `target` measurables and a params
bag. The Evidence cache walks those names → transitively closes
via the @measurable graph → computes every reached scalar per
cell → writes a flat-columnar parquet next to each corpus's
`runs.parquet`.

Two surfaces:

- `build_cache(bridges, runs_path, traces_path, out_path)` — one
  corpus. Reads runs + the trace columns the measurables actually
  consume; computes per-cell with `evaluate_with_measurables`'s
  per-record memoisation; writes `runs_with_mediators.parquet`.

- `build_universal_cache(bridges, data_root, out_path)` —
  multi-corpus orchestrator. Auto-discovers corpora under
  `data_root` (any subdir with a non-empty `runs.parquet`), runs
  `build_cache` per corpus, then merges all per-corpus caches
  into one universal parquet via `stream_concat_parquets(
  how='diagonal_relaxed')`. The merge tags each row with a
  `corpus` column = corpus directory name.

Heterogeneous columns (extra HPs in one sweep, extra invariants
in another) flow through the diagonal_relaxed concat with null
padding — no manual schema curation, no hardcoded corpus list.
"""
from __future__ import annotations

import os

# Pure numpy on persisted traces. Force CPU before any JAX import.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from corroborate.claim_bridge import Bridge, measurable_names_for_bridges
from corroborate.measurable import (
    evaluate_with_measurables, get_registered, transitive_reads,
)
from corroborate.persistence import stream_concat_parquets


def _noop_log(_msg: str) -> None:
    return None


__all__ = [
    'build_cache',
    'build_universal_cache',
    'discover_corpora',
]


# ============ Per-corpus cache ============

def build_cache(
    bridges: Sequence[Bridge],
    runs_path: Path,
    traces_path: Path,
    out_path: Path,
    *,
    quiet: bool = False,
) -> None:
    """Walk `bridges` for the measurables it consumes (transitively
    via the @measurable graph), compute each per cell from `runs_path
    × traces_path`, write `out_path` with the original run cols + new
    measurable cols.

    Trace projection: only the columns the measurables' `reads`-set
    references are read from `traces.parquet`. Loading the full file
    is multi-GB on long sweeps; narrowing avoids OOM.

    Cells where a measurable can't resolve (missing leaf reads, type
    mismatch, ...) get None for that column — downstream consumers
    NaN-skip without losing the row.
    """
    log: Callable[[str], None] = _noop_log if quiet else print

    names = sorted(measurable_names_for_bridges(bridges))
    if not names:
        log('no measurables required by these bridges; nothing to cache')
        return
    log(f'measurables to cache ({len(names)}):')
    for n in names:
        log(f'  {n}')

    runs_df = pl.read_parquet(runs_path)
    log(f'runs:   {runs_df.height} rows × {len(runs_df.columns)} cols')

    trace_reads: set[str] = set()
    for n in names:
        trace_reads |= transitive_reads(n)
    runs_cols = set(runs_df.columns)
    needed_trace_cols = sorted(
        (k for k in trace_reads if k not in runs_cols),
    )
    if (
        traces_path.exists()
        and traces_path.stat().st_size > 0
        and needed_trace_cols
    ):
        # Some corpora don't carry every column the bridges'
        # measurables would read (e.g. older sweeps without
        # `reward_scale`). Project to the intersection with the
        # traces schema; missing columns surface as None per cell
        # via the measurable's defensive `record.get(...)` path.
        trace_schema_cols = set(
            pl.scan_parquet(traces_path).collect_schema().names(),
        )
        present_trace_cols = [
            c for c in needed_trace_cols if c in trace_schema_cols
        ]
        missing = [
            c for c in needed_trace_cols if c not in trace_schema_cols
        ]
        if present_trace_cols:
            traces_df = pl.read_parquet(
                traces_path, columns=['id', *present_trace_cols],
            )
            df = runs_df.join(traces_df, on='id', how='inner')
            log(
                f'traces: {traces_df.height} rows × '
                f'{len(present_trace_cols)} cols (filtered from full '
                f'file); joined → {df.height} cells',
            )
        else:
            df = runs_df
            log(
                f'traces: no needed cols present; '
                f'missing={missing[:8]}; using runs only',
            )
        if missing:
            log(
                f'traces: {len(missing)} needed cols missing from this '
                f'corpus — measurables that read them will yield None',
            )
    else:
        df = runs_df
        log('traces: not needed or unavailable; using runs only')

    # Cache-first discipline at the measurable layer too: if the
    # measurable's name already exists as a column in runs.parquet
    # (cell_runner persisted it at sweep time), the persisted
    # value is authoritative — recomputing would NaN-overwrite
    # values for cells without their leaf trace reads (e.g.
    # `eval_final_mean` was emitted by cell_runner reading the
    # online `ep_return`/`done` series; post-hoc on a runs-only
    # cache the measurable's leaf reads are absent and it returns
    # None, which would clobber the original finite value).
    runs_cols_set = set(runs_df.columns)
    names_to_compute = [n for n in names if n not in runs_cols_set]
    names_skipped = [n for n in names if n in runs_cols_set]
    if names_skipped:
        log(
            f'measurables already persisted in runs.parquet '
            f'({len(names_skipped)}): {names_skipped[:8]}'
            f'{"..." if len(names_skipped) > 8 else ""}',
        )
    new_cols: dict[str, list[object]] = {n: [] for n in names_to_compute}
    for cell in df.iter_rows(named=True):
        cache: dict[str, object] = {}
        for n in names_to_compute:
            m = get_registered(n)
            if m is None:
                new_cols[n].append(None)
                continue
            try:
                v = evaluate_with_measurables(m.fn, cell, cache=cache)
            except (KeyError, TypeError, ValueError):
                # Missing leaf reads → measurable can't resolve;
                # store None so downstream can NaN-skip.
                v = None
            new_cols[n].append(_to_polars_value(v))

    # Persist runs + scalar measurables + a NARROW set of 2-D
    # trace cols. Per-step 1-D trajectories (`online_max_q_per_
    # step` etc. — shape `(n_steps,)`, hundreds of thousands of
    # entries per cell) are used to compute the scalar
    # measurables but get dropped from the cache; preserving
    # them on every cell would inflate the cache by orders of
    # magnitude (1-D series at 1M+ steps × n_cells × n_corpora
    # = tens of GB).
    #
    # 2-D structured cols (e.g. `mc_return` shape `(n_bursts,
    # n_episodes)` ≈ 100 entries per cell) ARE preserved: they're
    # consumed raw by downstream per-burst analyses
    # (`paired_g_per_burst`, `mundlak_paired_g_per_burst`,
    # `paired_link_per_burst`) which read the 2-D layout. Cost
    # is small: each cell carries only `~n_bursts × n_episodes`
    # scalars, not the full per-step trajectory.
    preserve_trace_cols: list[str] = []
    runs_col_set = set(runs_df.columns)
    for c in df.columns:
        if c in runs_col_set or c == 'id':
            continue
        if not isinstance(df.schema[c], pl.List):
            continue
        # 2-D = `List(List(<scalar>))`; 1-D = `List(<scalar>)`.
        # Polars exposes nested-list dtype via the inner type.
        inner = df.schema[c].inner  # pyright: ignore[reportAttributeAccessIssue]
        if isinstance(inner, pl.List):
            preserve_trace_cols.append(c)
    if preserve_trace_cols:
        enriched = (
            runs_df
            .join(
                df.select(['id', *preserve_trace_cols]),
                on='id', how='left',
            )
            .with_columns([
                pl.Series(n, new_cols[n]) for n in names_to_compute
            ])
        )
        log(
            f'preserving 2-D trace cols ({len(preserve_trace_cols)}): '
            f'{preserve_trace_cols[:6]}'
            f'{"..." if len(preserve_trace_cols) > 6 else ""}',
        )
    else:
        enriched = runs_df.with_columns([
            pl.Series(n, new_cols[n]) for n in names_to_compute
        ])
    enriched.write_parquet(out_path)
    log(
        f'wrote: {out_path}  '
        f'({enriched.height} rows × {len(enriched.columns)} cols)',
    )


def _to_polars_value(v: object) -> object:
    """Coerce a measurable's output to something polars accepts:
    scalars stay scalar; numpy arrays become Python lists (polars
    encodes as list-of-float); None passes through."""
    if v is None:
        return None
    if isinstance(v, (int, float, bool, str)):
        return v
    if isinstance(v, np.ndarray):
        # `ndarray.tolist()` is `Any` per typeshed; the runtime
        # invariant is "nested Python list of scalars", which polars
        # accepts as a List(<inferred>) column. Cast at the boundary.
        return cast(object, v.tolist())
    return v


# ============ Multi-corpus orchestrator ============

def discover_corpora(
    data_root: Path,
    *,
    min_runs_size: int = 1024,
) -> list[Path]:
    """Auto-discover corpora under `data_root`. A corpus is a
    direct subdirectory containing a `runs.parquet` of size at
    least `min_runs_size` bytes (default 1KB — strictly above
    "empty parquet metadata only").

    Returns sorted list of corpus directories. Subdirectories
    without `runs.parquet`, or with a too-small `runs.parquet`,
    are silently skipped.
    """
    if not data_root.is_dir():
        raise ValueError(f'data_root not a directory: {data_root}')
    out: list[Path] = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        runs = d / 'runs.parquet'
        if not runs.exists():
            continue
        try:
            sz = runs.stat().st_size
        except OSError:
            continue
        if sz < min_runs_size:
            continue
        out.append(d)
    return out


def build_universal_cache(
    bridges: Sequence[Bridge],
    *,
    data_root: Path,
    out_path: Path,
    out_name: str = 'runs_with_bridge_cache.parquet',
    skip_up_to_date: bool = True,
    quiet_per_corpus: bool = False,
) -> None:
    """Build an evidence cache across every corpus under
    `data_root`, then merge into one universal parquet at
    `out_path`.

    Per-corpus cache files (`<corpus>/<out_name>`) are reused when
    fresher than their `runs.parquet` (override with
    `skip_up_to_date=False`). Heterogeneous-column corpora flow
    through `stream_concat_parquets(how='diagonal_relaxed')` —
    extra HPs / invariants null-pad cleanly, no manual schema
    curation.

    Each merged row carries a `corpus` column (= corpus directory
    name) so downstream slices can stratify by corpus. The tag is
    written into temp files at merge time; per-corpus caches are
    untouched and remain consumable on their own.
    """
    corpora = discover_corpora(data_root)
    if not corpora:
        raise SystemExit(f'no corpora found under {data_root}')
    print(f'discovered {len(corpora)} corpora under {data_root}')

    per_corpus_caches: list[Path] = []
    skipped: list[str] = []
    for corpus in corpora:
        runs_path = corpus / 'runs.parquet'
        traces_path = corpus / 'traces.parquet'
        cache_path = corpus / out_name

        rebuild = True
        if skip_up_to_date and cache_path.exists():
            try:
                cache_mtime = cache_path.stat().st_mtime
                runs_mtime = runs_path.stat().st_mtime
                rebuild = cache_mtime < runs_mtime
            except OSError:
                rebuild = True

        if rebuild:
            print(f'\n[{corpus.name}] building cache...')
            try:
                build_cache(
                    bridges, runs_path, traces_path, cache_path,
                    quiet=quiet_per_corpus,
                )
            except (KeyError, ValueError, RuntimeError) as exc:
                # Per-corpus failure (missing measurable, schema
                # quirk) shouldn't kill the whole sweep. Log and
                # skip; the merge proceeds without this corpus.
                print(f'[{corpus.name}] FAILED: {exc!r}')
                skipped.append(corpus.name)
                continue
        else:
            print(f'[{corpus.name}] cache fresh; reusing')

        if cache_path.exists():
            per_corpus_caches.append(cache_path)
        else:
            skipped.append(corpus.name)

    if not per_corpus_caches:
        raise SystemExit(
            'no per-corpus caches available to merge '
            f'(skipped: {skipped})',
        )

    # Tag each per-corpus cache with a `corpus` column at merge
    # time. Done in temp files so per-corpus caches remain untouched
    # and consumable on their own.
    tmp_dir = Path(tempfile.mkdtemp(prefix='universal_evidence_'))
    try:
        tagged_paths: list[Path] = []
        for cache_path in per_corpus_caches:
            corpus_name = cache_path.parent.name
            tagged = tmp_dir / f'{corpus_name}.parquet'
            df = pl.read_parquet(cache_path)
            df = df.with_columns(
                pl.lit(corpus_name).alias('corpus'),
            )
            df.write_parquet(tagged)
            tagged_paths.append(tagged)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        stream_concat_parquets(
            tagged_paths, out_path, type_widening=True,
        )
        merged = pl.read_parquet(out_path)
        print()
        print(
            f'wrote universal evidence: {out_path}  '
            f'({merged.height} rows × {len(merged.columns)} cols, '
            f'{len(tagged_paths)} corpora)',
        )
        if skipped:
            print(f'skipped corpora ({len(skipped)}): {skipped}')
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
