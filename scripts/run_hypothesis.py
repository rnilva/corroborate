"""CLI thin-wrapper around `corroborate.runner.run`.

Run any bridges-module-as-hypothesis (`experiments/findings/<X>.py`
exporting `INTERVENTION` + `BRIDGES`) on a data input, with the
per-hypothesis cache:

    python scripts/run_hypothesis.py experiments.findings.ddqn \\
        --data experiments/data/

Library code lives in `corroborate.runner`; this file is purely the
argparse + verdict-printing surface."""
from __future__ import annotations

import os as _os

# Force JAX onto CPU before any substrate import. The bridge-eval /
# ingest paths don't need GPU — they're numpy/polars work. Substrate
# modules unavoidably `import jax.numpy as jnp` at module-load time
# (Replay, MLP, train_phase, ...), which probes the GPU and
# pre-allocates ~80% of VRAM via XLA's default preallocator, starving
# any concurrent GPU work (e.g. a sweep running in another shell).
# Set BEFORE importlib / runner imports — JAX latches on first init.
_os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import argparse
import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from corroborate.bridge import Bridge, BridgeEvaluation
from corroborate.core.finding import Finding
from corroborate.graph.causal import (
    EMPTY_EXTENT_HASH, PostEvalEntry,
    cluster_verdict, clusters_by_extent, composed_verdict, evaluated_graph,
)
from corroborate.runner import check, check_cache_sources, evict, run
from corroborate.runner.runner import (
    _default_cache_path,  # pyright: ignore[reportPrivateUsage]
    _validate_hypothesis,  # pyright: ignore[reportPrivateUsage]
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='run_hypothesis',
        description='Run a hypothesis-module on a data input, with cache.',
    )
    parser.add_argument(
        'module',
        help='dotted module path, e.g. experiments.findings.ddqn',
    )
    parser.add_argument(
        '--ingest', type=str, default=None, metavar='CORPUS[,CORPUS...]',
        help='ingest specific named corpora (CACHE_ADDITIVITY.md '
             'CA3). Names are resolved relative to '
             'experiments/data/<name>/ unless absolute. Use this '
             'when you know what new data to load — much faster '
             'than --ingest-all.',
    )
    parser.add_argument(
        '--ingest-all', type=Path, default=None, metavar='ROOT',
        help='walk every corpus under ROOT and append new cells '
             'to the cache. The "I don\'t know what\'s new" path. '
             'Today\'s --data <root> behavior, renamed for honesty.',
    )
    parser.add_argument(
        '--ingest-file', type=Path, default=None, metavar='PATH',
        help='ingest a single .parquet file as the data source '
             '(no directory walk).',
    )
    parser.add_argument(
        '--data', type=Path, default=None,
        help=argparse.SUPPRESS,  # deprecated alias for --ingest-all
    )
    parser.add_argument(
        '--cache-path', type=Path, default=None,
        help='explicit cache path; defaults to '
             'experiments/data/cache/<short>.parquet',
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='compute fresh, no cache read or write',
    )
    parser.add_argument(
        '--no-write-cache', action='store_true',
        help='read cache for speedup but don\'t persist updates',
    )
    parser.add_argument(
        '--rebuild', action='store_true',
        help='invalidate the per-hypothesis cache before running',
    )
    parser.add_argument(
        '--no-restore', action='store_true',
        help='don\'t restore archived corpora from cloud on miss',
    )
    parser.add_argument(
        '--profile', dest='profile', default=None, type=str,
        help='AWS profile name for the cloud preflight + downstream '
             'restore calls. Only used when an ingest mode is set '
             'AND --no-restore is NOT set AND the ingested corpora '
             'have `_remote.json` (cloud-backed). Falls back to '
             'AWS_PROFILE env var, then the default credential chain.',
    )
    parser.add_argument(
        '--skip-preflight', action='store_true',
        help='Skip the upfront cloud-auth check on --ingest paths. '
             'Use when iterating against known-good creds; saves '
             '~100-300ms per run. Off by default.',
    )
    parser.add_argument(
        '--no-report', action='store_true',
        help='skip writing the post-run JSON audit report',
    )
    parser.add_argument(
        '--report-path', type=Path, default=None,
        help='explicit JSON report path; defaults to '
             'experiments/findings/<short>.run.json',
    )
    parser.add_argument(
        '-k', '--filter', dest='bridge_filter', type=str, default=None,
        help='substring match against bridge names (pytest\'s -k '
             'shape). Run only bridges whose name contains the '
             'pattern. Faster iteration when debugging one bridge.',
    )
    parser.add_argument(
        '--check', action='store_true',
        help='**CACHE_ADDITIVITY.md Phase 2** drift-visibility '
             'mode. Reports per-corpus drift / missing columns '
             'vs the current registry, no compute / no ingest. '
             'Exits 0 if cache is current, 2 if drift detected.',
    )
    parser.add_argument(
        '--evict', type=str, default=None, metavar='CORPUS[,CORPUS...]',
        help='filter the cache parquet to drop all cells from the '
             'named corpora. Per-corpus stores under '
             'experiments/data/<corpus>/ are NOT touched. Useful '
             'to temporarily exclude a corpus from analysis '
             'without rm-ing its data. NOTE: a subsequent '
             '--ingest-all walk re-projects all per-corpus stores '
             'and will re-include the evicted ones; for permanent '
             'exclusion, delete the corpus directory.',
    )
    args = parser.parse_args(argv)

    # --evict: cache-only eviction, no bridge eval.
    evict_arg = cast(str | None, args.evict)
    if evict_arg is not None:
        names = [n.strip() for n in evict_arg.split(',') if n.strip()]
        if not names:
            raise SystemExit('--evict needs at least one corpus name')
        total, counts = evict(cast(str, args.module), names)
        for name in names:
            n = counts.get(name, 0)
            mark = '✓' if n > 0 else '·'
            print(f'  {mark} {name}: dropped {n} cells')
        print(f'evict: {total} cells removed from cache.')
        if total == 0:
            return 0
        return 0

    # CACHE_ADDITIVITY.md Phase 2: --check mode — drift report,
    # no run.
    if cast(bool, args.check):
        return _check_and_report(cast(str, args.module))

    # CACHE_ADDITIVITY.md CA2/CA3: resolve ingest mode from flags.
    # Mutually exclusive: at most one of {--ingest, --ingest-all,
    # --ingest-file, --data}. No flag → cache-only.
    ingest_arg = cast(str | None, args.ingest)
    ingest_all = cast(Path | None, args.ingest_all)
    ingest_file = cast(Path | None, args.ingest_file)
    legacy_data = cast(Path | None, args.data)
    set_count = sum(
        1 for v in (ingest_arg, ingest_all, ingest_file, legacy_data)
        if v is not None
    )
    if set_count > 1:
        raise SystemExit(
            'choose at most one of --ingest, --ingest-all, '
            '--ingest-file, --data',
        )
    if legacy_data is not None:
        import sys
        print(
            'run_hypothesis: --data is deprecated; use '
            '--ingest-all <root> for the same behavior.',
            file=sys.stderr,
        )
    data: Path | str | list[Path] | None = None
    if ingest_arg is not None:
        names = [n.strip() for n in ingest_arg.split(',') if n.strip()]
        data = [
            Path(n) if Path(n).is_absolute()
            else Path('experiments/data') / n
            for n in names
        ]
    elif ingest_all is not None:
        data = ingest_all
    elif ingest_file is not None:
        data = ingest_file
    elif legacy_data is not None:
        data = legacy_data

    bridge_filter = cast(str | None, args.bridge_filter)
    write_cache = not cast(bool, args.no_write_cache)
    write_report = not cast(bool, args.no_report)
    restore_from_cloud = not cast(bool, args.no_restore)
    profile = cast(str | None, args.profile)
    skip_preflight = cast(bool, args.skip_preflight)

    # AWS_PROFILE export is independent of preflight. If the user
    # passes --profile, downstream cloud ops (lazy restore, archive)
    # need it on the env regardless of whether preflight ran.
    if profile is not None:
        import os as _os
        _os.environ['AWS_PROFILE'] = profile

    # Cloud preflight — only when (a) we're ingesting AND (b)
    # cloud-restore is on AND (c) at least one corpus under the
    # ingest scope has a `_remote.json` (else there's no cloud touch
    # to verify against). Fails fast instead of letting the lazy
    # restore step in `runner.run` crash mid-load.
    if data is not None and restore_from_cloud and not skip_preflight:
        from corroborate._internals.cloud_auth import (
            CloudAuthError, preflight as _preflight,
        )
        from corroborate.corpus.cloud import load_manifest, MANIFEST_NAME
        manifest_remote_root: str | None = None
        # Walk the ingest scope and find ANY corpus carrying a
        # `_remote.json`. Use its remote_root for preflight.
        candidate_dirs: list[Path] = []
        if isinstance(data, list):
            candidate_dirs.extend(data)
        elif data.is_dir():
            candidate_dirs.extend(
                d for d in data.iterdir() if d.is_dir()
            )
        for d in candidate_dirs:
            if not (d / MANIFEST_NAME).exists():
                continue
            try:
                m = load_manifest(d)
            except (ValueError, TypeError, OSError) as e:
                # Corrupt `_remote.json` — surface as a clean
                # preflight error rather than letting the unhandled
                # exception bubble.
                import sys
                print(
                    f'run_hypothesis: preflight aborted — corrupt '
                    f'manifest at {d / MANIFEST_NAME}: {e}',
                    file=sys.stderr,
                )
                return 1
            if m is not None:
                manifest_remote_root = m.remote_root
                break
        if manifest_remote_root is not None:
            try:
                _preflight(manifest_remote_root, profile=profile)
            except CloudAuthError as e:
                import sys
                print(
                    f'run_hypothesis: cloud preflight FAILED — '
                    f'aborting before ingest.\n  {e}',
                    file=sys.stderr,
                )
                return 1
    if bridge_filter is not None:
        # Filter mode runs a subset of bridges → measurable
        # computation is the filtered subset's deps, not the full
        # bridges file's. Writing back to the canonical cache /
        # `<module>.run.json` would overwrite the full-run baseline
        # with a partial slice. Force-disable both writes; user
        # gets a single-shot read-only debug iteration. Pass
        # `--report-path /tmp/foo.json` if a partial report is
        # actually wanted at a non-canonical path.
        if write_cache or write_report:
            import sys
            print(
                f'run_hypothesis: bridge_filter={bridge_filter!r} '
                f'set → forcing write_cache=False, '
                f'write_report=False (avoid overwriting canonical '
                f'cache / run.json with a partial slice).',
                file=sys.stderr,
            )
        write_cache = False
        write_report = False

    module_name = cast(str, args.module)
    h_module = importlib.import_module(module_name)
    bridges = cast(tuple[Bridge, ...], h_module.BRIDGES)
    findings = cast(tuple[Finding, ...], h_module.FINDINGS)
    results = run(
        module_name,
        data=data,
        cache_path=cast(Path | None, args.cache_path),
        use_cache=not cast(bool, args.no_cache),
        write_cache=write_cache,
        rebuild=cast(bool, args.rebuild),
        restore_from_cloud=restore_from_cloud,
        report_path=cast(Path | None, args.report_path),
        write_report=write_report,
        bridge_filter=bridge_filter,
    )
    _print_verdicts(results, bridges, findings)
    return 0


def _check_and_report(module: str) -> int:
    """**CACHE_ADDITIVITY.md Phase 2** drift report.

    Two halves:
    - **Output-side** (measurable drift): `runner.check(module)`
      compares per-corpus `measurements.hashes.json` against the
      current registry's closure hashes.
    - **Input-side** (cache-sources drift):
      `runner.check_cache_sources(cache_path)` compares the cache
      parquet's per-corpus cell counts against the on-disk
      `runs.parquet` of each source.

    Exits 0 when both clean, 2 when either reports drift —
    pipeline-friendly (`run_hypothesis.py <m> --check &&
    run_hypothesis.py <m>`).
    """
    report = check(module)
    # Resolve cache path for the input-side check.
    h = _validate_hypothesis(importlib.import_module(module))
    cache_path = _default_cache_path(h)
    input_drift = check_cache_sources(cache_path)

    # Output-side rendering (existing behavior).
    output_exit = 0
    if not report.per_corpus:
        print('check: no corpora found under experiments/data/')
    elif report.is_clean:
        print(
            f'check: {len(report.per_corpus)} corpora — '
            f'no measurable drift, no missing columns.',
        )
    else:
        affected = report.affected_corpus_names()
        print(f'check (output): drift detected across {len(affected)} corpora')
        for c in report.per_corpus:
            if c.is_clean:
                continue
            bits: list[str] = []
            if c.drifted:
                bits.append(f'drifted=[{", ".join(c.drifted)}]')
            if c.missing:
                bits.append(f'missing=[{", ".join(c.missing)}]')
            print(f'  {c.corpus_dir.name:50s}  {" ".join(bits)}')
        print()
        print(f'  → refresh affected corpora with:')
        print(f'     --ingest {",".join(affected)}')
        print(f'     OR --ingest-all experiments/data')
        output_exit = 2

    # Input-side rendering (cache-sources drift).
    bad = [d for d in input_drift if d.status != 'MATCHED']
    input_exit = 0
    if input_drift and bad:
        print()
        print(
            f'check (input): {len(bad)} of {len(input_drift)} '
            f'cache-source entries are not MATCHED',
        )
        for d in bad:
            line = (
                f'  {d.corpus:50s}  status={d.status}  '
                f'cache={d.cache_cell_count}  '
                f'current={d.current_cell_count}'
            )
            if d.remote_root is not None:
                line += f'  remote={d.remote_root}'
            print(line)
        input_exit = 2
    elif input_drift:
        print(
            f'check (input): {len(input_drift)} cache-source entries — '
            f'all MATCHED.',
        )
    return max(output_exit, input_exit)


def _print_verdicts(
    results: dict[str, BridgeEvaluation],
    bridges: tuple[Bridge, ...],
    findings: tuple[Finding, ...],
) -> None:
    counts: dict[str, int] = {}
    refutation_counts: dict[tuple[str, str], int] = {}
    # `assumption_violations` are pre-prefixed with `<fixture>: ` by
    # the bridge layer (bridge.py:846-859). Roll-up groups by the
    # raw fixture:flag string so the operator can see both WHERE
    # the flag fired and WHAT it was. Wiring per
    # UNCONSUMED_PRIMITIVES_AUDIT.md Round 2.
    violation_counts: dict[str, int] = {}
    n_with_violations = 0
    # Extent clusters: bridges keyed by `(source_name, target_name,
    # extent_hash)` admit identical cell-sets — refutation-cluster
    # identity under extent-based grouping. Two bridges with the
    # same key are corroborating the same edge on the same cells.
    # Track per-member verdict so we can compose a cluster-level
    # verdict (supported / refuted / underpowered / empty_extent).
    extent_clusters: dict[
        tuple[str, str, int], list[tuple[str, int, str]],
    ] = {}
    for name, ev in results.items():
        v = ev.verdict.value
        counts[v] = counts.get(v, 0) + 1
        if ev.refutation_class is not None:
            key = (v, ev.refutation_class.value)
            refutation_counts[key] = refutation_counts.get(key, 0) + 1
        bits: list[str] = []
        for ar in ev.analysis_results.values():
            bits.append(_summarize(ar))
        suffix = ' | '.join(bits) if bits else ''
        cls = (
            f' ({ev.refutation_class.value})'
            if ev.refutation_class is not None else ''
        )
        av_inline = ''
        if ev.assumption_violations:
            n_with_violations += 1
            for flag in ev.assumption_violations:
                violation_counts[flag] = violation_counts.get(flag, 0) + 1
            av_inline = f' [av: {"; ".join(ev.assumption_violations)}]'
        cluster_key = (ev.source_name, ev.target_name, ev.extent_hash)
        extent_clusters.setdefault(cluster_key, []).append(
            (name, ev.n_cells_in_scope, v),
        )
        print(f'{name:60s}  {v:24s}{cls}{av_inline}  {suffix}')
    print()
    print('verdict counts:')
    for k in sorted(counts):
        # Indented sub-classification breakdown when present.
        sub = sorted(
            (cls, n) for (verdict, cls), n in refutation_counts.items()
            if verdict == k
        )
        print(f'  {k:24s}  {counts[k]}')
        for cls, n in sub:
            print(f'    └── {cls:20s}  {n}')
    if n_with_violations:
        print()
        print(f'assumption violations: {n_with_violations} bridge(s)')
        for flag in sorted(violation_counts):
            print(f'  └── {flag:50s}  {violation_counts[flag]}')

    # Extent clusters: report multi-bridge groups (refutation
    # clusters or sibling-pair patterns) and the empty-extent group
    # separately. Singletons are the default and don't need a
    # roll-up. Cluster verdict label comes from the framework's
    # `cluster_verdict` primitive — see
    # `corroborate.graph.causal` for the implementation; the
    # `experiments/findings/ddqn/finding_*.py` walks show the
    # canonical use pattern for Findings.
    post_eval = {
        name: PostEvalEntry(verdict=ev.verdict, extent_hash=ev.extent_hash)
        for name, ev in results.items()
    }
    g = evaluated_graph(bridges, post_eval)
    graph_clusters = clusters_by_extent(g)
    multi = [(k, v) for k, v in extent_clusters.items() if len(v) >= 2]
    empty_members = [
        (name, n) for (_, _, h), members in extent_clusters.items()
        for name, n, _ in members
        if h == EMPTY_EXTENT_HASH
    ]
    if multi or empty_members:
        print()
        print(f'extent clusters: {len(multi)} multi-bridge group(s)')
        for (src, tgt, h), members in sorted(
            multi, key=lambda x: (-len(x[1]), x[0][:2]),
        ):
            n = members[0][1]
            src_short = src if len(src) < 50 else src[:47] + '...'
            edge_members = graph_clusters.get((src, tgt, h), ())
            verdict_label = cluster_verdict(edge_members).value
            print(
                f'  {verdict_label:14s}  ({len(members)} bridges, '
                f'n_cells={n})  {src_short} → {tgt}',
            )
            for member_name, _, _ in members:
                print(f'      {member_name}')
        if empty_members:
            n_empty = len(empty_members)
            print(
                f'  {n_empty} bridge(s) in empty-extent group(s) — '
                f'scope admits zero cells on current cache.',
            )

    # Findings rollup: every Hypothesis declares FINDINGS (possibly
    # empty). For each, evaluate the cluster-shaped claim against
    # the post-eval graph and compare to the author's EXPECTED.
    # Drift is the operational signal — a claim flipping from
    # SUPPORTED to REFUTED (or anywhere else) shows up here on
    # every run.
    if findings:
        print()
        drift_count = 0
        blocked_count = 0
        for f in findings:
            verdict = composed_verdict(g, bridges=f.BRIDGES)
            drift = verdict != f.EXPECTED
            blocked = f.BLOCKED_ON is not None
            if drift:
                drift_count += 1
            if blocked:
                blocked_count += 1
            # `← DRIFT` = verdict CHANGED relative to author's
            # pinned state (regression or improvement — both
            # warrant attention). `[blocked]` = state matches
            # EXPECTED but the finding is intentionally pinned
            # to a sub-optimal state pending data; quiet status,
            # no investigation needed.
            if drift:
                state_marker = ' ← DRIFT'
            elif blocked:
                state_marker = ' [blocked]'
            else:
                state_marker = ''
            n_bridges = len(f.BRIDGES)
            # Distinct-extent count surfaces the structural shape
            # without naming it: `N bridges, 1 extent` = cluster
            # pattern (parallel edges, shared admitted cells);
            # `N bridges, N extents` = envelope (per-scope sub-
            # claims). Framework reports counts, not labels —
            # cluster/envelope/chain interpretation lives where
            # human prose lives (the docstring).
            bridge_names = {b.name for b in f.BRIDGES}
            distinct_extents = len({
                e.metadata.extent_hash for e in g.edges
                if e.metadata.bridge_name in bridge_names
            })
            extents_label = (
                f'{distinct_extents} extents'
                if distinct_extents != 1 else '1 extent'
            )
            short_name = f.__name__.rsplit('.', 1)[-1]
            print(
                f'  {verdict.value:14s}  '
                f'({n_bridges} bridges, {extents_label})  '
                f'{short_name}{state_marker}',
            )
            if drift:
                doc = (f.__doc__ or '').strip().split('\n', 1)[0]
                print(f'      expected: {f.EXPECTED.value}')
                if doc:
                    print(f'      claim:    {doc}')
            if blocked:
                print(f'      blocked-on: {f.BLOCKED_ON}')
        print(
            f'findings: {len(findings)} declared, '
            f'{drift_count} drift, {blocked_count} blocked',
        )


def _summarize(result: object) -> str:
    type_name = type(result).__name__
    parts: list[str] = [type_name]
    for attr in ('g', 'se', 'mean_diff', 'p_value', 'n_pairs'):
        v = getattr(result, attr, None)
        if isinstance(v, (int, float)):
            parts.append(f'{attr}={v:.3g}')
    return ' '.join(parts)


if __name__ == '__main__':
    import sys
    sys.exit(main())
