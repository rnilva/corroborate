"""CLI thin-wrapper around `corroborate.runner.run`.

Run any bridges-module-as-hypothesis (`experiments/findings/<X>.py`
exporting `INTERVENTION` + `BRIDGES`) on a data input, with the
per-hypothesis cache:

    python scripts/run_hypothesis.py experiments.findings.ddqn \\
        --data experiments/data/

Library code lives in `corroborate.runner`; this file is purely the
argparse + verdict-printing surface."""
from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from corroborate.bridge import Bridge, BridgeEvaluation
from corroborate.core.finding import Finding
from corroborate.graph.causal import (
    PostEvalEntry,
    cluster_verdict, clusters_by_extent, composed_verdict, evaluated_graph,
)
from corroborate.runner import check, evict, run


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
        restore_from_cloud=not cast(bool, args.no_restore),
        report_path=cast(Path | None, args.report_path),
        write_report=write_report,
        bridge_filter=bridge_filter,
    )
    _print_verdicts(results, bridges, findings)
    return 0


def _check_and_report(module: str) -> int:
    """**CACHE_ADDITIVITY.md Phase 2** drift report. Calls
    `runner.check(module)`, prints per-corpus drift / missing
    summary + a pasteable `--ingest` command for the affected
    corpora. Exits 0 when clean, 2 when drift detected — usable
    in shell pipelines (`run_hypothesis.py <m> --check &&
    run_hypothesis.py <m>`)."""
    report = check(module)
    if not report.per_corpus:
        print('check: no corpora found under experiments/data/')
        return 0
    if report.is_clean:
        print(
            f'check: {len(report.per_corpus)} corpora — all current. '
            f'No drift, no missing required columns.',
        )
        return 0
    affected = report.affected_corpus_names()
    print(f'check: drift detected across {len(affected)} corpora')
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
    return 2


_EMPTY_EXTENT_HASH = hash(frozenset[str]())


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
        if h == _EMPTY_EXTENT_HASH
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
        for f in findings:
            verdict = composed_verdict(g, bridges=f.BRIDGES)
            drift = verdict != f.EXPECTED
            if drift:
                drift_count += 1
            drift_marker = ' ← DRIFT' if drift else ''
            n_bridges = len(f.BRIDGES)
            short_name = f.__name__.rsplit('.', 1)[-1]
            print(
                f'  {verdict.value:14s}  ({n_bridges} bridges)  '
                f'{short_name}{drift_marker}',
            )
            if drift:
                doc = (f.__doc__ or '').strip().split('\n', 1)[0]
                print(f'      expected: {f.EXPECTED.value}')
                if doc:
                    print(f'      claim:    {doc}')
        print(
            f'findings: {len(findings)} declared, {drift_count} drift',
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
