"""CLI thin-wrapper around `corroborate.runner.run`.

Run any bridges-module-as-hypothesis (`experiments/findings/<X>.py`
exporting `INTERVENTION` + `BRIDGES`) on a data input, with the
per-hypothesis cache:

    python scripts/run_hypothesis.py experiments.findings.ddqn_universe \\
        --data experiments/data/

Library code lives in `corroborate.runner`; this file is purely the
argparse + verdict-printing surface."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from corroborate.bridge import BridgeEvaluation
from corroborate.runner import check, run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='run_hypothesis',
        description='Run a hypothesis-module on a data input, with cache.',
    )
    parser.add_argument(
        'module',
        help='dotted module path, e.g. experiments.findings.ddqn_universe',
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
    args = parser.parse_args(argv)

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

    results = run(
        cast(str, args.module),
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
    _print_verdicts(results)
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


def _print_verdicts(results: dict[str, BridgeEvaluation]) -> None:
    counts: dict[str, int] = {}
    for name, ev in results.items():
        v = ev.verdict.value
        counts[v] = counts.get(v, 0) + 1
        bits: list[str] = []
        for ar in ev.analysis_results.values():
            bits.append(_summarize(ar))
        suffix = ' | '.join(bits) if bits else ''
        print(f'{name:60s}  {v:24s}  {suffix}')
    print()
    print('verdict counts:')
    for k in sorted(counts):
        print(f'  {k:24s}  {counts[k]}')


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
