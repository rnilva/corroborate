"""Read stable-baselines3 Logger output into cells.

SB3's `Logger` (`configure(folder, ['json', 'csv'])`) writes one
record per `dump()`: a flat mapping of author-namespaced keys
(`rollout/ep_rew_mean`, `train/loss`, `eval/mean_reward`,
`time/total_timesteps`, plus whatever the producer `record()`s).
That is already the shape the framework wants — configs +
records, all columns — so this reader is mechanical:

- **One log → one row.** A progress file is one producer log;
  it becomes one cell. Time lives *inside* the row: every record
  key becomes an array column with one slot per dump (NaN for
  numeric keys, null otherwise, where the key wasn't recorded
  at that dump), aligned by the
  `time/total_timesteps` column — the same wide shape the native
  trace store uses (`TraceRow` array leaves + index arrays).
- **Keys are kept verbatim.** The reader renames nothing and
  reduces nothing: no terminal means, no AUC, no "outcome"
  classification. Reductions are claim-side (`@measurable` /
  `corroborate.measurables.reductions`) — the claim decides what
  is scoped over, targeted or ignored.
- **Configuration is the producer's statement, not forensics.**
  A key under `config/` (one `logger.record(f'config/{k}', v)`
  loop in the training script) is registry; recorded values must
  agree across dumps. No checkpoint-zip excavation, hence no
  stable-baselines3 import anywhere in this module.

What makes two rows the same experimental unit — pairing, arms,
scope — is declared by each claim (`pair_by`, `DoEffect`,
`scope`), never inferred here; `load_sb3_logger_runs` is a
zero-semantics walk (every file the glob matches is a row), and
callers with a different disk layout assemble their own frame
from `read_progress` + `progress_columns` and
`Panel.from_dataframe`.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl

from corroborate.corpus.schema import MeasurementLeaf
from corroborate.data.derive import is_scalar_leaf
from corroborate.data.panel import CorpusSource, Panel

PROGRESS_GLOB = '**/progress.json'
CONFIG_PREFIX = 'config/'

type CellValue = (
    MeasurementLeaf
    | list[int | None]
    | list[float | None]
    | list[str | None]
    | list[bool | None]
)


_INT_TEXT = re.compile(r'-?\d+')
_FLOAT_TEXT = re.compile(r'-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?')


def _parse_csv_value(text: str) -> MeasurementLeaf | None:
    """CSV erased the types; recover them mechanically — and
    narrowly. Empty string is how the CSV writer renders a key
    absent from that dump: read it back as absence, not a value.
    Only unambiguous decimal literals become numbers, plus the
    three texts `str(float)` itself produces — 'nan', 'inf',
    '-inf' — because SB3's CSV writer renders values with str(),
    so those ARE the floats the producer had (a routine
    `rollout/ep_rew_mean` before the first episode completes is
    the text 'nan'). Everything else ambiguous — '1_000',
    'True' — stays the string the file carries; int()/float()
    leniency would silently retype it."""
    if text == '':
        return None
    if text in ('nan', 'inf', '-inf'):
        return float(text)
    if _INT_TEXT.fullmatch(text):
        return int(text)
    if _FLOAT_TEXT.fullmatch(text):
        return float(text)
    return text


def read_progress(path: Path | str) -> list[dict[str, MeasurementLeaf]]:
    """One Logger output file → its dumps, verbatim: one mapping
    per `dump()`, keys exactly as recorded, absent keys absent.

    `progress.json` preserves types and absence natively (one
    JSON object per line, keys omitted when not recorded).
    `progress.csv` conflates absence with empty text and renders
    numbers as strings; parsing recovers both readings — prefer
    the json format when configuring the producer."""
    file = Path(path)
    dumps: list[dict[str, MeasurementLeaf]] = []
    if file.name.endswith('.json') or file.suffix == '.jsonl':
        for n, line in enumerate(
            file.read_text(encoding='utf-8').splitlines(), start=1,
        ):
            if not line.strip():
                continue
            raw: object = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f'sb3_logger: {file}:{n} is not a JSON object',
                )
            dump: dict[str, MeasurementLeaf] = {}
            for key_obj, value in raw.items():
                key = str(key_obj)
                if value is None:
                    continue  # explicit null == not recorded
                if not is_scalar_leaf(value):
                    raise ValueError(
                        f'sb3_logger: {file}:{n} key {key!r} has '
                        f'non-scalar value {value!r}',
                    )
                dump[key] = value
            dumps.append(dump)
    else:
        with file.open(newline='', encoding='utf-8') as handle:
            for n, row in enumerate(csv.DictReader(handle), start=2):
                if None in row:
                    # DictReader parks a long row's extra fields
                    # under the None restkey — a truncated or
                    # hand-edited file; diagnose, don't crash on
                    # an int(list) deep in the parse.
                    raise ValueError(
                        f'sb3_logger: {file}:{n} has more fields '
                        f'than the header — ragged csv row',
                    )
                dumps.append({
                    key: parsed
                    for key, text in row.items()
                    if text is not None
                    for parsed in (_parse_csv_value(text),)
                    if parsed is not None
                })
    if not dumps:
        raise ValueError(f'sb3_logger: {file} contains no dumps')
    return dumps


def _config_scalar(
    key: str, values: list[MeasurementLeaf], where: str,
) -> MeasurementLeaf:
    """A config key is one fact; re-recording it must agree
    (NaN agrees with NaN — the repeated-unset-float case)."""
    first = values[0]
    for other in values[1:]:
        if (
            isinstance(first, float) and isinstance(other, float)
            and math.isnan(first) and math.isnan(other)
        ):
            continue
        if other != first:
            raise ValueError(
                f'sb3_logger: {where}: config key {key!r} recorded '
                f'with disagreeing values {first!r} and {other!r}',
            )
    return first


def _record_array(
    key: str, dumps: Sequence[Mapping[str, MeasurementLeaf]], where: str,
) -> CellValue:
    """One record key → one array slot per dump. Mixed int/float
    promotes to float (lossless); any other type mixture is a
    producer bug and raises rather than silently stringifying.

    Padding for absent dumps: numeric arrays pad with NaN, so
    they stay numpy-consumable end-to-end (the reductions layer's
    `from_key` does a bare `np.asarray`; a None would force
    object dtype and break every reduction). NaN already IS the
    framework's "no real value here" reading for floats
    (`compute_missing_columns` treats null and NaN alike). An
    all-present int array stays int; bool/str arrays pad with
    null — no numeric missing-marker exists for them."""
    values = [dump.get(key) for dump in dumps]
    present = [v for v in values if v is not None]
    kinds = {type(v) for v in present}
    if kinds <= {int, float}:
        if kinds == {int} and len(present) == len(values):
            return [v for v in values if isinstance(v, int)]
        floats: list[float | None] = [
            float(v) if isinstance(v, (int, float)) else float('nan')
            for v in values
        ]
        return floats
    if len(kinds) > 1:
        raise ValueError(
            f'sb3_logger: {where}: record key {key!r} mixes types '
            f'{sorted(k.__name__ for k in kinds)}',
        )
    if kinds == {bool}:
        return [v if isinstance(v, bool) else None for v in values]
    return [v if isinstance(v, str) else None for v in values]


def progress_columns(
    dumps: Sequence[Mapping[str, MeasurementLeaf]],
    *,
    config_prefix: str = CONFIG_PREFIX,
    where: str = '<dumps>',
) -> tuple[dict[str, CellValue], frozenset[str]]:
    """One log's dumps → one row's columns + its registry keys.

    Keys under `config_prefix` are configuration the producer
    declared: scalar columns, values required to agree across
    dumps, and membership in the returned registry. Every other
    key is a record: an array column aligned dump-by-dump (use
    `time/total_timesteps` — itself just a record column — as
    the time axis). Key order follows first appearance."""
    ordered: list[str] = []
    seen: set[str] = set()
    for dump in dumps:
        for key in dump:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    columns: dict[str, CellValue] = {}
    registry: set[str] = set()
    for key in ordered:
        if key.startswith(config_prefix):
            present = [v for d in dumps if (v := d.get(key)) is not None]
            columns[key] = _config_scalar(key, present, where)
            registry.add(key)
        else:
            columns[key] = _record_array(key, dumps, where)
    return columns, frozenset(registry)


def _column_kind(value: 'CellValue') -> str | None:
    """The scalar kind a column carries — a config scalar's own
    type, a record array's element type — or None when the array
    is all-missing. int and float are one numeric kind: polars
    upcasts them losslessly."""
    if isinstance(value, list):
        for element in value:
            if element is None:
                continue
            # NaN IS numeric evidence — an all-NaN float array
            # (a crashed run's safe_mean) must still conflict
            # loudly with a str array for the same key.
            return (
                'numeric' if not isinstance(element, bool)
                and isinstance(element, (int, float))
                else type(element).__name__
            )
        return None
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        return 'numeric'
    return type(value).__name__


def _progress_files(root: Path, glob: str) -> list[tuple[str, Path]]:
    """Every file matching `glob` under `root`, as (row id, file).
    The id is the file's path relative to `root` — the one
    identity the filesystem already guarantees unique, free of
    invented meaning; a directory is never a unit here, so no
    id can collide and no format is ever preferred over another.
    Semantic identity, when wanted, belongs to `config/*`
    columns the producer recorded."""
    pairs = [
        (str(file.relative_to(root)), file)
        for file in sorted(root.glob(glob))
        if file.is_file()
    ]
    if not pairs:
        raise ValueError(
            f'sb3_logger: no progress files matching {glob!r} '
            f'under {root}',
        )
    return pairs


def load_sb3_logger_runs(
    root: Path | str,
    *,
    corpus: str | None = None,
    config_prefix: str = CONFIG_PREFIX,
    glob: str = PROGRESS_GLOB,
) -> Panel:
    """Zero-semantics walk: every file matching `glob` under
    `root` is one row, `id` = the file's path relative to `root`
    (the identity the filesystem already guarantees unique;
    semantic identity belongs to `config/*` columns). SB3's json
    rendering is the default; csv logs pass
    `glob='**/progress.csv'`. Rows
    concatenate diagonally — heterogeneous keys null-pad — and
    the registry is the union of every log's `config/*` keys.
    The returned Panel carries cells + registry + provenance, so
    `evaluate(claim, panel)` needs nothing else.

    No unit semantics: whether two rows are the same seed, the
    same condition, or duplicates is the claim's declaration
    (`pair_by`, `DoEffect`, `scope`) checked by its admission
    gates over the cells it admits. A folder carrying both Logger
    renderings under a `progress.*` glob is two matched files —
    two declared rows: the paired family's duplicate-unit checks
    surface that (`dedupe_strategy='raise'`), but unpaired
    analyses pool what they are given, so point the glob at ONE
    rendering unless duplication is meant."""
    root_path = Path(root)
    corpus_name = corpus if corpus is not None else root_path.name
    rows: list[dict[str, CellValue]] = []
    registry: set[str] = set()
    kind_by_column: dict[str, tuple[str, str]] = {}
    for row_id, file in _progress_files(root_path, glob):
        columns, leaves = progress_columns(
            read_progress(file),
            config_prefix=config_prefix,
            where=str(file),
        )
        for reserved in ('id', 'corpus'):
            if reserved in columns:
                raise ValueError(
                    f'sb3_logger: {file} records the reserved key '
                    f'{reserved!r}',
                )
        # Cross-file kind agreement: polars' diagonal concat
        # unifies conflicting list dtypes to List(String)
        # SILENTLY — a bool logged by one run and a 'True' string
        # by another (a json/csv mixed folder) must be a loud
        # producer error, never a stringified column.
        for key, value in columns.items():
            kind = _column_kind(value)
            if kind is None:
                continue
            earlier = kind_by_column.get(key)
            if earlier is not None and earlier[0] != kind:
                raise ValueError(
                    f'sb3_logger: column {key!r} is {earlier[0]} '
                    f'in {earlier[1]} but {kind} in {file} — '
                    f'mixed-type records do not pool',
                )
            kind_by_column.setdefault(key, (kind, str(file)))
        rows.append({'id': row_id, 'corpus': corpus_name, **columns})
        registry |= leaves
    return Panel.from_dataframe(
        pl.from_dicts(rows, infer_schema_length=None),
        sources=(
            CorpusSource(corpus=corpus_name, data_root=root_path),
        ),
        leaves=frozenset(registry),
    )


__all__ = [
    'CONFIG_PREFIX',
    'PROGRESS_GLOB',
    'load_sb3_logger_runs',
    'progress_columns',
    'read_progress',
]
