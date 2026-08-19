"""Load externally-produced run records into a Panel.

`load_runs` is a convenience reader, not a gatekeeper: it turns a
directory of plain producer files into the framework's canonical
cell shape (one row per seeded run, path-keyed scalar columns) and
returns it as a `Panel` — the typed carrier for the two facts a
bare DataFrame cannot hold: provenance (`sources`) and the
configuration registry (`leaves`, read off the record's own
config files). `evaluate()` accepts the Panel directly, so the
registry travels with the cells instead of being re-derived and
hand-passed. The reader holds no opinion about study design —
pairing, configuration isolation, and evidence quality are claims
about a *contrast*, so they are checked by the admission gates of
the claim being evaluated, scoped to exactly the cells that claim
admits. A producer who already has a DataFrame skips this module
entirely (`Panel.from_dataframe(df, leaves=...)`, or pass the
bare frame plus `leaves=` to `evaluate`).

Directory layout (only ``runs.jsonl`` is required)::

    <root>/
      runs.jsonl           one record per run; ``run_id`` required,
                           other scalar fields become columns,
                           ``config_path`` points at the resolved
                           configuration actually used
      configs/<run>.json   flattened into dotted-path columns
                           (``gamma``, ``optimizer.lr``, ...)
      evaluations.jsonl    one record per (run, checkpoint[, eval
                           seed]); numeric fields are outcomes
      provenance.json      optional; ``producer`` becomes the
                           ``program`` column

From the evaluation records the loader derives, per outcome
field: the mean at the RECORD-WIDE terminal checkpoint
(``<outcome>_mean`` — null for a run not evaluated there, never
silently rebased to an earlier horizon), the finite/attempted
sample counts behind it (``<outcome>_terminal_n`` /
``<outcome>_terminal_attempted``), a checkpoint-normalised area
under the curve (``<outcome>_auc``, derived only for runs
covering the record-wide grid), and the trajectory as one scalar
column per checkpoint (``<outcome>_mean_at_<checkpoint>``) —
means taken over the finite samples logged at that checkpoint.
The definitions live in `corroborate.data.derive`, shared with
every other run reader.

Malformed structure raises a plain ``ValueError`` (duplicate run
ids, duplicate evaluation records, evaluations for unknown runs,
conflicting values for one column, colliding dotted paths). There
is no verdict vocabulary here; evidence is a live record that
this module merely reads.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeIs

from corroborate.data._run_io import read_json, read_jsonl, safe_run_path
from corroborate.data.derive import (
    derive_outcomes,
    flatten_config,
    is_scalar_leaf,
    outcome_globals,
    put_column,
)
from corroborate.data.panel import CorpusSource, Panel
from corroborate.corpus.schema import MeasurementLeaf

import polars as pl


def _is_number(value: object) -> TypeIs[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_checkpoint(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f'load_runs: {where}: checkpoint must be an integer, '
            f'got {value!r}',
        )
    return value


def _read_evaluations(
    root: Path,
) -> dict[str, dict[int, dict[str, list[float]]]]:
    """``evaluations.jsonl`` → per-run, per-checkpoint outcome
    samples (one sample per logged evaluation seed; non-finite
    samples are kept — the derivation counts them as attempted)."""
    path = root / 'evaluations.jsonl'
    by_run: dict[str, dict[int, dict[str, list[float]]]] = {}
    if not path.is_file():
        return by_run
    seen: set[tuple[str, int, object]] = set()
    for record in read_jsonl(path):
        run_id = record.get('run_id')
        if not isinstance(run_id, str):
            raise ValueError(
                'load_runs: evaluation record without a string run_id: '
                f'{record!r}',
            )
        checkpoint = _as_checkpoint(
            record.get('checkpoint'), f'evaluation of run {run_id!r}',
        )
        eval_seed = record.get('eval_seed')
        key = (run_id, checkpoint, eval_seed)
        if key in seen:
            raise ValueError(
                f'load_runs: duplicate evaluation record for run '
                f'{run_id!r} at checkpoint {checkpoint} '
                f'(eval_seed={eval_seed!r}) — distinguish repeated '
                'evaluations by eval_seed',
            )
        seen.add(key)
        outcomes = by_run.setdefault(run_id, {}).setdefault(checkpoint, {})
        for field, value in record.items():
            if field in ('run_id', 'checkpoint', 'eval_seed'):
                continue
            if _is_number(value):
                outcomes.setdefault(field, []).append(float(value))
    return by_run


def load_runs(
    root: Path | str,
    *,
    corpus: str | None = None,
) -> Panel:
    """Read a producer's run directory into a Panel of one row per
    run, carrying the configuration registry and provenance.

    ``corpus`` names the run set in the ``corpus`` column
    (defaults to the directory name) so rows from several
    directories stay distinguishable after pooling — batches of a
    growing study pool with
    `corroborate.data.concat_panels([a, b])`, one run set that
    happens to arrive in parts."""
    root_path = Path(root)
    runs_path = root_path / 'runs.jsonl'
    if not runs_path.is_file():
        raise ValueError(f'load_runs: {runs_path} does not exist')
    raw_runs = read_jsonl(runs_path)
    if not raw_runs:
        raise ValueError(f'load_runs: {runs_path} is empty')

    program: str | None = None
    provenance_path = root_path / 'provenance.json'
    if provenance_path.is_file():
        provenance = read_json(provenance_path)
        if isinstance(provenance, Mapping):
            producer = provenance.get('producer')
            if isinstance(producer, str):
                program = producer

    evaluations = _read_evaluations(root_path)
    terminal_by_outcome, grid_by_outcome = outcome_globals(evaluations)
    corpus_name = corpus if corpus is not None else root_path.name

    rows: list[dict[str, MeasurementLeaf]] = []
    leaves: set[str] = set()
    seen_run_ids: set[str] = set()
    for raw in raw_runs:
        run_id = raw.get('run_id')
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(
                f'load_runs: run record without a string run_id: {raw!r}',
            )
        if run_id in seen_run_ids:
            raise ValueError(f'load_runs: duplicate run_id {run_id!r}')
        seen_run_ids.add(run_id)
        row: dict[str, MeasurementLeaf] = {'id': run_id}
        put_column(row, 'corpus', corpus_name, run_id=run_id)
        if program is not None:
            put_column(row, 'program', program, run_id=run_id)
        for field, value in raw.items():
            if field in ('run_id', 'config_path'):
                continue
            if is_scalar_leaf(value):
                put_column(row, field, value, run_id=run_id)
        config_path = raw.get('config_path')
        if config_path is not None:
            if not isinstance(config_path, str):
                raise ValueError(
                    f'load_runs: run {run_id!r} config_path must be a '
                    f'string, got {config_path!r}',
                )
            config = read_json(safe_run_path(root_path, config_path))
            if not isinstance(config, Mapping):
                raise ValueError(
                    f'load_runs: {config_path} must contain a JSON '
                    'object',
                )
            # Runtime invariant: json.loads mapping keys are str.
            typed_config = {str(k): v for k, v in config.items()}
            for key, value in flatten_config(typed_config).items():
                put_column(row, key, value, run_id=run_id)
                leaves.add(key)
        per_checkpoint = evaluations.pop(run_id, None)
        if per_checkpoint is not None:
            derive_outcomes(
                row, per_checkpoint, run_id=run_id,
                terminal_by_outcome=terminal_by_outcome,
                grid_by_outcome=grid_by_outcome,
            )
        rows.append(row)

    if evaluations:
        unknown = sorted(evaluations)
        raise ValueError(
            f'load_runs: evaluations.jsonl references run id(s) not in '
            f'runs.jsonl: {unknown!r}',
        )
    return Panel.from_dataframe(
        pl.from_dicts(rows, infer_schema_length=None),
        sources=(
            CorpusSource(corpus=corpus_name, data_root=root_path),
        ),
        leaves=frozenset(leaves),
    )


def config_columns(root: Path | str) -> frozenset[str]:
    """The configuration leaves of a run directory: the union of
    dotted-path column names its resolved-config files flatten to.

    This is the external record's counterpart of the native
    substrate's `walk_paths(claim, regime='leaf')` — in both cases
    the leaf registry is read off an artifact that already exists
    (the claim composition there, the resolved-config files here),
    never authored separately. `load_runs` carries the same set on
    the returned Panel; this standalone form serves callers who
    build their frame another way and pass
    `evaluate(..., leaves=...)` explicitly. The registry means
    "what the producer's record says was configured" — a config
    file that logs junk registers junk; a zero-authority reader
    cannot do better than the record."""
    root_path = Path(root)
    runs_path = root_path / 'runs.jsonl'
    if not runs_path.is_file():
        raise ValueError(f'config_columns: {runs_path} does not exist')
    names: set[str] = set()
    for raw in read_jsonl(runs_path):
        config_path = raw.get('config_path')
        if config_path is None:
            continue
        if not isinstance(config_path, str):
            run_id = raw.get('run_id')
            raise ValueError(
                f'config_columns: run {run_id!r} config_path must be '
                f'a string, got {config_path!r}',
            )
        config = read_json(safe_run_path(root_path, config_path))
        if not isinstance(config, Mapping):
            raise ValueError(
                f'config_columns: {config_path} must contain a JSON '
                'object',
            )
        # Runtime invariant: json.loads mapping keys are str.
        typed_config = {str(k): v for k, v in config.items()}
        names.update(flatten_config(typed_config))
    return frozenset(names)


__all__ = ['config_columns', 'load_runs']
