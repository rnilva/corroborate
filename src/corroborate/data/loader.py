"""Load externally-produced run records into a DataFrame.

`load_runs` is a convenience reader, not a gatekeeper: it turns a
directory of plain producer files into the framework's canonical
cell shape (one row per seeded run, path-keyed scalar columns) and
nothing else. It holds no opinion about study design — pairing,
configuration isolation, and evidence quality are claims about a
*contrast*, so they are checked by the admission gates of the
claim being evaluated, scoped to exactly the cells that claim
admits. A producer who already has a DataFrame skips this module
entirely.

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

Malformed structure raises a plain ``ValueError`` (duplicate run
ids, duplicate evaluation records, evaluations for unknown runs,
conflicting values for one column). There is no verdict
vocabulary here; evidence is a live record that this module
merely reads.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TypeIs

import polars as pl

from corroborate.data._run_io import read_json, read_jsonl, safe_run_path
from corroborate.corpus.schema import MeasurementLeaf


def _is_scalar(value: object) -> TypeIs[MeasurementLeaf]:
    return isinstance(value, (str, int, float, bool))


def _is_number(value: object) -> TypeIs[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _put(
    row: dict[str, MeasurementLeaf],
    key: str,
    value: MeasurementLeaf,
    *,
    run_id: str,
) -> None:
    """Set a column on a run's row; a repeated key is tolerated
    only when the values agree (producers often stamp e.g. ``seed``
    on both the run record and the configuration), because that is
    the one case where accepting it loses nothing."""
    if key in row and row[key] != value:
        raise ValueError(
            f'load_runs: run {run_id!r} has conflicting values for '
            f'column {key!r}: {row[key]!r} != {value!r}',
        )
    row[key] = value


def _flatten_config(
    config: Mapping[str, object],
    *,
    prefix: str = '',
    _flat: dict[str, MeasurementLeaf] | None = None,
) -> dict[str, MeasurementLeaf]:
    """Nested configuration mapping → dotted-path leaves.

    Scalars pass through; array-valued leaves (`net_arch: [64, 64]`)
    are encoded as canonical JSON strings so a structured
    configuration difference between arms stays visible to the
    isolation gate rather than silently vanishing — the registry
    must not be a lossy projection of what was configured. Null
    leaves are skipped: an absent column and a stored null read
    identically (`row.get -> None`), so nothing observable is lost.

    Dotted paths are not injective (`{'a': {'b': 1}, 'a.b': 2}`
    collide), so a duplicate flattened path is rejected rather
    than silently overwritten."""
    flat: dict[str, MeasurementLeaf] = {} if _flat is None else _flat
    for key, value in config.items():
        path = f'{prefix}{key}'
        if isinstance(value, Mapping):
            # Runtime invariant: json.loads mapping keys are str.
            nested = {str(k): v for k, v in value.items()}
            _flatten_config(nested, prefix=f'{path}.', _flat=flat)
            continue
        if value is None:
            continue
        if path in flat:
            raise ValueError(
                f'configuration flattens two entries to the same '
                f'path {path!r} — dotted keys collide with nesting',
            )
        if _is_scalar(value):
            flat[path] = value
        else:
            # Array-valued configuration: canonical JSON keeps the
            # difference observable and equality-comparable.
            flat[path] = json.dumps(value, sort_keys=True)
    return flat


def _normalised_auc(
    checkpoints: tuple[int, ...],
    means: tuple[float, ...],
) -> float:
    """Trapezoid area over the checkpoint axis, normalised by its
    span — reduces to the single checkpoint mean when the run was
    evaluated once."""
    if len(checkpoints) == 1:
        return means[0]
    area = 0.0
    for index in range(len(checkpoints) - 1):
        step = float(checkpoints[index + 1] - checkpoints[index])
        area += step * (means[index] + means[index + 1]) / 2.0
    return area / float(checkpoints[-1] - checkpoints[0])


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
    samples (one sample per logged evaluation seed)."""
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


def _outcome_globals(
    evaluations: Mapping[str, Mapping[int, Mapping[str, list[float]]]],
) -> tuple[dict[str, int], dict[str, frozenset[int]]]:
    """Per outcome, the record-wide terminal checkpoint (the
    largest at which ANY run evaluated it) and the record-wide
    checkpoint grid. The terminal defines what `<outcome>_mean`
    MEANS for every row — one horizon, not "whatever this run
    reached"."""
    grids: dict[str, set[int]] = {}
    for per_checkpoint in evaluations.values():
        for checkpoint, outcomes in per_checkpoint.items():
            for name in outcomes:
                grids.setdefault(name, set()).add(checkpoint)
    return (
        {name: max(grid) for name, grid in grids.items()},
        {name: frozenset(grid) for name, grid in grids.items()},
    )


def _derive_outcomes(
    row: dict[str, MeasurementLeaf],
    per_checkpoint: Mapping[int, Mapping[str, list[float]]],
    *,
    run_id: str,
    terminal_by_outcome: Mapping[str, int],
    grid_by_outcome: Mapping[str, frozenset[int]],
) -> None:
    """Derived outcome columns, comparable by construction.

    `<outcome>_mean` is the finite-sample mean AT THE RECORD-WIDE
    TERMINAL CHECKPOINT — a run not evaluated there (or with no
    finite sample there) gets null, never a silent rebase to an
    earlier horizon: two arms evaluated to different training
    budgets must not manufacture an effect through the terminal
    summary. `<outcome>_terminal_n` / `<outcome>_terminal_attempted`
    retain how many finite samples the terminal mean stands on and
    how many evaluations were attempted there. `<outcome>_auc` is
    derived only when the run covers the record-wide grid with
    finite means (partial-horizon areas are not comparable).
    `<outcome>_mean_at_<checkpoint>` stays per-run and null-pads —
    the explicit-horizon surface for claims at a chosen budget."""
    checkpoints = tuple(sorted(per_checkpoint))
    outcome_names = sorted(
        {name for samples in per_checkpoint.values() for name in samples},
    )
    for outcome in outcome_names:
        grid: list[int] = []
        means: list[float] = []
        for checkpoint in checkpoints:
            if outcome not in per_checkpoint[checkpoint]:
                continue
            finite = [
                s for s in per_checkpoint[checkpoint][outcome]
                if math.isfinite(s)
            ]
            if not finite:
                continue
            grid.append(checkpoint)
            means.append(math.fsum(finite) / len(finite))
        terminal = terminal_by_outcome[outcome]
        terminal_samples = per_checkpoint.get(terminal, {}).get(outcome, [])
        terminal_finite = [
            s for s in terminal_samples if math.isfinite(s)
        ]
        _put(
            row, f'{outcome}_terminal_attempted', len(terminal_samples),
            run_id=run_id,
        )
        _put(
            row, f'{outcome}_terminal_n', len(terminal_finite),
            run_id=run_id,
        )
        if terminal_finite:
            _put(
                row, f'{outcome}_mean',
                math.fsum(terminal_finite) / len(terminal_finite),
                run_id=run_id,
            )
        if grid and frozenset(grid) == grid_by_outcome[outcome]:
            _put(
                row, f'{outcome}_auc',
                _normalised_auc(tuple(grid), tuple(means)),
                run_id=run_id,
            )
        # The trajectory as flat checkpoint-keyed scalar columns —
        # null-padded on diagonal concat across run sets with
        # different checkpoint grids.
        for checkpoint, mean in zip(grid, means):
            _put(
                row, f'{outcome}_mean_at_{checkpoint}', mean,
                run_id=run_id,
            )


def load_runs(
    root: Path | str,
    *,
    corpus: str | None = None,
) -> pl.DataFrame:
    """Read a producer's run directory into one row per run.

    ``corpus`` names the run set in the ``corpus`` column
    (defaults to the directory name) so rows from several
    directories stay distinguishable after concatenation — the
    concatenation itself is plain ``pl.concat(..., how='diagonal')``,
    batches of a growing study being one run set that happens to
    arrive in parts."""
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
    terminal_by_outcome, grid_by_outcome = _outcome_globals(evaluations)

    rows: list[dict[str, MeasurementLeaf]] = []
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
        _put(
            row, 'corpus',
            corpus if corpus is not None else root_path.name,
            run_id=run_id,
        )
        if program is not None:
            _put(row, 'program', program, run_id=run_id)
        for field, value in raw.items():
            if field in ('run_id', 'config_path'):
                continue
            if _is_scalar(value):
                _put(row, field, value, run_id=run_id)
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
            for key, value in _flatten_config(typed_config).items():
                _put(row, key, value, run_id=run_id)
        per_checkpoint = evaluations.pop(run_id, None)
        if per_checkpoint is not None:
            _derive_outcomes(
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
    return pl.from_dicts(rows, infer_schema_length=None)


def config_columns(root: Path | str) -> frozenset[str]:
    """The configuration leaves of a run directory: the union of
    dotted-path column names its resolved-config files flatten to.

    The names are read from resolved-config files that already exist,
    never authored as a second manual list. Pass the result to
    `evaluate(..., leaves=...)` to compare observable configuration
    values between declared arms within pairing units. The registry
    means only "what the producer's record says was configured": it
    cannot witness assignment, randomisation, completeness, producer
    authenticity, or the absence of hidden confounding. A config file
    that logs junk registers junk; a zero-authority reader cannot do
    better than the supplied record."""
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
        names.update(_flatten_config(typed_config))
    return frozenset(names)


__all__ = ['config_columns', 'load_runs']
