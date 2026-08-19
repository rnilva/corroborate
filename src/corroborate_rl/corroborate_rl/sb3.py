"""Read stable-baselines3 training artifacts into the neutral shape.

The framework side (`corroborate.data.load_runs`) reads a neutral
directory format and stays library-blind; this module is the
RL-side integration that turns what SB3 users ALREADY have — a
folder of runs, each with a checkpoint zip and an `EvalCallback`
`evaluations.npz` — into the same one-row-per-run DataFrame plus
the configuration-leaf registry, with zero changes to their
training script.

Expected layout (what `model.save()` + `EvalCallback` produce)::

    <root>/
      <run>/model.zip           (or best_model.zip / any *.zip)
      <run>/evaluations.npz     timesteps × per-episode returns

Two artifact facts drive the design:

- A checkpoint zip's ``data`` entry is a JSON dump of the
  algorithm's resolved state — configuration AND runtime state
  (``num_timesteps``, ``exploration_rate``) mixed, because
  ``save()`` exists to let ``load()`` resume. The configuration
  view is recovered by intersecting with the constructor's
  signature: a leaf is a parameter the entry point accepts, the
  same registration principle as the native substrate's
  ``walk_paths(claim, regime='leaf')``. Entries cloudpickled by
  SB3 (dicts carrying ``:serialized:``) are dropped — they are
  callables/spaces, not scalar configuration.
- ``evaluations.npz`` carries ``timesteps`` (one per evaluation
  point) and ``results`` (per-episode returns at each point) —
  exactly the per-checkpoint evaluation record the neutral loader
  aggregates, with the episode index in place of an evaluation
  seed.

Import discipline (cf. `dqn_sweep`): nothing heavy at module
level. `stable_baselines3` is needed ONLY when `algo` is given as
a string; pass the algorithm class itself and this module runs
without SB3 installed.
"""
from __future__ import annotations

import inspect
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

# In-repo private reuse: the derivation semantics (dotted-path
# flattening, per-checkpoint aggregation, collision policy) must
# stay bit-identical to the neutral loader's, so the helpers are
# shared rather than re-implemented.
from corroborate.corpus.schema import MeasurementLeaf
from corroborate.data.loader import (
    _derive_outcomes,
    _flatten_config,
    _outcome_globals,
    _put,
)

_SERIALIZED_MARKER = ':serialized:'


def _constructor_parameters(algo: type | str) -> frozenset[str]:
    """Parameter names of the algorithm's constructor — the leaf
    registry ("assignable" means "the entry point accepts it").

    A string resolves against `stable_baselines3` (imported here,
    lazily, so the rest of the module works without it); a class
    is inspected directly."""
    if isinstance(algo, str):
        try:
            import stable_baselines3
        except ImportError as exc:
            raise ImportError(
                f'corroborate_rl.sb3: resolving algo={algo!r} needs '
                f'stable-baselines3 installed; pass the algorithm '
                f'class itself to avoid the dependency',
            ) from exc
        resolved: object = getattr(stable_baselines3, algo, None)
        if not isinstance(resolved, type):
            raise ValueError(
                f'corroborate_rl.sb3: stable_baselines3 has no '
                f'algorithm class named {algo!r}',
            )
        algo_class = resolved
    else:
        algo_class = algo
    return frozenset(inspect.signature(algo_class.__init__).parameters) - {
        'self',
    }


def checkpoint_config(
    zip_path: Path | str,
    algo: type | str,
) -> dict[str, object]:
    """The resolved configuration inside an SB3 checkpoint zip:
    the ``data`` JSON entries that are constructor parameters.
    Nested mappings survive (they flatten to dotted paths
    downstream). Cloudpickled entries (``train_freq``, class
    references — anything SB3 could not JSON-encode) are kept as
    their opaque serialised payload strings: not human-readable,
    but equality-comparable, so a configuration difference in
    them stays visible to the isolation gate instead of silently
    vanishing from the registry. Constructor parameters absent
    from the ``data`` record (``policy``, ``env``, ``device``)
    are genuinely unrecoverable and stay absent — the registry is
    exactly the recoverable slice, no more claimed than that."""
    parameters = _constructor_parameters(algo)
    path = Path(zip_path)
    with zipfile.ZipFile(path) as archive:
        raw: object = json.loads(archive.read('data').decode('utf-8'))
    if not isinstance(raw, dict):
        raise ValueError(
            f'corroborate_rl.sb3: {path} has a non-object data entry',
        )
    # Runtime invariant: json.loads object keys are always str.
    entries: dict[str, object] = {str(k): v for k, v in raw.items()}
    config: dict[str, object] = {}
    for name, value in entries.items():
        if name not in parameters:
            continue  # runtime state (num_timesteps, ...) — not a leaf
        if isinstance(value, Mapping) and _SERIALIZED_MARKER in value:
            payload = value.get(_SERIALIZED_MARKER)
            config[name] = (
                payload if isinstance(payload, str)
                else json.dumps(payload, sort_keys=True)
            )
            continue
        config[name] = value
    return config


def _first_zip(run_dir: Path, checkpoint: str | None) -> Path | None:
    """The run's checkpoint zip; None for a directory that is not
    a run (a tensorboard folder, a stray subdirectory) — real log
    folders carry those, and a reader should walk past them.

    Selection must be deliberate, never lexicographic: an explicit
    `checkpoint` filename wins; otherwise `model.zip` /
    `best_model.zip`; otherwise a SOLE archive is unambiguous, and
    several (a `CheckpointCallback` series) raise — picking one
    silently would bind the claim to an arbitrary training
    budget."""
    if checkpoint is not None:
        candidate = run_dir / checkpoint
        return candidate if candidate.is_file() else None
    preferred = [run_dir / 'model.zip', run_dir / 'best_model.zip']
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    others = sorted(run_dir.glob('*.zip'))
    if len(others) > 1:
        raise ValueError(
            f'corroborate_rl.sb3: {run_dir} carries several '
            f'checkpoint zips ({[p.name for p in others]}) and none '
            f'named model.zip/best_model.zip — pass '
            f'`checkpoint=<filename>` to select one deliberately.',
        )
    return others[0] if others else None


def _run_zips(
    root: Path, checkpoint: str | None,
) -> list[tuple[Path, Path]]:
    """(run_dir, checkpoint zip) for each subdirectory of `root`
    that carries one; raises only when none do."""
    pairs = [
        (run_dir, zip_path)
        for run_dir in sorted(d for d in root.iterdir() if d.is_dir())
        for zip_path in [_first_zip(run_dir, checkpoint)]
        if zip_path is not None
    ] if root.is_dir() else []
    if not pairs:
        raise ValueError(
            f'corroborate_rl.sb3: no run directories with a '
            f'checkpoint zip under {root}',
        )
    return pairs


def _evaluations(
    npz_path: Path,
) -> dict[int, dict[str, list[float]]]:
    """`evaluations.npz` → the per-checkpoint sample mapping the
    neutral loader's derivation consumes (`return` samples keyed
    by checkpoint, one per evaluation episode)."""
    with np.load(npz_path) as archive:
        # numpy's untyped __getitem__/`tolist` yield Any; launder
        # to object, then cast to the npz contract EvalCallback
        # writes (1-D numeric timesteps, 2-D float results) at
        # this one boundary.
        timesteps_obj: object = archive['timesteps'].tolist()
        results_obj: object = archive['results'].tolist()
    timesteps = cast('list[float]', timesteps_obj)
    results = cast('list[list[float]]', results_obj)
    if len(timesteps) != len(results):
        raise ValueError(
            f'corroborate_rl.sb3: {npz_path} timesteps/results '
            f'length mismatch',
        )
    per_checkpoint: dict[int, dict[str, list[float]]] = {}
    for step, episodes in zip(timesteps, results):
        if not float(step).is_integer():
            raise ValueError(
                f'corroborate_rl.sb3: {npz_path} non-integer '
                f'timestep {step!r}',
            )
        checkpoint = int(step)
        if checkpoint in per_checkpoint:
            raise ValueError(
                f'corroborate_rl.sb3: {npz_path} duplicate '
                f'evaluation timestep {checkpoint}',
            )
        # Keep every attempted episode, non-finite included — the
        # shared derivation filters to finite samples per
        # checkpoint and retains the attempted count, so a failed
        # evaluation stays visible instead of shrinking silently.
        per_checkpoint[checkpoint] = {
            'return': [float(r) for r in episodes],
        }
    return per_checkpoint


def load_sb3_runs(
    root: Path | str,
    algo: type | str,
    *,
    corpus: str | None = None,
    checkpoint: str | None = None,
) -> pl.DataFrame:
    """One row per run subdirectory of `root`: configuration from
    the checkpoint zip (flattened to dotted-path columns) plus the
    derived evaluation aggregates (`return_mean` at the
    record-wide terminal evaluation point — null for runs not
    evaluated there — with `return_terminal_n` /
    `return_terminal_attempted` counts, `return_auc` for runs
    covering the full grid, and one `return_mean_at_<step>` column
    per evaluation point).

    Same shape, derivation, and collision policy as
    `corroborate.data.load_runs`; the run id is the subdirectory
    name. `checkpoint` selects a specific zip filename when runs
    carry several archives. Pair with
    `sb3_config_columns(root, algo)` for the leaf registry
    `evaluate(..., leaves=...)` consumes."""
    root_path = Path(root)
    run_zips = _run_zips(root_path, checkpoint)
    evaluations: dict[str, dict[int, dict[str, list[float]]]] = {}
    for run_dir, _zip_path in run_zips:
        npz_path = run_dir / 'evaluations.npz'
        if npz_path.is_file():
            evaluations[run_dir.name] = _evaluations(npz_path)
    terminal_by_outcome, grid_by_outcome = _outcome_globals(evaluations)
    rows: list[dict[str, MeasurementLeaf]] = []
    for run_dir, zip_path in run_zips:
        run_id = run_dir.name
        row: dict[str, MeasurementLeaf] = {'id': run_id}
        _put(
            row, 'corpus',
            corpus if corpus is not None else root_path.name,
            run_id=run_id,
        )
        config = checkpoint_config(zip_path, algo)
        for key, value in _flatten_config(config).items():
            _put(row, key, value, run_id=run_id)
        per_checkpoint = evaluations.get(run_id)
        if per_checkpoint is not None:
            _derive_outcomes(
                row, per_checkpoint, run_id=run_id,
                terminal_by_outcome=terminal_by_outcome,
                grid_by_outcome=grid_by_outcome,
            )
        rows.append(row)
    return pl.from_dicts(rows, infer_schema_length=None)


def sb3_config_columns(
    root: Path | str,
    algo: type | str,
    *,
    checkpoint: str | None = None,
) -> frozenset[str]:
    """The configuration-leaf registry of an SB3 run folder: the
    union of dotted-path column names the runs' checkpoint configs
    flatten to — the recoverable slice of what was configured
    (constructor parameters present in the checkpoint's ``data``
    record; see `checkpoint_config` for what that includes and
    excludes). Counterpart of `corroborate.data.config_columns`
    for records whose config artifact is the checkpoint zip."""
    names: set[str] = set()
    for _run_dir, zip_path in _run_zips(Path(root), checkpoint):
        config = checkpoint_config(zip_path, algo)
        names.update(_flatten_config(config))
    return frozenset(names)


__all__ = [
    'checkpoint_config',
    'load_sb3_runs',
    'sb3_config_columns',
]
