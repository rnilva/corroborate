"""Shared derivation kernel for run readers.

`load_runs` (neutral JSON layout) and `corroborate_rl.sb3`
(SB3 artifacts) must produce bit-identical column semantics —
same flattening, same collision policy, same terminal-summary
definition — or the meaning of a derived column would depend on
which reader built the frame. This module is the single home of
those semantics; readers import it rather than re-implementing.

The load-bearing definitions:

- **Flattening** (`flatten_config`): nested configuration →
  dotted-path leaves; array values encode as canonical JSON so a
  structured difference between arms stays observable; colliding
  dotted paths raise rather than silently overwrite.
- **Column collision** (`put_column`): a repeated column is
  tolerated only when the values agree — the one case where
  accepting it loses nothing.
- **Terminal summary** (`derive_outcomes` + `outcome_globals`):
  ``<outcome>_mean`` means the mean at the RECORD-WIDE terminal
  checkpoint — null for a run not evaluated there, never a
  silent rebase to an earlier horizon — with
  ``<outcome>_terminal_n`` / ``<outcome>_terminal_attempted``
  retaining what it stands on; ``<outcome>_auc`` only for runs
  covering the record-wide grid; ``<outcome>_mean_at_<cp>`` as
  the per-run explicit-horizon surface.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import TypeIs

from corroborate.corpus.schema import MeasurementLeaf


def is_scalar_leaf(value: object) -> TypeIs[MeasurementLeaf]:
    return isinstance(value, (str, int, float, bool))


def put_column(
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
            f'run {run_id!r} has conflicting values for '
            f'column {key!r}: {row[key]!r} != {value!r}',
        )
    row[key] = value


def flatten_config(
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
            flatten_config(nested, prefix=f'{path}.', _flat=flat)
            continue
        if value is None:
            continue
        if path in flat:
            raise ValueError(
                f'configuration flattens two entries to the same '
                f'path {path!r} — dotted keys collide with nesting',
            )
        if is_scalar_leaf(value):
            flat[path] = value
        else:
            # Array-valued configuration: canonical JSON keeps the
            # difference observable and equality-comparable.
            flat[path] = json.dumps(value, sort_keys=True)
    return flat


def normalised_auc(
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


def outcome_globals(
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


def derive_outcomes(
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
        put_column(
            row, f'{outcome}_terminal_attempted', len(terminal_samples),
            run_id=run_id,
        )
        put_column(
            row, f'{outcome}_terminal_n', len(terminal_finite),
            run_id=run_id,
        )
        if terminal_finite:
            put_column(
                row, f'{outcome}_mean',
                math.fsum(terminal_finite) / len(terminal_finite),
                run_id=run_id,
            )
        if grid and frozenset(grid) == grid_by_outcome[outcome]:
            put_column(
                row, f'{outcome}_auc',
                normalised_auc(tuple(grid), tuple(means)),
                run_id=run_id,
            )
        # The trajectory as flat checkpoint-keyed scalar columns —
        # null-padded on diagonal concat across run sets with
        # different checkpoint grids.
        for checkpoint, mean in zip(grid, means):
            put_column(
                row, f'{outcome}_mean_at_{checkpoint}', mean,
                run_id=run_id,
            )


__all__ = [
    'derive_outcomes',
    'flatten_config',
    'is_scalar_leaf',
    'normalised_auc',
    'outcome_globals',
    'put_column',
]
