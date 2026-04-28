"""Sweep — multi-cell runner over (env, seed) grid.

Takes a `Hypothesis[R]` and a caller-provided `runner` that knows
how to execute one cell (run the intervention against one
(env, seed) pair and produce a record of type `R`). The framework
doesn't know about training, JAX, env-stepping, etc. — those live
in the theory layer (step 3) and the runner is the boundary
between framework-side data flow and domain-specific execution.

For each cell, sweep:
1. Calls `runner(h, env_name, seed, total_steps)` and collects the
   record.
2. Applies each `h.bridges[i]` to the record, flattens the result
   into `bridge.<name>.*` / `invariant.<name>.*` measurements.
3. Builds a RunRow with substrate-metadata measurements
   (`env_name`, `seed`, `total_steps`, `intervention_name`),
   the primary outcome scalar (`outcome.late_window_mean`), and
   the bridge measurements.
4. Captures any exception as a `CellFailure` rather than crashing
   the sweep — the caller sees both successful rows and failures.

Subprocess isolation (one process per env, to prevent JIT cache
OOM as v9 documented) is deferred. v0 runs in-process; large
sweeps that need isolation can switch to a process-pool runner
later without changing the Hypothesis / Bridge API."""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from corroborate.aggregate import aggregate_cell_verdict
from corroborate.bridge import BridgeResult
from corroborate.hypothesis import Hypothesis
from corroborate.schema import MeasurementLeaf, RunRow


@dataclass(frozen=True, slots=True)
class CellFailure:
    """One cell that raised during execution. Captures provenance
    + the exception's str representation. Sweeps return failures
    alongside successful rows so callers see the gap explicitly
    (no silent drops, per v9's Pong-misc lesson)."""
    intervention_name: str
    env_name: str
    seed: int
    error: str
    duration_s: float


def _bridge_result_to_measurements(
    result: BridgeResult,
) -> dict[str, MeasurementLeaf]:
    """Flatten a BridgeResult into path-keyed measurements.

    `bridge.<name>.verdict` (or `invariant.<name>.verdict` when
    `stats['kind'] == 'tautological'`) carries the verdict; each
    scalar entry of `result.stats` lands under
    `<prefix>.<name>.stats.<key>`. Non-scalar stats (rare; not
    expected in v0) are silently dropped — the BridgeResult
    contract types `stats` to scalar primitives, so this is a
    defensive guard, not the common path."""
    is_invariant = result.stats.get('kind') == 'tautological'
    prefix = f'invariant.{result.name}' if is_invariant else f'bridge.{result.name}'
    out: dict[str, MeasurementLeaf] = {
        f'{prefix}.verdict': result.verdict.value,
    }
    # `BridgeResult.stats` is typed `Mapping[str, float | int |
    # bool | str]` — every value already satisfies
    # MeasurementLeaf. Forward each entry verbatim.
    for stat_key, stat_value in result.stats.items():
        out[f'{prefix}.stats.{stat_key}'] = stat_value
    return out


def sweep[R: Mapping[str, object]](
    h: Hypothesis[R],
    *,
    env_names: tuple[str, ...],
    seeds: tuple[int, ...],
    total_steps: int,
    runner: Callable[[Hypothesis[R], str, int, int], R],
    primary_outcome_extractor: Callable[[R], float],
    cycle_id: str | None = None,
) -> tuple[list[RunRow], list[CellFailure]]:
    """Run `h` on each (env_name, seed) cell. Returns
    `(successful_rows, failures)`.

    `runner(h, env_name, seed, total_steps) -> R` is the
    domain-specific cell executor. `primary_outcome_extractor` pulls
    the headline scalar from the record (e.g. `final_return` mean)
    that downstream `ArmRow` aggregation summarises across seeds."""
    rows: list[RunRow] = []
    failures: list[CellFailure] = []

    for env_name in env_names:
        for seed in seeds:
            t0 = time.monotonic()
            try:
                record = runner(h, env_name, seed, total_steps)
            except Exception as exc:  # noqa: BLE001 - intentional broad capture
                failures.append(CellFailure(
                    intervention_name=h.name,
                    env_name=env_name,
                    seed=seed,
                    error=f'{type(exc).__name__}: {exc}',
                    duration_s=time.monotonic() - t0,
                ))
                continue

            duration = time.monotonic() - t0
            primary = primary_outcome_extractor(record)
            bridge_results = tuple(b(record) for b in h.bridges)
            verdict = aggregate_cell_verdict(
                tuple(r.verdict for r in bridge_results),
            )

            measurements: dict[str, MeasurementLeaf] = {
                'intervention_name': h.name,
                'env_name': env_name,
                'seed': seed,
                'total_steps': total_steps,
                'outcome.late_window_mean': primary,
                'duration_s': duration,
            }
            for result in bridge_results:
                measurements.update(_bridge_result_to_measurements(result))

            rows.append(RunRow(
                id=str(uuid.uuid4()),
                parent_id=None,
                cycle_id=cycle_id,
                timestamp=datetime.now(UTC).isoformat(timespec='seconds'),
                verdict=verdict,
                measurements=measurements,
            ))

    return rows, failures
