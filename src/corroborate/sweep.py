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
2. Applies each `h.bridges[i]` to the record.
3. Wraps the bridge results into FactRows and a RunRow.
4. Captures any exception as a `CellFailure` rather than crashing
   the sweep — the caller sees both successful rows and failures.

Subprocess isolation (one process per env, to prevent JIT cache
OOM as v9 documented) is deferred. v0 runs in-process; large
sweeps that need isolation can switch to a process-pool runner
later without changing the Hypothesis / Bridge API.

Statistical computation (Hedges' g, derived q, ΔI) is deferred
to the statistics module (step 5). The FactRows produced here
carry verdict and stats from each bridge; natural_strength and
delta_i are zeroed (placeholder), to be filled in by the
statistics layer when ArmRow / ComparisonRow / CorpusRow are
constructed."""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from corroborate.bridge import Bridge, BridgeResult
from corroborate.hypothesis import Hypothesis, MechanismKey
from corroborate.schema import FactRow, RunRow
from corroborate.verdict import Verdict


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
            intervention_sig = _intervention_signature_leaves(h.mechanism_key)
            facts = tuple(
                _bridge_result_to_fact(b(record), b, intervention_sig)
                for b in h.bridges
            )
            verdict = _aggregate_cell_verdict(facts)
            reads_set: frozenset[str] = frozenset()
            for f in facts:
                reads_set = reads_set | f.reads
            row_meta: dict[str, str | int | float | bool] = {
                'duration_s': duration,
            }
            rows.append(RunRow(
                id=str(uuid.uuid4()),
                parent_id=None,
                intervention_name=h.name,
                cycle_id=cycle_id,
                timestamp=datetime.now(UTC).isoformat(timespec='seconds'),
                env_name=env_name,
                total_steps=total_steps,
                seed=seed,
                mechanism_key=h.mechanism_key,
                primary_outcome_summary=primary,
                record_keys=tuple(sorted(record.keys())),
                facts=facts,
                reads_set=reads_set,
                verdict=verdict,
                meta=row_meta,
            ))

    return rows, failures


# ============ Bridge-result → FactRow conversion ============

def _bridge_result_to_fact[R: Mapping[str, object]](
    result: BridgeResult,
    bridge: Bridge[R],
    intervention_signature: frozenset[str],
) -> FactRow:
    """Stub conversion: BridgeResult → FactRow. The framework's
    statistics layer (step 5) populates `natural_strength` and
    `delta_i` from the result's stats; the verdict layer
    (step 6) populates `evidentiary_level` per axiom 19. Until
    those land, placeholders are 0.0 and the verdict's string
    value, respectively.

    `intervention_signature` is the leaf-flattened form of the
    parent hypothesis's mechanism_key.intervention_signature —
    feeds axiom 19's redundancy primitive's intervention factor.

    `kind='bridge'` unless the result's stats carry the
    'tautological' tag (set by `@invariant`); then `'invariant'`."""
    is_invariant = result.stats.get('kind') == 'tautological'
    return FactRow(
        name=result.name if result.name else bridge.name,
        kind='invariant' if is_invariant else 'bridge',
        targets=result.targets if result.targets else bridge.targets,
        reads=frozenset(result.targets if result.targets else bridge.targets),
        verdict=result.verdict,
        natural_strength=0.0,
        delta_i=0.0,
        evidentiary_level=result.verdict.value,
        stats=result.stats,
        intervention_signature=intervention_signature,
    )


def _intervention_signature_leaves(mk: MechanismKey) -> frozenset[str]:
    """Flatten `mechanism_key.intervention_signature` to a
    frozenset of leaf strings. Each (slot, value) pair contributes
    both halves; the redundancy primitive's intervention-similarity
    factor uses Jaccard over these leaves. Empty signatures (e.g.
    a baseline arm with no overrides) yield an empty frozenset."""
    leaves: set[str] = set()
    for slot, value in mk.intervention_signature:
        leaves.add(slot)
        leaves.add(value)
    return frozenset(leaves)


def _aggregate_cell_verdict(facts: tuple[FactRow, ...]) -> Verdict:
    """Cell-level verdict from per-bridge facts.

    Precedence (highest first):
    1. Any tautological-tagged NO_EFFECT (invariant violated;
       mechanism didn't operate) → INVARIANT_VIOLATION.
    2. Any NO_EFFECT → NO_EFFECT (claim was tested and failed).
    3. All HELD → HELD.
    4. Otherwise (mixed HELD + POWER_INSUFFICIENT) →
       POWER_INSUFFICIENT (the framework's 'cannot tell' tag).
    Empty facts → POWER_INSUFFICIENT (no test was performed).

    The statistics layer (step 5) refines this with MDE+power-
    aware trichotomy at the comparison level (ArmRow + ComparisonRow).
    Cell-level here is the coarse aggregate.

    Returns `Verdict` directly (typed enum), not a string — the
    framework's primary discrimination is the trichotomy, and
    de-typing it to str at the row boundary loses semantic
    information. Pyright catches a Verdict variant rename instead
    of silently mismatching strings."""
    if not facts:
        return Verdict.POWER_INSUFFICIENT
    has_invariant_violation = any(
        f.kind == 'invariant' and f.verdict is Verdict.NO_EFFECT
        for f in facts
    )
    if has_invariant_violation:
        return Verdict.INVARIANT_VIOLATION
    has_no_effect = any(f.verdict is Verdict.NO_EFFECT for f in facts)
    if has_no_effect:
        return Verdict.NO_EFFECT
    all_held = all(f.verdict is Verdict.HELD for f in facts)
    if all_held:
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT
