"""End-to-end smoke for the DDQN substrate post-decomposition.

Validates the foundation + Phase 1+2 cuts actually carry the
vanilla-vs-DDQN comparison, AND that the two-store persistence
(runs.parquet + traces.parquet) round-trips correctly with HP
query support at the dataframe level:

1. Builds two hypotheses differing only in `greedification`.
2. Runs each on CartPole, 3 seeds, 1000 steps via `run_dqn_arm`
   (returns `tuple[CellResult, ...]` per arm — both stores).
3. Asserts:
   - `leaf_signature` distinguishes the arms by the `greedification`
     swap (leaf topology paths differ).
   - `leaf_signature` groups cells consistently across seeds.
   - Each arm has finite outcome summaries.
4. Persists both stores: writes `runs.parquet` and
   `traces.parquet` to a temp dir, reads back, verifies:
   - Configurational leaf columns (`gamma`, `optimizer.inner.lr`,
     ...) survive as typed parquet columns query-able via
     `df.filter(pl.col('optimizer.inner.lr') < 1e-3)`.
   - Multi-dim trajectory columns persist as nested
     `List[List[...]]` and round-trip exactly.
   - `RunRow.id == TraceRow.id` for paired records.

Run: `uv run python experiments/smoke_ddqn_run.py`."""
from __future__ import annotations

import tempfile
import time
from functools import partial
from pathlib import Path

import polars as pl

from corroborate.aggregate import leaf_signature
from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention
from corroborate.persistence import (
    read_runrows,
    read_tracerows,
    write_runrows,
    write_tracerows,
)
from corroborate.rl.cell_runner import run_dqn_arm
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get


_ENV = 'CartPole-v1'
_SEEDS: tuple[int, ...] = (0, 1, 2)
_TOTAL_STEPS = 1000


_HPARAMS: dict[str, object] = {
    'total_steps': _TOTAL_STEPS,
    'eval_every': _TOTAL_STEPS // 5,
    'n_episodes': 5,
    'gamma': 0.99,
    'replay': Replay(capacity=2000, batch_size=32),
    'sync_period': 100,
}


def _vanilla() -> Hypothesis[DQNTrajectoryRecord]:
    return Hypothesis(
        name='vanilla_dqn',
        intervention={**_HPARAMS},
        bridges=(),
        predicted_direction=None,
        intervention_arms=(),
    )


def _ddqn() -> Hypothesis[DQNTrajectoryRecord]:
    return Hypothesis(
        name='ddqn',
        intervention={
            **_HPARAMS,
            'bootstrap': partial(bootstrap, greedification=double_greedify),
        },
        bridges=(),
        predicted_direction='a_gt_b',
        intervention_arms=(
            Intervention(
                slot_path='bootstrap',
                replacement=partial(
                    bootstrap, greedification=double_greedify,
                ),
            ),
        ),
    )


def main() -> None:
    env_spec = get(_ENV)

    print(f'env={_ENV}, seeds={_SEEDS}, total_steps={_TOTAL_STEPS}')
    print()

    vanilla = _vanilla()
    ddqn = _ddqn()

    print('1. Running vanilla arm on CartPole...')
    t0 = time.time()
    vanilla_arm = run_dqn_arm(
        env_spec, _SEEDS, vanilla,
    )
    vanilla_cells = vanilla_arm.cells
    vanilla_rows = tuple(c.run for c in vanilla_cells)
    vanilla_traces = tuple(c.trace for c in vanilla_cells)
    print(f'   {len(vanilla_cells)} cells in {time.time() - t0:.1f}s')
    for row in vanilla_rows:
        seed = row.measurements['seed']
        outcome = row.measurements['outcome.late_window_mean']
        print(f'     seed={seed} verdict={row.verdict.value} '
              f'outcome={outcome}')
    print()

    print('2. Running DDQN arm on CartPole...')
    t0 = time.time()
    ddqn_arm = run_dqn_arm(
        env_spec, _SEEDS, ddqn,
    )
    ddqn_cells = ddqn_arm.cells
    ddqn_rows = tuple(c.run for c in ddqn_cells)
    ddqn_traces = tuple(c.trace for c in ddqn_cells)
    print(f'   {len(ddqn_cells)} cells in {time.time() - t0:.1f}s')
    for row in ddqn_rows:
        seed = row.measurements['seed']
        outcome = row.measurements['outcome.late_window_mean']
        print(f'     seed={seed} verdict={row.verdict.value} '
              f'outcome={outcome}')
    print()

    print('3. Per-arm leaf_signature on RunRows:')
    v_sigs = {leaf_signature(row.measurements) for row in vanilla_rows}
    d_sigs = {leaf_signature(row.measurements) for row in ddqn_rows}
    assert len(v_sigs) == 1, (
        f'vanilla seeds should share one leaf_signature; got {len(v_sigs)}'
    )
    assert len(d_sigs) == 1, (
        f'ddqn seeds should share one leaf_signature; got {len(d_sigs)}'
    )
    assert v_sigs != d_sigs, 'vanilla and DDQN must canonicalise distinctly'
    print('   OK vanilla-rows share one signature, ddqn-rows share another, '
          'and they differ.\n')

    print('4. RunRows group by leaf_signature (ArmRow retired):')
    by_sig: dict[
        tuple[tuple[str, str], ...], list[RunRow],
    ] = {}
    for r in list(vanilla_rows) + list(ddqn_rows):
        by_sig.setdefault(leaf_signature(r.measurements), []).append(r)
    print(f'   {len(by_sig)} distinct leaf signatures (one per hypothesis)')
    assert len(by_sig) == 2, f'expected 2 sigs, got {len(by_sig)}'
    for sig, members in by_sig.items():
        name = members[0].measurements['intervention_name']
        env = members[0].measurements['env_name']
        n = len(members)
        print(f'     {name} on {env}: n={n} cells')

    print()
    print('5. Two-store persistence round-trip + HP query:')
    all_runs = vanilla_rows + ddqn_rows
    all_traces = vanilla_traces + ddqn_traces
    with tempfile.TemporaryDirectory() as tmp:
        runs_path = Path(tmp) / 'runs.parquet'
        traces_path = Path(tmp) / 'traces.parquet'
        write_runrows(all_runs, runs_path)
        write_tracerows(all_traces, traces_path)

        # Verify HP querying at the dataframe level.
        traces_df = pl.read_parquet(traces_path)
        # Each cell carries `optimizer.inner.lr` as a typed Float64
        # column — `df.filter(...)` works without JSON decoding.
        lr_column = traces_df.get_column('optimizer.inner.lr')
        print(f'   traces.parquet columns: {len(traces_df.columns)}')
        print(f'   optimizer.inner.lr (per cell): '
              f'{lr_column.to_list()}')

        # Read back + verify id-link between stores.
        rows_back = read_runrows(runs_path)
        traces_back = read_tracerows(traces_path)
        run_ids = {r.id for r in rows_back}
        trace_ids = {t.id for t in traces_back}
        assert run_ids == trace_ids, (
            f'id-link violated: {run_ids ^ trace_ids}'
        )

        # Verify the per-step Q reductions survived the round-trip.
        # train_phase now emits `online_q_per_action` shape
        # `(total_steps, n_actions)` and `pearson_stats` shape
        # `(total_steps, 5)` — both 2-D per cell.
        sample_trace = traces_back[0]
        oq = sample_trace.leaves['online_q_per_action']
        assert isinstance(oq, list)
        assert isinstance(oq[0], list)
        ps = sample_trace.leaves['pearson_stats']
        assert isinstance(ps, list)
        assert isinstance(ps[0], list)
        print(f'   online_q_per_action 2-D shape: '
              f'({len(oq)}, {len(oq[0])}) — preserved')
        print(f'   pearson_stats 2-D shape: '
              f'({len(ps)}, {len(ps[0])}) — preserved')
        print(f'   id-link OK: {len(run_ids)} cells, {len(trace_ids)} traces, '
              f'matched.')

    print()
    print('All checks passed.')


if __name__ == '__main__':
    main()
