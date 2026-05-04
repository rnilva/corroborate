"""Tests for `rl.convergence` — per-env classification + filters."""
from __future__ import annotations

import math

from corroborate.rl.convergence import (
    EnvConvergence, PathDifferential,
    classify_envs, envs_in_class, filter_to_classes,
    mediator_differential,
)
from corroborate.rl.env_solve_thresholds import SolveThreshold
from corroborate.schema import RunRow
from corroborate.bridge.verdict import Verdict


def _run(
    cell_id: str, *, env: str, best: float, final: float,
) -> RunRow:
    return RunRow(
        id=cell_id, parent_id=None, cycle_id=None,
        timestamp='2026-04-29T00:00:00Z',
        verdict=Verdict.HELD, arm_key='baseline',
        measurements={
            'env_name': env,
            'eval_best_burst_mean': best,
            'eval_final_mean': final,
        },
    )


_TABLE = {
    'EasyEnv': SolveThreshold(
        env_name='EasyEnv', threshold=0.5,
        source='test', confidence='literature',
    ),
    'HardEnv': SolveThreshold(
        env_name='HardEnv', threshold=10.0,
        source='test', confidence='literature',
    ),
    'AbsentEnv': SolveThreshold(
        env_name='AbsentEnv', threshold=None,
        source='no-criterion', confidence='absent',
    ),
}


# ============ classify_envs ============

def test_classify_envs_solved_when_above_threshold() -> None:
    """All baseline cells exceed threshold → solved."""
    runs = [
        _run(f'c{i}', env='EasyEnv', best=0.8, final=0.7)
        for i in range(10)
    ]
    out = classify_envs(runs, table=_TABLE)
    assert 'EasyEnv' in out
    c = out['EasyEnv']
    assert c.classification == 'solved'
    assert c.final_solve_rate == 1.0
    assert c.n_cells == 10


def test_classify_envs_unsolved_when_no_cells_meet_threshold() -> None:
    runs = [
        _run(f'c{i}', env='HardEnv', best=2.0, final=1.0)
        for i in range(10)
    ]
    out = classify_envs(runs, table=_TABLE)
    c = out['HardEnv']
    assert c.classification == 'unsolved'
    assert c.final_solve_rate == 0.0


def test_classify_envs_partial_for_intermediate_rate() -> None:
    """30% solve rate → partial (below default 0.5 threshold but
    above 0)."""
    runs = (
        [_run(f's{i}', env='EasyEnv', best=0.8, final=0.7)
         for i in range(3)] +
        [_run(f'u{i}', env='EasyEnv', best=0.4, final=0.3)
         for i in range(7)]
    )
    out = classify_envs(runs, table=_TABLE)
    c = out['EasyEnv']
    assert c.classification == 'partial'
    assert c.final_solve_rate == 0.3


def test_classify_envs_absent_for_missing_threshold() -> None:
    """Envs with confidence='absent' classify as 'absent'
    regardless of outcome values."""
    runs = [
        _run(f'c{i}', env='AbsentEnv', best=999.0, final=999.0)
        for i in range(5)
    ]
    out = classify_envs(runs, table=_TABLE)
    c = out['AbsentEnv']
    assert c.classification == 'absent'
    assert c.best_solve_rate is None
    assert c.final_solve_rate is None


def test_classify_envs_handles_envs_not_in_table() -> None:
    """Envs that appear in runs but not in the threshold table
    classify as 'absent' with a synthetic SolveThreshold marker
    (so consumers know we *considered* the env, didn't silently
    skip)."""
    runs = [
        _run(f'c{i}', env='UnknownEnv', best=1.0, final=1.0)
        for i in range(3)
    ]
    out = classify_envs(runs, table=_TABLE)
    c = out['UnknownEnv']
    assert c.classification == 'absent'
    assert c.threshold.confidence == 'absent'
    assert c.threshold.source == 'not-in-table'


def test_classify_envs_drops_nan_and_non_numeric_outcomes() -> None:
    """NaN / non-numeric outcomes are filtered before averaging.
    A cell with NaN outcome doesn't contribute to the mean or the
    solve rate."""
    runs = [
        _run('valid_1', env='EasyEnv', best=0.8, final=0.8),
        _run('valid_2', env='EasyEnv', best=0.6, final=0.6),
        # NaN cell — dropped.
        RunRow(
            id='nan_cell', parent_id=None, cycle_id=None,
            timestamp='2026-04-29T00:00:00Z',
            verdict=Verdict.HELD, arm_key='baseline',
            measurements={
                'env_name': 'EasyEnv',
                'eval_best_burst_mean': float('nan'),
                'eval_final_mean': float('nan'),
            },
        ),
    ]
    out = classify_envs(runs, table=_TABLE)
    c = out['EasyEnv']
    # Only 2 finite cells; both above threshold → solve rate 1.0.
    assert c.final_solve_rate == 1.0


def test_classify_envs_solved_threshold_parameter() -> None:
    """Lowering `solved_threshold` reclassifies partial → solved."""
    runs = (
        [_run(f's{i}', env='EasyEnv', best=0.8, final=0.7)
         for i in range(3)] +
        [_run(f'u{i}', env='EasyEnv', best=0.4, final=0.3)
         for i in range(7)]
    )
    # Default: 0.5 → partial.
    out_default = classify_envs(runs, table=_TABLE)
    assert out_default['EasyEnv'].classification == 'partial'
    # Lowered to 0.2 → solved.
    out_lowered = classify_envs(
        runs, table=_TABLE, solved_threshold=0.2,
    )
    assert out_lowered['EasyEnv'].classification == 'solved'


# ============ envs_in_class / filter_to_classes ============

def test_envs_in_class_returns_sorted_names() -> None:
    classifications = {
        'B': EnvConvergence(
            env_name='B', threshold=_TABLE['EasyEnv'],
            best_mean=0.8, final_mean=0.7,
            best_solve_rate=1.0, final_solve_rate=1.0,
            n_cells=5, classification='solved',
        ),
        'A': EnvConvergence(
            env_name='A', threshold=_TABLE['EasyEnv'],
            best_mean=0.4, final_mean=0.3,
            best_solve_rate=0.0, final_solve_rate=0.0,
            n_cells=5, classification='unsolved',
        ),
        'C': EnvConvergence(
            env_name='C', threshold=_TABLE['EasyEnv'],
            best_mean=0.7, final_mean=0.6,
            best_solve_rate=0.8, final_solve_rate=0.6,
            n_cells=5, classification='solved',
        ),
    }
    assert envs_in_class(classifications, 'solved') == ('B', 'C')
    assert envs_in_class(classifications, 'unsolved') == ('A',)
    assert envs_in_class(classifications, 'partial') == ()


def test_filter_to_classes_keeps_only_target_envs() -> None:
    classifications = {
        'A': EnvConvergence(
            env_name='A', threshold=_TABLE['EasyEnv'],
            best_mean=0.8, final_mean=0.7,
            best_solve_rate=1.0, final_solve_rate=1.0,
            n_cells=2, classification='solved',
        ),
        'B': EnvConvergence(
            env_name='B', threshold=_TABLE['EasyEnv'],
            best_mean=0.4, final_mean=0.3,
            best_solve_rate=0.0, final_solve_rate=0.0,
            n_cells=2, classification='unsolved',
        ),
    }
    runs = [
        _run('a1', env='A', best=0.8, final=0.7),
        _run('a2', env='A', best=0.6, final=0.6),
        _run('b1', env='B', best=0.4, final=0.3),
        _run('b2', env='B', best=0.4, final=0.3),
    ]
    kept = filter_to_classes(runs, classifications, ('solved',))
    assert len(kept) == 2
    assert all(r.measurements.get('env_name') == 'A' for r in kept)


def test_filter_to_classes_accepts_multiple_targets() -> None:
    """`targets=('solved', 'partial')` is the looser scope filter."""
    classifications = {
        'A': EnvConvergence(
            env_name='A', threshold=_TABLE['EasyEnv'],
            best_mean=0.8, final_mean=0.7,
            best_solve_rate=1.0, final_solve_rate=1.0,
            n_cells=2, classification='solved',
        ),
        'B': EnvConvergence(
            env_name='B', threshold=_TABLE['EasyEnv'],
            best_mean=0.4, final_mean=0.3,
            best_solve_rate=0.5, final_solve_rate=0.3,
            n_cells=2, classification='partial',
        ),
        'C': EnvConvergence(
            env_name='C', threshold=_TABLE['EasyEnv'],
            best_mean=0.2, final_mean=0.1,
            best_solve_rate=0.0, final_solve_rate=0.0,
            n_cells=2, classification='unsolved',
        ),
    }
    runs = [
        _run('a', env='A', best=0.8, final=0.7),
        _run('b', env='B', best=0.5, final=0.4),
        _run('c', env='C', best=0.2, final=0.1),
    ]
    kept = filter_to_classes(
        runs, classifications, ('solved', 'partial'),
    )
    assert {r.measurements.get('env_name') for r in kept} == {'A', 'B'}


# ============ mediator_differential ============

def _run_with_mediator(
    cell_id: str, *, env: str, mediator_value: float,
) -> RunRow:
    return RunRow(
        id=cell_id, parent_id=None, cycle_id=None,
        timestamp='2026-04-29T00:00:00Z',
        verdict=Verdict.HELD, arm_key='baseline',
        measurements={
            'env_name': env,
            'mediator.test_path': mediator_value,
            'mediator.constant_path': 1.0,
        },
    )


_DIFFERENTIAL_CLASSIFICATIONS = {
    'EnvSolved': EnvConvergence(
        env_name='EnvSolved', threshold=_TABLE['EasyEnv'],
        best_mean=0.9, final_mean=0.9,
        best_solve_rate=1.0, final_solve_rate=1.0,
        n_cells=10, classification='solved',
    ),
    'EnvUnsolved': EnvConvergence(
        env_name='EnvUnsolved', threshold=_TABLE['EasyEnv'],
        best_mean=0.1, final_mean=0.1,
        best_solve_rate=0.0, final_solve_rate=0.0,
        n_cells=10, classification='unsolved',
    ),
}


def test_mediator_differential_cell_strong_separation() -> None:
    """Cell-mode: per-cell pooling with clear class separation
    yields large |g|."""
    import random
    rng = random.Random(0)
    runs = (
        [
            _run_with_mediator(
                f's{i}', env='EnvSolved',
                mediator_value=rng.gauss(0.0, 0.1),
            )
            for i in range(10)
        ] + [
            _run_with_mediator(
                f'u{i}', env='EnvUnsolved',
                mediator_value=rng.gauss(5.0, 0.1),
            )
            for i in range(10)
        ]
    )
    diffs = mediator_differential(
        runs, _DIFFERENTIAL_CLASSIFICATIONS,
        paths=['mediator.test_path'],
        aggregation='cell',
    )
    assert len(diffs) == 1
    d = diffs[0]
    assert d.path == 'mediator.test_path'
    assert d.g > 5.0
    assert d.n_a == 10
    assert d.n_b == 10


def test_mediator_differential_constant_path_yields_nan_g() -> None:
    """All cells have the same value → pooled variance is 0 → NaN."""
    runs = [
        _run_with_mediator(
            f'c{i}', env='EnvSolved' if i < 10 else 'EnvUnsolved',
            mediator_value=1.0,
        )
        for i in range(20)
    ]
    diffs = mediator_differential(
        runs, _DIFFERENTIAL_CLASSIFICATIONS,
        paths=['mediator.constant_path'],
        aggregation='cell',
    )
    assert math.isnan(diffs[0].g)


def test_mediator_differential_cell_sorts_by_abs_g() -> None:
    """Cell-mode: paths with larger |g| come first."""
    import random
    rng = random.Random(1)
    runs = []
    for i in range(10):
        runs.append(RunRow(
            id=f's{i}', parent_id=None, cycle_id=None,
            timestamp='ts', verdict=Verdict.HELD, arm_key='baseline',
            measurements={
                'env_name': 'EnvSolved',
                'mediator.weak': rng.gauss(0.0, 1.0),
                'mediator.strong': rng.gauss(0.0, 0.1),
            },
        ))
        runs.append(RunRow(
            id=f'u{i}', parent_id=None, cycle_id=None,
            timestamp='ts', verdict=Verdict.HELD, arm_key='baseline',
            measurements={
                'env_name': 'EnvUnsolved',
                'mediator.weak': rng.gauss(0.5, 1.0),
                'mediator.strong': rng.gauss(5.0, 0.1),
            },
        ))
    diffs = mediator_differential(
        runs, _DIFFERENTIAL_CLASSIFICATIONS,
        paths=['mediator.weak', 'mediator.strong'],
        aggregation='cell',
    )
    assert diffs[0].path == 'mediator.strong'
    assert diffs[1].path == 'mediator.weak'
    assert abs(diffs[0].g) > abs(diffs[1].g)


def test_mediator_differential_cell_handles_nan_in_paths() -> None:
    """Cell-mode: cells with NaN at the path are dropped."""
    runs = [
        _run_with_mediator(
            f's{i}', env='EnvSolved', mediator_value=1.0,
        )
        for i in range(10)
    ] + [
        RunRow(
            id=f'u{i}', parent_id=None, cycle_id=None,
            timestamp='ts', verdict=Verdict.HELD, arm_key='baseline',
            measurements={
                'env_name': 'EnvUnsolved',
                'mediator.test_path': (
                    float('nan') if i < 3 else 5.0
                ),
            },
        )
        for i in range(10)
    ]
    diffs = mediator_differential(
        runs, _DIFFERENTIAL_CLASSIFICATIONS,
        paths=['mediator.test_path'],
        aggregation='cell',
    )
    assert diffs[0].n_a == 7
    assert diffs[0].n_b == 10


def test_mediator_differential_returns_path_differential_dataclass() -> None:
    """Each entry in the result is a typed PathDifferential."""
    runs = [
        _run_with_mediator(
            f's{i}', env='EnvSolved', mediator_value=float(i),
        )
        for i in range(5)
    ] + [
        _run_with_mediator(
            f'u{i}', env='EnvUnsolved', mediator_value=float(i + 10),
        )
        for i in range(5)
    ]
    diffs = mediator_differential(
        runs, _DIFFERENTIAL_CLASSIFICATIONS,
        paths=['mediator.test_path'],
        aggregation='cell',
    )
    assert isinstance(diffs[0], PathDifferential)


def test_mediator_differential_env_mean_default() -> None:
    """Default mode is env_mean: each env contributes one mean
    value; g is computed across env means. With multi-env fixtures."""
    classifications = {
        'Solved1': EnvConvergence(
            env_name='Solved1', threshold=_TABLE['EasyEnv'],
            best_mean=0.9, final_mean=0.9,
            best_solve_rate=1.0, final_solve_rate=1.0,
            n_cells=5, classification='solved',
        ),
        'Solved2': EnvConvergence(
            env_name='Solved2', threshold=_TABLE['EasyEnv'],
            best_mean=0.85, final_mean=0.85,
            best_solve_rate=1.0, final_solve_rate=1.0,
            n_cells=5, classification='solved',
        ),
        'Solved3': EnvConvergence(
            env_name='Solved3', threshold=_TABLE['EasyEnv'],
            best_mean=0.95, final_mean=0.95,
            best_solve_rate=1.0, final_solve_rate=1.0,
            n_cells=5, classification='solved',
        ),
        'Unsolved1': EnvConvergence(
            env_name='Unsolved1', threshold=_TABLE['EasyEnv'],
            best_mean=0.1, final_mean=0.1,
            best_solve_rate=0.0, final_solve_rate=0.0,
            n_cells=5, classification='unsolved',
        ),
        'Unsolved2': EnvConvergence(
            env_name='Unsolved2', threshold=_TABLE['EasyEnv'],
            best_mean=0.05, final_mean=0.05,
            best_solve_rate=0.0, final_solve_rate=0.0,
            n_cells=5, classification='unsolved',
        ),
        'Unsolved3': EnvConvergence(
            env_name='Unsolved3', threshold=_TABLE['EasyEnv'],
            best_mean=0.15, final_mean=0.15,
            best_solve_rate=0.0, final_solve_rate=0.0,
            n_cells=5, classification='unsolved',
        ),
    }
    runs = []
    # Solved envs: mediator.test_path centered at 0 (env-mean=0).
    for env_idx, env_name in enumerate(('Solved1', 'Solved2', 'Solved3')):
        for i in range(5):
            runs.append(_run_with_mediator(
                f's{env_idx}c{i}', env=env_name,
                mediator_value=float(env_idx) * 0.01 + i * 0.001,
            ))
    # Unsolved envs: mediator.test_path centered at 5 (env-mean=5).
    for env_idx, env_name in enumerate(('Unsolved1', 'Unsolved2', 'Unsolved3')):
        for i in range(5):
            runs.append(_run_with_mediator(
                f'u{env_idx}c{i}', env=env_name,
                mediator_value=5.0 + float(env_idx) * 0.01 + i * 0.001,
            ))
    diffs = mediator_differential(
        runs, classifications,
        paths=['mediator.test_path'],
    )
    d = diffs[0]
    # 3 envs per class; each contributes one mean.
    assert d.n_a == 3
    assert d.n_b == 3
    # Unsolved env-means ≈ 5; solved env-means ≈ 0 → very large g.
    assert d.g > 10.0
