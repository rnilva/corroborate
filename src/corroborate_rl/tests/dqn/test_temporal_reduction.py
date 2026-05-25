"""Tests for `@temporal_reduction` — the pair-registration decorator."""
from __future__ import annotations

import numpy as np
import pytest

import corroborate_rl.dqn.measurables  # noqa: F401 — ensures substrate measurables registered
from corroborate.measurables.measurable import get_registered
from corroborate_rl.dqn._temporal_reduction import temporal_reduction


def test_decorator_registers_both_late_and_per_burst() -> None:
    """When both names are supplied, the decorator registers both
    measurables and they share the reduction kernel."""
    @temporal_reduction(
        reads=('test_per_step_input',),
        late_name='_test_double_late',
        per_burst_name='_test_double_per_burst',
    )
    def _double(w: np.ndarray) -> float:
        return float(w.mean() * 2)

    late = get_registered('_test_double_late')
    pb = get_registered('_test_double_per_burst')
    assert late is not None
    assert pb is not None
    # Late reads only the trace col
    assert late.reads == ('test_per_step_input',)
    # Per-burst reads add eval_step_index
    assert pb.reads == ('test_per_step_input', 'eval_step_index')


def test_late_computes_on_late_half() -> None:
    @temporal_reduction(
        reads=('arr_per_step',),
        late_name='_test_sum_late',
    )
    def _sum(w: np.ndarray) -> float:
        return float(w.sum())

    late = get_registered('_test_sum_late')
    assert late is not None
    # arr of length 10; late half is indices 5..9 → sum = 5+6+7+8+9 = 35
    record = {'arr_per_step': list(range(10))}
    assert late.fn(record) == pytest.approx(35.0)


def test_per_burst_slices_via_eval_step_index() -> None:
    @temporal_reduction(
        reads=('arr_per_step',),
        per_burst_name='_test_mean_per_burst',
    )
    def _mean(w: np.ndarray) -> float:
        return float(w.mean()) if w.size > 0 else float('nan')

    pb = get_registered('_test_mean_per_burst')
    assert pb is not None
    # arr length 12 split into 3 bursts → windows [0:4], [4:8], [8:12]
    record = {
        'arr_per_step': list(range(12)),
        'eval_step_index': [4, 8, 12],
    }
    result = pb.fn(record)
    assert isinstance(result, np.ndarray)
    assert result.tolist() == pytest.approx([1.5, 5.5, 9.5])


def test_per_burst_handles_missing_inputs() -> None:
    @temporal_reduction(
        reads=('arr_per_step',),
        per_burst_name='_test_robust_per_burst',
    )
    def _mean(w: np.ndarray) -> float:
        return float(w.mean()) if w.size > 0 else float('nan')

    pb = get_registered('_test_robust_per_burst')
    assert pb is not None
    # Missing trace column → empty array
    assert pb.fn({'eval_step_index': [4]}).shape == (0,)
    # Missing eval_step_index → empty array
    assert pb.fn({'arr_per_step': [1.0]}).shape == (0,)


def test_window_invariant_false_rejects_per_burst() -> None:
    """The decorator refuses to auto-pair when the reduction's
    interpretation depends on window size."""
    with pytest.raises(ValueError, match='window_invariant=False'):
        @temporal_reduction(
            reads=('arr_per_step',),
            late_name='_test_late_only',
            per_burst_name='_test_per_burst_forbidden',
            window_invariant=False,
        )
        def _f(w: np.ndarray) -> float:
            return 0.0


def test_requires_single_per_step_column() -> None:
    """Multi-column reductions (like policy_churn) must use @measurable
    directly, not the decorator."""
    with pytest.raises(ValueError, match='exactly one per-step trace column'):
        @temporal_reduction(
            reads=('a_per_step', 'b_per_step'),
            late_name='_test_multi_col_late',
        )
        def _f(w: np.ndarray) -> float:
            return 0.0


def test_requires_at_least_one_name() -> None:
    with pytest.raises(ValueError, match='at least one'):
        temporal_reduction(reads=('arr_per_step',))


def test_substrate_state_hash_per_burst_measurables_registered() -> None:
    """The three state-coverage per-burst measurables added via
    @temporal_reduction are registered and compute the same statistic
    as the existing _late siblings when handed the same window."""
    nu_pb = get_registered('state_hash_n_unique_per_burst')
    h_pb = get_registered('state_hash_entropy_per_burst')
    rr_pb = get_registered('state_repeat_rate_window64_per_burst')
    assert nu_pb is not None
    assert h_pb is not None
    assert rr_pb is not None
    # Each reads state_hash_per_step + eval_step_index (auto-injected)
    for m in (nu_pb, h_pb, rr_pb):
        assert 'state_hash_per_step' in m.reads
        assert 'eval_step_index' in m.reads


def test_per_burst_state_hash_n_unique_matches_late_on_equivalent_window() -> None:
    """If the per-burst version is given a single burst spanning the
    full late-half, it should produce the same n_unique that the
    _late scalar produces on the late half."""
    nu_late = get_registered('state_hash_n_unique_late')
    nu_pb = get_registered('state_hash_n_unique_per_burst')
    assert nu_late is not None and nu_pb is not None
    # Synthetic per-step state-hash sequence of length 20
    state_hashes = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
                    5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
    record_late = {'state_hash_per_step': state_hashes}
    # Late half is indices 10..19 → {5, 6, 7, 8, 9} → 5 unique
    assert nu_late.fn(record_late) == 5.0

    # Per-burst with one burst spanning the late half
    record_pb = {
        'state_hash_per_step': state_hashes,
        'eval_step_index': [20],  # one burst
    }
    result = nu_pb.fn(record_pb)
    assert isinstance(result, np.ndarray)
    # Single burst spans full array (length 20) → 10 unique total
    assert result.tolist() == [10.0]
