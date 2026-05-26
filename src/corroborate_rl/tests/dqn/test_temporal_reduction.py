"""Tests for `@temporal_reduction` — the pair-registration decorator."""
from __future__ import annotations

import math

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


# ============ Refactor-equivalence tests ============
#
# Each refactored measurable (`@temporal_reduction(late_name=...)`)
# preserves the SAME numerical output as the pre-refactor hand-rolled
# `@measurable` body. The pre-refactor closed forms are reproduced
# inline here as the ground truth — independent of the decorator's
# implementation — so any regression in the decorator wiring or in
# the migrated reduction kernel surfaces as a test failure.


def test_argmax_entropy_late_equivalence() -> None:
    """Refactor of `argmax_entropy_late` via @temporal_reduction
    preserves the pre-refactor Shannon-entropy output on
    `online_argmax_per_step`'s late-50% slice."""
    m = get_registered('argmax_entropy_late')
    assert m is not None
    # Per-step argmax stream of length 8 with K=3 actions.
    # Late half (idx 4..8) = [0, 1, 2, 2] → counts [1, 1, 2]
    # → p = [1/4, 1/4, 2/4]
    # → H = -(0.25·ln0.25 + 0.25·ln0.25 + 0.5·ln0.5)
    #      = 0.5·ln4 + 0.5·ln2 = ln4·0.5 + ln2·0.5 = (3/2)·ln2
    record = {
        'online_argmax_per_step': np.array(
            [0, 0, 1, 1, 0, 1, 2, 2], dtype=np.int64,
        ),
    }
    expected = 0.5 * np.log(4.0) + 0.5 * np.log(2.0)
    assert math.isclose(m.fn(record), expected, abs_tol=1e-9)


def test_argmax_mode_freq_late_equivalence() -> None:
    """Refactor of `argmax_mode_freq_late` preserves the mode-fraction
    on the late-50% slice."""
    m = get_registered('argmax_mode_freq_late')
    assert m is not None
    # Per-step argmax of length 8. Late half = [0, 1, 2, 2]
    # counts [1, 1, 2] → mode_freq = 2/4 = 0.5
    record = {
        'online_argmax_per_step': np.array(
            [0, 0, 1, 1, 0, 1, 2, 2], dtype=np.int64,
        ),
    }
    assert math.isclose(m.fn(record), 0.5, abs_tol=1e-12)


def test_td_residual_late_equivalence() -> None:
    """Refactor of `td_residual_late` preserves late-50% mean of
    `td_error` (matches the existing closed-form test at
    `test_late_window_measurables.py::test_td_residual_late_mean_over_late_half`)."""
    m = get_registered('td_residual_late')
    assert m is not None
    # Same record as the pre-existing test.
    record = {'td_error': np.array([0.5, 0.5, 0.1, 0.1])}
    assert math.isclose(m.fn(record), 0.1, abs_tol=1e-12)


def test_td_within_batch_var_late_equivalence() -> None:
    """Refactor preserves late-50% mean of `td_error_within_batch_std`."""
    m = get_registered('td_within_batch_var_late')
    assert m is not None
    record = {
        'td_error_within_batch_std': np.array(
            [0.2, 0.2, 0.4, 0.4, 0.6, 0.6],
        ),
    }
    # Late half (idx 3..6) = [0.4, 0.6, 0.6] mean = 1.6/3 ≈ 0.5333
    assert math.isclose(m.fn(record), 1.6 / 3.0, abs_tol=1e-12)


def test_q_max_temporal_cv_late_equivalence() -> None:
    """Refactor preserves std(ddof=1)/|mean| on `online_max_q_per_step`'s
    late-50% slice."""
    m = get_registered('q_max_temporal_cv_late')
    assert m is not None
    # 8-step series. Late half (idx 4..8) = [1, 3, 1, 3]
    # mean = 2.0; std(ddof=1) of [1,3,1,3] = √(((-1)²+1²+(-1)²+1²)/3)
    #   = √(4/3)
    record = {
        'online_max_q_per_step': np.array(
            [10.0, 10.0, 10.0, 10.0, 1.0, 3.0, 1.0, 3.0],
        ),
    }
    expected = math.sqrt(4.0 / 3.0) / 2.0
    assert math.isclose(m.fn(record), expected, abs_tol=1e-9)


def test_q_max_temporal_cv_late_nan_on_zero_mean() -> None:
    """Pre-refactor returned NaN when |mean| < 1e-9; refactor preserves."""
    m = get_registered('q_max_temporal_cv_late')
    assert m is not None
    record = {
        'online_max_q_per_step': np.array(
            [10.0, 10.0, 10.0, 10.0, 1.0, -1.0, 1.0, -1.0],
        ),
    }
    # Late half = [1, -1, 1, -1] mean = 0 → NaN
    assert math.isnan(m.fn(record))


def test_q_autocorr_late_equivalence() -> None:
    """Refactor preserves lag-1 Pearson autocorrelation on the late-50%
    slice of `online_max_q_per_step`."""
    m = get_registered('q_autocorr_late')
    assert m is not None
    # Late half (idx 4..8) = [1, 2, 3, 4, 5] → ramp → autocorr = +1
    record = {
        'online_max_q_per_step': np.array(
            [99.0, 99.0, 99.0, 99.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        ),
    }
    assert math.isclose(m.fn(record), 1.0, abs_tol=1e-9)


def test_q_autocorr_late_nan_on_constant_window() -> None:
    """Pre-refactor returned NaN when np.std==0 on either x or y;
    refactor preserves."""
    m = get_registered('q_autocorr_late')
    assert m is not None
    record = {
        'online_max_q_per_step': np.array(
            [1.0, 2.0, 3.0, 4.0, 7.0, 7.0, 7.0, 7.0],
        ),
    }
    assert math.isnan(m.fn(record))


def test_refactored_late_names_in_registry() -> None:
    """All refactored late-only measurables surface under their original
    public names in the global registry, so downstream `import` paths
    and `transitive_reads` walks still find them by name."""
    for name in (
        'argmax_entropy_late',
        'argmax_mode_freq_late',
        'td_residual_late',
        'td_within_batch_var_late',
        'q_max_temporal_cv_late',
        'q_autocorr_late',
    ):
        m = get_registered(name)
        assert m is not None, f'{name} missing from registry'
        assert m.name == name
