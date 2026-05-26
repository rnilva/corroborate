"""Tests for the late-window scalar Measurables in
`corroborate_rl.dqn.measurables` — typed reductions over per-step
columns, used as inputs to PAPER §5/§6's mediator analysis. The
"mediator" framing is paper-section domain language; in the
framework these are plain `Measurable[Mapping, float]`s."""
from __future__ import annotations

import math

import numpy as np

from corroborate_rl.dqn.measurables import (
    eval_full_auc_mean,
    eval_late_burst_mean,
    fill_ratio_late,
    greedy_match_late,
    mutual_info_state_argmax_late,
    q_autocorr_late,
    q_gap_growth,
    q_gap_late,
    q_max_growth,
    state_conditional_argmax_entropy_late,
    td_residual_late,
    v_vs_max_delta_late,
)


def test_q_gap_late_matches_late_half_mean() -> None:
    record = {
        'online_max_q_per_step': np.array([1.0, 2.0, 3.0, 4.0]),
        'online_min_q_per_step': np.array([0.0, 0.5, 0.5, 1.0]),
    }
    # gap = [1.0, 1.5, 2.5, 3.0]; late half (idx 2..4) mean = 2.75
    assert q_gap_late(record) == 2.75


def test_q_gap_late_returns_nan_when_min_missing() -> None:
    """Function-level defensive NaN-return: substrate bodies catch
    `KeyError` on missing declared reads and return NaN directly,
    rather than letting the framework's `compute_missing_columns`
    wrapper catch it at the per-cell boundary. Both paths end at
    NaN in `runs.parquet` — the function-level catch is the
    stricter (lower-latency) form: the function never escapes
    with an exception even for a defensively malformed record."""
    record = {
        'online_max_q_per_step': np.array([1.0, 2.0]),
    }
    assert math.isnan(q_gap_late(record))


def test_q_gap_growth_late_minus_early() -> None:
    record = {
        'online_max_q_per_step': np.array([1.0, 1.0, 4.0, 4.0]),
        'online_min_q_per_step': np.array([0.0, 0.0, 1.0, 1.0]),
    }
    # gap = [1, 1, 3, 3]; early=1.0, late=3.0; growth=2.0
    assert q_gap_growth(record) == 2.0


def test_q_max_growth_late_quarter_over_early_quarter() -> None:
    record = {
        'online_max_q_per_step': np.array(
            [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0],
        ),
    }
    # early quarter (idx 0..2) mean=1; late quarter (idx 6..8) mean=5
    assert q_max_growth(record) == 5.0


def test_q_max_growth_handles_zero_early_via_floor() -> None:
    record = {
        'online_max_q_per_step': np.array(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ),
    }
    # early=0 → divisor floors at 1e-9; result ≈ 1 / 1e-9 = 1e9
    assert math.isclose(q_max_growth(record), 1e9, rel_tol=1e-6)


def test_q_autocorr_late_perfect_smooth_returns_one() -> None:
    """A late half that is a linear ramp has lag-1 autocorr = 1.0
    by construction — adjacent pairs differ by a constant."""
    record = {
        'online_max_q_per_step': np.array(
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
    }
    # late half (idx 5..10): [2,3,4,5,6]; pairs (2,3),(3,4),(4,5),(5,6) → r=1
    assert math.isclose(q_autocorr_late(record), 1.0, abs_tol=1e-9)


def test_q_autocorr_late_alternating_returns_minus_one() -> None:
    """Alternating high-low late-window has lag-1 autocorr = -1.0
    (perfect anti-correlation: each pair is one above-mean and one
    below)."""
    record = {
        'online_max_q_per_step': np.array(
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        ),
    }
    assert math.isclose(q_autocorr_late(record), -1.0, abs_tol=1e-9)


def test_q_autocorr_late_constant_returns_nan() -> None:
    """Constant late-window has zero std → corrcoef is undefined.
    Measurable returns NaN rather than propagate inf/divide errors."""
    record = {
        'online_max_q_per_step': np.array([1.0, 1.0, 1.0, 1.0]),
    }
    assert math.isnan(q_autocorr_late(record))


def test_q_autocorr_late_returns_nan_when_key_missing() -> None:
    """Missing-key contract: see
    `test_q_gap_late_returns_nan_when_min_missing` — function-level
    defensive NaN-return rather than letting KeyError escape."""
    assert math.isnan(q_autocorr_late({}))


def test_q_autocorr_late_too_short_returns_nan() -> None:
    """Late half must have at least 2 values for lag-1 pairs."""
    record = {'online_max_q_per_step': np.array([1.0])}
    assert math.isnan(q_autocorr_late(record))


def test_q_action_grad_overlap_late_linear_fa_gives_zero() -> None:
    """Closed-form: linear FA `Q(s,a) = W_a · obs + b_a` has
    independent W_a rows → ∂Q(s,a_i)/∂θ and ∂Q(s,a_j)/∂θ are
    orthogonal → off-diagonal cosine α = 0.

    Test by constructing per-step gradient-overlap values that
    a linear-FA training loop would produce (≈0 + small numerical
    noise), and verifying the late-window measurable recovers ≈0."""
    from corroborate_rl.dqn.measurables import q_action_grad_overlap_late

    # 20 steps of small-noise around-zero overlap
    record = {
        'q_action_grad_overlap_per_step': np.array(
            [0.0, 0.001, -0.002, 0.0005] * 5, dtype=np.float64,
        ),
    }
    val = q_action_grad_overlap_late(record)
    assert math.isfinite(val)
    assert abs(val) < 0.01


def test_q_action_grad_overlap_late_deep_mlp_gives_positive() -> None:
    """Closed-form: shared-trunk MLP has trunk gradients flowing
    into all action heads → cosine overlap > 0. Test recovers
    the late-window mean of high-overlap synthetic values."""
    from corroborate_rl.dqn.measurables import q_action_grad_overlap_late

    record = {
        # Early-training (lower overlap) then late-training (high)
        'q_action_grad_overlap_per_step': np.array(
            [0.2] * 10 + [0.7, 0.75, 0.8, 0.85, 0.78, 0.82, 0.79, 0.81, 0.77, 0.83],
            dtype=np.float64,
        ),
    }
    val = q_action_grad_overlap_late(record)
    # Late half mean ≈ 0.79
    assert math.isclose(val, 0.79, abs_tol=0.05)


def test_q_action_grad_overlap_late_missing_returns_nan() -> None:
    from corroborate_rl.dqn.measurables import q_action_grad_overlap_late
    assert math.isnan(q_action_grad_overlap_late({}))


def test_q_inter_state_grad_overlap_late_recovers_mean() -> None:
    """Late-window mean of synthetic inter-state α values."""
    from corroborate_rl.dqn.measurables import q_inter_state_grad_overlap_late

    record = {
        'q_inter_state_grad_overlap_per_step': np.array(
            [0.3] * 10 + [0.6, 0.65, 0.7, 0.75, 0.7, 0.6, 0.65, 0.7, 0.68, 0.72],
            dtype=np.float64,
        ),
    }
    v = q_inter_state_grad_overlap_late(record)
    # Late half mean ≈ 0.67
    assert math.isclose(v, 0.675, abs_tol=0.05)


def test_q_inter_state_grad_overlap_late_missing_returns_nan() -> None:
    from corroborate_rl.dqn.measurables import q_inter_state_grad_overlap_late
    assert math.isnan(q_inter_state_grad_overlap_late({}))


def test_bootstrap_action_mismatch_late_late_half_mean() -> None:
    """Late-window mean over synthetic per-step mismatch rates."""
    from corroborate_rl.dqn.measurables import bootstrap_action_mismatch_late

    record = {
        'bootstrap_action_mismatch_per_step': np.array(
            [0.0] * 10 + [0.5, 0.6, 0.7, 0.8, 0.6, 0.5, 0.7, 0.6, 0.5, 0.6],
            dtype=np.float64,
        ),
    }
    # Late half mean ≈ 0.61
    v = bootstrap_action_mismatch_late(record)
    assert math.isclose(v, 0.61, abs_tol=0.05)


def test_bootstrap_action_mismatch_late_missing_returns_nan() -> None:
    from corroborate_rl.dqn.measurables import bootstrap_action_mismatch_late
    assert math.isnan(bootstrap_action_mismatch_late({}))


def test_v_vs_max_delta_late_abs_diff_late_half() -> None:
    record = {
        'online_max_q_per_step': np.array([2.0, 2.0, 4.0, 4.0]),
        'online_mean_q_per_step': np.array([1.0, 1.0, 3.0, 3.0]),
    }
    # delta = |mean - max| = [1, 1, 1, 1]; late half mean = 1.0
    assert v_vs_max_delta_late(record) == 1.0


def test_td_residual_late_mean_over_late_half() -> None:
    record = {
        'td_error': np.array([0.5, 0.5, 0.1, 0.1]),
    }
    # late half (idx 2..4) mean = 0.1
    assert math.isclose(td_residual_late(record), 0.1)


def test_greedy_match_late_fraction_of_argmax_agreement() -> None:
    record = {
        'online_argmax_per_step':  np.array([0, 1, 0, 1, 1, 1, 1, 1]),
        'target_argmax_per_step':  np.array([0, 1, 1, 0, 1, 1, 0, 0]),
    }
    # late half (idx 4..8): online=[1,1,1,1] target=[1,1,0,0]
    # match = [1,1,0,0] → mean = 0.5
    assert greedy_match_late(record) == 0.5


def test_fill_ratio_late_uses_explicit_capacity() -> None:
    record = {
        'buf_size': np.array([0, 100, 500, 1000]),
    }
    # late half (idx 2..4): mean(buf_size) = 750; / capacity 1000 = 0.75
    assert fill_ratio_late(record, capacity=1000) == 0.75


def test_fill_ratio_late_returns_nan_on_zero_capacity() -> None:
    record = {'buf_size': np.array([0, 100])}
    assert math.isnan(fill_ratio_late(record, capacity=0))


# ============ Measurable-contract tests ============

def test_q_trajectory_autocorr_late_smooth_trajectory_returns_one() -> None:
    """Late-half burst with monotone-increasing Q on the active
    trajectory → lag-1 autocorr = 1."""
    from corroborate_rl.dqn.measurables import q_trajectory_autocorr_late

    # 2 bursts, 1 episode, 5 step cap. Late half = burst index 1.
    # Burst 1 Q = [1, 2, 3, 4, 5] all active.
    q = np.zeros((2, 1, 5), dtype=np.float64)
    q[1, 0, :] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    active = np.zeros((2, 1, 5), dtype=np.float64)
    active[1, 0, :] = 1.0
    record = {'predicted_q_per_step': q, 'active_per_step': active}
    assert math.isclose(q_trajectory_autocorr_late(record), 1.0, abs_tol=1e-9)


def test_q_trajectory_autocorr_late_mask_dropped() -> None:
    """Inactive steps (post-done) excluded from the autocorr."""
    from corroborate_rl.dqn.measurables import q_trajectory_autocorr_late

    q = np.zeros((2, 1, 6), dtype=np.float64)
    # late burst Q = [1,2,3,4,5, 999] with 999 inactive
    q[1, 0, :] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 999.0])
    active = np.zeros((2, 1, 6), dtype=np.float64)
    active[1, 0, :5] = 1.0  # last step inactive
    record = {'predicted_q_per_step': q, 'active_per_step': active}
    # The 999 must be excluded; autocorr over [1,2,3,4,5] = 1.0
    assert math.isclose(q_trajectory_autocorr_late(record), 1.0, abs_tol=1e-9)


def test_q_trajectory_autocorr_late_alternating_returns_minus_one() -> None:
    """Alternating high-low Q gives autocorr = -1."""
    from corroborate_rl.dqn.measurables import q_trajectory_autocorr_late

    q = np.zeros((2, 1, 6), dtype=np.float64)
    q[1, 0, :] = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    active = np.ones((2, 1, 6), dtype=np.float64)
    record = {'predicted_q_per_step': q, 'active_per_step': active}
    assert math.isclose(q_trajectory_autocorr_late(record), -1.0, abs_tol=1e-9)


def test_q_trajectory_autocorr_late_averaged_across_episodes() -> None:
    """Two episodes in late burst: one with r=+1, one with r=-1.
    Mean = 0."""
    from corroborate_rl.dqn.measurables import q_trajectory_autocorr_late

    q = np.zeros((2, 2, 5), dtype=np.float64)
    q[1, 0, :] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])     # autocorr=1
    q[1, 1, :] = np.array([0.0, 1.0, 0.0, 1.0, 0.0])     # autocorr=-1
    active = np.ones((2, 2, 5), dtype=np.float64)
    record = {'predicted_q_per_step': q, 'active_per_step': active}
    assert math.isclose(q_trajectory_autocorr_late(record), 0.0, abs_tol=1e-9)


def test_q_trajectory_autocorr_late_missing_inputs() -> None:
    from corroborate_rl.dqn.measurables import q_trajectory_autocorr_late
    assert math.isnan(q_trajectory_autocorr_late({}))


def test_q_autocorr_per_burst_chunked_independence() -> None:
    """Two-burst case: first chunk perfect ramp (autocorr=1), second
    chunk perfect anti-correlation (autocorr=-1). Verifies chunks
    are computed independently and ordering matches eval_step_index."""
    from corroborate_rl.dqn.measurables import q_autocorr_per_burst

    record = {
        # 10-step series: first 5 = ramp; next 5 = alternating
        'online_max_q_per_step': np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0,  # ramp → autocorr=1
             0.0, 1.0, 0.0, 1.0, 0.0],  # alternating → autocorr=-1
            dtype=np.float64,
        ),
        'eval_step_index': np.array([5, 10], dtype=np.int32),
    }
    out = q_autocorr_per_burst(record)
    assert out.shape == (2,)
    assert math.isclose(out[0], 1.0, abs_tol=1e-9)
    assert math.isclose(out[1], -1.0, abs_tol=1e-9)


def test_q_autocorr_per_burst_constant_chunk_returns_nan() -> None:
    """Chunks with zero std (constant Q) → NaN for that chunk only;
    other chunks still computed."""
    from corroborate_rl.dqn.measurables import q_autocorr_per_burst

    record = {
        'online_max_q_per_step': np.array(
            [1.0, 1.0, 1.0, 1.0,  # constant → NaN
             1.0, 2.0, 3.0, 4.0],  # ramp → 1.0
            dtype=np.float64,
        ),
        'eval_step_index': np.array([4, 8], dtype=np.int32),
    }
    out = q_autocorr_per_burst(record)
    assert out.shape == (2,)
    assert math.isnan(out[0])
    assert math.isclose(out[1], 1.0, abs_tol=1e-9)


def test_q_autocorr_per_burst_missing_inputs_returns_empty() -> None:
    """Missing inputs / 0-d arrays return empty vector (caller
    NaN-propagates)."""
    from corroborate_rl.dqn.measurables import q_autocorr_per_burst

    assert q_autocorr_per_burst({}).shape == (0,)
    assert q_autocorr_per_burst({
        'online_max_q_per_step': np.array([1.0, 2.0]),
    }).shape == (0,)  # no eval_step_index


def test_reward_nonzero_frac_sparse_terminal_only() -> None:
    """FR-style sparse terminal-reward: episode of length 5 with
    `mc[t] = γ^(T-1-t)` (only r[T-1]=1, rest=0). Active steps =
    5; nonzero rewards = 1; density = 1/5."""
    from corroborate_rl.dqn.measurables import reward_nonzero_frac

    gamma = 0.99
    # Single burst, single episode, episode_cap=6 (one inactive
    # padded step).
    mc = np.array([[[
        gamma ** 4, gamma ** 3, gamma ** 2, gamma, 1.0, 0.0,
    ]]], dtype=np.float64)
    active = np.array([[[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]], dtype=np.float64)
    record = {
        'mc_return_from_step': mc,
        'active_per_step': active,
        'gamma': gamma,
    }
    # Reconstructed r: [0, 0, 0, 0, 1.0]. Active steps = 5,
    # nonzero = 1, density = 0.2.
    assert math.isclose(
        reward_nonzero_frac(record), 0.2, abs_tol=1e-6,
    )


def test_reward_nonzero_frac_dense_every_step() -> None:
    """CartPole-style dense reward: r[t] = 1 every step. Density = 1.0."""
    from corroborate_rl.dqn.measurables import reward_nonzero_frac

    gamma = 0.99
    # 4-step episode, r=1 every step → mc[t] = Σ_{j≥t} γ^(j-t)
    mc_vals = np.array([
        1 + gamma + gamma ** 2 + gamma ** 3,
        1 + gamma + gamma ** 2,
        1 + gamma,
        1.0,
    ])
    mc = mc_vals.reshape(1, 1, 4)
    active = np.ones((1, 1, 4), dtype=np.float64)
    record = {
        'mc_return_from_step': mc,
        'active_per_step': active,
        'gamma': gamma,
    }
    assert math.isclose(reward_nonzero_frac(record), 1.0, abs_tol=1e-6)


def test_reward_nonzero_frac_all_zero_reward() -> None:
    """Pure-zero-reward episode (vanilla-collapse FR γ=0.999 case):
    mc[t]=0 throughout → density = 0."""
    from corroborate_rl.dqn.measurables import reward_nonzero_frac

    mc = np.zeros((1, 1, 5), dtype=np.float64)
    active = np.ones((1, 1, 5), dtype=np.float64)
    record = {
        'mc_return_from_step': mc, 'active_per_step': active,
        'gamma': 0.99,
    }
    assert reward_nonzero_frac(record) == 0.0


def test_reward_nonzero_frac_missing_inputs() -> None:
    from corroborate_rl.dqn.measurables import reward_nonzero_frac

    assert math.isnan(reward_nonzero_frac({}))
    # Missing gamma
    assert math.isnan(reward_nonzero_frac({
        'mc_return_from_step': np.zeros((1, 1, 3)),
        'active_per_step': np.ones((1, 1, 3)),
    }))


def test_late_window_measurables_registered_under_their_function_names() -> None:
    """`@measurable` registers each late-window scalar under its
    declared name in the global registry; lookup via
    `get_registered` returns the same instance."""
    from corroborate.measurables import get_registered

    for name in (
        'q_gap_late', 'q_gap_growth', 'q_max_growth',
        'q_autocorr_late', 'q_autocorr_per_burst',
        'q_trajectory_autocorr_late',
        'reward_nonzero_frac',
        'q_action_grad_overlap_late',
        'q_inter_state_grad_overlap_late',
        'bootstrap_action_mismatch_late',
        'v_vs_max_delta_late', 'td_residual_late',
        'greedy_match_late', 'fill_ratio_late',
    ):
        m = get_registered(name)
        assert m is not None, f'{name} not in measurable registry'
        assert m.name == name


def test_late_window_measurable_reads_match_declared_record_keys() -> None:
    """Each measurable's `reads` declares the exact record keys
    its fn body consumes — used downstream by
    `Bridge.transitive_reads` for the redundancy primitive."""
    assert q_gap_late.reads == (
        'online_max_q_per_step', 'online_min_q_per_step',
    )
    assert q_max_growth.reads == ('online_max_q_per_step',)
    assert td_residual_late.reads == ('td_error',)
    assert greedy_match_late.reads == (
        'online_argmax_per_step', 'target_argmax_per_step',
    )
    assert fill_ratio_late.reads == ('buf_size',)


def test_fill_ratio_late_is_measurable_with_extra_kwarg() -> None:
    """fill_ratio_late wraps as Measurable but takes an extra
    `capacity` kwarg the framework's auto-resolver doesn't fill.
    Caller must pass capacity directly."""
    from corroborate.measurables import Measurable

    assert isinstance(fill_ratio_late, Measurable)
    record = {'buf_size': np.array([0, 100, 500, 1000])}
    # Direct call with capacity works (Measurable.__call__ proxies
    # to fn(record, **deps) — passing capacity as a dep).
    assert fill_ratio_late(record, capacity=1000) == 0.75


# ============ state-conditional argmax measurables ============

def _argmax_record(argmax: list[int], state: list[int]) -> dict[str, np.ndarray]:
    """Build a record exposing `online_argmax_per_step` +
    `state_hash_per_step` for the state-conditional argmax measurables."""
    return {
        'online_argmax_per_step': np.asarray(argmax, dtype=np.int64),
        'state_hash_per_step': np.asarray(state, dtype=np.int64),
    }


def test_state_conditional_argmax_entropy_zero_when_policy_deterministic() -> None:
    """Late slice = second half of trajectory. If every state-bucket
    deterministically maps to one action in the late half, then
    H(argmax | s) = 0 in every bucket → weighted average = 0."""
    # 8 steps; late half is last 4. States [A, A, B, B] in late half;
    # actions [0, 0, 1, 1] — perfect state→action mapping.
    record = _argmax_record(
        argmax=[0, 1, 0, 1, 0, 0, 1, 1],
        state=[0, 0, 0, 0, 1, 1, 2, 2],
    )
    assert state_conditional_argmax_entropy_late(record) == 0.0


def test_state_conditional_argmax_entropy_log2_under_uniform_within_state() -> None:
    """Two equally-frequent actions within each state-bucket → H per
    bucket = log(2) ≈ 0.693. Weighted average = log(2)."""
    record = _argmax_record(
        argmax=[0, 0, 0, 0, 0, 1, 0, 1],  # late half: [0, 1, 0, 1] split across buckets
        state=[0, 0, 0, 0, 0, 0, 1, 1],   # late half: bucket 0 (× 2: 0,1), bucket 1 (× 2: 0,1)
    )
    assert math.isclose(
        state_conditional_argmax_entropy_late(record), math.log(2), abs_tol=1e-9,
    )


def test_state_conditional_argmax_entropy_includes_singleton_buckets() -> None:
    """A singleton bucket (n_s=1) contributes h_s=0 — the empirical
    conditional is a point mass. Critical: this is what makes the
    estimator's support match the marginal H(argmax), so chain-rule
    `H(X|Y) ≤ H(X)` holds in the MI consumer."""
    # Late half (last 4): state=[0, 0, 1, 2]; actions=[0, 1, 0, 0].
    # Bucket 0 has 2 obs uniform over {0,1} → H = log(2).
    # Buckets 1, 2 are singletons → H = 0 each.
    # Weighted average: (2 · log(2) + 1 · 0 + 1 · 0) / 4 = log(2) / 2.
    record = _argmax_record(
        argmax=[1, 1, 1, 1, 0, 1, 0, 0],
        state=[0, 0, 0, 0, 0, 0, 1, 2],
    )
    assert math.isclose(
        state_conditional_argmax_entropy_late(record),
        math.log(2) / 2, abs_tol=1e-9,
    )


def test_state_conditional_argmax_entropy_nan_when_single_bucket() -> None:
    """`unique_s.size < 2` short-circuits to NaN — no
    state-conditional structure to measure."""
    record = _argmax_record(
        argmax=[0, 1, 0, 1, 0, 1, 0, 1],
        state=[5, 5, 5, 5, 5, 5, 5, 5],
    )
    assert math.isnan(state_conditional_argmax_entropy_late(record))


def test_state_conditional_argmax_entropy_nan_when_state_hash_missing() -> None:
    """No state_hash_per_step key in record → NaN sentinel
    (env-side hash not registered)."""
    record = {'online_argmax_per_step': np.array([0, 1, 0, 1])}
    assert math.isnan(state_conditional_argmax_entropy_late(record))


def test_mutual_info_chain_rule_nonneg() -> None:
    """`I(s; argmax) = H(argmax) − H(argmax | s) ≥ 0` by the chain
    rule of entropy. Closed-form: 4 late-half steps with marginal
    actions {0, 0, 0, 1} → marginal H = -(3/4)log(3/4) - (1/4)log(1/4).
    Conditional: bucket 0 = [0, 0] → H=0; bucket 1 = [0, 1] → H=log(2).
    Weighted conditional = (2·0 + 2·log(2)) / 4 = log(2)/2.
    MI = marginal_H − log(2)/2."""
    record = _argmax_record(
        argmax=[1, 1, 1, 1, 0, 0, 0, 1],
        state=[0, 0, 0, 0, 0, 0, 1, 1],
    )
    h_marg = -(3 / 4) * math.log(3 / 4) - (1 / 4) * math.log(1 / 4)
    expected = h_marg - math.log(2) / 2
    assert math.isclose(
        mutual_info_state_argmax_late(record), expected, abs_tol=1e-9,
    )
    assert mutual_info_state_argmax_late(record) >= 0.0


def test_mutual_info_zero_under_independent_state_action() -> None:
    """When p(a | s) ≈ p(a) for every s, MI ≈ 0. Construct: in both
    state buckets in the late half, the actions are uniformly {0, 1}."""
    record = _argmax_record(
        argmax=[0, 0, 0, 0, 0, 1, 0, 1],
        state=[0, 0, 0, 0, 0, 0, 1, 1],
    )
    # late half: state=[0, 0, 1, 1]; argmax=[0, 1, 0, 1].
    # marginal H = log(2); conditional H per bucket = log(2); MI = 0.
    assert math.isclose(
        mutual_info_state_argmax_late(record), 0.0, abs_tol=1e-9,
    )


def test_mutual_info_nan_when_marginal_collapses_to_single_action() -> None:
    """If the marginal argmax is a single action in the late half,
    MI is undefined (the marginal has zero entropy and conditioning
    can't reduce it further). The measurable returns NaN."""
    record = _argmax_record(
        argmax=[1, 1, 1, 1, 0, 0, 0, 0],
        state=[0, 0, 0, 0, 0, 1, 2, 3],
    )
    assert math.isnan(mutual_info_state_argmax_late(record))


# ============ per-burst-window outcome measurables ============

def test_eval_full_auc_mean_averages_all_bursts_and_episodes() -> None:
    """Closed-form: `mean(mc_return)` over all 10×5 entries.
    Construct a record where DDQN-like trajectory rises from 0
    to 10 across 10 bursts (each burst's 5 episodes share the
    same value for the test). Expected mean = (0+1+...+9)/10 = 4.5."""
    mc = np.tile(np.arange(10, dtype=np.float64)[:, None], (1, 5))
    assert math.isclose(
        eval_full_auc_mean({'mc_return': mc}), 4.5, abs_tol=1e-9,
    )


def test_eval_full_auc_mean_nan_when_missing() -> None:
    assert math.isnan(eval_full_auc_mean({}))
    assert math.isnan(eval_full_auc_mean({'mc_return': np.array([[]])}))


def test_eval_late_burst_mean_picks_ceil_n_over_3() -> None:
    """For n_bursts=10, late window is ceil(10/3)=4 → bursts 6..9
    (0-indexed). Closed-form values 0..9 → mean of [6,7,8,9] = 7.5."""
    mc = np.tile(np.arange(10, dtype=np.float64)[:, None], (1, 5))
    assert math.isclose(
        eval_late_burst_mean({'mc_return': mc}), 7.5, abs_tol=1e-9,
    )


def test_eval_late_burst_mean_single_burst_returns_full_mean() -> None:
    """With 1 burst total, last-30% rounds up to 1 burst → mean
    over all episodes of the only burst."""
    mc = np.array([[1.0, 2.0, 3.0]])
    assert math.isclose(
        eval_late_burst_mean({'mc_return': mc}), 2.0, abs_tol=1e-9,
    )


def test_eval_late_burst_diverges_from_best_on_late_collapse() -> None:
    """The use-case: vanilla peaks mid-training then degrades.
    Best-burst picks the peak; late-burst reads the degraded end.
    Construct: bursts = [10, 20, 30, 25, 20] (peak at burst 2,
    drift down after). best = 30; late-30% (last 2 bursts) mean
    = 22.5. The 7.5-unit gap is the signature `eval_best > late`
    captures (memory `findings_per_burst_acrobot_k_sweep`)."""
    mc = np.array([[10.0], [20.0], [30.0], [25.0], [20.0]])
    late = eval_late_burst_mean({'mc_return': mc})
    # late-30% of 5 = ceil(5/3) = 2 → last 2 bursts [25, 20] → 22.5.
    assert math.isclose(late, 22.5, abs_tol=1e-9)
