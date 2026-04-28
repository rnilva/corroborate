"""Tests for `corroborate.rl.dqn.mediators` — §5 candidate-mediator
projections from per-cell DQN traces."""
from __future__ import annotations

import math

import numpy as np

from corroborate.rl.dqn.mediators import (
    epsilon_late,
    fill_ratio_late,
    greedy_match_late,
    q_gap_growth,
    q_gap_late,
    q_max_growth,
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


def test_epsilon_late_closed_form_linear_schedule() -> None:
    # eps_init=1.0, eps_final=0.1, anneal_steps=100, total=200.
    # late half = steps 100..199; all clipped to anneal_steps so
    # progress=1.0; ε = eps_final = 0.1.
    eps = epsilon_late(
        eps_init=1.0, eps_final=0.1,
        anneal_steps=100, total_steps=200,
    )
    assert math.isclose(eps, 0.1)


def test_epsilon_late_partial_anneal_in_late_window() -> None:
    # If anneal_steps spans into the late window, the late-window
    # mean is intermediate. eps_init=1.0, eps_final=0.0,
    # anneal_steps=200, total=200. Late half = steps 100..199.
    # progress = step/200 ∈ [0.5, 0.995]; ε = 1 - progress.
    # mean(1 - progress) on [100..199] = 1 - mean([100..199])/200
    #   = 1 - 149.5/200 = 0.2525.
    eps = epsilon_late(
        eps_init=1.0, eps_final=0.0,
        anneal_steps=200, total_steps=200,
    )
    assert math.isclose(eps, 0.2525)


def test_epsilon_late_returns_nan_on_zero_anneal() -> None:
    assert math.isnan(epsilon_late(
        eps_init=1.0, eps_final=0.0,
        anneal_steps=0, total_steps=200,
    ))
