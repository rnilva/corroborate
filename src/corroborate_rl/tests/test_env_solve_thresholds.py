"""Tests for `env_solve_thresholds` — table coverage + predicate
behavior."""
from __future__ import annotations

import pytest

from corroborate_rl.env_catalogue import ENV_REGISTRY
from corroborate_rl.env_catalogue import (
    SOLVE_THRESHOLDS,
    envs_with_threshold, is_solved,
)


# ============ Coverage ============

def test_all_registered_envs_have_solve_thresholds() -> None:
    """Every env in the catalogue must appear in SOLVE_THRESHOLDS
    (even if confidence is 'absent') — silent missing entries
    risk mis-classification."""
    registered = set(ENV_REGISTRY)
    thresholded = set(SOLVE_THRESHOLDS)
    missing = registered - thresholded
    assert not missing, (
        f'envs in ENV_REGISTRY but missing from SOLVE_THRESHOLDS: '
        f'{sorted(missing)!r}'
    )


def test_no_extra_thresholds_outside_catalogue() -> None:
    """SOLVE_THRESHOLDS should not contain envs that aren't
    registered — drift means a threshold is for a fictional env."""
    registered = set(ENV_REGISTRY)
    thresholded = set(SOLVE_THRESHOLDS)
    extras = thresholded - registered
    assert not extras, (
        f'envs in SOLVE_THRESHOLDS but not in ENV_REGISTRY: '
        f'{sorted(extras)!r}'
    )


def test_absent_envs_have_none_threshold() -> None:
    """Confidence='absent' must coincide with threshold=None.
    Consistency check on the table."""
    for t in SOLVE_THRESHOLDS.values():
        if t.confidence == 'absent':
            assert t.threshold is None, (
                f'{t.env_name}: confidence=absent but threshold='
                f'{t.threshold}'
            )
        else:
            assert t.threshold is not None, (
                f'{t.env_name}: confidence={t.confidence} but '
                f'threshold=None'
            )


# ============ Predicate ============

def test_is_solved_true_for_value_at_or_above_threshold() -> None:
    """CartPole-v1's threshold is 99.0 (discounted at γ=0.99);
    outcome 99.34 → True (max possible discounted return)."""
    assert is_solved('CartPole-v1', 99.34) is True
    assert is_solved('CartPole-v1', 99.0) is True


def test_is_solved_false_for_value_below_threshold() -> None:
    """CartPole-v1 outcome=80 (well below the 99.0 discounted
    threshold) → False."""
    assert is_solved('CartPole-v1', 80.0) is False


def test_is_solved_handles_negative_thresholds() -> None:
    """Acrobot's threshold is -63.4 (discounted from raw -100).
    -50 ≥ -63.4 → solved; -100 < -63.4 → not."""
    assert is_solved('Acrobot-v1', -50.0) is True
    assert is_solved('Acrobot-v1', -100.0) is False


def test_is_solved_returns_none_when_threshold_absent() -> None:
    """Misc envs with confidence='absent' return None — caller
    must decide how to treat 'unknown'."""
    assert is_solved('GaussianBandit-misc', 100.0) is None
    assert is_solved('Pong-misc', 50.0) is None


def test_is_solved_unknown_env_raises() -> None:
    """Unregistered envs raise KeyError loudly — silent False
    would mis-classify cells from those envs."""
    with pytest.raises(KeyError, match='not registered'):
        _ = is_solved('NotAnEnv', 0.0)


# ============ envs_with_threshold ============

def test_envs_with_threshold_excludes_absent() -> None:
    """Helper returns only the envs we can defensibly classify."""
    with_thresh = set(envs_with_threshold())
    for name, t in SOLVE_THRESHOLDS.items():
        if t.confidence == 'absent':
            assert name not in with_thresh
        else:
            assert name in with_thresh


def test_envs_with_threshold_count() -> None:
    """At pin-time: 9 literature + 4 derived = 13 envs with
    defensible thresholds. The 5 misc envs are 'absent'."""
    assert len(envs_with_threshold()) == 13


# Custom-table override: removed in the env_catalogue
# consolidation. No production caller used it; sensitivity
# analyses can read `EnvSpec.solve_threshold` directly off the
# registry and apply their own comparison.
