"""Tests for the DDQN measurables — declared scalar derivations
of the per-step record. Verifies values, shapes, registration,
and transitive resolution via `evaluate_with_measurables`."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from corroborate.measurable import (
    evaluate_with_measurables,
    get_registered,
)
from corroborate.rl.dqn.measurables import (
    pearson_r_online_target,
    q_gap_per_step,
    q_max_per_step,
    q_mean_per_step,
    q_std_per_step,
    target_q_max_per_step,
    target_q_mean_per_step,
    td_error_norm_per_step,
)


# ============ Registration ============

def test_all_dqn_measurables_registered() -> None:
    """Every measurable defined in `rl.dqn.measurables` is
    indexed in the global registry by its function name."""
    for m in (
        q_mean_per_step, q_max_per_step, q_std_per_step,
        q_gap_per_step, target_q_mean_per_step,
        target_q_max_per_step, td_error_norm_per_step,
        pearson_r_online_target,
    ):
        assert get_registered(m.name) is m


# ============ Q-distribution measurables ============

def _q_record() -> Mapping[str, object]:
    """Synthetic record with per-step Q-vectors over 3 actions."""
    online = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 4.0, 5.0],
        [0.0, 1.0, 2.0],
    ])
    target = np.array([
        [0.5, 1.5, 2.5],
        [3.0, 3.5, 4.5],
        [0.5, 0.5, 1.5],
    ])
    return {
        'online_q_per_action': online,
        'target_q_per_action': target,
        'td_error': np.array([0.1, 0.2, -0.3]),
        'pearson_stats': np.zeros((3, 5)),  # placeholder
    }


def test_q_mean_per_step_value() -> None:
    rec = _q_record()
    out = q_mean_per_step(rec)
    assert out.tolist() == [2.0, (4 + 4 + 5) / 3, 1.0]


def test_q_max_per_step_value() -> None:
    rec = _q_record()
    out = q_max_per_step(rec)
    assert out.tolist() == [3.0, 5.0, 2.0]


def test_q_std_per_step_finite() -> None:
    rec = _q_record()
    out = q_std_per_step(rec)
    assert all(math.isfinite(v) for v in out.tolist())


def test_q_gap_per_step_value() -> None:
    """Gap = max - second_max. For [1, 2, 3], gap = 3 - 2 = 1.
    For [4, 4, 5], gap = 5 - 4 = 1."""
    rec = _q_record()
    out = q_gap_per_step(rec)
    assert out.tolist() == [1.0, 1.0, 1.0]


def test_target_q_mean_per_step_value() -> None:
    rec = _q_record()
    out = target_q_mean_per_step(rec)
    expected = [1.5, (3.0 + 3.5 + 4.5) / 3, (0.5 + 0.5 + 1.5) / 3]
    for got, want in zip(out.tolist(), expected, strict=True):
        assert math.isclose(got, want)


def test_target_q_max_per_step_value() -> None:
    rec = _q_record()
    out = target_q_max_per_step(rec)
    assert out.tolist() == [2.5, 4.5, 1.5]


def test_td_error_norm_per_step_value() -> None:
    """td_error 1-D: per-step magnitude is just abs(td_error)."""
    rec = _q_record()
    out = td_error_norm_per_step(rec)
    assert out.tolist() == [0.1, 0.2, 0.3]


# ============ Transitive resolution via evaluate_with_measurables ============

def test_consumer_can_declare_measurables_as_params() -> None:
    """A consumer fn that declares q_max_per_step and
    q_mean_per_step as parameters has both auto-injected by
    `evaluate_with_measurables`."""
    def consumer(
        record: Mapping[str, object],
        q_max_per_step: np.ndarray,
        q_mean_per_step: np.ndarray,
    ) -> float:
        del record
        # Average gap between max and mean across steps.
        return float((q_max_per_step - q_mean_per_step).mean())

    out = evaluate_with_measurables(consumer, _q_record())
    # Per-step (max - mean) for the 3 steps in _q_record:
    # (3 - 2) = 1; (5 - 13/3) ≈ 0.667; (2 - 1) = 1. Mean ≈ 0.889.
    assert math.isclose(out, (1.0 + (5.0 - 13/3) + 1.0) / 3, abs_tol=1e-9)


def test_pearson_r_online_target_via_resolver() -> None:
    """Pearson-r measurable computed through the resolver. Inputs
    designed so r ≈ 1 (online and target have the same shape)."""
    online = np.arange(3 * 4 * 2, dtype=np.float64).reshape((3, 4, 2))
    on_flat = online.reshape(online.shape[0], -1)
    target_flat = on_flat.copy()
    pearson_stats = np.stack([
        on_flat.mean(axis=-1),
        target_flat.mean(axis=-1),
        (on_flat ** 2).mean(axis=-1),
        (target_flat ** 2).mean(axis=-1),
        (on_flat * target_flat).mean(axis=-1),
    ], axis=-1)
    rec: Mapping[str, object] = {'pearson_stats': pearson_stats}

    def consumer(
        record: Mapping[str, object],
        pearson_r_online_target: float,
    ) -> float:
        del record
        return pearson_r_online_target

    r = evaluate_with_measurables(consumer, rec)
    assert math.isclose(r, 1.0, abs_tol=1e-5)
