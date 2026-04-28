"""DDQN measurables — declared scalar derivations of the per-step
record. Each is a `@measurable` registered in the framework's
name-keyed registry.

These are the *substrate-level* derivations bridges and
diagnostics consume across the DDQN claim graph. Reductions live
here (not in `train_phase`) so that:

- The training loop only emits raw record fields plus the
  cheapest in-loop sum-stats.
- Multiple bridges sharing a derivation (q_mean reads in three
  places) compute it exactly once per record, memoized by the
  framework.
- Authors who want a NEW measurable add it here without changing
  `train_phase` or any bridge body.

`reads` declarations are explicit so the redundancy primitive
(`compute_redundancy(h, G)`) can fingerprint each bridge by its
transitive read-set.

The current set covers Q-distribution shape (`q_mean`, `q_max`,
`q_std`, `q_gap`), target-network coverage (`target_q_mean`,
`target_q_max`), and TD-error magnitude (`td_error_norm`,
`td_error_max`)."""
from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from corroborate.measurable import measurable


# ============ Online Q distribution ============

# Inputs are `online_q_per_action` shape `(steps, n_actions)` — the
# in-loop batch-averaged per-step Q vector. Reductions collapse the
# action axis to per-step scalars.

@measurable(reads=('online_q_per_action',))
def q_mean_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Mean Q across actions, per training step. Series shape
    `(steps,)`."""
    arr = np.asarray(record['online_q_per_action']).astype(np.float64)
    return arr.mean(axis=-1) if arr.ndim >= 1 else arr


@measurable(reads=('online_q_per_action',))
def q_max_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Max Q across actions, per training step."""
    arr = np.asarray(record['online_q_per_action']).astype(np.float64)
    return arr.max(axis=-1) if arr.ndim >= 1 else arr


@measurable(reads=('online_q_per_action',))
def q_std_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Std Q across actions, per step. Captures the spread of
    the action-value distribution at each training step."""
    arr = np.asarray(record['online_q_per_action']).astype(np.float64)
    return arr.std(axis=-1) if arr.ndim >= 1 else arr


@measurable(reads=('online_q_per_action',))
def q_gap_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Gap between max and second-max Q-values per step. The
    DDQN-relevant signal: a small gap means the greedy action is
    barely distinguishable from a runner-up — the regime where
    overestimation matters most."""
    arr = np.asarray(record['online_q_per_action']).astype(np.float64)
    if arr.ndim < 2 or arr.shape[-1] < 2:
        return np.zeros(arr.shape[:-1], dtype=np.float64)
    sorted_arr = np.sort(arr, axis=-1)
    return sorted_arr[..., -1] - sorted_arr[..., -2]


# ============ Target Q distribution ============

@measurable(reads=('target_q_per_action',))
def target_q_mean_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Mean target-Q across actions, per step."""
    arr = np.asarray(record['target_q_per_action']).astype(np.float64)
    return arr.mean(axis=-1) if arr.ndim >= 1 else arr


@measurable(reads=('target_q_per_action',))
def target_q_max_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Max target-Q across actions, per step."""
    arr = np.asarray(record['target_q_per_action']).astype(np.float64)
    return arr.max(axis=-1) if arr.ndim >= 1 else arr


# ============ TD-error magnitude ============

@measurable(reads=('td_error',))
def td_error_norm_per_step(
    record: Mapping[str, object],
) -> npt.NDArray[np.float64]:
    """Per-step √mean(td_error²) — a positive scalar magnitude."""
    arr = np.asarray(record['td_error']).astype(np.float64)
    if arr.ndim == 0:
        return np.abs(arr)
    if arr.ndim == 1:
        return np.abs(arr)
    return np.sqrt(np.mean(arr ** 2, axis=tuple(range(1, arr.ndim))))


# ============ Pearson r — online vs target Q populations ============

@measurable(reads=('pearson_stats',))
def pearson_r_online_target(record: Mapping[str, object]) -> float:
    """Population-level Pearson r between online-Q and target-Q
    over all (step, sample, action) triples observed during
    training. Aggregates the per-step `pearson_stats` sufficient
    statistics. Same scalar `hasselt_covariance_gap` reduces to;
    exposed here as a reusable measurable so other bridges /
    diagnostics can read it without redoing the math."""
    ps = jnp.asarray(record['pearson_stats'])
    agg = ps.mean(axis=0)
    m_on, m_tg, m_on_sq, m_tg_sq, m_xy = (
        float(agg[0]), float(agg[1]), float(agg[2]),
        float(agg[3]), float(agg[4]),
    )
    var_on = m_on_sq - m_on ** 2
    var_tg = m_tg_sq - m_tg ** 2
    if var_on <= 1e-12 or var_tg <= 1e-12:
        return float('nan')
    cov = m_xy - m_on * m_tg
    return cov / (var_on * var_tg) ** 0.5
