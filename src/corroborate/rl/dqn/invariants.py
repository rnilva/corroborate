"""Theorem-gap measurables for the DQN claims.

Each entry here is a `Measurable[DQNTrajectoryRecord, float]`
returning a *gap magnitude* — how far this run sits from the
literal theorem condition. Gap = 0 means the theorem condition
holds; gap > 0 means it doesn't, and the magnitude is the
research-actionable signal: "method X reduces gap Y by Δ".

Three roles consume each gap differently (see `invariant.py`
module docstring):

- Intervention: read the scalar into RunRow.stats; per-comparison
  Δ-gap on ComparisonRow.stats.
- Falsification: wrap with `at_most(gap, threshold,
  of_claim=...)` in Hypothesis.bridges. The threshold is the
  author's scope commitment.
- Causal analysis: same `at_most`-wraps gate discovery sample
  selection.

**Honest scope.** Only the gaps whose data the v0 record can
already support are implemented here. Gaps that need richer
logging (Q-snapshots per step for empirical contraction rate;
MC-return ground truth for Jensen bias; Q-VALUES per batch
state for Hasselt covariance; per-env state hashes for Watkins
(s, a)-coverage) are deferred — see FUTURE_WORKS.md."""
from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from corroborate.measurable import Measurable


# Convenience alias. Bridge / Measurable accept Mapping[str,
# jnp.ndarray] because Mapping is covariant in its value type.
type DQNTrajectoryRecord = Mapping[str, jnp.ndarray]


# ============ FQI within-window decay gap ============

def fqi_decay_gap(
    sync_period: int,
    *,
    gamma: float = 0.99,
) -> Measurable[DQNTrajectoryRecord, float]:
    """Empirical per-window decay rate of the TD-error vs. the
    γ-contraction theoretical rate. Gap is the magnitude by which
    observed decay falls short of γ.

    Theory: Mnih 2015 §3 + Munos 2003 fitted-Q-iteration. Within
    each [k·τ, (k+1)·τ] window, the regression target is frozen,
    so the gradient step is supervised regression — FQI is a
    γ-contraction in sup-norm under Lipschitz function-class
    assumptions. The signature: `||TD_error_late_window||` /
    `||TD_error_early_window||` ≤ γ per window, asymptotically.

    What we compute: for each sync window, the ratio `mean(td_error
    in window's late half) / mean(td_error in window's early half)`.
    Average those ratios. Gap = max(0, average_ratio − γ): non-zero
    when observed decay is slower than γ-contraction predicts.

    `sync_period` must match the experiment's sync_period so the
    windows align with target-net resets."""
    assert sync_period > 0, f'sync_period must be positive; got {sync_period}'
    name = f'fqi_decay_gap[τ={sync_period},γ={gamma:g}]'

    def fn(record: DQNTrajectoryRecord) -> float:
        td = jnp.asarray(record['td_error'])
        n = int(td.shape[0])
        # Discard incomplete trailing window if any.
        n_windows = n // sync_period
        if n_windows == 0:
            # Run too short to have even one full window; report 0
            # (no-data; not a violation).
            return 0.0
        ratios: list[float] = []
        for k in range(n_windows):
            start = k * sync_period
            end = start + sync_period
            window = td[start:end]
            half = sync_period // 2
            if half == 0:
                continue
            early = float(jnp.mean(window[:half]))
            late = float(jnp.mean(window[half:]))
            ratio = late / max(abs(early), 1e-9)
            ratios.append(ratio)
        if not ratios:
            return 0.0
        avg_ratio = sum(ratios) / len(ratios)
        return float(max(0.0, avg_ratio - gamma))

    return Measurable(fn=fn, name=name, reads=('td_error',))


# ============ Lin 1992 i.i.d. sampling gap ============

def lin_iid_gap(
    capacity: int,
) -> Measurable[DQNTrajectoryRecord, float]:
    """KL divergence of the empirical sampling distribution over
    buffer indices from the uniform distribution. Gap = the KL
    itself — 0 means perfectly uniform, >0 means biased.

    Theory: Lin 1992 + Singh-Sutton 1996. Q-learning + replay
    convergence assumes uniform i.i.d. resampling from the buffer.
    Uniform replay's empirical sampling distribution should match
    Uniform(0, buf_size). Bias toward recent transitions, hot
    indices, etc. shows up as KL > 0.

    What we compute: histogram of `sample_indices` flattened over
    the trajectory, normalised; KL(empirical || uniform) over the
    populated buffer support. Larger KL = more biased sampling.

    `capacity` is the buffer's max capacity, used to size the
    uniform reference distribution."""
    assert capacity > 0, f'capacity must be positive; got {capacity}'
    name = f'lin_iid_gap[cap={capacity}]'

    def fn(record: DQNTrajectoryRecord) -> float:
        indices = jnp.asarray(record['sample_indices']).flatten()
        if indices.size == 0:
            return 0.0
        # Histogram count per buffer index.
        # bincount returns shape (capacity,) when minlength=capacity.
        counts = jnp.bincount(indices, length=capacity)
        total = float(jnp.sum(counts))
        if total == 0.0:
            return 0.0
        empirical = counts / total
        # KL(empirical || uniform). Uniform = 1/capacity per slot.
        # KL = Σ p · log(p / q) over support where p > 0.
        # log(p / q) = log(p · capacity).
        eps = 1e-12
        nonzero = empirical > eps
        log_ratio = jnp.where(
            nonzero,
            jnp.log(empirical * capacity + eps),
            0.0,
        )
        kl = float(jnp.sum(empirical * log_ratio))
        return float(max(0.0, kl))

    return Measurable(fn=fn, name=name, reads=('sample_indices',))


# ============ Hasselt independence gap (Pearson correlation) ============

def hasselt_covariance_gap() -> Measurable[DQNTrajectoryRecord, float]:
    """Empirical |Pearson correlation| between Q_online and
    Q_target across the sampling distribution.

    Theory: Hasselt 2010, 2016. DDQN's bias-correction relies on
    online and target nets being roughly independent estimators
    of Q*. Independence ⇒ correlation ≈ 0 ⇒ gap ≈ 0. Perfect
    correlation (vanilla DQN at sync, online ≡ target) ⇒ gap = 1.

    Computes Pearson r over flattened `(T, batch, n_actions)`
    arrays of Q-values from the always-on `_value_probe` in
    `train_phase`. The probe runs both networks on the bootstrap's
    batch each step; values are stored raw (not pre-reduced) so
    the correlation is a post-hoc reduction here."""
    name = 'hasselt_covariance_gap[Pearson_r]'

    def fn(record: DQNTrajectoryRecord) -> float:
        on = jnp.asarray(record['online_q_values']).flatten()
        tg = jnp.asarray(record['target_q_values']).flatten()
        on_centered = on - jnp.mean(on)
        tg_centered = tg - jnp.mean(tg)
        cov = jnp.mean(on_centered * tg_centered)
        std_on = jnp.sqrt(jnp.mean(on_centered ** 2))
        std_tg = jnp.sqrt(jnp.mean(tg_centered ** 2))
        denom = std_on * std_tg
        # If either side is constant (zero variance), correlation
        # is undefined; treat as 0 (no information about
        # independence either way — conservative).
        r = jnp.where(denom > 1e-9, cov / jnp.where(denom > 1e-9, denom, 1.0), 0.0)
        return float(jnp.abs(r))

    return Measurable(
        fn=fn, name=name, reads=('online_q_values', 'target_q_values'),
    )


# ============ Action coverage proxy (Watkins, caveated) ============

def action_coverage_gap(
    *,
    expected_min_unique: int = 2,
) -> Measurable[DQNTrajectoryRecord, float]:
    """Gap = max(0, expected_min_unique - unique_actions_seen).

    Watkins 1992 requires every (s, a) visited infinitely often.
    The literal (s, a)-coverage condition is NOT measured here —
    that needs per-env state-hashing (deferred to step 4). What
    we measure: the trivial necessary floor that ε-greedy
    explored at least `expected_min_unique` distinct actions.

    Gap = 0 when unique actions ≥ expected_min_unique (the floor
    holds); gap = (expected_min_unique − unique_count) when it
    doesn't. Failing this means ε-greedy collapsed to fewer
    actions than expected, which definitely violates Watkins;
    passing it does NOT imply (s, a) coverage."""
    name = f'action_coverage_gap[≥{expected_min_unique}]'

    def fn(record: DQNTrajectoryRecord) -> float:
        actions = jnp.asarray(record['action']).flatten()
        n_unique = int(jnp.unique(actions).shape[0])
        return float(max(0, expected_min_unique - n_unique))

    return Measurable(fn=fn, name=name, reads=('action',))


# ============ Jensen overestimation gap (reads EvalRecord) ============

def jensen_overestimation_gap() -> Measurable[Mapping[str, jnp.ndarray], float]:
    """Empirical mean (predicted_q_at_start − mc_return) over an
    eval-pass record. Hasselt 2010, 2016 §3 — vanilla DQN's
    Jensen-inequality bias is positive (Q̂ overestimates Q*); DDQN
    aims to reduce it. Gap = max(0, mean bias) — clipped because
    only positive bias (over) is the Jensen signature; negative
    means under-estimating and is a different phenomenon.

    Reads from the `EvalTrajectoryRecord` produced by
    `train_with_eval`'s eval pass. The `(n_bursts, K)`-shaped
    `predicted_q_at_start` and `mc_return` arrays are flattened
    over both axes to compute the global mean bias across all
    eval episodes."""
    name = 'jensen_overestimation_gap'

    def fn(record: Mapping[str, jnp.ndarray]) -> float:
        predicted = jnp.asarray(record['predicted_q_at_start'])
        actual = jnp.asarray(record['mc_return'])
        bias = float(jnp.mean(predicted - actual))
        return float(max(0.0, bias))

    return Measurable(
        fn=fn, name=name,
        reads=('predicted_q_at_start', 'mc_return'),
    )


# ============ Watkins (s, a)-coverage gap (env-parameterised) ============

def state_action_coverage_gap(
    *,
    state_hash_cardinality: int | None,
    n_actions: int,
) -> Measurable[DQNTrajectoryRecord, float]:
    """Watkins-style (s, a) coverage gap. Gap = `1 −
    unique_pairs / max_unique_pairs`. Coverage 1 (every (s, a)
    visited) ⇒ gap = 0; coverage 0 ⇒ gap = 1.

    Theory: Watkins 1992 tabular Q-learning convergence requires
    every (s, a) ∞-often. The empirical proxy: count distinct
    `(state_hash, action)` pairs over the trajectory, normalise
    by `state_hash_cardinality * n_actions`. Gap measures how far
    the realised coverage falls short of the theorem's
    assumption.

    `state_hash_cardinality=None` (image envs without a declared
    state_hash) produces a `gap=0` no-data Measurable — the
    theorem isn't measurable in that regime, so we report no
    information rather than a misleading number."""
    if state_hash_cardinality is None or n_actions <= 0:
        # No-data sentinel: env has no state_hash discretization.
        no_data_name = 'state_action_coverage_gap[no_data]'

        def no_data_fn(_record: DQNTrajectoryRecord) -> float:
            del _record
            return 0.0
        return Measurable(fn=no_data_fn, name=no_data_name, reads=())

    max_unique = state_hash_cardinality * n_actions
    name = (
        f'state_action_coverage_gap[card={state_hash_cardinality},'
        f'A={n_actions}]'
    )

    def fn(record: DQNTrajectoryRecord) -> float:
        sh = jnp.asarray(record['state_hash'])
        ac = jnp.asarray(record['action'])
        # Encode (s, a) pair as a combined integer in
        # [0, cardinality * n_actions).
        pairs = sh.flatten() * n_actions + ac.flatten()
        n_unique = int(jnp.unique(pairs).shape[0])
        coverage = n_unique / max_unique
        return float(max(0.0, 1.0 - coverage))

    return Measurable(fn=fn, name=name, reads=('state_hash', 'action'))


# ============ Convenience: all v0-implementable gaps ============

# Note this list contains *Measurable factories* — call each with
# the experiment's parameters (sync_period, capacity, ...) to get
# the actual gap Measurable. The author then either reads the
# scalar directly (intervention) or wraps with `at_most(gap,
# threshold, of_claim=...)` in Hypothesis.bridges (falsification
# / scope).
__all__ = [
    'DQNTrajectoryRecord',
    'fqi_decay_gap',
    'lin_iid_gap',
    'hasselt_covariance_gap',
    'action_coverage_gap',
    'jensen_overestimation_gap',
    'state_action_coverage_gap',
]
