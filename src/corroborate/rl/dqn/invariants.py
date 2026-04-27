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
    """Across-window sup-norm decay rate of the TD-error vs the
    γ-contraction theoretical rate.

    Theory: Munos 2003 fitted-Q-iteration. Within each
    [k·τ, (k+1)·τ] window the regression target is frozen, so
    the inner gradient step is supervised regression — FQI's
    actual contraction property is **between** iterations
    (windows, k → k+1) in **sup-norm**, not within a single
    window's mean. The principled signature:

        ‖TD_error‖_∞ at window k+1 ≤ γ · ‖TD_error‖_∞ at window k

    What we compute: per-window `max|TD_error|` (sup-norm), then
    consecutive-window decay ratios across windows, averaged.
    Gap = max(0, avg_ratio − γ) — non-zero when observed
    across-window contraction falls short of the γ rate.

    `sync_period` must match the experiment's sync_period so
    windows align with target-net resets. Returns 0.0 when the
    trajectory has fewer than 2 complete windows (no across-
    window ratio computable)."""
    assert sync_period > 0, f'sync_period must be positive; got {sync_period}'
    name = f'fqi_decay_gap[τ={sync_period},γ={gamma:g}]'

    def fn(record: DQNTrajectoryRecord) -> float:
        td = jnp.asarray(record['td_error'])
        n = int(td.shape[0])
        n_windows = n // sync_period
        if n_windows < 2:
            # Need at least two windows for an across-window ratio.
            return 0.0
        # Per-window sup-norm |TD_error|.
        window_sup_norms: list[float] = []
        for k in range(n_windows):
            window = td[k * sync_period : (k + 1) * sync_period]
            window_sup_norms.append(float(jnp.max(jnp.abs(window))))
        # Across-window decay ratios.
        ratios = [
            window_sup_norms[k] / max(abs(window_sup_norms[k - 1]), 1e-9)
            for k in range(1, n_windows)
        ]
        avg_ratio = sum(ratios) / len(ratios)
        return float(max(0.0, avg_ratio - gamma))

    return Measurable(fn=fn, name=name, reads=('td_error',))


# ============ Lin 1992 i.i.d. sampling gap ============

def lin_iid_gap(
    capacity: int,
) -> Measurable[DQNTrajectoryRecord, float]:
    """KL divergence of the empirical sampling distribution over
    buffer indices from `Uniform(0, capacity)`, computed only
    over steps where the replay buffer is fully filled
    (`buf_size == capacity`). Gap = the KL — 0 means perfectly
    uniform, >0 means biased.

    Theory: Lin 1992 + Singh-Sutton 1996. Q-learning + replay
    convergence assumes uniform i.i.d. resampling from the
    buffer. Uniform replay's empirical sampling distribution
    should match `Uniform(0, buf_size)` at each step. Bias
    toward recent transitions, hot indices, etc. shows up as
    KL > 0.

    Filtering to `buf_size == capacity` avoids a structural
    confound: during the fill phase the buffer is small, so
    `sample_indices` is mechanically biased toward low values
    (because high values aren't available yet). Including those
    steps would inflate the KL for reasons unrelated to Lin's
    sampling-uniformity claim. Returns `gap=0` if the buffer
    never fills (no-data, not a violation).

    `capacity` must match the experiment's `buffer_capacity` so
    the post-fill filter aligns."""
    assert capacity > 0, f'capacity must be positive; got {capacity}'
    name = f'lin_iid_gap[cap={capacity}]'

    def fn(record: DQNTrajectoryRecord) -> float:
        indices = jnp.asarray(record['sample_indices'])  # (T, batch)
        buf_size = jnp.asarray(record['buf_size'])       # (T,)
        full_mask = buf_size >= capacity                 # (T,)
        n_full = int(jnp.sum(full_mask))
        if n_full == 0:
            return 0.0
        # Filter rows where buffer was full; flatten across (T_full, batch).
        full_indices = indices[full_mask].flatten()
        if int(full_indices.size) == 0:
            return 0.0
        counts = jnp.bincount(full_indices, length=capacity)
        total = float(jnp.sum(counts))
        if total == 0.0:
            return 0.0
        empirical = counts / total
        eps = 1e-12
        nonzero = empirical > eps
        log_ratio = jnp.where(
            nonzero,
            jnp.log(empirical * capacity + eps),
            0.0,
        )
        kl = float(jnp.sum(empirical * log_ratio))
        return float(max(0.0, kl))

    return Measurable(fn=fn, name=name, reads=('sample_indices', 'buf_size'))


# ============ Hasselt independence gap (Pearson correlation) ============

def hasselt_covariance_gap() -> Measurable[DQNTrajectoryRecord, float]:
    """Empirical positive Pearson correlation between Q_online
    and Q_target across the sampling distribution. Gap =
    `max(0, r)` — only positive correlation degrades DDQN's
    decoupling; anti-correlation actually *helps* (estimators
    cancel), so the gap is the asymmetric `max(0, r)`, not `|r|`.

    Theory: Hasselt 2010, 2016. DDQN's bias-correction relies on
    online and target nets being roughly *uncorrelated*
    estimators of Q*. Uncorrelated (or anti-correlated) ⇒
    gap ≈ 0. Perfect positive correlation (vanilla DQN at sync,
    online ≡ target) ⇒ gap = 1.

    Computes Pearson r over flattened `(T, batch, n_actions)`
    arrays of Q-values from the always-on `value_probe` in
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
        # Constant-variance side ⇒ undefined correlation ⇒ 0.
        r = jnp.where(denom > 1e-9, cov / jnp.where(denom > 1e-9, denom, 1.0), 0.0)
        # Asymmetric: only positive correlation is the failure
        # mode for DDQN's decoupling assumption.
        return float(jnp.maximum(0.0, r))

    return Measurable(
        fn=fn, name=name, reads=('online_q_values', 'target_q_values'),
    )


# Note: `action_coverage_gap` was retired — it was a
# numerical-sanity assert ("did ε-greedy explore ≥ 2 distinct
# actions?") dressed as a Watkins invariant. The literal Watkins
# (s, a)-coverage measurement is `state_action_coverage_gap`,
# which lives below; the trivial action-floor check (if needed)
# is a unit test, not a theorem-condition gap.


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
    'jensen_overestimation_gap',
    'state_action_coverage_gap',
]
