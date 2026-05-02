"""Theorem-gap measurables for the DQN claims.

Each entry here is a `Measurable[DQNTrajectoryRecord, float]`
returning a *gap magnitude* — how far this run sits from the
literal theorem condition. Gap = 0 means the theorem condition
holds; gap > 0 means it doesn't, and the magnitude is the
research-actionable signal: "method X reduces gap Y by Δ".

**No-data sentinel: NaN, not 0.** When a gap can't be computed
from the available data (replay never filled, fewer than 2 sync
windows, etc.), the measurable returns `float('nan')`. The
`at_most(gap, threshold)` wrap maps NaN → POWER_INSUFFICIENT
verdict (rather than HELD), so a hypothesis with a scope-commit
that never had data won't spuriously hold. Downstream consumers
must handle NaN: aggregation uses `nanmean`-style reductions.

Three roles consume each gap differently (see `invariant.py`
module docstring):

- Intervention: read the scalar into RunRow.measurements; the
  paired comparison surface (`HypothesisComparisonRow.from_cells`)
  computes Δ-gap across (treatment, baseline) pairs. NaN means
  "no data" — different from "gap = 0" (which means "data
  confirmed compliance").
- Falsification: wrap with `at_most(gap, threshold,
  of_claim=...)` in Hypothesis.bridges. NaN propagates to
  POWER_INSUFFICIENT.
- Causal analysis: same `at_most`-wraps gate discovery sample
  selection; NaN-bearing facts can be filtered out of the
  in-scope set.

**Honest scope.** Only the gaps whose data the v0 record can
already support are implemented here. Gaps that need richer
logging (Q-snapshots per step for empirical contraction rate;
MC-return ground truth for Jensen bias) are deferred — see
FUTURE_WORKS.md."""
from __future__ import annotations

import math
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
    windows align with target-net resets. Returns `NaN` when the
    trajectory has fewer than 2 complete windows (no across-
    window ratio computable) — the no-data sentinel that
    `at_most` maps to POWER_INSUFFICIENT."""
    assert sync_period > 0, f'sync_period must be positive; got {sync_period}'
    name = f'fqi_decay_gap[τ={sync_period},γ={gamma:g}]'

    def fn(record: DQNTrajectoryRecord) -> float:
        td = jnp.asarray(record['td_error'])
        n = int(td.shape[0])
        n_windows = n // sync_period
        if n_windows < 2:
            # No across-window ratio computable — NaN sentinel.
            return float('nan')
        # Per-window sup-norm |TD_error|.
        window_sup_norms: list[float] = []
        for k in range(n_windows):
            window = td[k * sync_period : (k + 1) * sync_period]
            window_sup_norms.append(float(jnp.max(jnp.abs(window))))
        # Across-window decay ratios — geometric mean, principled
        # for multiplicative-decay statistics (Munos 2003 FQI's
        # contraction is multiplicative). Skip ratios where the
        # denominator window had near-zero sup-norm (no signal to
        # decay from); if all are skipped, return NaN.
        log_ratios: list[float] = []
        for k in range(1, n_windows):
            denom = abs(window_sup_norms[k - 1])
            if denom < 1e-9:
                continue
            ratio = window_sup_norms[k] / denom
            log_ratios.append(float(jnp.log(jnp.maximum(ratio, 1e-12))))
        if not log_ratios:
            return float('nan')
        geomean_ratio = float(jnp.exp(sum(log_ratios) / len(log_ratios)))
        return float(max(0.0, geomean_ratio - gamma))

    return Measurable(fn=fn, name=name, reads=('td_error',))


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

    def fn(
        record: DQNTrajectoryRecord,
        pearson_r_online_target: float,
    ) -> float:
        """Declares `pearson_r_online_target` as a measurable dep.
        The framework's `evaluate_with_measurables` resolver looks
        up the registered name, computes once per record (memoized
        in the per-cell cache so multiple bridges sharing the dep
        share the work), and injects.

        Asymmetric: only positive correlation is the failure mode
        for DDQN's decoupling assumption. Anti-correlation is
        favourable ⇒ gap = 0."""
        del record  # value comes via the resolved measurable
        if pearson_r_online_target != pearson_r_online_target:  # NaN
            return float('nan')
        return max(0.0, float(pearson_r_online_target))

    return Measurable(
        fn=fn, name=name, reads=('pearson_stats',),
    )


# Note: `action_coverage_gap` was retired — it was a
# numerical-sanity assert ("did ε-greedy explore ≥ 2 distinct
# actions?") dressed as a Watkins invariant. The literal Watkins
# (s, a)-coverage measurement is `state_action_coverage_gap`,
# which lives below; the trivial action-floor check (if needed)
# is a unit test, not a theorem-condition gap.


# ============ Jensen overestimation gap (reads EvalRecord) ============

def jensen_overestimation_gap() -> Measurable[DQNTrajectoryRecord, float]:
    """Empirical mean (predicted_q_at_start − mc_return) over the
    eval-burst fields of the merged record. Hasselt 2010, 2016
    §3 — vanilla DQN's Jensen-inequality bias is positive (Q̂
    overestimates Q*); DDQN aims to reduce it. Gap = max(0, mean
    bias) — clipped because only positive bias (over) is the
    Jensen signature; negative means under-estimating and is a
    different phenomenon.

    Typed at `DQNTrajectoryRecord` (the merged training+eval
    record produced by `train_with_eval`). Reads the
    `(n_bursts, K)`-shaped `predicted_q_at_start` and `mc_return`
    arrays, flattening both axes to compute the global mean bias
    across all eval episodes."""
    name = 'jensen_overestimation_gap'

    def fn(record: DQNTrajectoryRecord) -> float:
        predicted = jnp.asarray(record['predicted_q_at_start'])
        actual = jnp.asarray(record['mc_return'])
        bias = float(jnp.mean(predicted - actual))
        return float(max(0.0, bias))

    return Measurable(
        fn=fn, name=name,
        reads=('predicted_q_at_start', 'mc_return'),
    )


# ============ Jensen structural floor + dormancy gap ============
#
# The Jensen-overestimation theorem (Hasselt 2010) gives, for |A|
# iid noisy Q estimators with std σ:
#
#     E[max_a Q̂_a] − max_a E[Q̂_a]  ≳  σ · √(2 log |A|)
#
# This is the *structural floor* of the bias under the iid-Gaussian
# regime — the minimum overestimation that exists by Jensen alone.
# DDQN's `double_greedify` claims to reduce overestimation by
# decoupling selection from valuation; for the claim to bite, the
# observed bias has to actually be in this regime (≥ the floor).
# When the observed bias is *below* the structural floor, the
# Jensen-premise of `double_greedify` is dormant — there's no max-
# of-noisy-estimators bias to correct, so the mechanism's edge in
# the causal chain is structurally weak.
#
# Implementation note (load-bearing). The structural floor depends
# on (action_dim, q_noise). Both surface naturally from the per-
# step Q distribution `online_q_per_action` shape `(steps,
# n_actions)`:
# - `n_actions = q.shape[-1]` — action dim is the last axis's
#   length, no separate measurement plumbing needed.
# - `σ_q = mean_t std_a(q[t, :])` — std *across actions* per step,
#   averaged across the late training half. Each axis-collapse is
#   spelled out explicitly: `jnp.std(..., axis=-1)` for the action
#   axis (where Jensen-max bias is computed), then `jnp.mean(...)`
#   for the time axis to produce the scalar floor. Aggregating in
#   one shot would hide which axis is which.

def _jensen_floor_late_from_reduced() -> Measurable[
    DQNTrajectoryRecord, float,
]:
    """Sibling of `jensen_floor_late` that reads the persisted
    1-D per-step σ_Q (`online_std_q_per_step`, emitted by
    `Q_TRACE_REDUCTIONS`) plus the joined `n_actions` scalar
    instead of the dropped 2-D `online_q_per_action`. Same
    formula `σ_late × √(2 log |A|)`, distinct name so persistence
    keeps observations from this path under their own column."""
    name = 'jensen_floor_late_from_reduced'

    def fn(record: DQNTrajectoryRecord) -> float:
        std = jnp.asarray(record['online_std_q_per_step'])
        if std.ndim != 1 or std.shape[0] < 2:
            return float('nan')
        n_actions_v = record['n_actions']
        if not isinstance(n_actions_v, int) or n_actions_v < 2:
            return float('nan')
        late_lo = int(std.shape[0]) // 2
        sigma = float(jnp.mean(std[late_lo:]))
        if not math.isfinite(sigma):
            return float('nan')
        return sigma * math.sqrt(2.0 * math.log(n_actions_v))

    return Measurable(
        fn=fn, name=name,
        reads=('online_std_q_per_step', 'n_actions'),
    )


def jensen_floor_late() -> Measurable[DQNTrajectoryRecord, float]:
    """Structural Jensen floor: `σ_late × √(2 log |A|)`. Reads
    `online_q_per_action` shape `(steps, n_actions)`. Returns the
    minimum-bias-by-Jensen-alone scalar at the late-half regime.

    The σ collapse is over the action axis (per-step std-across-
    actions), then averaged over the late training half. The mean
    is inevitable to make the metric scalar; making it explicit
    keeps the action-axis dependence visible in the code path.

    `fallbacks` adds `jensen_floor_late_from_reduced` — fires when
    persistence has already collapsed the action axis
    (`online_std_q_per_step` from `Q_TRACE_REDUCTIONS`) and joined
    env metadata supplies `n_actions` as a scalar field. Same
    formula, distinct persisted name."""
    name = 'jensen_floor_late'

    def fn(record: DQNTrajectoryRecord) -> float:
        q = jnp.asarray(record['online_q_per_action'])
        # Shape contract: (steps, n_actions). Action dim emerges
        # from `q.shape[-1]` — explicit, no plumbed leaf.
        if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] < 2:
            return float('nan')
        n_actions = int(q.shape[-1])
        late_lo = int(q.shape[0]) // 2
        late = q[late_lo:]                          # (late_steps, n_actions)
        per_step_action_std = jnp.std(late, axis=-1)  # (late_steps,)
        sigma = float(jnp.mean(per_step_action_std))
        if not math.isfinite(sigma):
            return float('nan')
        return sigma * math.sqrt(2.0 * math.log(n_actions))

    return Measurable(
        fn=fn, name=name, reads=('online_q_per_action',),
        fallbacks=(_jensen_floor_late_from_reduced(),),
    )


def jensen_dormancy_gap() -> Measurable[DQNTrajectoryRecord, float]:
    """Gap between the *structural* Jensen floor and the *observed*
    overestimation: `max(0, floor − observed_gap)`.

    Convention: gap = 0 ⇒ Jensen-premise active (observed bias ≥
    structural floor), so `double_greedify` has something to
    correct. gap > 0 ⇒ Jensen-premise dormant (observed bias is
    *below* what Jensen-alone would predict at this |A| and σ_Q),
    so the mechanism's edge in DDQN's causal chain is structurally
    weak — the data isn't in the regime where decoupling matters.

    Wrap with `at_most(jensen_dormancy_gap, threshold=0,
    of_claim=double_greedify)` to commit the scope: HELD when
    premise active, INVARIANT_VIOLATION when premise dormant.

    Reads `(predicted_q_at_start, mc_return, online_q_per_action)`.
    """
    observed = jensen_overestimation_gap()

    def _build(floor: Measurable[DQNTrajectoryRecord, float], name: str) -> (
        Measurable[DQNTrajectoryRecord, float]
    ):
        def fn(record: DQNTrajectoryRecord) -> float:
            obs = observed(record)
            flr = floor(record)
            if math.isnan(obs) or math.isnan(flr):
                return float('nan')
            return float(max(0.0, flr - obs))
        return Measurable(
            fn=fn, name=name,
            reads=tuple(observed.reads) + tuple(floor.reads),
        )

    primary = _build(jensen_floor_late(), 'jensen_dormancy_gap')
    return Measurable(
        fn=primary.fn, name=primary.name, reads=primary.reads,
        fallbacks=(_build(
            _jensen_floor_late_from_reduced(),
            'jensen_dormancy_gap_from_reduced',
        ),),
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
    state_hash) produces a `NaN` no-data Measurable — the
    theorem isn't measurable in that regime; `at_most` maps the
    NaN to POWER_INSUFFICIENT rather than misleading HELD."""
    if state_hash_cardinality is None or n_actions <= 0:
        # No-data: env has no state_hash discretization. NaN
        # sentinel — distinguishes "no data" from "perfect
        # coverage" (gap = 0).
        no_data_name = 'state_action_coverage_gap[no_data]'

        def no_data_fn(_record: DQNTrajectoryRecord) -> float:
            del _record
            return float('nan')
        return Measurable(fn=no_data_fn, name=no_data_name, reads=())

    max_unique = state_hash_cardinality * n_actions
    name = (
        f'state_action_coverage_gap[card={state_hash_cardinality},'
        f'A={n_actions}]'
    )

    def fn(record: DQNTrajectoryRecord) -> float:
        sh = jnp.asarray(record['state_hash']).flatten()
        ac = jnp.asarray(record['action']).flatten()
        # Defensive: contract says state_hash ∈ [0, cardinality);
        # if violated, the pair encoding overflows and unique-count
        # is misleading. Clip to the valid range so gap stays
        # honest even on contract-broken inputs.
        sh_clipped = jnp.clip(sh, 0, state_hash_cardinality - 1)
        ac_clipped = jnp.clip(ac, 0, n_actions - 1)
        pairs = sh_clipped * n_actions + ac_clipped
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
    'hasselt_covariance_gap',
    'jensen_dormancy_gap',
    'jensen_floor_late',
    'jensen_overestimation_gap',
    'state_action_coverage_gap',
]
