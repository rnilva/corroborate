"""Theorem-condition invariants for the DQN claims.

Each invariant tests whether a *theorem-level* property holds in
the run's trajectory. Failure is `INVARIANT_VIOLATION`, not
`NO_EFFECT`: the claim's mechanism didn't operate within the
theorem's domain of applicability under this run, so the outcome
test is out of scope.

Composed via `bounded(of=Measurable, threshold, theorem, of_claim)`
from framework primitives (`reductions`, `invariant`). No DQN-
specific resolver — every invariant is a value-built composition.

The thresholds are deliberately generous: these are *divergence
detectors*, not tight bounds. A `q_max__max_abs` of 100 is fine
for CartPole; 1e3 indicates the deadly-triad-triggered runaway
that target-net + replay are supposed to prevent.

For tighter bounds (linked to the published Q* range per env),
the invariants would parameterise on the env's reward range —
deferred to step 4's env catalogue."""
from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from corroborate.bridge import Bridge
from corroborate.invariant import bounded
from corroborate.reductions import (
    from_key,
    growth_window,
    max_abs,
)
from corroborate.rl.dqn.claims.bootstrap import vanilla_bootstrap
from corroborate.rl.dqn.claims.loss import squared_error
from corroborate.rl.dqn.claims.q_network import mlp_q
from corroborate.rl.dqn.claims.target_sync import periodic_copy


# Convenience type alias for the per-step DQN trajectory record:
# `dict[str, jax.Array]` after `python_loop` / `scan_loop`
# stacking. Bridges accept `Mapping[str, jnp.ndarray]` because
# Mapping is covariant in its value type.
type DQNTrajectoryRecord = Mapping[str, jnp.ndarray]


# ============ Q-network: Banach contraction ============

q_bounded: Bridge[DQNTrajectoryRecord] = bounded(
    max_abs(from_key('max_q')),
    threshold=1e3,
    theorem=(
        'Banach contraction on T* (Bertsekas-Tsitsiklis 1996, §6.3): '
        'γ-contraction implies |Q*| ≤ R_max / (1−γ). '
        'Q diverging past this bound signals the deadly-triad '
        '(off-policy + bootstrap + FA) overpowering target-net + '
        'replay stabilisation.'
    ),
    of_claim=mlp_q,
)


# ============ Bootstrap: Jensen-bias overestimation drift ============

max_q_overestimation_bounded: Bridge[DQNTrajectoryRecord] = bounded(
    # Skip the first quarter of the trajectory in the early window:
    # warmup leaves max_q at random-init magnitudes, and the
    # late/early ratio is dominated by initial-learning growth
    # rather than the Jensen-bias drift the invariant targets.
    growth_window(from_key('max_q'), early=(0.25, 0.5), late=(0.75, 1.0)),
    threshold=1e3,
    theorem=(
        'Hasselt 2010, 2016: vanilla DQN exhibits Jensen-bias '
        'overestimation, E[max_a Q̂] ≥ max_a E[Q̂]. The bias is '
        'bounded under stationary policy + target-net; runaway '
        'late/early growth ratio of max_q indicates diverging '
        'overestimation. Threshold is generous (divergence '
        'detector, not tight bound) — tightening requires an '
        'env-specific Q* range, deferred to step 4.'
    ),
    of_claim=vanilla_bootstrap,
)


# ============ Squared-error loss: semi-gradient stability ============

loss_bounded: Bridge[DQNTrajectoryRecord] = bounded(
    max_abs(from_key('loss')),
    threshold=1e6,
    theorem=(
        'Semi-gradient TD (Sutton-Barto 11.2): L = E[(y − Q)²] is '
        'bounded iff bootstrap target y and prediction Q are both '
        'bounded. Loss explosion ⇒ either Q diverged or numerical '
        'instability — semi-gradient is unbounded under deadly '
        'triad off-policy.'
    ),
    of_claim=squared_error,
)


# ============ Periodic-copy target sync: residual boundedness ============

td_error_bounded: Bridge[DQNTrajectoryRecord] = bounded(
    max_abs(from_key('td_error')),
    threshold=1e3,
    theorem=(
        'Mnih 2015 §3 / Munos 2003 FQI: target-net freezes the '
        'regression target y for τ steps, making |y − Q| bounded '
        'within each τ-window. Unbounded TD-error means target-net '
        'stabilisation failed — claim out of scope.'
    ),
    of_claim=periodic_copy,
)


# Note on static invariants. Kolmogorov-axiom-style checks
# (ε ∈ [0, 1], action ∈ [0, n_actions), batch shapes match) are
# guaranteed by construction at the claim's body or call site.
# Those belong as plain `assert` lines inside the @claim, not as
# `INVARIANT_VIOLATION` bridges — invariants here are reserved
# for *theorem-level* scope conditions (when does the run sit
# inside the theorem's domain of applicability).


# ============ Public registry ============

DQN_INVARIANTS: tuple[Bridge[DQNTrajectoryRecord], ...] = (
    q_bounded,
    max_q_overestimation_bounded,
    loss_bounded,
    td_error_bounded,
)
"""Theorem-condition invariants for the v0 DQN claims, attached
to: mlp_q (Banach contraction), vanilla_bootstrap (Jensen bias),
squared_error (semi-gradient stability), periodic_copy (FQI
contraction). A trajectory is in-scope of all four theorems iff
every invariant returns HELD."""
