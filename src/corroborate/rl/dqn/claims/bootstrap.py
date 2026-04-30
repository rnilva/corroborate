"""Bootstrap — Bellman target = `r + γ · gradient_rule(greedification(s'))`.

Decomposed paper-honestly into two orthogonal components:

- **`greedification`** — how the next-state value v(s') is
  computed from Q. The DDQN-vs-vanilla axis lives here:
    - `max_greedify` (vanilla): v(s') = max_a Q_target(s', a)
    - `double_greedify` (DDQN): v(s') = Q_target(s',
      argmax_a Q_online(s', a))
  Same Protocol; same call signature; the swap is one kwarg.

- **`gradient_rule`** — what backprops through the target. The
  semi-gradient-vs-full-gradient axis lives here:
    - `semi_gradient` (Mnih 2015 default): stop_gradient(target)
      so loss only differentiates through Q_online's prediction.
    - `full_gradient`: differentiate through everything (rarely
      used; included for ablations).

`bootstrap` itself is the composition: `r + γ · (1−done) ·
gradient_rule(greedification(...))`. It's still a Claim
(intervention can swap the whole composition wholesale), but the
*usual* DDQN intervention is now `{'greedification':
double_greedify}` — sharper structural identity than swapping a
monolithic bootstrap.

**Theorem references.** Bellman optimality T* is a γ-contraction
on Q* (Bertsekas-Tsitsiklis 1996 §6.3); vanilla bootstrap
implements T* directly. Hasselt 2010, 2016: vanilla DQN exhibits
a Jensen-inequality bias E[max_a Q̂(s,a)] ≥ max_a E[Q̂(s,a)],
because max is convex in Q. DDQN decouples action selection
(online net) from value evaluation (target net) so the two
estimators are roughly independent → unbiased target."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from corroborate.claim import claim
from corroborate.invariant import at_most, attach_invariant
from corroborate.rl.dqn.invariants import jensen_dormancy_gap
from corroborate.rl.dqn.types import (
    Greedification,
    GradientRule,
    Params,
    QFunction,
)


# ============ Greedification: how v(s') is computed ============

@claim
def max_greedify(
    *,
    online_params: Params,
    target_params: Params,
    q_network: QFunction,
    next_obs: jax.Array,
) -> jax.Array:
    """Vanilla greedification: v(s') = max_a Q_target(s', a).

    Both action selection AND evaluation use the target network.
    `online_params` ignored (DDQN's flag is unused in vanilla)."""
    del online_params
    next_q = q_network(target_params, next_obs)
    return jnp.max(next_q, axis=-1)


@claim
def double_greedify(
    *,
    online_params: Params,
    target_params: Params,
    q_network: QFunction,
    next_obs: jax.Array,
) -> jax.Array:
    """DDQN greedification: v(s') = Q_target(s', argmax_a Q_online(s', a)).

    Action selection uses online network; value evaluation uses
    target network. Hasselt 2016: this asymmetry reduces vanilla
    DQN's Q-overestimation bias by decorrelating the two
    estimators (Jensen-bias signature).

    **Premise dependency.** The mechanism's bite scales with the
    structural Jensen floor `σ_Q · √(2 log |A|)`: at |A|=2 with
    low Q-noise the floor is small and the correction is
    structurally weak. The attached `jensen_dormancy_gap`
    invariant fires INVARIANT_VIOLATION when the *observed*
    overestimation is below this floor — there's no Jensen-bias
    above noise to correct, so the mechanism's causal-chain edge
    is dormant on this run regardless of activation."""
    next_q_online = q_network(online_params, next_obs)
    a_star = jnp.argmax(next_q_online, axis=-1)
    next_q_target = q_network(target_params, next_obs)
    return jnp.take_along_axis(
        next_q_target, a_star[..., None], axis=-1,
    ).squeeze(-1)


# Attach the Jensen-dormancy invariant to `double_greedify`. The
# claim graph now exposes the dependency on (action_dim, σ_Q):
# composition discovery surfaces the bridge whenever
# `double_greedify` is in a theory tree, and `gap_value > 0`
# (premise dormant) preempts an outcome-positive verdict.
attach_invariant(
    at_most(
        jensen_dormancy_gap(), threshold=0.0,
        of_claim=double_greedify,
    ),
    to=double_greedify,
)


# ============ Gradient rule: what backprops through target ============

@claim
def semi_gradient(target: jax.Array) -> jax.Array:
    """Mnih 2015's choice: stop gradient at the target so the loss
    only differentiates through `predicted` (Q_online's output).
    Without this, the bootstrap target moves with online updates
    and training is unstable."""
    return jax.lax.stop_gradient(target)


@claim
def full_gradient(target: jax.Array) -> jax.Array:
    """Differentiate through the target. Rarely used (Mnih 2015
    explicitly avoids it); included for ablation studies."""
    return target


# ============ Bootstrap: the composition ============

@claim
def bootstrap(
    *,
    online_params: Params,
    target_params: Params,
    q_network: QFunction,
    next_obs: jax.Array,
    reward: jax.Array,
    done: jax.Array,
    gamma: float,
    n_step: int = 1,
    greedification: Greedification = max_greedify,
    gradient_rule: GradientRule = semi_gradient,
) -> jax.Array:
    """Bellman target: `R^n + γ^n · (1−done) · gradient_rule(
    greedification(...))`.

    For 1-step (n_step=1, default) this collapses to the standard
    `r + γ · (1−done) · v(s')`. For n_step=k, the input `reward`
    is already the k-step accumulated return Σ_{j=0}^{k-1} γʲ
    r_{t+j} (computed by the `n_step_return` Free Claim during
    rollout, before the transition is added to Replay), and the
    bootstrap discount on v(s_{t+k}) is `γ^k` because the
    bootstrap target lies k steps in the future.

    Single source of truth: `n_step` is a top-level kwarg of
    `dqn` / `dqn_step` and flows here from there. Both rollout's
    `n_step_return` and bootstrap's discount read the same
    value; the Hypothesis intervention dict sets it ONCE at the
    dqn level (`{'n_step': 3}`), not separately on Replay and
    bootstrap.

    `greedification` is the DDQN-vs-vanilla axis; `gradient_rule`
    is the semi-gradient-vs-full-gradient axis. Authors swap
    either independently via partial:

        partial(bootstrap, greedification=double_greedify)
        partial(bootstrap, gradient_rule=full_gradient)
        partial(bootstrap, greedification=double_greedify,
                gradient_rule=full_gradient)

    `n_step` is NOT typically set via partial on bootstrap —
    callers pass it as a top-level dqn kwarg, and dqn_step
    forwards it to bootstrap internally."""
    v_next = greedification(
        online_params=online_params,
        target_params=target_params,
        q_network=q_network,
        next_obs=next_obs,
    )
    discount = gamma ** n_step
    target = reward + discount * (1.0 - done) * v_next
    return gradient_rule(target)


