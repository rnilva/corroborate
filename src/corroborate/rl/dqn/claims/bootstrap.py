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

from corroborate.core.claim import claim
from corroborate.rl.dqn.claims.q_network import Params, QFunction
from corroborate.rl.dqn.types import (
    Greedification,
    GradientRule,
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


# Jensen-dormancy invariant lives on the measurable channel
# (`rl/dqn/measurables.py:at_most_jensen_dormancy_gap_zero_verdict`)
# after Phase 4 of the Bridge-collapse refactor. Substrate factories
# include it via `dqn_default_measurables()`; cell_runner persists
# the per-cell verdict at `at_most[jensen_dormancy_gap<=0].
# verdict`. The framework no longer carries a per-record Bridge
# channel, so attach_invariant is gone.


@claim
def expectile_greedify(
    *,
    online_params: Params,
    target_params: Params,
    q_network: QFunction,
    next_obs: jax.Array,
    tau: float = 0.7,
    n_iters: int = 8,
) -> jax.Array:
    """Expectile-pessimistic greedification: v(s') =
    expectile_τ(Q_target(s', ·)).

    The asymmetric expectile of a vector v at level τ is the
    fixed-point of the weighted mean μ where w_i = τ if v_i ≥ μ
    else (1−τ). For τ=1 the weights collapse to "max only",
    recovering vanilla. For τ=0.5 the weights are uniform,
    recovering mean. Intermediate τ ∈ (0.5, 1) is pessimistic
    relative to max.

    `online_params` is unused — expectile-greedify, like vanilla
    `max_greedify`, doesn't decouple selection from evaluation.
    The mechanism is structurally distinct from DDQN's
    selection-evaluation decoupling; the contrast tests whether
    the residual `bootstrap_fraction → g_link | g_mech` (DDQN
    200k corpus, ATE=+0.88) is DDQN-specific (different
    bias-correction mechanism fixes it) or sparse-reward-
    intrinsic (same residual under both).

    Garg et al 2023 (Extreme Q-Learning, "XQL"): the residual
    max-bias after DDQN's action-noise correction is σ-
    proportional to Q-vector spread; an expectile target is the
    consistent estimator under extreme-value statistics for
    finite action sets.

    `n_iters`: fixed-point iterations. 8 converges for typical
    n_actions ≤ 16 (RL standard); raise for larger spaces."""
    del online_params  # not used — pessimistic operator on target
    next_q = q_network(target_params, next_obs)
    mu = jnp.mean(next_q, axis=-1, keepdims=True)
    for _ in range(n_iters):
        weights = jnp.where(next_q >= mu, tau, 1.0 - tau)
        mu = (
            jnp.sum(weights * next_q, axis=-1, keepdims=True)
            / jnp.sum(weights, axis=-1, keepdims=True)
        )
    return mu.squeeze(-1)


@claim
def dampened_double_greedify(
    *,
    online_params: Params,
    target_params: Params,
    q_network: QFunction,
    next_obs: jax.Array,
    alpha: float = 1.0,
) -> jax.Array:
    """Linearly-interpolated DDQN: v(s') = α·v_DDQN + (1−α)·v_vanilla.

    α=0 → vanilla `max_greedify`.
    α=1 → full `double_greedify`.
    Intermediate α scales the per-step Jensen correction ε by α.

    Used to test the multiplicative claim
    `g_link ≈ ε · effective_horizon` by varying ε at fixed
    γ (i.e. fixed effective_horizon). If g_link is linear in α,
    the multiplicative model is corroborated; if not (e.g.
    threshold response or non-monotone), refuted.

    Both branches compute (it's an interpolation, not a switch),
    so per-batch cost is the sum of vanilla + DDQN target ops."""
    v_ddqn = double_greedify.fn(
        online_params=online_params, target_params=target_params,
        q_network=q_network, next_obs=next_obs,
    )
    v_vanilla = max_greedify.fn(
        online_params=online_params, target_params=target_params,
        q_network=q_network, next_obs=next_obs,
    )
    return alpha * v_ddqn + (1.0 - alpha) * v_vanilla


@claim
def adaptive_dormancy_greedify(
    *,
    online_params: Params,
    target_params: Params,
    q_network: QFunction,
    next_obs: jax.Array,
    sigma_floor_factor: float = 1.0,
) -> jax.Array:
    """Adaptive greedification that switches between
    `max_greedify` (vanilla) and `double_greedify` (DDQN) per
    batch based on a per-batch dormancy heuristic.

    The framework's `jensen_dormancy_gap` invariant tests whether
    the structural Jensen floor `σ_Q · √(2 log |A|)` is exceeded
    by the observed bias. We don't have observed bias inside a
    training batch, so we use a per-batch in-state proxy:

        excess  = mean_b(max_a Q_target(s'_b, a) − mean_a Q_target(s'_b, a))
        floor   = σ_Q · √(2 log |A|)  with σ_Q = mean_b(std_a Q_target(s'_b, a))
        active  = excess ≥ sigma_floor_factor · floor

    When `active`, dispatch to DDQN's `double_greedify` (premise
    active → bias to correct). When dormant, dispatch to
    `max_greedify` (vanilla → avoid over-correction on a tightly-
    distributed Q).

    `sigma_floor_factor` is a knob:
      - 1.0 matches the strict dormancy threshold
      - <1.0 makes DDQN MORE active (lower bar)
      - >1.0 makes DDQN LESS active (higher bar)

    Both branches compute (jax.lax.select), so JIT tracing is
    uniform — ~10% extra compute on the bootstrap target."""
    next_q_target = q_network(target_params, next_obs)
    n_actions = next_q_target.shape[-1]
    # Per-state action-axis stats, then averaged across batch.
    per_state_std = jnp.std(next_q_target, axis=-1)
    per_state_max = jnp.max(next_q_target, axis=-1)
    per_state_mean = jnp.mean(next_q_target, axis=-1)
    sigma_q = jnp.mean(per_state_std)
    excess = jnp.mean(per_state_max - per_state_mean)
    floor = sigma_q * jnp.sqrt(
        2.0 * jnp.log(jnp.maximum(jnp.float32(n_actions), 2.0)),
    )
    is_active = excess >= sigma_floor_factor * floor

    v_active = double_greedify.fn(
        online_params=online_params, target_params=target_params,
        q_network=q_network, next_obs=next_obs,
    )
    v_dormant = max_greedify.fn(
        online_params=online_params, target_params=target_params,
        q_network=q_network, next_obs=next_obs,
    )
    return jax.lax.select(is_active, v_active, v_dormant)


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
    greedification: Greedification = max_greedify,
    gradient_rule: GradientRule = semi_gradient,
) -> jax.Array:
    """Bellman target: `reward + gamma · (1−done) · gradient_rule(
    greedification(...))`.

    `gamma` here is the BOOTSTRAP DISCOUNT on v(s'), which equals
    γⁿ for an n-step return and γ¹ = γ for the standard 1-step
    case. dqn_step computes `gamma**n_step` once and passes that
    as `gamma`, so bootstrap doesn't need to know `n_step`
    separately — single leaf in the configuration surface, no
    duplication.

    `reward` is the (potentially-aggregated) n-step return
    `Σⱼ γʲ rⱼ` precomputed by the `n_step_return` Free Claim
    during rollout. For 1-step this is just rₜ and the formula
    collapses to the textbook `rₜ + γ·(1−d)·v(s_{t+1})`.

    `greedification` is the DDQN-vs-vanilla axis; `gradient_rule`
    is the semi-gradient-vs-full-gradient axis. Authors swap
    either independently via partial:

        partial(bootstrap, greedification=double_greedify)
        partial(bootstrap, gradient_rule=full_gradient)
        partial(bootstrap, greedification=double_greedify,
                gradient_rule=full_gradient)"""
    v_next = greedification(
        online_params=online_params,
        target_params=target_params,
        q_network=q_network,
        next_obs=next_obs,
    )
    target = reward + gamma * (1.0 - done) * v_next
    return gradient_rule(target)


