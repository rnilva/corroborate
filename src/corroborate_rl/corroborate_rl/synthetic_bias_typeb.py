"""Synthetic bias Type-A/B controlled-substrate env (v2).

v1 → v2 redesign rationale (see `/tmp/synthetic_env_roast.md`
for the brutal review of v1). v1 was "a bandit in a tuxedo":
action-independent transitions (`s' = (s+1) mod L`) made
`γ·max_b Q*(s', b)` action-independent, so the max-of-K bias
contributed a constant to every action's Q-value that cancelled
out of `argmax`. No chain-amplified bias preserved or destroyed.
v1 also conflated `reward_variance_scale` with |Q|, Δ_v, AND
Var_a[Q*] in lockstep, pinned γ=0.99, and had no FA-capacity
axis. The five concrete redesign axes (verbatim from the roast):

1. Action-dependent transitions — `s' = (s + a) mod L`. Action
   selects WHICH successor state to land in, so `max_b Q*(s', b)`
   is action-discriminating and chain-amplified bias can exist.
2. Decouple Var_a[Q*] from Δ_v. Best-action immediate reward
   pinned at `mu_best`; the (K-1) non-best actions have tied
   mean 0. The TYPE-A/B AXIS is `anisotropy_alpha`: per-action
   reward NOISE σ_a varies independently of action MEAN.
   Var_a[Q*] grows with σ_a's heterogeneity; Δ_v stays pinned
   at mu_best.
3. L-axis: `n_states ∈ {8, 64, 256}` registered as separate
   envs; the 32-unit MLP becomes capacity-bound at large L
   (true Q* has K × n_states entries; a 2-layer 32-unit MLP
   on one-hot input has fewer effective parameters than the
   K × 256 = 1024 distinct Q values it must represent).
4. γ sweep ∈ {0.95, 0.99, 0.999} via the intervention YAML's
   `base: {gamma: ...}` mechanism.
5. Knife-edge regime: `mu_best = 0.05`, `sigma_base = 0.5` ⇒
   σ/Δ ≈ 10 (10× the natural-env Asterix 1% but order-of-
   magnitude closer than v1's 30-50% / 2× regime). The
   anisotropy_alpha knob can push σ_best/Δ into the 1-3% range
   (when anisotropy_alpha=0, σ_best = sigma_base; when
   anisotropy_alpha < 0, σ_best is suppressed so the best
   action stays "clean" — Type-A).

## MDP definition (v2)

- States: `s ∈ {0, ..., L-1}`. Initial state sampled uniformly
  at reset (no fixed start so all states are visited even at
  short horizons).
- Actions: `K=4` discrete.
- Transition: `s' = (s + a + 1) mod L`. Each action goes to a
  DIFFERENT next state. (The +1 ensures action=0 doesn't
  self-loop, which would otherwise create a trivial maximizing
  policy.)
- Reward at step t given state s, action a:
    - The state-conditional "best action" rotates by state:
      `a_best(s) = s mod K`. Calling action a_best(s) yields
      mean = `mu_best`; all other actions yield mean 0.
    - Per-action reward noise σ_a(s, a) follows the anisotropy
      profile:
        - For the best action: σ_best = sigma_base × exp(α)
        - For all other actions: σ_other = sigma_base × exp(-α/(K-1))
      where α = `anisotropy_alpha`. This holds total cross-action
      noise variance ≈ constant in α (geometric-mean preserved)
      while shifting the noise mass between the best and
      non-best actions.
    - Reward = mean_a + Normal(0, σ_a). NO sparsity gate (drops
      a confounded knob from v1).
- Termination: after `max_steps_in_episode` steps.

## Type-A vs Type-B regime characterisation

The state-rotating best-action structure makes the optimal
policy a TABLE indexed by state: a*(s) = s mod K. A linear FA
(MLP) can represent this perfectly given enough capacity, but
becomes capacity-bound at large L (specifically, when L > 4 ×
hidden_width). FA-bound vanilla DQN learns approximate Q-values
where per-state argmax errors are driven by the SD of the
function approximator's residual; the σ_best vs σ_other
anisotropy modulates how policy-informative that residual SD
is.

**Type-A** (anisotropy_alpha < 0): the best action has LOW noise
(σ_best < sigma_base); non-best actions have HIGHER noise.
Vanilla's max-of-K bias picks up noise from non-best actions
inflating their Q-estimates; DDQN's clip denoises uniformly →
DDQN helps.

**Type-B** (anisotropy_alpha > 0): the best action has HIGH
noise (σ_best > sigma_base); non-best actions are quiet. The
high noise on the best action is what makes it argmax-able
(its Q-estimate fluctuates above the tied non-best plateau
under vanilla's optimistic-max bias). DDQN's symmetric clip
removes the inflation that was carrying the signal → DDQN
harms.

**Type-A REGIME REPRODUCIBILITY**: high confidence. The "noise
helps non-best actions cross above the best" story is the
classical Hasselt 2010 max-bias result and reproduces in any
finite-sample max-of-K estimator.

**Type-B REGIME REPRODUCIBILITY**: open question. The natural-
env Asterix story has vanilla's σ_action correlated with the
true Q-value structure (high-Q states/actions have high SD as
a feature of the FA's local approximation quality). Whether
synthetic env construction with hand-injected per-action noise
asymmetry reproduces this is the substantive empirical
question this sweep tests. If only Type-A appears, that's a
finding: the natural-env Asterix pattern doesn't reproduce in
clean controllable substrate.

## Q* closed form

With deterministic transition `s' = (s + a + 1) mod L`, the
optimal value satisfies Q*(s, a) = mu(s, a) + γ V*((s+a+1) mod L)
where V*(s) = max_a Q*(s, a). Across the state cycle the
optimal policy is "always take a_best(current_state)", giving
a per-state reward of `mu_best`; the geometric-series ceiling is
V* ≤ mu_best / (1 - γ). At γ=0.999, that's ~50× mu_best, so
mu_best=0.05 lands Q* in the 0-50 magnitude regime where
finite-sample effects on argmax are meaningful.

Var_a[Q*(s, ·)] over the K actions has two structurally
distinct contributions:
1. The mean term: best action gets `mu_best`, others get 0
   → Var_a[mean] = (K-1)/K² × mu_best².
2. The γ-chain term: action a leads to state (s+a+1) mod L
   with V*-value that varies across the state cycle.

(1) is pinned at mu_best (no rvs confound). (2) is what
`anisotropy_alpha` controls *via the empirical residual*, NOT
in the true Q* (the true Q* is action-noise-independent because
expected reward is action-noise-independent). The substrate
test is: does FA-bound vanilla's argmax extract this true Q*
correctly, or does the empirical (σ_best, σ_other) anisotropy
warp it?

## Per-cell parameters

`BiasTypeBParams` carries:
- `n_states`, `n_actions` (READ-ONLY metadata; structural).
- `mu_best` (best-action immediate-reward mean, pinned per env
  for the sweep).
- `sigma_base` (the baseline noise SD; `sigma_base × exp(±α/...)`
  is the per-action SD).
- `anisotropy_alpha` (Type-A/B axis; 0 = isotropic, < 0 = best
  action quiet, > 0 = best action loud).
- `max_steps_in_episode`.

The env factory `make_synthetic_bias_typeb` exposes all of
these as keyword args; per-name registrations in
`env_catalogue._register_synthetic_bias_typeb_panel` bake the
structural axes (n_states, anisotropy_alpha) into the env name.

API matches the gymnax `Env` Protocol structurally:
`reset(rng, params) → (obs, state)`, `step(rng, state, action,
params) → (obs, state, reward, done, info)`. Per the substrate
convention, the env is config-free (no class fields) and all
per-cell configuration flows through `BiasTypeBParams`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

if TYPE_CHECKING:
    from gymnax import Box, Discrete


@struct.dataclass
class BiasTypeBParams:
    """Per-cell parameters for the synthetic bias Type-A/B env v2.

    Structural axes (READ-ONLY metadata baked at registration):
    - `n_states`: chain length L; the FA-capacity axis.
    - `n_actions`: action count K.
    - `max_steps_in_episode`: episode horizon.

    Sweep axes:
    - `mu_best`: immediate-reward mean for the state-conditional
      best action. Pinned per env (typically 0.05); chosen with
      `sigma_base` to land σ/Δ in the knife-edge regime.
    - `sigma_base`: baseline per-action reward noise SD; the
      anisotropy_alpha distorts this asymmetrically across actions.
    - `anisotropy_alpha`: Type-A/B axis. 0 = isotropic noise;
      negative = best action quiet (Type-A predicted); positive
      = best action loud (Type-B predicted). Typically swept
      across {-0.5, -0.25, 0, +0.25, +0.5}.
    """
    n_states: int = struct.field(pytree_node=False, default=16)
    n_actions: int = struct.field(pytree_node=False, default=4)
    mu_best: float = 0.05
    sigma_base: float = 0.5
    anisotropy_alpha: float = 0.0
    max_steps_in_episode: int = struct.field(
        pytree_node=False, default=128,
    )


@struct.dataclass
class BiasTypeBState:
    """Per-step env state. Step counter + chain position."""
    step: jax.Array  # int32 scalar
    state: jax.Array  # int32 scalar in [0, n_states)


@dataclass(frozen=True, slots=True)
class BiasTypeBEnv:
    """Synthetic K-action chain MDP for bias Type-A/B causal
    testing (v2).

    Construction is config-free; per-cell config flows through
    `BiasTypeBParams`. Mirrors `LunarLanderEnv`'s class shape.

    Action-dependent transitions: `s' = (s + a + 1) mod n_states`,
    so each action leads to a distinct successor state — the
    pre-requisite for chain-amplified bias to be policy-informative.
    """

    def reset(
        self, rng: jax.Array, params: BiasTypeBParams,
    ) -> tuple[jax.Array, BiasTypeBState]:
        # Sample initial state uniformly so all states are visited
        # equally at short horizons; vital for the FA-bound
        # regime where some states might otherwise be under-sampled.
        s0 = jax.random.randint(
            rng, shape=(), minval=0, maxval=params.n_states,
        )
        state = BiasTypeBState(
            step=jnp.int32(0), state=s0.astype(jnp.int32),
        )
        return self._obs(state.state, params.n_states), state

    def step(
        self,
        rng: jax.Array,
        state: BiasTypeBState,
        action: jax.Array,
        params: BiasTypeBParams,
    ) -> tuple[
        jax.Array, BiasTypeBState, jax.Array, jax.Array,
        dict[str, jax.Array],
    ]:
        # Best action at this state rotates with state index: the
        # state-conditional optimal action is (s mod K). Makes
        # Q*(s, ·) action-discriminating in a state-dependent way.
        a_best = state.state % params.n_actions
        is_best = (action.astype(jnp.int32) == a_best.astype(jnp.int32))

        # Action-mean: pinned mu_best for the best action, 0 for
        # others. Var_a[mean] = (K-1)/K² × mu_best², INDEPENDENT
        # of sigma_base / anisotropy_alpha — the decoupling axis.
        mean_a = jnp.where(
            is_best,
            jnp.float32(params.mu_best),
            jnp.float32(0.0),
        )

        # Per-action noise SD under anisotropy profile:
        # - Best action: sigma_base × exp(+α)
        # - Other K-1 actions: sigma_base × exp(-α/(K-1))
        # The exponents are chosen so that the GEOMETRIC mean of
        # per-action SDs equals sigma_base regardless of α (closed-
        # form: 1×exp(α) + (K-1)×exp(-α/(K-1)) → log-geometric-mean
        # = (α + (K-1)·(-α/(K-1)))/K = 0). This keeps "total noise
        # budget" approximately fixed while shifting where the
        # noise is concentrated.
        k_minus_1 = jnp.maximum(params.n_actions - 1, 1).astype(
            jnp.float32,
        )
        alpha = jnp.float32(params.anisotropy_alpha)
        sigma_best = params.sigma_base * jnp.exp(alpha)
        sigma_other = params.sigma_base * jnp.exp(-alpha / k_minus_1)
        sigma_a = jnp.where(is_best, sigma_best, sigma_other)

        # Gaussian reward noise.
        eps = jax.random.normal(rng) * sigma_a
        reward = mean_a + eps

        # Action-dependent transition: s' = (s + a + 1) mod L.
        # The "+ 1" prevents action=0 from self-looping (which would
        # otherwise create a degenerate optimal policy where action=0
        # always wins by virtue of immediate-vs-discounted reward).
        new_state_idx = (
            state.state + action.astype(jnp.int32) + jnp.int32(1)
        ) % params.n_states
        new_step = state.step + 1
        done = new_step >= params.max_steps_in_episode

        new_state = BiasTypeBState(
            step=new_step, state=new_state_idx,
        )
        obs = self._obs(new_state_idx, params.n_states)
        return obs, new_state, reward.astype(jnp.float32), done, {}

    def _obs(
        self, state_idx: jax.Array, n_states: int,
    ) -> jax.Array:
        """One-hot encoding of chain position. The FA-capacity
        axis (n_states ∈ {8, 64, 256}) exercises the MLP's
        representational capacity through the input dimension."""
        return jax.nn.one_hot(
            state_idx, num_classes=n_states, dtype=jnp.float32,
        )

    def action_space(self, params: BiasTypeBParams) -> 'Discrete':
        return spaces.Discrete(params.n_actions)

    def observation_space(self, params: BiasTypeBParams) -> 'Box':
        return spaces.Box(
            low=0.0, high=1.0,
            shape=(params.n_states,),
            dtype=jnp.float32,
        )


def make_synthetic_bias_typeb(
    *,
    n_states: int = 16,
    n_actions: int = 4,
    mu_best: float = 0.05,
    sigma_base: float = 0.5,
    anisotropy_alpha: float = 0.0,
    max_steps_in_episode: int = 128,
) -> tuple[BiasTypeBEnv, BiasTypeBParams]:
    """Factory matching the lunar_lander pattern. Builds an env
    instance + its default params; substrate's cell_runner consumes
    the pair via the gymnax-style API.

    For the parametric Type-A/B sweep, register one factory closure
    per named config (see `env_catalogue.py`)."""
    env = BiasTypeBEnv()
    params = BiasTypeBParams(
        n_states=int(n_states),
        n_actions=int(n_actions),
        mu_best=float(mu_best),
        sigma_base=float(sigma_base),
        anisotropy_alpha=float(anisotropy_alpha),
        max_steps_in_episode=int(max_steps_in_episode),
    )
    return env, params
