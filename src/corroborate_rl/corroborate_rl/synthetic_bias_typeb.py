"""Synthetic bias Type-A/B controlled-substrate env (v3.1).

## Evolution: v1 → v2 → v3 → v3.1 (all prior versions scrapped pre-launch)

**v1** (2026-05-19 morning, scrapped pre-sweep): a "bandit in a
tuxedo" — `s' = (s+1) mod L` made transitions action-independent;
`γ·max_b Q*(s', b)` was a constant added to every action's Q-value
that cancelled out of `argmax`. No chain-amplified policy-
informative bias was possible. Also: `reward_variance_scale` knob
confounded |Q*|, Δ_v, AND Var_a[Q*] in lockstep; no FA-capacity
axis; γ pinned at 0.99. See `/tmp/synthetic_env_roast.md`.

**v2** (2026-05-19 afternoon, scrapped pre-sweep): added action-
dependent transitions (`s' = (s + a + 1) mod L`) and an
`anisotropy_alpha` knob. But v2's α knob modulated per-step
REWARD-SAMPLING NOISE (the per-action σ of the immediate reward),
NOT the Q-target-side `Var_a[V*(s')]` that Cor 3.2's σ_clip
actually concerns. v2 also had |Q*| ≈ 50 (μ_best=0.05 / (1-γ=0.999)) —
50× under the natural-env Asterix Q≈436 scale; σ/Δ ≈ 1000% (vs the
natural-env knife-edge of ~1%); L ∈ {16, 64} with hidden=[32, 32]
gave 12.8× over-parameterized FA at the L=64 corner; n_seeds=12
(under-powered); pre-registration walk-back paths pre-laundered
every observed-data shape as publishable.
See `/tmp/synthetic_v2_roast.md`.

**v3** (2026-05-19 evening, scrapped pre-sweep): introduced
state-baked deterministic payoffs `mu_state(s) = peak_value · β^(s
mod K)`. Value iteration on the v3 MDP revealed two STRUCTURAL
flaws (`/tmp/synthetic_v3_review.md`):

1. **`Var_a[V*(s'_a)] = 0` at every β.** Under the optimal
   policy, every reachable successor sits on the same V* orbit
   (the optimal trajectory cycles through intra-block-idx=0
   states). The β knob actually controls `Var_a[mu_state(s')]` —
   the IMMEDIATE-REWARD spread — not the Q-target-side variance
   the v3 docstring claimed. Same conceptual error as v2 (per-step
   reward variance ≠ Q-target-side anisotropy), just relocated
   from σ_a knob to mu_state knob.
2. **Q* has only 16 distinct values across L=1024 states.** Q*
   is exactly periodic in s with period K=4. The MLP needs to
   learn K × K = 16 distinct Q*-values; a 16-unit hidden layer
   trivially represents this. The L=1024 axis tested gradient
   sparsity / replay coverage, NOT FA capacity.

## v3.1 design: random per-state payoffs break the periodicity

The substantive fix: each state has an INDEPENDENT random payoff
drawn from a seeded distribution at env initialization. This
addresses both v3 flaws cleanly:

- **Var_a[V*(s'_a)] is genuinely non-zero** at every spread > 0.
  With random per-state payoffs, the K successors `(s+a+1) mod L`
  for `a ∈ 0..K-1` lead to states with DIFFERENT V*-values
  (verified by value iteration in
  `tests/test_synthetic_bias_typeb.py`).
- **Q* has L distinct values** (no modular collapse). At L=1024,
  Q* spans ~L=1024 unique values; the 16-unit hidden bottleneck
  must alias 1024 distinct V*-values into 16-dim hidden →
  genuine FA-capacity binding.

### Concrete construction

- States: `s ∈ {0, ..., L-1}`. One-hot observations.
- Actions: K=4 discrete.
- Transition (action-dependent, deterministic):
    `s' = (s + a + 1) mod L`. Same as v3.
- **State-payoff vector** (the v3.1 fix):
    `mu_state[s] = peak_value · (1 - payoff_spread + payoff_spread · U_s)`
    where `U_s ~ U(0, 1)` is drawn deterministically from a seeded
    `payoff_seed`. At `payoff_spread=0`, all states have payoff
    `peak_value` (degenerate; V* = peak_value/(1-γ) everywhere
    → Var_a[V*(s')] = 0). At `payoff_spread=1`, states span
    `[0, peak_value]` uniformly.
- Reward on transition to `s'`: `r = mu_state[s'] + N(0, noise_sigma)`.
- Episode terminates after `max_steps_in_episode` steps.

### The `payoff_spread` knob (the v3.1 anisotropy axis)

`payoff_spread ∈ [0, 1]` controls the per-state payoff range:

- **payoff_spread ≈ 0 (degenerate / "isotropic")**: All states
    have payoff ≈ peak_value. Var_a[V*(s'_a)] ≈ 0. DDQN's
    optimism-bias correction has nothing structural to act on
    → DDQN ≈ vanilla.
- **payoff_spread ≈ 0.5 ("moderate")**: States span
    `[0.5·peak, peak]`. Per-state V* spread is non-trivial; the
    K successors have meaningfully different V*-values. The
    argmax-margin at each state depends on the random payoff
    configuration of the K immediate successors.
- **payoff_spread ≈ 1 ("high anisotropy")**: States span
    `[0, peak]`. Maximum Var_a[V*(s'_a)] under this construction;
    knife-edge argmax-margins at many states; vanilla's max-of-K
    bias picks up large policy-informative bootstrap noise from
    successor V*; DDQN's clip introduces argmax-asymmetry that
    corrupts the knife-edge selection → DDQN HARMS.

`payoff_seed` allows cross-env averaging over random payoff
realisations (multiple envs at the same `payoff_spread` with
different `payoff_seed` see different mu_state vectors).

### Closed-form Q*

There is NO simple closed-form for Q* under random per-state
payoffs — Q* must be computed by value iteration on the (L × K)
matrix. The reference implementation in
`tests/test_synthetic_bias_typeb.py::compute_v_star` runs VI to
convergence and is used to verify the Var_a[V*(s'_a)] property at
each (L, payoff_spread).

V*_max upper bound: `peak_value / (1-γ)`. At `peak_value=1.0`,
γ=0.999, V*_max ≤ 1000 — matches natural-env Asterix Q≈436
scale (critic rec #1, preserved from v3).

### Why the FA capacity-binding regime works at L ≥ 1024

Under hidden=[16], the MLP must represent K · L Q-values through
a 16-unit bottleneck. Crucially, in v3.1 these K · L Q-values
are GENUINELY DISTINCT (no modular collapse): at L=1024, the
1024 V*-values are ~uniform-on-[0, peak/(1-γ)] (verified by VI),
so the MLP cannot compress them via the K=4 equivalence class
that v3 admitted. THIS is the FA-binding regime: 16-dim hidden
cannot encode 1024 distinct V*-values without aliasing,
introducing structured approximation error. vanilla's
`max_b Q(s',b)` picks up the worst FA-residual on each (s,a);
DDQN's clip via online-argmax reroutes the bootstrap through
a less-biased estimator. The L-axis differential isolates the
FA-binding regime contribution.

### Calibrated knife-edge regime (preserved from v3)

`noise_sigma = 0.02 * peak_value` (default). At `payoff_spread=1`,
the empirical distribution of per-state argmax-margins is
heavy-tailed (some states have wide margins, others knife-edge);
the σ/Δ regime is heterogeneous across states. This is closer
to the natural-env Asterix mechanism (FA-residual + env-baked
heterogeneity) than v3's uniform Δ_v = peak·(1-β) per env.

### Per-cell parameters

`BiasTypeBParams` carries:

- `n_states` (L): chain length, the FA-capacity axis (READ-ONLY
    metadata).
- `n_actions` (K): action count (READ-ONLY).
- `max_steps_in_episode`: episode horizon (READ-ONLY).
- `peak_value`: maximum possible per-step payoff. Pinned at 1.0
    across the v3.1 panel; sets the |Q| magnitude scale.
- `payoff_spread`: the v3.1 anisotropy knob (per-state payoff
    range / peak_value).
- `payoff_seed`: deterministic seed for the per-state payoff
    realisation.
- `noise_sigma`: per-step reward noise SD; pinned at 0.02 across
    the v3.1 panel (relative to peak_value=1.0).
- `mu_state`: the L-dimensional per-state payoff vector,
    precomputed deterministically from `payoff_seed` at env-
    factory time. Stored as a pytree leaf so `step` is fully
    jit-able without re-deriving the random payoffs each step.

API matches the gymnax `Env` Protocol: `reset(rng, params) →
(obs, state)`, `step(rng, state, action, params) → (obs, state,
reward, done, info)`. Per the substrate convention, the env is
config-free (no class fields).
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
    """Per-cell parameters for the v3.1 synthetic bias Type-A/B env.

    Structural axes (READ-ONLY metadata baked at registration):

    - `n_states` (L): chain length; the FA-capacity axis.
    - `n_actions` (K): action count.
    - `max_steps_in_episode`: episode horizon.

    Sweep axes:

    - `peak_value`: maximum per-step payoff. Pinned at 1.0 in the
      v3.1 panel so |Q| at γ=0.999 lands near 1000, matching
      natural-env Asterix/Acrobot.
    - `payoff_spread`: the v3.1 anisotropy knob. Per-state payoff
      ranges from `peak_value · (1 - payoff_spread)` (when U_s=0)
      to `peak_value` (when U_s=1). At `payoff_spread=0` all
      states have payoff `peak_value` (degenerate, Var_a[V*]=0);
      at `payoff_spread=1` states span [0, peak_value] uniformly
      (maximum Var_a[V*]).
    - `payoff_seed`: deterministic seed for the per-state payoff
      realisation. Different `payoff_seed` at the same
      `payoff_spread` give independent random payoff vectors;
      cross-env averaging over payoff realisations.
    - `noise_sigma`: per-step Gaussian reward noise SD. Pinned at
      0.02 across the v3.1 panel — 2% of peak_value, matching the
      natural-env Asterix knife-edge σ/Δ ≈ 1-3% regime.
    - `mu_state`: the L-vector of per-state payoffs, precomputed
      deterministically from `payoff_seed` at env-factory time.
    """
    n_states: int = struct.field(pytree_node=False, default=32)
    n_actions: int = struct.field(pytree_node=False, default=4)
    peak_value: float = 1.0
    payoff_spread: float = 0.0
    payoff_seed: int = struct.field(pytree_node=False, default=0)
    noise_sigma: float = 0.02
    max_steps_in_episode: int = struct.field(
        pytree_node=False, default=128,
    )
    mu_state: jax.Array = struct.field(
        default_factory=lambda: jnp.ones((32,), dtype=jnp.float32),
    )


@struct.dataclass
class BiasTypeBState:
    """Per-step env state. Step counter + chain position."""
    step: jax.Array  # int32 scalar
    state: jax.Array  # int32 scalar in [0, n_states)


@dataclass(frozen=True, slots=True)
class BiasTypeBEnv:
    """Synthetic K-action chain MDP with RANDOM per-state payoffs
    (v3.1) for bias Type-A/B causal testing.

    The state-payoff vector `mu_state[s]` is drawn deterministically
    from `payoff_seed` at env-factory time:
        `mu_state[s] = peak_value · (1 - payoff_spread + payoff_spread · U_s)`
    where `U_s ~ U(0, 1)`. This breaks v3's modular periodicity:
    each state has a distinct V*, so the K-action successors visit
    states with DIFFERENT V*-values → Var_a[V*(s'_a)] is genuinely
    non-zero (the v3 reviewer's structural critique).

    Action-dependent transition `s' = (s + a + 1) mod L` (preserved
    from v3) makes each action visit a distinct successor — the
    precondition for chain-amplified bias to be policy-informative.
    Config-free, params-carry-everything (including `mu_state`).
    """

    def reset(
        self, rng: jax.Array, params: BiasTypeBParams,
    ) -> tuple[jax.Array, BiasTypeBState]:
        # Sample initial state uniformly so all states are visited
        # equally at short horizons; vital for the FA-bound regime
        # where some states might otherwise be under-sampled.
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
        # Action-dependent transition: s' = (s + a + 1) mod L.
        # Same as v3 — each action visits a distinct successor.
        # The v3.1 fix is in the PAYOFF (mu_state[s']), not the
        # transition: with random per-state payoffs, the K
        # successors from any state s visit K states with K
        # distinct V*-values → Var_a[V*(s'_a)] > 0.
        new_state_idx = (
            state.state + action.astype(jnp.int32) + jnp.int32(1)
        ) % params.n_states

        # State-baked payoff: mu_state[s'] is looked up from the
        # precomputed L-vector (drawn from payoff_seed at factory
        # time). The cross-action successor-payoff spread is set
        # by the RANDOM realisation of mu_state, NOT by a closed-
        # form modular shape — addressing the v3 reviewer's
        # "Q* periodic with K=4" critique.
        mu_target = params.mu_state[new_state_idx]

        # Small Gaussian noise calibrated to natural-env knife-edge
        # SNR (2% of peak_value by default).
        eps = jax.random.normal(rng) * jnp.float32(params.noise_sigma)
        reward = mu_target + eps

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
        """One-hot encoding of chain position. With a small hidden
        bottleneck (hidden ≤ 16) the L=1024 setting forces the FA
        to alias 1024 distinct V*-values (one per state, no modular
        collapse) into a 16-dim hidden subspace → genuine
        FA-binding regime."""
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


def build_mu_state(
    *,
    n_states: int,
    peak_value: float,
    payoff_spread: float,
    payoff_seed: int,
) -> jax.Array:
    """Build the deterministic per-state payoff vector.

    `mu_state[s] = peak_value · (1 - payoff_spread + payoff_spread · U_s)`
    where `U_s ~ U(0, 1)` is drawn from `jax.random.uniform` seeded
    by `payoff_seed`. The construction guarantees:

    - At `payoff_spread = 0`: `mu_state[s] = peak_value` for all s
      (degenerate isotropic case; Var_a[V*] = 0).
    - At `payoff_spread = 1`: `mu_state[s] ∈ [0, peak_value]`
      uniformly distributed (maximum anisotropy).
    - Monotone in `payoff_spread`: per-state payoff variance scales
      as `(peak_value · payoff_spread)² / 12` (closed-form U(0,1)
      variance × scale²).

    The returned array is float32 with shape `(n_states,)`. Same
    `payoff_seed` + `n_states` always produces the same vector
    (deterministic; vital for reproducibility across processes
    and JAX RNG-platform variations).
    """
    rng = jax.random.PRNGKey(int(payoff_seed))
    u = jax.random.uniform(
        rng, shape=(int(n_states),),
        minval=0.0, maxval=1.0, dtype=jnp.float32,
    )
    return jnp.float32(peak_value) * (
        jnp.float32(1.0 - payoff_spread) + jnp.float32(payoff_spread) * u
    )


def make_synthetic_bias_typeb(
    *,
    n_states: int = 32,
    n_actions: int = 4,
    peak_value: float = 1.0,
    payoff_spread: float = 0.0,
    payoff_seed: int = 0,
    noise_sigma: float = 0.02,
    max_steps_in_episode: int = 128,
) -> tuple[BiasTypeBEnv, BiasTypeBParams]:
    """Factory matching the lunar_lander pattern. Builds an env
    instance + its default params; substrate's cell_runner consumes
    the pair via the gymnax-style API.

    For the parametric Type-A/B sweep, register one factory closure
    per named config (see `env_catalogue.py`). The `mu_state` vector
    is baked into `params` at factory time so all sweep cells with
    the same (n_states, payoff_spread, payoff_seed) see byte-
    identical payoff realisations."""
    env = BiasTypeBEnv()
    mu_state = build_mu_state(
        n_states=int(n_states),
        peak_value=float(peak_value),
        payoff_spread=float(payoff_spread),
        payoff_seed=int(payoff_seed),
    )
    params = BiasTypeBParams(
        n_states=int(n_states),
        n_actions=int(n_actions),
        peak_value=float(peak_value),
        payoff_spread=float(payoff_spread),
        payoff_seed=int(payoff_seed),
        noise_sigma=float(noise_sigma),
        max_steps_in_episode=int(max_steps_in_episode),
        mu_state=mu_state,
    )
    return env, params
