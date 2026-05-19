"""Synthetic bias Type-A/B controlled-substrate env (v3).

## Evolution: v1 → v2 → v3 (both prior versions scrapped)

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
actually concerns. Vanilla DDQN's bootstrap max is over `Q(s', b)`,
whose action-wise fluctuation comes from FA residual + replay
gradient noise atop the TRUE Q-target spread — NOT from immediate-
reward noise. v2 also had |Q*| ≈ 50 (μ_best=0.05 / (1-γ=0.999)) —
50× under the natural-env Asterix Q≈436 scale; σ/Δ ≈ 1000% (vs the
natural-env knife-edge of ~1%); L ∈ {16, 64} with hidden=[32, 32]
gave 12.8× over-parameterized FA at the L=64 corner (no FA-binding
regime); n_seeds=12 (under-powered); pre-registration walk-back
paths pre-laundered every observed-data shape as publishable.
See `/tmp/synthetic_v2_roast.md`.

## v3 design: anisotropy primitive on the Q-TARGET side

The substantive fix: the Type-A/B axis IS Var_a[V*(s')]. State
`s'_a = (s + a + 1) mod L` is the action-`a` successor of state
`s`. Each state has a DETERMINISTIC, ENV-BAKED scalar payoff
`mu_state(s)` collected on transition INTO that state. The
cross-action distribution `{mu_state(s'_a) : a ∈ 0..K-1}`
determines the TRUE per-state argmax-margin and Var_a[V*(s')]
DIRECTLY — no immediate-reward-noise conflation.

### Concrete construction

- States: `s ∈ {0, ..., L-1}`. One-hot observations.
- Actions: K=4 discrete.
- Transition (action-dependent, deterministic):
    `s' = (s + a + 1) mod L`. Each action visits a distinct
    successor; from any s the K actions visit a contiguous block
    of K successors `{(s+1) mod L, ..., (s+K) mod L}` in order.
- State-payoff vector: `mu_state(s) = peak_value * beta ** (s mod K)`.
    Each state-block of K consecutive states has payoffs
    `(peak_value, peak·β, peak·β², peak·β³)` cycling. From any
    state `s`, the K successors visit ALL K intra-block positions
    (modulo L wrap) → cross-action payoff spread is exactly the
    K-tuple `(peak, peak·β, peak·β², peak·β³)`.
- Reward on transition to `s'`: `r = mu_state(s') + N(0, noise_sigma)`.
    Small Gaussian noise calibrated to natural-env knife-edge SNR.
- Episode terminates after `max_steps_in_episode` steps.

### The β knob (Type-A/B axis)

`beta` ∈ [0, 1] controls the K-action payoff spread:

- **β ≈ 0 (Type-A "peaked")**: Only the best-position successor
    has nonzero payoff; the other K-1 have 0. Knife-edge
    argmax-margin Δ_v = peak_value (full payoff). Low
    σ_clip / Δ_v on the Q-target side because the value of
    non-optimal actions IS structurally zero. Vanilla's max-of-K
    bias INFLATES the Q-estimates of the K-1 tied non-best
    actions via FA noise + replay noise; DDQN's clip denoises
    them uniformly → DDQN HELPS.
- **β ≈ 0.5-0.9 (Type-B "graded")**: All K actions have
    nonzero successor-payoff; (1, β, β², β³) is monotone
    decreasing. Best-vs-second-best margin Δ_v = peak·(1-β).
    At β=0.9, Δ_v = 0.1·peak — knife-edge regime where the
    optimal argmax is fragile to small perturbations of Q.
    DDQN's symmetric clip on `Q_target(s', argmax_online_Q)`
    introduces argmax-asymmetry that corrupts the knife-edge
    selection → DDQN HARMS.

### Closed-form Q*

Optimal policy `a*(s)`: pick the action whose successor
intra-block index is 0 (highest payoff). For any state with
intra-block index `j = s mod K`, the action `a* = (K - j - 1)
mod K` lands at intra-block-idx 0. Under the optimal policy the
agent cycles indefinitely through intra-block-idx-0 states (or
its modular orbit when L is not a multiple of K), collecting
`peak_value` each step. Discounted return ceiling:

    V*(intra=0) = peak_value · (1 + γ + γ² + …) = peak_value / (1-γ)

At `peak_value = 1.0` and γ=0.999, V* ≈ 1000 — matches the
natural-env Asterix Q≈436 / Acrobot Q≈100 magnitude regime
(critic rec #1).

### Why the FA capacity-binding regime works at L ≥ 1024

Under hidden=[16] (single 16-unit hidden layer, critic rec #3),
the MLP must represent K · L distinct Q-values through a 16-unit
bottleneck. At L=32, K·L=128 Q-values through 16 hidden units is
representable (8 Q-values per hidden unit on average); at
L=1024, K·L=4096 Q-values through 16 hidden units is fundamentally
NOT representable without aliasing — the FA is forced to compress
the Q-table into a 16-dim subspace, introducing structured
approximation error. THIS is where the Hasselt max-of-K bias
becomes load-bearing: vanilla's `max_b Q(s',b)` picks up the
worst FA-residual on each (s,a); DDQN's clip via online-argmax
reroutes the bootstrap through a less-biased estimator. The
L-axis differential isolates the FA-binding regime contribution.

### Calibrated knife-edge regime (critic rec #2)

`noise_sigma = 0.02 * peak_value` (default). At β=0 (Type-A):
σ/Δ = 2% — true knife-edge. At β=0.5: σ/Δ = 4% (still knife-
edge). At β=0.9: σ/Δ = 20% (margin compressed below per-step
noise, FA-residual dominates argmax). The β axis SCANS the
σ/Δ regime cleanly — v2's σ/Δ ≈ 1000% issue is structurally
resolved.

### Per-cell parameters

`BiasTypeBParams` carries:

- `n_states` (L): chain length, the FA-capacity axis (READ-ONLY
    metadata).
- `n_actions` (K): action count (READ-ONLY).
- `max_steps_in_episode`: episode horizon (READ-ONLY).
- `peak_value`: highest payoff in the per-block shape. Pinned at
    1.0 across the v3 panel; sets the |Q| magnitude scale.
- `beta`: Type-A/B knob (per-block payoff shape geometric ratio).
- `noise_sigma`: per-step reward noise SD; pinned at 0.02 across
    the v3 panel (relative to peak_value=1.0).

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
    """Per-cell parameters for the v3 synthetic bias Type-A/B env.

    Structural axes (READ-ONLY metadata baked at registration):

    - `n_states` (L): chain length; the FA-capacity axis.
    - `n_actions` (K): action count.
    - `max_steps_in_episode`: episode horizon.

    Sweep axes:

    - `peak_value`: highest payoff in the per-block shape. Pinned
      at 1.0 in the v3 panel so |Q| at γ=0.999 lands near 1000,
      matching natural-env Asterix/Acrobot.
    - `beta`: Type-A/B knob. Per-block payoff shape is
      `(peak, peak·β, peak·β², peak·β³)`. β=0 → one-peaked
      Type-A; β=0.9 → graded knife-edge Type-B.
    - `noise_sigma`: per-step Gaussian reward noise SD. Pinned at
      0.02 across the panel — 2% of peak_value, matching the
      natural-env Asterix knife-edge σ/Δ ≈ 1-3% regime.
    """
    n_states: int = struct.field(pytree_node=False, default=32)
    n_actions: int = struct.field(pytree_node=False, default=4)
    peak_value: float = 1.0
    beta: float = 0.0
    noise_sigma: float = 0.02
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
    """Synthetic K-action chain MDP with state-baked anisotropic
    payoffs (v3) for bias Type-A/B causal testing.

    The state-payoff vector `mu_state(s) = peak_value · β^(s mod K)`
    encodes the Type-A/B axis directly on the Q-TARGET side:
    Var_a[V*(s'_a)] is determined by the cross-action spread of
    the K-tuple `(peak, peak·β, peak·β², peak·β³)`, NOT by per-
    step reward-sampling noise (v2's conceptual error).

    Action-dependent transition `s' = (s + a + 1) mod L` makes
    each action visit a distinct successor — the precondition for
    chain-amplified bias to be policy-informative. Mirrors
    `LunarLanderEnv`'s class shape; config-free, params-carry-
    everything.
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
        # Each action visits a distinct intra-block successor; from
        # state s with intra-block idx j = s mod K, action a lands
        # at intra-block idx (j + a + 1) mod K. The K actions visit
        # ALL K intra-block positions, so Var_a over the K successor
        # payoffs equals the variance of the per-block shape vector.
        new_state_idx = (
            state.state + action.astype(jnp.int32) + jnp.int32(1)
        ) % params.n_states

        # State-baked payoff: mu_state(s) = peak_value · β^(s mod K).
        # The Q-target-side anisotropy primitive. The per-block
        # K-tuple is (peak, peak·β, peak·β², peak·β³); cycling β=0
        # gives the one-peaked Type-A regime, β→1 gives the graded
        # Type-B regime where the best/second-best margin
        # peak·(1-β) shrinks below the per-step noise floor.
        intra_block = (
            new_state_idx % jnp.int32(params.n_actions)
        ).astype(jnp.float32)
        mu_target = jnp.float32(params.peak_value) * jnp.power(
            jnp.float32(params.beta), intra_block,
        )

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
        to alias K · L = 4096 distinct Q-values into a 16-dim
        hidden subspace → genuine FA-binding regime."""
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
    n_states: int = 32,
    n_actions: int = 4,
    peak_value: float = 1.0,
    beta: float = 0.0,
    noise_sigma: float = 0.02,
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
        peak_value=float(peak_value),
        beta=float(beta),
        noise_sigma=float(noise_sigma),
        max_steps_in_episode=int(max_steps_in_episode),
    )
    return env, params
