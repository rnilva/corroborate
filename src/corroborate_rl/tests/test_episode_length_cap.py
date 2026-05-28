"""Truncation flag on `EpisodeLengthCappedEnv`.

Verifies the Sutton-Barto §6.6 / Gymnasium-API distinction the
wrapper publishes via `info['truncated']`:

- Pre-cap steps: `done=False`, `info['truncated']=False`.
- At-cap step (no natural termination yet): `done=True`,
  `info['truncated']=True`.
- Natural termination before the cap: `done=True`,
  `info['truncated']=False`.

The substrate's `bootstrap` claim consumes this signal — at
truncation the trajectory continues bootstrap (target = r + γ·v(s')),
at termination it zeros (target = r). Without the wrapper-published
flag, capping at e.g. 200 on a 1000-step env would change the
learned MDP to "the game ends at step 200" rather than "the
experiment chose to stop observing at step 200."
"""
from __future__ import annotations

import gymnax
import jax
import jax.numpy as jnp

from corroborate_rl.env_catalogue import EpisodeLengthCappedEnv


def test_cap_emits_truncated_flag_at_cap() -> None:
    """Asterix-MinAtar capped at 10 steps. The inner env doesn't
    naturally terminate quickly (death depends on enemy positions
    with random seeds; the 10-step cap reliably fires first). Step
    10 must report `done=True, truncated=True`; steps 1..9 must
    report `done=False, truncated=False`."""
    inner, params = gymnax.make('Asterix-MinAtar')
    wrapped = EpisodeLengthCappedEnv(inner=inner, max_steps=10)
    rng = jax.random.PRNGKey(42)
    reset_key, run_key = jax.random.split(rng)
    obs, state = wrapped.reset(reset_key, params)

    keys = jax.random.split(run_key, 10)
    saw_pre_cap_truncated = False
    saw_pre_cap_done = False
    final_step_info: dict[str, object] | None = None
    final_done: jax.Array | None = None
    for i, step_key in enumerate(keys):
        # NOOP action (index 0) — minimal interaction with env;
        # MinAtar Asterix doesn't terminate solely from NOOP within
        # 10 steps under most seeds. Tested with seed 42.
        action = jnp.int32(0)
        obs, state, _reward, done, info = wrapped.step(
            step_key, state, action, params,
        )
        del obs
        trunc_obj = info.get('truncated')
        trunc_val = (
            float(trunc_obj) if isinstance(trunc_obj, jax.Array) else 0.0
        )
        done_val = float(done)
        if i < 9:
            if trunc_val > 0.5:
                saw_pre_cap_truncated = True
            if done_val > 0.5:
                saw_pre_cap_done = True
        else:
            final_step_info = info
            final_done = done

    # Pre-cap steps must NOT trigger truncation (sanity: the cap is
    # at step 10).
    assert not saw_pre_cap_truncated, (
        'pre-cap step reported truncated=True; wrapper fires the '
        'flag too eagerly (should fire only at step >= max_steps)'
    )
    # The agent shouldn't be naturally dying within 10 NOOPs at
    # this seed — if this asserts it just means we picked a bad
    # seed and the env terminated before the cap. Re-pick seed.
    assert not saw_pre_cap_done, (
        'inner env terminated within the 10-step cap at seed 42; '
        'pick a different seed for this test'
    )
    assert final_step_info is not None and final_done is not None
    # At the cap step, both done and truncated must fire.
    assert float(final_done) > 0.5, (
        f'cap step did not set done=True; got done={float(final_done)}'
    )
    final_trunc = final_step_info.get('truncated')
    assert isinstance(final_trunc, jax.Array), (
        f'cap step info missing truncated jax.Array: '
        f'got {type(final_trunc)}'
    )
    assert float(final_trunc) > 0.5, (
        f'cap step did not set truncated=True; got '
        f'truncated={float(final_trunc)} despite done=True'
    )


def test_natural_termination_emits_truncated_zero() -> None:
    """Cartpole capped at a value larger than the env's natural
    horizon. When the env naturally terminates (pole falls /
    out-of-bounds), `done=True` AND `truncated=False`. The cap is
    set high enough that it can't fire on this episode (CartPole
    natural fall typically inside ~30 steps from random NOOP)."""
    inner, params = gymnax.make('CartPole-v1')
    # Cap at 500 (default CartPole horizon). Natural termination
    # from poor actions occurs well before this — confirms the
    # `truncated=False` branch.
    wrapped = EpisodeLengthCappedEnv(inner=inner, max_steps=500)
    rng = jax.random.PRNGKey(0)
    reset_key, run_key = jax.random.split(rng)
    obs, state = wrapped.reset(reset_key, params)

    saw_natural_termination_with_trunc_zero = False
    # Keep stepping until we see a done.
    keys = jax.random.split(run_key, 500)
    for i, step_key in enumerate(keys):
        # Always push right (action 1) — biases the pole to fall
        # outside the safe region within ~30 steps deterministically.
        action = jnp.int32(1)
        obs, state, _reward, done, info = wrapped.step(
            step_key, state, action, params,
        )
        del obs
        if float(done) > 0.5:
            trunc_obj = info.get('truncated')
            assert isinstance(trunc_obj, jax.Array), (
                f'natural-done step info missing truncated jax.Array: '
                f'got {type(trunc_obj)}'
            )
            # The cap is 500; natural termination happens well
            # before that → cap_reached==False → truncated==False.
            assert float(trunc_obj) < 0.5, (
                f'natural termination at step {i + 1} (well below cap '
                f'500) erroneously marked truncated=True'
            )
            saw_natural_termination_with_trunc_zero = True
            break

    assert saw_natural_termination_with_trunc_zero, (
        'CartPole with action=push-right never terminated within '
        '500 steps; test setup wrong (pole should fall fast)'
    )


def test_no_truncated_flag_on_unwrapped_env() -> None:
    """Sanity: vanilla (un-capped) env emits no `truncated` key.
    The rollout-phase default path then synthesizes `truncated=0`,
    so the substrate behaves identically to its pre-refactor
    semantics."""
    inner, params = gymnax.make('CartPole-v1')
    rng = jax.random.PRNGKey(0)
    obs, state = inner.reset(rng, params)
    del obs
    _, _, _, _, info = inner.step(rng, state, jnp.int32(0), params)
    # Inner env publishes nothing at the truncated key.
    assert 'truncated' not in info, (
        f'vanilla CartPole now emits info["truncated"]; the '
        f'truncation-flag plumbing has accidentally leaked into '
        f'unwrapped envs. info keys: {list(info)}'
    )
