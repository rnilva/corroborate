"""Tests for the env catalogue — `EnvSpec` registry, `bucket_hash`
factory, and lookup helpers.

Verifies all gymnax envs in v9's table register successfully
under corroborate's EnvSpec shape, with introspected fields
auto-populated and corroborate-specific extensions
(state_hash + cardinality) wired correctly per benchmark
family."""
from __future__ import annotations

import jax.numpy as jnp

from corroborate_rl.env_catalogue import (
    ENV_REGISTRY,
    EnvSpec,
    all_envs,
    bucket_hash,
    envs_in_family,
    get,
    introspect_env,
)


# ============ Registration ============

def test_all_v9_envs_registered() -> None:
    """The catalogue ships v9's 17-env table verbatim."""
    expected = {
        'CartPole-v1', 'Acrobot-v1', 'MountainCar-v0',
        'Catch-bsuite', 'DeepSea-bsuite', 'MemoryChain-bsuite',
        'UmbrellaChain-bsuite', 'DiscountingChain-bsuite',
        'MNISTBandit-bsuite',
        'Asterix-MinAtar', 'Breakout-MinAtar', 'Freeway-MinAtar',
        'SpaceInvaders-MinAtar',
        'FourRooms-misc', 'MetaMaze-misc', 'Pong-misc',
        'BernoulliBandit-misc', 'GaussianBandit-misc',
    }
    assert set(ENV_REGISTRY.keys()) == expected


def test_classic_control_envs_have_state_hash() -> None:
    """Vector-obs classic-control envs declare a bucket-hash
    discretization for the (s, a)-coverage gap."""
    for name in ('CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'):
        spec = get(name)
        assert spec.state_hash is not None, name
        assert spec.state_hash_cardinality is not None, name
        assert spec.state_hash_cardinality > 0, name


def test_minatar_envs_have_no_state_hash() -> None:
    """Image-obs minatar envs ship `state_hash=None` —
    bucket cardinality is astronomical, KL invariant signal-free."""
    for name in (
        'Asterix-MinAtar', 'Breakout-MinAtar',
        'Freeway-MinAtar', 'SpaceInvaders-MinAtar',
    ):
        spec = get(name)
        assert spec.state_hash is None, name
        assert spec.state_hash_cardinality is None, name


# ============ Introspection ============

def test_introspect_returns_correct_shape_for_cartpole() -> None:
    """CartPole has 4-dim obs, 2 discrete actions, 500-step
    horizon."""
    info = introspect_env('CartPole-v1')
    assert info['action_type'] == 'discrete'
    assert info['action_dim'] == 2
    assert info['observation_shape'] == (4,)
    assert info['observation_type'] == 'vector'
    assert info['horizon'] == 500


def test_introspect_classifies_minatar_as_image() -> None:
    info = introspect_env('Breakout-MinAtar')
    assert info['observation_type'] == 'image'
    shape = info['observation_shape']
    assert isinstance(shape, tuple)
    assert len(shape) == 3


# ============ EnvSpec fields ============

def test_envspec_obs_dim_is_total_flattened_size() -> None:
    cartpole = get('CartPole-v1')
    assert cartpole.obs_dim == 4

    breakout = get('Breakout-MinAtar')
    # 10 × 10 × 4 = 400
    assert breakout.obs_dim == 400


def test_envspec_eval_episode_cap_reads_horizon() -> None:
    cartpole = get('CartPole-v1')
    assert cartpole.eval_episode_cap == 500


def test_envspec_metadata_fields_populated() -> None:
    cartpole = get('CartPole-v1')
    assert cartpole.r_min == 0.0
    assert cartpole.r_max == 1.0
    assert cartpole.reward_regime == 'per_step'
    assert cartpole.benchmark_family == 'classic_control'

    deepsea = get('DeepSea-bsuite')
    assert deepsea.r_min == -0.01
    assert deepsea.reward_regime == 'event_triggered'
    assert deepsea.benchmark_family == 'bsuite'


# ============ bucket_hash factory ============

def test_bucket_hash_returns_int_in_range() -> None:
    state_hash, cardinality = bucket_hash(
        lows=jnp.array([0.0, 0.0]),
        highs=jnp.array([10.0, 10.0]),
        n_buckets_per_dim=5,
    )
    # 5^2 = 25
    assert cardinality == 25

    h = state_hash(jnp.array([5.0, 5.0]))
    assert 0 <= int(h) < cardinality


def test_bucket_hash_distinct_obs_distinct_buckets() -> None:
    """Two obs in different bucket cells produce different hashes."""
    state_hash, _ = bucket_hash(
        lows=jnp.array([0.0, 0.0]),
        highs=jnp.array([10.0, 10.0]),
        n_buckets_per_dim=5,
    )
    h_a = int(state_hash(jnp.array([1.0, 1.0])))   # bucket (0, 0)
    h_b = int(state_hash(jnp.array([8.0, 8.0])))   # bucket (4, 4)
    assert h_a != h_b


def test_bucket_hash_clips_out_of_bounds() -> None:
    """Out-of-bounds obs values clip into boundary buckets, not
    out of range."""
    state_hash, cardinality = bucket_hash(
        lows=jnp.array([0.0, 0.0]),
        highs=jnp.array([10.0, 10.0]),
        n_buckets_per_dim=5,
    )
    # Both above the high bound — should still return a valid
    # in-range bucket id.
    h = int(state_hash(jnp.array([100.0, 100.0])))
    assert 0 <= h < cardinality


def test_cartpole_state_hash_callable_on_real_obs() -> None:
    spec = get('CartPole-v1')
    assert spec.state_hash is not None
    obs = jnp.array([0.1, 0.0, 0.05, 0.0])
    h = int(spec.state_hash(obs))
    assert 0 <= h < (spec.state_hash_cardinality or 0)


# ============ Lookup helpers ============

def test_get_raises_keyerror_with_helpful_message() -> None:
    try:
        get('NonexistentEnv-v0')
        raise AssertionError('expected KeyError')
    except KeyError as e:
        assert 'NonexistentEnv-v0' in str(e)
        # Message lists known envs
        assert 'CartPole-v1' in str(e)


def test_envs_in_family_filters_correctly() -> None:
    classic = envs_in_family('classic_control')
    names = {e.name for e in classic}
    assert names == {'CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'}

    minatar = envs_in_family('minatar')
    assert len(minatar) == 4
    assert all(e.benchmark_family == 'minatar' for e in minatar)


def test_all_envs_returns_complete_set() -> None:
    assert len(all_envs()) == len(ENV_REGISTRY)
    assert all(isinstance(e, EnvSpec) for e in all_envs())
