"""Auto-registry smoke: walk the DQN claims namespace, confirm
every Claim and Module class round-trips by name, and that the
resolved handles are the *same objects* the existing sweep scripts
import directly. The contract: YAML referring to `'double_greedify'`
must produce the identical `FnClaim` that
`from corroborate_rl.dqn.claims.bootstrap import double_greedify`
yields, otherwise the YAML path silently diverges from the
hand-authored Python path."""
from __future__ import annotations

from functools import partial

import pytest

from dataclasses import dataclass

from corroborate.core.claim import FnClaim
from corroborate.runner.registry import Registry


# Module set every DQN sweep script transitively binds.
DQN_CLAIM_MODULES = (
    'corroborate_rl.dqn.claims.bootstrap',
    'corroborate_rl.dqn.claims.action_select',
    'corroborate_rl.dqn.claims.replay',
    'corroborate_rl.dqn.claims.q_network',
    'corroborate_rl.dqn.claims.optimizer',
    'corroborate_rl.dqn.claims.target_sync',
    'corroborate_rl.dqn.claims.loss',
)


@pytest.fixture
def dqn_registry() -> Registry:
    reg = Registry()
    reg.add_modules(DQN_CLAIM_MODULES)
    return reg


def test_fn_discovery_covers_every_authored_claim(
    dqn_registry: Registry,
) -> None:
    """Every `@claim`-decorated free function in the DQN implementation
    is reachable by name."""
    expected_fns = {
        'max_greedify', 'double_greedify', 'expectile_greedify',
        'semi_gradient', 'full_gradient', 'bootstrap',
        'epsilon_greedy', 'linear_epsilon',
        'uniform_sample', 'n_step_return',
        'periodic_copy', 'squared_error',
    }
    missing = expected_fns - set(dqn_registry.fns)
    assert not missing, f'missing FnClaim registrations: {missing}'


def test_class_discovery_covers_every_config_bundle(
    dqn_registry: Registry,
) -> None:
    """Every config bundle (`MLP`, `CNN`, `Replay`) is reachable
    by name through the `classes` map. Optimizers and action-
    select are now `@claim`-functions (in `fns`); see
    `test_fn_discovery_covers_optimizers` and
    `test_fn_discovery_covers_every_authored_claim`."""
    expected_classes = {
        'MLP', 'CNN',
        'Replay',
    }
    missing = expected_classes - set(dqn_registry.classes)
    assert not missing, (
        f'missing class registrations: {missing}'
    )


def test_fn_discovery_covers_optimizers(
    dqn_registry: Registry,
) -> None:
    """Optimizer factories (`adam`, `rmsprop`, `sgd`,
    `warmed_update`) are `@claim`-decorated functions registered
    in the `fns` map. Authors compose via
    `partial(adam, lr=...)`."""
    expected_fns = {'adam', 'rmsprop', 'sgd', 'warmed_update'}
    assert expected_fns <= set(dqn_registry.fns), (
        f'missing fn registrations: {expected_fns - set(dqn_registry.fns)}'
    )


def test_resolved_fn_is_identical_to_imported_handle(
    dqn_registry: Registry,
) -> None:
    """The whole point: YAML path matches Python-import path. The
    same memoization in `@claim` (`_FN_CACHE`) makes
    `claim(f) is claim(f)`; the registry must store *those* same
    instances, not fresh wrappers."""
    from corroborate_rl.dqn.claims.bootstrap import (
        bootstrap, double_greedify, expectile_greedify,
        max_greedify,
    )
    assert dqn_registry.fn('bootstrap') is bootstrap
    assert dqn_registry.fn('double_greedify') is double_greedify
    assert dqn_registry.fn('expectile_greedify') is expectile_greedify
    assert dqn_registry.fn('max_greedify') is max_greedify


def test_resolved_class_is_identical_to_imported_class(
    dqn_registry: Registry,
) -> None:
    from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
    from corroborate_rl.dqn.claims.q_network import MLP
    from corroborate_rl.dqn.claims.replay import Replay
    assert dqn_registry.fn('adam') is adam
    assert dqn_registry.fn('warmed_update') is warmed_update
    assert dqn_registry.cls('MLP') is MLP
    assert dqn_registry.cls('Replay') is Replay


def test_round_trip_reconstructs_expectile_intervention(
    dqn_registry: Registry,
) -> None:
    """End-to-end: YAML-style name resolution reconstructs the
    exact `intervention['bootstrap']` partial that the canonical
    expectile recipe builds via direct imports. Identity (not
    just equality) of the inner FnClaims is the round-trip
    guarantee."""
    bootstrap = dqn_registry.fn('bootstrap')
    expectile = dqn_registry.fn('expectile_greedify')

    boot_yaml = partial(
        bootstrap, greedification=partial(expectile, tau=0.7),
    )

    from corroborate_rl.dqn.claims.bootstrap import (
        bootstrap as bootstrap_direct,
        expectile_greedify as expectile_direct,
    )
    boot_direct = partial(
        bootstrap_direct,
        greedification=partial(expectile_direct, tau=0.7),
    )

    # Both partials wrap the same underlying FnClaim and carry
    # the same nested partial (a partial of the same FnClaim with
    # the same kwargs).
    assert boot_yaml.func is boot_direct.func
    assert boot_yaml.keywords['greedification'].func is (
        boot_direct.keywords['greedification'].func
    )
    assert boot_yaml.keywords['greedification'].keywords == (
        boot_direct.keywords['greedification'].keywords
    )


def test_unknown_name_raises_with_known_set(
    dqn_registry: Registry,
) -> None:
    with pytest.raises(KeyError, match='no FnClaim'):
        dqn_registry.fn('not_a_real_claim')
    with pytest.raises(KeyError, match='no class'):
        dqn_registry.cls('NotARealClass')


def test_collision_on_different_handle_raises() -> None:
    """Re-registering the *same* name with a *different* class
    is a implementation-author error; the registry refuses."""
    reg = Registry()

    @dataclass(frozen=True)
    class A:
        pass

    @dataclass(frozen=True)
    class B:
        pass

    # Spoof the names so they collide.
    A.__name__ = 'Same'
    B.__name__ = 'Same'

    reg.add_class(A)
    with pytest.raises(ValueError, match='already registered'):
        reg.add_class(B)


def test_idempotent_re_add_is_a_noop(dqn_registry: Registry) -> None:
    """Re-walking the same module twice doesn't raise — the
    second pass sees the same objects and skips."""
    dqn_registry.add_modules(DQN_CLAIM_MODULES)
    from corroborate_rl.dqn.claims.bootstrap import double_greedify
    assert dqn_registry.fn('double_greedify') is double_greedify


def test_registered_handles_are_the_typed_shape(
    dqn_registry: Registry,
) -> None:
    """No type erasure: `fn` returns FnClaim, `cls` returns a
    `type`. Config bundles (`MLP`, `Replay`) round-trip as plain
    frozen-dataclass classes."""
    assert isinstance(dqn_registry.fn('bootstrap'), FnClaim)
    assert isinstance(dqn_registry.fn('epsilon_greedy'), FnClaim)
    mlp_cls = dqn_registry.cls('MLP')
    assert isinstance(mlp_cls, type)
    replay_cls = dqn_registry.cls('Replay')
    assert isinstance(replay_cls, type)
