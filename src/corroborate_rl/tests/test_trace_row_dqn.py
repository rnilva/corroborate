"""Substrate-side `walk_paths` smoke: a DQN-configured intervention
surfaces its nested leaves at dotted topology paths.

The framework's `walk_paths` primitive is exercised in the
framework's unit tests for `signature`; this case verifies the
end-to-end shape on a real implementation composition where
`partial(dqn, optimizer=partial(warmed_update,
inner=partial(adam, lr=...), warmup_steps=100))` produces the
nested `optimizer.inner.lr` path the corpus column-namer
depends on."""
from __future__ import annotations


def test_walk_paths_surfaces_nested_leaves_at_dotted_paths() -> None:
    """`signature.walk_paths` must produce dotted topology paths
    keyed at each leaf — `optimizer.inner.lr` (nested leaf under
    WarmedUpdate(inner=Adam(...))) — not the flat `lr`."""
    from functools import partial

    from corroborate_rl.dqn.dqn import dqn
    from corroborate.core.signature import walk, walk_paths

    configured = partial(dqn, optimizer=_make_warmed_adam())
    paths = walk_paths(walk(configured), regime='leaf')

    # Top-level leaves (gamma is dqn's direct kwarg).
    assert 'gamma' in paths
    # Nested: optimizer is a Module field, inner is its inner
    # Module, lr is Adam's leaf.
    assert 'optimizer.inner.lr' in paths
    # Sibling at the same depth resolves to its own path —
    # NOT colliding with optimizer.inner.lr.
    assert 'optimizer.warmup_steps' in paths


def _make_warmed_adam() -> object:
    from functools import partial

    from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
    return partial(
        warmed_update, inner=partial(adam, lr=1e-3), warmup_steps=100,
    )
