"""Substrate-side integration of the `@claim` machinery.

Two cases that need the JAX/DQN substrate:

1. JIT silence — `@claim`-decorated calls fire records even when
   their args are JAX Tracer objects. This is the structural-
   extraction guarantee the framework needs for scan-bodies; lives
   here because verifying it depends on importing JAX.
2. Pickle round-trip on a real substrate-authored claim — the
   framework-side pickle test (`test_claim.py::test_pickle_*`)
   uses a synthetic module-level claim. This test confirms the
   contract holds for an actual `@claim`-decorated DQN function
   in `corroborate_rl.dqn.claims.bootstrap`."""
from __future__ import annotations

from corroborate.core.claim import claim, trace_context


def test_trace_records_under_jit_for_structural_extraction() -> None:
    """When @claim'd functions run inside `jax.jit` / `lax.scan` /
    `vmap`, args are Tracer objects. Recording fires anyway —
    that's exactly the structural information `computation_graph.
    build_*` extracts (which Claim called which during the tracing
    pass). Earlier versions skipped tracer-arg calls; that
    silently dropped every claim inside a scan loop, making the
    full `dqn` (which uses scan) unprofileable."""
    import jax
    import jax.numpy as jnp

    @claim
    def double(x: jax.Array) -> jax.Array:
        return x * 2

    with trace_context() as records:
        # Inside jit the arg is a Tracer → still records (one call
        # per tracing pass).
        result = jax.jit(double)(jnp.float32(3.0))
        # Concrete call outside jit → records.
        _ = double(jnp.float32(4.0))

    assert float(result) == 6.0
    # Both calls record: one from the jit tracing pass + one from
    # the concrete eager call.
    assert len(records) == 2
    assert all(r.claim is double for r in records)


def test_pickle_round_trip_function_claim() -> None:
    """Real substrate-authored claim round-trips through pickle.
    Memoization in `@claim` (`_FN_CACHE`) gives singleton identity
    even after a pickle/unpickle cycle."""
    import pickle
    from corroborate_rl.dqn.claims.bootstrap import bootstrap as vanilla_bootstrap

    blob = pickle.dumps(vanilla_bootstrap)
    restored = pickle.loads(blob)
    assert restored is vanilla_bootstrap


def test_pickle_round_trip_partial_over_claim() -> None:
    """`functools.partial` over a substrate claim pickles natively."""
    import pickle
    from functools import partial
    from corroborate_rl.dqn.claims.bootstrap import bootstrap as vanilla_bootstrap

    baked = partial(vanilla_bootstrap, gamma=0.95)
    blob = pickle.dumps(baked)
    restored = pickle.loads(blob)
    assert restored.func is vanilla_bootstrap
    assert restored.keywords == {'gamma': 0.95}
