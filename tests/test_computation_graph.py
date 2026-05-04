"""Tests for `corroborate.computation_graph` — claim-graph
extracted from a `trace_context()` recording.

Covers:
1. Free-function (`@claim`) calls produce an edge when one's
   output is consumed by the next's arg.
2. Structured returns (dict, dataclass) produce field-pathed
   edges.
3. Multi-iteration traces (same edge firing N times in a scan-
   like loop) deduplicate to one structural edge.
4. The two-arg structural signature is hashable and order-
   independent.
5. Faithful-intervention property: a slot-swap intervention
   produces a graph diff; an HP-only tweak produces no diff."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from corroborate.claim import claim, trace_context
from corroborate.graph.computation import (
    ComputationEdge,
    build_computation_graph,
    signature,
)


# ============ Synthetic Claims used across tests ============

@claim
def alpha(x: object) -> dict[str, object]:
    """Synthetic Claim returning a dict so we exercise the
    structured-return path. A new dict per call (object-identity)."""
    return {'value': [x], 'meta': {'tag': 'alpha'}}


@claim
def beta(value: object) -> object:
    """Reads `alpha`'s 'value' field via path matching."""
    return ('beta-out', value)


@claim
def gamma(meta: object) -> object:
    """Reads `alpha`'s 'meta' field via path matching."""
    return ('gamma-out', meta)


@claim
def delta(prev: object) -> object:
    """Bare-return reader: consumes alpha's whole dict."""
    return ('delta-out', prev)


# ============ Basic extraction ============

def test_bare_return_edge_emitted() -> None:
    """`delta(alpha(...))` — delta's `prev` arg matches alpha's
    full return value (`source_path=''`)."""
    with trace_context() as records:
        a_out = alpha('seed')
        delta(prev=a_out)
    g = build_computation_graph(records)
    edges = g.edges_between('alpha', 'delta')
    assert len(edges) == 1
    assert edges[0].metadata == ComputationEdge(
        reader_arg='prev', source_path='',
    )


def test_dict_field_edge_emitted_with_path() -> None:
    """`beta(value=alpha(...)['value'])` — beta's `value` arg
    matches a sub-leaf at path `value`."""
    with trace_context() as records:
        a_out = alpha('seed')
        beta(value=a_out['value'])
    g = build_computation_graph(records)
    edges = g.edges_between('alpha', 'beta')
    assert len(edges) == 1
    assert edges[0].metadata == ComputationEdge(
        reader_arg='value', source_path='value',
    )


def test_nested_dict_path_resolved() -> None:
    """`alpha`'s `meta` is itself a dict; consuming it as a whole
    matches at path `meta`."""
    with trace_context() as records:
        a_out = alpha('seed')
        meta = a_out['meta']
        # Wrap so `gamma`'s arg-id is the meta dict.
        gamma(meta=meta)
    g = build_computation_graph(records)
    edges = g.edges_between('alpha', 'gamma')
    assert len(edges) == 1
    assert edges[0].metadata.source_path == 'meta'


def test_nodes_include_claims_with_no_outgoing_edges() -> None:
    """A Claim that fires but isn't consumed anywhere still
    appears as a node — `with_node` records every Claim seen."""
    with trace_context() as records:
        _ = alpha('seed')  # no consumer
    g = build_computation_graph(records)
    assert 'alpha' in g.nodes


def test_multi_iteration_dedupes_to_one_structural_edge() -> None:
    """The same edge firing across N loop iterations dedupes to
    one structural edge in the ComputationGraph (call indices
    collapsed)."""
    with trace_context() as records:
        for _ in range(5):
            a_out = alpha('seed')
            delta(prev=a_out)
    g = build_computation_graph(records)
    # 5 raw edges in records, but only 1 structural edge:
    # alpha -> delta with reader_arg='prev', source_path=''.
    assert len(g.edges_between('alpha', 'delta')) == 1


# ============ Signature ============

def test_signature_is_hashable_and_deterministic() -> None:
    """Two graphs built from traces emitting the same edges in
    different orders produce identical signatures."""
    with trace_context() as records1:
        a = alpha('seed')
        beta(value=a['value'])
        delta(prev=a)
    with trace_context() as records2:
        a = alpha('seed')
        delta(prev=a)
        beta(value=a['value'])
    g1 = build_computation_graph(records1)
    g2 = build_computation_graph(records2)
    assert signature(g1) == signature(g2)
    # Hashable.
    _ = {signature(g1): 'ok'}


def test_signature_distinguishes_structural_difference() -> None:
    """Two graphs with different edges produce different
    signatures."""
    with trace_context() as records1:
        a = alpha('seed')
        beta(value=a['value'])
    with trace_context() as records2:
        a = alpha('seed')
        delta(prev=a)
    assert signature(build_computation_graph(records1)) != signature(
        build_computation_graph(records2),
    )


# ============ Faithful-intervention property ============

# Build a tiny "theory": a function that orchestrates Claim calls.
# Two interventions:
#   - HP-only:  partial(theory, hp=99)             — same edges
#   - slot:     partial(theory, downstream=gamma)  — different edges

def _theory(
    seed: object, *, hp: int = 1, downstream: object = beta,
) -> object:
    """`hp` is a leaf-scalar tweak (no graph change). `downstream`
    is a slot-swap (the actual reader Claim changes — graph
    changes)."""
    del hp
    a = alpha(seed)
    if downstream is beta:
        return beta(value=a['value'])
    if downstream is gamma:
        return gamma(meta=a['meta'])
    raise ValueError('unknown downstream slot')


def test_hp_only_tweak_has_empty_graph_diff() -> None:
    """A `partial(theory, hp=99)` intervention produces the SAME
    computation graph as baseline. The diff is empty — anti-
    laundering: a pure HP tweak isn't a structural intervention."""
    baseline_theory = partial(_theory, hp=1)
    tweaked_theory = partial(_theory, hp=99)
    with trace_context() as recs_b:
        baseline_theory('seed')
    with trace_context() as recs_t:
        tweaked_theory('seed')
    g_b = build_computation_graph(recs_b)
    g_t = build_computation_graph(recs_t)
    assert g_b.diff(g_t).is_empty()
    assert signature(g_b) == signature(g_t)


def test_slot_swap_intervention_changes_graph() -> None:
    """A `partial(theory, downstream=gamma)` intervention swaps a
    Claim slot. The diff is non-empty: alpha→gamma replaces
    alpha→beta. Faithful intervention shows up at the structural
    level."""
    baseline_theory = partial(_theory)  # downstream=beta default
    intervened_theory = partial(_theory, downstream=gamma)
    with trace_context() as recs_b:
        baseline_theory('seed')
    with trace_context() as recs_i:
        intervened_theory('seed')
    g_b = build_computation_graph(recs_b)
    g_i = build_computation_graph(recs_i)
    diff = g_b.diff(g_i)
    assert not diff.is_empty()
    # baseline has alpha→beta; intervened has alpha→gamma.
    only_b_targets = {e.target for e in diff.edges_only_in_self}
    only_i_targets = {e.target for e in diff.edges_only_in_other}
    assert 'beta' in only_b_targets
    assert 'gamma' in only_i_targets
    # Signatures differ.
    assert signature(g_b) != signature(g_i)


# ============ Class-based Claim integration (escape hatch) ============

@dataclass(frozen=True, slots=True)
class _ClassAdder:
    """Minimal class-based Claim using the manual-dataclass +
    `record_call` escape hatch. Tests that
    `_claim_callable(class.__call__)` resolves cleanly and `self`
    is stripped from the parameter list."""
    bias: int = 0

    @property
    def name(self) -> str:
        return 'class_adder'

    @property
    def invariants(self) -> tuple[object, ...]:
        return ()

    def __call__(self, payload: object) -> object:
        from corroborate.claim import record_call
        result = ('class-out', payload, self.bias)
        record_call(self, (payload,), {}, result)
        return result


def test_class_based_claim_emits_edge_with_param_name() -> None:
    """A class-based Claim using the escape hatch records via
    `record_call`; `_claim_callable` uses `type(self).__call__`
    to recover the parameter name (`payload`). The edge from
    `alpha` reaches the class by that arg name."""
    adder = _ClassAdder(bias=1)
    with trace_context() as records:
        a_out = alpha('seed')
        adder(payload=a_out)
    g = build_computation_graph(records)
    edges = g.edges_between('alpha', 'class_adder')
    assert len(edges) == 1
    assert edges[0].metadata.reader_arg == 'payload'
