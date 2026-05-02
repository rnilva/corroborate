"""Tests for the Phase 6B topology query helpers.

`measurables_by_attachment(graph, claim_name)` — registered
measurables whose `reads` intersect the source-paths the named
claim emitted in `graph`.
`measurable_scope(graph, measurable_name)` — `ScopeInfo` with
`producing_claims` + `unmatched_reads`.

The helpers close the loop between the measurable graph (record-
key reads) and the claim graph (Claim-output source-paths)."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.computation_graph import (
    ComputationEdge,
    ComputationGraph,
    ScopeInfo,
    measurable_scope,
    measurables_by_attachment,
    producing_paths,
)
from corroborate.graph import Graph
from corroborate.measurable import measurable


# A graph that mirrors a tiny substrate: `bootstrap` emits
# `mc_return` (read by an outcome measurable) and
# `predicted_q_at_start` (read by a mechanism measurable). A
# downstream `loss` claim consumes `td_error` from `bootstrap`.

@measurable(reads=('mc_return',))
def _t_outcome_mean(record: Mapping[str, object]) -> float:
    """Test-only measurable: reads `mc_return` (produced by
    bootstrap) → outcome-side."""
    del record
    return 0.0


@measurable(reads=('predicted_q_at_start', 'mc_return'))
def _t_mechanism_gap(record: Mapping[str, object]) -> float:
    """Test-only measurable: reads two bootstrap-emitted paths
    → mechanism-side."""
    del record
    return 0.0


@measurable(reads=('env_name', 'seed'))
def _t_exogenous_only(record: Mapping[str, object]) -> float:
    """Test-only measurable: reads only substrate exogenous
    keys (`env_name`, `seed`) — neither is emitted by any Claim
    in the graph."""
    del record
    return 0.0


def _toy_graph() -> ComputationGraph:
    g: ComputationGraph = Graph()
    g = g.with_edge(
        'bootstrap', 'loss',
        ComputationEdge(reader_arg='mc_return', source_path='mc_return'),
    )
    g = g.with_edge(
        'bootstrap', 'mechanism_diag',
        ComputationEdge(
            reader_arg='predicted',
            source_path='predicted_q_at_start',
        ),
    )
    g = g.with_edge(
        'bootstrap', 'loss',
        ComputationEdge(reader_arg='td', source_path='td_error'),
    )
    # Bare-return edge — should NOT count as a record-key emission.
    g = g.with_edge(
        'optimizer', 'updater',
        ComputationEdge(reader_arg='params', source_path=''),
    )
    return g


# ============ producing_paths ============

def test_producing_paths_collects_named_emissions() -> None:
    g = _toy_graph()
    assert producing_paths(g, 'bootstrap') == frozenset({
        'mc_return', 'predicted_q_at_start', 'td_error',
    })


def test_producing_paths_filters_bare_return_edges() -> None:
    """Edges with empty source_path are claim-to-claim flow-
    through, NOT record-key emissions."""
    g = _toy_graph()
    assert producing_paths(g, 'optimizer') == frozenset()


def test_producing_paths_unknown_claim_returns_empty() -> None:
    g = _toy_graph()
    assert producing_paths(g, 'nonexistent_claim') == frozenset()


# ============ measurables_by_attachment ============

def test_measurables_by_attachment_finds_outcome_and_mechanism() -> None:
    """Both `_t_outcome_mean` and `_t_mechanism_gap` read
    bootstrap-produced paths; both should be attached."""
    g = _toy_graph()
    attached = set(measurables_by_attachment(g, 'bootstrap'))
    assert '_t_outcome_mean' in attached
    assert '_t_mechanism_gap' in attached


def test_measurables_by_attachment_excludes_unrelated() -> None:
    """Measurables whose `reads` don't intersect bootstrap's
    paths shouldn't be in the attached set."""
    g = _toy_graph()
    attached = set(measurables_by_attachment(g, 'bootstrap'))
    assert '_t_exogenous_only' not in attached


def test_measurables_by_attachment_unknown_claim_returns_empty() -> None:
    g = _toy_graph()
    assert measurables_by_attachment(g, 'nonexistent') == ()


# ============ measurable_scope ============

def test_measurable_scope_records_producing_claims() -> None:
    """`_t_outcome_mean.reads = ('mc_return',)` and bootstrap
    emits 'mc_return' → producing_claims contains 'bootstrap',
    no unmatched reads."""
    g = _toy_graph()
    info = measurable_scope(g, '_t_outcome_mean')
    assert info == ScopeInfo(
        measurable_name='_t_outcome_mean',
        producing_claims=('bootstrap',),
        unmatched_reads=(),
    )


def test_measurable_scope_lists_unmatched_reads() -> None:
    """Reads that no Claim emits in the graph land in
    `unmatched_reads` — substrate-side exogenous keys live
    here."""
    g = _toy_graph()
    info = measurable_scope(g, '_t_exogenous_only')
    assert info.measurable_name == '_t_exogenous_only'
    assert info.producing_claims == ()
    assert set(info.unmatched_reads) == {'env_name', 'seed'}


def test_measurable_scope_unknown_name_raises() -> None:
    g = _toy_graph()
    try:
        measurable_scope(g, 'definitely_not_registered')
    except KeyError as e:
        assert 'definitely_not_registered' in str(e)
    else:
        raise AssertionError('expected KeyError')
