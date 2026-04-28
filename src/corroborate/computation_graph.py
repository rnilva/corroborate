"""Computation graph — claim-graph derived from a trace.

Nodes are Claim names. Edges are `ComputationEdge`s carrying the
reader's parameter name + the source's output path:

  bootstrap.v_next ← greedify_target  # bare-return source
  loss.is_weights ← buffer_sample.is_weights  # dict-field source

Subsumes `extract_edges` as a typed Graph[str, ComputationEdge].
The graph adds reachability, paths, subgraph, diff (vs another
graph), and a structural `signature(g)` — a hashable fingerprint
of the graph topology.

Use cases:

- **Faithful intervention check.** Run a baseline hypothesis under
  `trace_context()`; build_computation_graph; then run the
  intervened hypothesis the same way. `g_baseline.diff(g_intervened)`
  shows EXACTLY which edges changed. A `partial(theory, gamma=0.95)`-
  style HP tweak produces an empty diff (no structural change); a
  slot-swap (`partial(theory, bootstrap=double_greedify)`) produces
  a node/edge diff. Anti-laundering at the structural level.

- **Mechanism signature.** `signature(g)` is hashable, suitable as
  a tuple identity component for declared-but-derived mechanism
  keys. Two interventions producing the same signature are
  structurally identical mechanisms.

- **Substrate-agnostic.** Operates on `CallRecord`s, which only
  require Claim's structural Protocol + the `record_call`
  registration in the active `trace_context()`. No JAX dependency
  beyond what's already in `record_call`.

The extraction algorithm:

1. For each call, bind positional args to parameter names via
   `inspect.signature`. Parameter name IS the edge's reader_arg.
2. For each bound arg value, look up its `id()` in a registry of
   prior calls' outputs (walked recursively — dataclass, NamedTuple,
   dict, tuple leaves are all registered with their path).
3. A hit emits an `_RawEdge`: `reader_call.reader_arg ←
   source_call.source_path`. Source path is empty for bare return;
   for structured returns it's the dotted/indexed path to the
   specific field (`'value'`, `'is_weights'`, `'[1].v_next'`, etc.).

Identity-based matching is reliable in eager Python with JAX
arrays (every operation creates a fresh array, no interning).
Small primitive ints/floats/bools (which Python may intern) are
NOT registered — edges on those values would be indistinguishable
anyway."""
from __future__ import annotations

import dataclasses as _dc
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from corroborate.claim import CallRecord, Claim
from corroborate.graph import Graph


_NON_REGISTERABLE_PRIMITIVES = (int, float, bool, type(None), str, bytes)


@dataclass(frozen=True, slots=True)
class ComputationEdge:
    """One edge in the claim-graph: which kwarg of the reader Claim
    consumed which output path of the source Claim.

    `reader_arg` — the parameter name on the reader's signature.
    `source_path` — the dotted/indexed path inside the source
    Claim's return value. Empty when the reader consumed the bare
    return value."""
    reader_arg: str
    source_path: str = ''

    def __str__(self) -> str:
        if self.source_path:
            return f'.{self.reader_arg} ← .{self.source_path}'
        return f'.{self.reader_arg} ← <return>'


type ComputationGraph = Graph[str, ComputationEdge]


@dataclass(frozen=True, slots=True)
class _RawEdge:
    """Internal — pre-deduplication edge with call indices intact.
    `build_computation_graph` collapses these by structural key
    (reader_name, reader_arg, source_name, source_path)."""
    reader_call: int
    reader_name: str
    reader_arg: str
    source_call: int
    source_name: str
    source_path: str


def _claim_callable(claim: Claim[..., object]) -> Callable[..., object] | None:
    """Extract the callable to inspect for signature binding.

    For `FnClaim`, the underlying `fn` field. For `ClaimBase`
    instances, the class's `__call__` method (positional 0 is
    `self` — `inspect.signature` on a bound method strips `self`
    automatically, so we use `type(instance).__call__` and bind
    the instance manually below).

    Returns None if no signature is recoverable — those calls are
    skipped during edge extraction."""
    fn = getattr(claim, 'fn', None)
    if callable(fn):
        return fn
    method = getattr(type(claim), '__call__', None)
    if callable(method):
        return method
    return None


def extract_raw_edges(
    records: Sequence[CallRecord],
) -> list[_RawEdge]:
    """Walk records in order; emit `_RawEdge`s when a call's arg-
    value matches a prior call's output leaf by `id`.

    Internal — most consumers want `build_computation_graph(records)`
    which deduplicates structurally. Exposed for test introspection
    + multi-iteration analyses (e.g., does the same edge fire across
    every scan iteration)."""
    edges: list[_RawEdge] = []
    # id(value) -> (call_idx, claim_name, path_in_output)
    registry: dict[int, tuple[int, str, str]] = {}

    for i, record in enumerate(records):
        claim_name = record.claim.name
        callable_for_sig = _claim_callable(record.claim)
        if callable_for_sig is None:
            # Skip records without a recoverable signature, but
            # still register the result so downstream calls can
            # match against it.
            _register_leaves(record.result, i, claim_name, '', registry)
            continue

        try:
            sig = inspect.signature(callable_for_sig)
            # For ClaimBase.__call__ (a function, not a bound method),
            # the first parameter is `self`; bind the instance.
            params = list(sig.parameters.values())
            if (
                params and params[0].name == 'self'
                and not isinstance(callable_for_sig, type)
            ):
                bound = sig.bind_partial(
                    record.claim, *record.args, **record.kwargs,
                )
            else:
                bound = sig.bind_partial(*record.args, **record.kwargs)
        except (ValueError, TypeError):
            _register_leaves(record.result, i, claim_name, '', registry)
            continue

        for arg_name, arg_value in bound.arguments.items():
            if arg_name == 'self':
                continue  # don't emit self-edges from Module call sites
            _scan_arg_for_edges(
                arg_name, arg_value, i, claim_name, registry, edges,
            )

        _register_leaves(record.result, i, claim_name, '', registry)

    return edges


def _scan_arg_for_edges(
    arg_name: str,
    arg_value: object,
    reader_idx: int,
    reader_name: str,
    registry: dict[int, tuple[int, str, str]],
    edges: list[_RawEdge],
) -> None:
    """Look up `arg_value` in the registry. On hit, append an edge.
    Aggregate matches (the whole dict / dataclass) work because
    `_register_leaves` registers the aggregate alongside its
    leaves."""
    source = registry.get(id(arg_value))
    if source is not None:
        src_idx, src_name, src_path = source
        edges.append(_RawEdge(
            reader_call=reader_idx,
            reader_name=reader_name,
            reader_arg=arg_name,
            source_call=src_idx,
            source_name=src_name,
            source_path=src_path,
        ))


def _register_leaves(
    value: object,
    call_idx: int,
    claim_name: str,
    path: str,
    registry: dict[int, tuple[int, str, str]],
) -> None:
    """Recursively register a value's leaves (and the aggregate
    itself) into the id registry. The aggregate is registered
    first so a reader consuming the WHOLE structure still matches.

    Skips small primitives that Python may intern — id-equality on
    `42` is meaningless across calls.

    First-write wins: if a value's `id()` is already registered,
    we skip BOTH re-registration AND child traversal. The reason:
    if Claim X produced value V (registering V and V's children),
    and Claim Y's return value happens to contain V (e.g. Y
    returns `('result', V)`), then V's true source is X — Y is
    just passing it through. Without first-write, the registry
    would be overwritten to credit Y, and trace-order would
    determine which Claim got attribution. First-write keeps the
    signature deterministic."""
    if isinstance(value, _NON_REGISTERABLE_PRIMITIVES):
        return
    if id(value) in registry:
        return  # first-write wins; children already registered

    # Register the aggregate so a reader consuming the whole
    # dict / dataclass / tuple still matches.
    registry[id(value)] = (call_idx, claim_name, path)

    # Walk known container types. For opaque objects (e.g.
    # jax.Array, custom classes), the aggregate registration above
    # is the only entry — that's correct, those are leaves.
    if isinstance(value, dict):
        for k, v in value.items():
            sub = f'{path}.{k}' if path else str(k)
            _register_leaves(v, call_idx, claim_name, sub, registry)
        return
    if _dc.is_dataclass(value) and not isinstance(value, type):
        for f in _dc.fields(value):
            sub = f'{path}.{f.name}' if path else f.name
            _register_leaves(
                getattr(value, f.name), call_idx, claim_name, sub, registry,
            )
        return
    fields = getattr(value, '_fields', None)
    if isinstance(fields, tuple):  # NamedTuple
        for f_name in fields:
            if not isinstance(f_name, str):
                continue
            sub = f'{path}.{f_name}' if path else f_name
            _register_leaves(
                getattr(value, f_name), call_idx, claim_name, sub, registry,
            )
        return
    if isinstance(value, (tuple, list)):
        for j, v in enumerate(value):
            sub = f'{path}[{j}]' if path else f'[{j}]'
            _register_leaves(v, call_idx, claim_name, sub, registry)
        return


# ============ ComputationGraph constructor + signature ============

def build_computation_graph(
    records: Sequence[CallRecord],
) -> ComputationGraph:
    """Build a `ComputationGraph` from a list of `CallRecord`s.

    Each Claim call's output identity is matched against subsequent
    callers' arg values (via `extract_raw_edges`); each match
    becomes a `ComputationEdge`. The same edge from multiple
    iterations is deduplicated by (reader_name, reader_arg,
    source_name, source_path) — the structural shape independent
    of how many times the loop fired.

    Use under `trace_context()`:

        with trace_context() as records:
            theory(state, **intervention)
        g = build_computation_graph(records)
        print(g.to_tree())
    """
    raw_edges = extract_raw_edges(records)
    g: ComputationGraph = Graph()

    seen: set[tuple[str, str, str, str]] = set()
    for re in raw_edges:
        key = (
            re.reader_name, re.reader_arg,
            re.source_name, re.source_path,
        )
        if key in seen:
            continue
        seen.add(key)
        edge_meta = ComputationEdge(
            reader_arg=re.reader_arg,
            source_path=re.source_path,
        )
        g = g.with_edge(re.source_name, re.reader_name, edge_meta)
    # Also include any source-only nodes whose calls fired but had
    # no consumers in the trace. They're nodes; just no edges.
    for r in records:
        g = g.with_node(r.claim.name)
    return g


def signature(g: ComputationGraph) -> tuple[
    tuple[str, str, str, str], ...,
]:
    """Hashable structural signature of a computation graph.

    Two graphs with identical signatures are structurally identical
    mechanisms (same nodes, same edges by source/target + reader
    arg + source path).

    Tuple element shape: (source, target, reader_arg, source_path).
    Sorted for deterministic equality across runs that emit the
    same edge set in different orders."""
    return tuple(sorted(
        (e.source, e.target, e.metadata.reader_arg, e.metadata.source_path)
        for e in g.edges
    ))
