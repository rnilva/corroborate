"""Minimal dowhy stubs — `CausalModel` is the only surface
corroborate's analyses/_dowhy_internal reaches.

dowhy ships no upstream type stubs. The framework's strict-Any
contract rejects the resulting `Unknown`/`Any` leaks at the
import boundary (the analyses/ subtree already relaxes the rest
via its executionEnvironment). This stub narrows just the
identify/estimate/refute trio that corroborate calls.

Mirrors the gymnax / optax / scipy / statsmodels stubs in shape —
narrow at the boundary, only the surface that's actually used.
Estimand / estimate / refuter return shapes are typed loosely
(`object` for the dynamic dowhy result objects); corroborate's
helpers narrow via `getattr` at the consumption site (the
intentional dynamic-attr boundary).

Inputs use `object` (not `Any`) per CLAUDE.md's typing-discipline
rule — the framework treats dowhy's polymorphic input shapes as
opaque at the stub seam; corroborate's `_to_networkx` /
`_cells_to_dataframe` helpers narrow before reaching this API."""
from __future__ import annotations


class CausalModel:
    """`dowhy.CausalModel(data, treatment, outcome, graph)` —
    constructor accepts a pandas DataFrame, treatment/outcome
    column names, and a graph (networkx DiGraph or DOT/edge-list).

    `data` and `graph` are `object` because dowhy accepts multiple
    input shapes; corroborate's boundary helpers narrow before
    passing here."""
    def __init__(
        self,
        data: object,  # pandas.DataFrame at runtime
        treatment: str,
        outcome: str,
        graph: object,  # networkx.DiGraph at runtime (corroborate coerces)
    ) -> None: ...

    def identify_effect(
        self,
        proceed_when_unidentifiable: bool = ...,
    ) -> object:
        """Return the identified-estimand object. Has dynamic
        attributes `no_directed_path: bool`, `estimands:
        dict[str, Any] | None`, and a `__str__` impl. Narrow via
        `getattr` at the use site."""

    def estimate_effect(
        self,
        identified_estimand: object,
        method_name: str = ...,
    ) -> object:
        """Return the estimate object. Has dynamic attribute
        `value: float | numpy.floating`. Narrow via `getattr`."""

    def refute_estimate(
        self,
        identified_estimand: object,
        estimate: object,
        method_name: str = ...,
    ) -> object:
        """Return the refuter object. Carries either `new_effect`
        or `estimated_effect` depending on dowhy version
        (corroborate's `_refuter_effect` tries both)."""
