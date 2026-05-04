"""Tests for `corroborate.signature.canonical` and
`claim_graph_signature` — default-axis elision for program-instance
identity.

The canonical form strips kwargs whose binding is canonically
equivalent to the signature default (either `==`-equal or whose
walked-and-canonicalised structure matches). This gives us:

1. **Syntactic invariance**: `partial(claim)`,
   `partial(claim, kw=signature_default)`, and
   `partial(claim, kw=partial(signature_default))` all produce the
   same hash.
2. **Forward-compatibility under axis addition**: a future Protocol
   axis with a default implementation doesn't perturb the canonical
   forms of corpora that pre-date the axis (their bindings are
   implicitly at-default for the new slot).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from corroborate.core.claim import claim
from corroborate.core.signature import (
    canonical, claim_graph_signature, walk,
)


# ============ Synthetic Claims for the canonicalisation tests ============

@claim
def inner_default(*, x: int = 1) -> int:
    return x


@claim
def inner_alternative(*, x: int = 1) -> int:
    return x + 100


@claim
def outer(
    *,
    inner: object = inner_default,
    n: int = 0,
) -> int:
    """Outer claim with two kwargs: a slot (`inner`) defaulting
    to inner_default, and a scalar `n` defaulting to 0."""
    del inner, n
    return 0


@dataclass(frozen=True, slots=True)
class Cfg:
    """Frozen-dataclass config with two fields. Used to test
    default-elision on dataclass shape too."""
    a: int = 0
    b: str = 'def'


@claim
def with_cfg(*, cfg: Cfg = Cfg(), n: int = 0) -> int:
    del cfg, n
    return 0


# ============ Direct elision tests ============

def test_bare_claim_canonicalises_to_no_kwargs() -> None:
    """All defaults → empty canonical kwargs."""
    sig = canonical(walk(outer))
    assert sig.name == 'outer'
    assert sig.kwargs == ()


def test_explicit_default_kwarg_elided() -> None:
    """`partial(outer, n=0)` (n's signature default) === bare outer."""
    bound = partial(outer, n=0)
    assert claim_graph_signature(bound) == claim_graph_signature(outer)


def test_divergent_scalar_kept() -> None:
    """`partial(outer, n=42)` differs from bare outer."""
    bound = partial(outer, n=42)
    assert claim_graph_signature(bound) != claim_graph_signature(outer)


# ============ Recursive canonical-equivalence tests ============

def test_partial_wrapping_default_is_canonical_noop() -> None:
    """`partial(outer, inner=partial(inner_default))` === bare outer.
    The partial wrapper around the signature default has no divergent
    kwargs; canonical form elides it."""
    bound = partial(outer, inner=partial(inner_default))
    assert claim_graph_signature(bound) == claim_graph_signature(outer)


def test_partial_with_explicit_default_kwarg_is_noop() -> None:
    """`partial(outer, inner=partial(inner_default, x=1))` === bare
    outer. Explicitly binding x to its signature default inside the
    inner partial is also a no-op."""
    bound = partial(outer, inner=partial(inner_default, x=1))
    assert claim_graph_signature(bound) == claim_graph_signature(outer)


def test_inner_divergent_value_perturbs_hash() -> None:
    """`partial(outer, inner=partial(inner_default, x=999))` differs
    from bare outer because x diverges from its default."""
    bound = partial(outer, inner=partial(inner_default, x=999))
    assert claim_graph_signature(bound) != claim_graph_signature(outer)


def test_inner_alternative_claim_perturbs_hash() -> None:
    """Swapping the slot to a different Claim is a structural
    divergence, not at-default."""
    bound = partial(outer, inner=inner_alternative)
    assert claim_graph_signature(bound) != claim_graph_signature(outer)


# ============ Dataclass-shape tests ============

def test_default_constructed_dataclass_is_noop() -> None:
    """`partial(with_cfg, cfg=Cfg())` (all-default-fields Cfg) ===
    bare with_cfg."""
    bound = partial(with_cfg, cfg=Cfg())
    assert claim_graph_signature(bound) == claim_graph_signature(with_cfg)


def test_dataclass_field_divergence_perturbs_hash() -> None:
    """`partial(with_cfg, cfg=Cfg(a=42))` differs."""
    bound = partial(with_cfg, cfg=Cfg(a=42))
    assert claim_graph_signature(bound) != claim_graph_signature(with_cfg)


# ============ Forward-compat property: canonicalisation version ============

def test_signature_includes_canonical_version() -> None:
    """The version prefix in the hash means future canonicalisation
    rule changes produce different hashes (no silent collisions
    across corpora at different framework versions)."""
    from corroborate.core.signature import CANONICAL_VERSION
    assert isinstance(CANONICAL_VERSION, str)
    assert len(CANONICAL_VERSION) > 0
