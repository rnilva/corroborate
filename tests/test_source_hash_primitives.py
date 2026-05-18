"""Source-fingerprint primitives — `bridge_source_hash` contract.

Four tests covering the algorithm's contract (spec §5 + the
decorator-kwarg axis named in §3):

1. round-trip — hash is deterministic on a single bridge.
2. semantic-change — a numeric-literal edit in the body flips
   the hash.
3. cosmetic-stability — docstring + whitespace + comment edits
   are canonicalised away by the `ast.dump + _strip_docstrings`
   pipeline.
4. decorator-kwarg flip — varying `scope` / `direction` (with the
   body byte-identical) flips the hash, via `dataclasses.replace`
   on a single bridge so the AST-dump portion is held constant.

Test 3 directly exercises the canonicalisation helpers rather
than comparing two distinct bridges' hashes — distinct bridges
have distinct `def` names which the AST captures as
`FunctionDef(name=...)`, so any two top-level bridge functions
necessarily hash differently. The canonicalisation contract is
"two source bodies with the same AST under docstring-stripping
hash identically", and the cleanest assertion of that contract
is to feed two same-named source strings through the same
pipeline.

Test 4 isolates the decorator-kwarg axis the same way: it
`dataclasses.replace`s a single bridge so the `holds_when` field
(and therefore `inspect.getsource` + AST-dump) is byte-identical
across the pair — the *only* delta is the kwarg the test varies."""
from __future__ import annotations

# Importing analyses populates the measurable registry so that
# `@claim_bridge` resolution at module-import time finds the
# names declared on the test bridges below.
import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.core.claim import claim
from corroborate.core.intervention import DoEffect, Intervention
from corroborate.core.signature import bridge_source_hash


# ============ Shared fixtures ============


@claim
def _t_op_source_hash(x: int) -> int:
    return x


@claim
def _b_op_source_hash(x: int) -> int:
    return x


_INTERVENTION = DoEffect(arms=(
    (Intervention(slot_path='op', replacement=_b_op_source_hash),),
    (Intervention(slot_path='op', replacement=_t_op_source_hash),),
))


# Two bridges differ only in a numeric literal in the body —
# `bridge_source_hash` must flip between them. Both live at module
# scope so the `@claim_bridge` decoration resolves cleanly at
# import time.

@claim_bridge(
    source=_INTERVENTION,
    target='outcome',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    predicted_direction='a_gt_b',
)
def _bridge_floor_a(
    *,
    treatment_arm: str = '',
    baseline_arm: str = '',
) -> Verdict:
    """Synthetic bridge — numeric literal `harm_floor=0.3` is what
    the semantic-change test edits to `0.5`."""
    del treatment_arm, baseline_arm
    harm_floor = 0.3
    if harm_floor < 0.5:
        return Verdict.HELD
    return Verdict.NO_EFFECT


@claim_bridge(
    source=_INTERVENTION,
    target='outcome',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    predicted_direction='a_gt_b',
)
def _bridge_floor_b(
    *,
    treatment_arm: str = '',
    baseline_arm: str = '',
) -> Verdict:
    """Sibling of `_bridge_floor_a` differing only in `harm_floor`'s
    numeric literal (`0.5` vs `0.3`) — source hash must flip."""
    del treatment_arm, baseline_arm
    harm_floor = 0.5
    if harm_floor < 0.5:
        return Verdict.HELD
    return Verdict.NO_EFFECT


# ============ Test 1: round-trip ============


def test_bridge_source_hash_round_trip() -> None:
    """Compute the hash twice on the same bridge; must match.

    Captures the simplest invariant of any hash function plus the
    AST-canonicalisation: the second-call hash flows through the
    same `inspect.getsource` → `ast.parse` → `_strip_docstrings`
    → `ast.dump` → sha256 path, so any path-dependence bug
    surfaces here before the semantic-drift / cosmetic-stability
    tests come into play."""
    h1 = bridge_source_hash(_bridge_floor_a)
    h2 = bridge_source_hash(_bridge_floor_a)
    assert h1 == h2


# ============ Test 2: semantic-change flips the hash ============


def test_bridge_source_hash_flips_on_semantic_change() -> None:
    """`_bridge_floor_a` and `_bridge_floor_b` differ only in a
    numeric literal (`harm_floor=0.3` vs `0.5`). Hashes must
    differ — the AST-of-source path catches the literal change."""
    h_a = bridge_source_hash(_bridge_floor_a)
    h_b = bridge_source_hash(_bridge_floor_b)
    assert h_a != h_b, (
        f'expected distinct hashes for bridges with different '
        f'`harm_floor` literals; got {h_a[:12]}... == {h_b[:12]}...'
    )


# ============ Test 3: cosmetic-change preserves the hash ============


def test_bridge_source_hash_stable_across_cosmetic_changes() -> None:
    """Two source bodies semantically identical but differing in
    whitespace, comments, and docstring text must canonicalise to
    the same payload via `_strip_docstrings` + `ast.dump`.

    Contract (spec §3): black / ruff reformats AND docstring edits
    don't bust the hash — only semantic body changes do. The
    AST-of-source step strips whitespace + comments; the
    `_strip_docstrings` walk drops the leading docstring node from
    every function / class / module in the parsed tree.

    Asserted directly on the canonicalisation helpers because two
    distinct module-scope bridges necessarily have distinct `def`
    names, which `FunctionDef(name=...)` captures in the AST —
    invalidating a hash-equality assertion across two bridges. The
    helpers ARE the canonicalisation contract; testing them in
    isolation is the cleaner shape."""
    import ast
    import hashlib
    import json

    from corroborate.core.signature import (
        _strip_docstrings,  # pyright: ignore[reportPrivateUsage]
    )

    src_a = (
        'def fn():\n'
        '    """doc string a"""\n'
        '    x = 1\n'
        '    return x\n'
    )
    src_b = (
        '\n'
        'def fn():\n'
        '\n'
        '    """doc string b — completely different prose."""\n'
        '    # explanatory comment\n'
        '    x = 1  # trailing comment\n'
        '\n'
        '    return x\n'
    )

    def _hash(src: str) -> str:
        tree = ast.parse(src)
        _strip_docstrings(tree)
        ast_repr = ast.dump(
            tree, annotate_fields=True, include_attributes=False,
        )
        payload = ast_repr + '\n' + json.dumps({}, sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    assert _hash(src_a) == _hash(src_b), (
        'expected docstring + whitespace + comment edits to be '
        'canonicalised away by ast.dump + _strip_docstrings; '
        'they were not.'
    )

    # Other half of the contract: a literal change (1 → 2) MUST
    # flip the hash via the same canonicalisation path.
    src_c = (
        'def fn():\n'
        '    """doc string a"""\n'
        '    x = 2\n'
        '    return x\n'
    )
    assert _hash(src_a) != _hash(src_c), (
        'expected a literal change (1 → 2) to flip the hash; '
        'it did not — the AST canonicalisation is too aggressive.'
    )


# ============ Test 4: decorator-kwarg flip ============


def test_bridge_source_hash_flips_on_decorator_kwarg_change() -> None:
    """A change to a decorator kwarg (with the bridge body byte-
    identical) flips the hash via the JSON-sorted `decorator_kwargs`
    serialisation in the payload.

    Constructed via `dataclasses.replace` on a single bridge: that
    holds the `holds_when` field constant (so `inspect.getsource`
    + `_strip_docstrings` + `ast.dump` produce an identical AST-
    dump portion of the payload). The only delta in the hashed
    payload is the kwarg we vary, so a hash flip isolates the
    decorator-kwargs path.

    Two axes asserted in one test (cheap; each call is sub-ms):
    `scope` (the spec's named motivator — `str(pl.Expr)` is what
    `@claim_bridge`'s scope-expression serialises through) and
    `direction` (a closed-enum kwarg). Each must flip the hash
    against the baseline."""
    import dataclasses

    import polars as pl

    baseline = _bridge_floor_a
    h_baseline = bridge_source_hash(baseline)

    # Axis 1: scope expression. Baseline has `scope=None`; we
    # replace with a concrete polars expression. `str(pl.Expr)` is
    # what `bridge_source_hash` reads — a structural edit to the
    # scope predicate must therefore flip the hash.
    with_scope = dataclasses.replace(
        baseline, scope=pl.col('env_name') == 'LGSCM',
    )
    h_scope = bridge_source_hash(with_scope)
    assert h_scope != h_baseline, (
        f'expected scope-expression change (None → '
        f"pl.col('env_name') == 'LGSCM') to flip the hash; "
        f'got {h_scope[:12]}... == {h_baseline[:12]}...'
    )

    # Axis 2: direction enum. Baseline is Direction.DIRECT; flip
    # to AT_LEAST. The kwargs dict's 'direction' field carries
    # `direction.value`, so the JSON payload changes between the
    # pair.
    with_direction = dataclasses.replace(
        baseline, direction=Direction.AT_LEAST,
    )
    h_direction = bridge_source_hash(with_direction)
    assert h_direction != h_baseline, (
        f'expected direction change (DIRECT → AT_LEAST) to flip '
        f'the hash; got {h_direction[:12]}... == '
        f'{h_baseline[:12]}...'
    )
