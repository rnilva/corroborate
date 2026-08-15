"""Contract tests for `BridgeEvaluation.extent_hash`.

The extent_hash is the framework's load-bearing cluster-identity
primitive: `stable_extent_hash(admitted_cell_ids)` computed at
evaluation time. Two bridges with the same `(source_name,
target_name, extent_hash)` admitted identical cell-sets on the
current cache — automatic cluster on the post-evaluated graph.

These tests lock the five invariants the design relies on:

1. extent_hash equals `stable_extent_hash(admitted_ids)` literally.
2. Empty admission → `stable_extent_hash(())` (the honest-empty-extent
   invariant; AWAITING-DATA bridges all share this hash).
3. Stable under row permutation (frozenset semantics).
4. Bridges sharing a NAMED module-level scope predicate hash
   identically — the cluster-identity invariant.
5. Disjoint scopes admit disjoint cells → distinct hashes."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

import polars as pl

from corroborate.bridge.analysis import analysis
from corroborate.bridge.bridge import claim_bridge, evaluate
from corroborate.bridge.verdict import Verdict
from corroborate.graph.causal import Direction, Tier, stable_extent_hash


@dataclass(frozen=True, slots=True)
class _NoopResult:
    """Synthetic analysis result for extent_hash tests — content
    is irrelevant; we only inspect `BridgeEvaluation.extent_hash`."""
    admitted: int


@analysis
def _noop_analysis(
    cells: list[Mapping[str, object]],
) -> _NoopResult:
    """No-op analysis — returns admitted count. The extent_hash
    tests don't care about the result content; they read
    `BridgeEvaluation.extent_hash` directly."""
    return _NoopResult(admitted=len(cells))


def _cells(*ids: str, x: tuple[float, ...] | None = None) -> list[dict[str, object]]:
    """Build synthetic cells with `id` + optional `x` covariate.
    `x` defaults to `(1.0, 2.0, ...)` aligned to `ids`."""
    xs = x if x is not None else tuple(float(i + 1) for i in range(len(ids)))
    return [{'id': cid, 'x': xv} for cid, xv in zip(ids, xs)]


# Named scope predicate — shared across bridges to exercise the
# cluster-identity invariant (test #4).
_SHARED_POSITIVE_X_SCOPE = pl.col('x') > 0


@claim_bridge(
    source='x', target='x',
    direction=Direction.DIRECT, tier=Tier.ASSOCIATIONAL,
    scope=_SHARED_POSITIVE_X_SCOPE, pair_by=(),
)
def _bridge_positive_x_a(_noop_analysis: _NoopResult) -> Verdict:
    return Verdict.HELD


@claim_bridge(
    source='x', target='x',
    direction=Direction.DIRECT, tier=Tier.ASSOCIATIONAL,
    scope=_SHARED_POSITIVE_X_SCOPE, pair_by=(),
)
def _bridge_positive_x_b(_noop_analysis: _NoopResult) -> Verdict:
    return Verdict.HELD


@claim_bridge(
    source='x', target='x',
    direction=Direction.DIRECT, tier=Tier.ASSOCIATIONAL,
    scope=pl.col('x') < 0, pair_by=(),
)
def _bridge_negative_x(_noop_analysis: _NoopResult) -> Verdict:
    return Verdict.HELD


@claim_bridge(
    source='x', target='x',
    direction=Direction.DIRECT, tier=Tier.ASSOCIATIONAL,
    scope=pl.col('x') > 1000.0, pair_by=(),
)
def _bridge_empty_scope(_noop_analysis: _NoopResult) -> Verdict:
    return Verdict.HELD


def test_extent_hash_equals_frozenset_admitted_ids() -> None:
    """The load-bearing identity invariant: extent_hash IS
    `stable_extent_hash(admitted_cell_ids)` — not a digest, not
    a sorted-tuple hash. Frozenset semantics are the contract."""
    cells = _cells('c0', 'c1', 'c2', 'c3', x=(1.0, 2.0, -1.0, -2.0))
    out = evaluate(_bridge_positive_x_a, cells)
    expected = stable_extent_hash({'c0', 'c1'})
    assert out.extent_hash == expected


def test_extent_hash_empty_when_scope_admits_zero() -> None:
    """Honest-empty-extent invariant: scope admits zero cells →
    extent_hash is `stable_extent_hash(())`. AWAITING-DATA bridges
    cluster naturally under this single hash, reflecting that the
    framework cannot distinguish them on the current cache."""
    cells = _cells('c0', 'c1', 'c2', x=(1.0, 2.0, 3.0))
    out = evaluate(_bridge_empty_scope, cells)
    assert out.extent_hash == stable_extent_hash(())
    assert out.n_cells_in_scope == 0


def test_extent_hash_stable_under_row_permutation() -> None:
    """Frozenset semantics: same admitted IDs in any order →
    same hash. Locks the contract against accidental switches
    to ordered-hash (e.g. `hash(tuple(sorted(ids)))`)."""
    forward = _cells('a', 'b', 'c', 'd', x=(1.0, 2.0, 3.0, 4.0))
    reversed_order = _cells('d', 'c', 'b', 'a', x=(4.0, 3.0, 2.0, 1.0))
    out_forward = evaluate(_bridge_positive_x_a, forward)
    out_reversed = evaluate(_bridge_positive_x_a, reversed_order)
    assert out_forward.extent_hash == out_reversed.extent_hash


def test_extent_hash_stable_across_python_hash_seeds() -> None:
    """Saved reports must compare across fresh interpreter processes.

    In particular, this fails for ``hash(frozenset(str_ids))`` because
    CPython salts string hashes independently in each process.
    """
    root = Path(__file__).parents[1]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root / 'src')
    command = [
        sys.executable,
        '-c',
        (
            'from corroborate.graph._extent import stable_extent_hash; '
            'print(stable_extent_hash({"alpha", "beta"}))'
        ),
    ]
    outputs = []
    for hash_seed in ('1', '987654'):
        env['PYTHONHASHSEED'] = hash_seed
        outputs.append(subprocess.check_output(
            command,
            cwd=root,
            env=env,
            text=True,
        ).strip())
    assert outputs[0] == outputs[1]


def test_extent_hash_shared_when_same_named_scope() -> None:
    """Cluster-identity invariant — the load-bearing claim of
    the whole design. Two bridges importing the same named
    module-level scope constant produce identical `extent_hash`
    values on the same cache. Authors who want a refutation /
    sibling cluster MUST share the named expression."""
    cells = _cells('c0', 'c1', 'c2', 'c3', x=(1.0, 2.0, -1.0, -2.0))
    out_a = evaluate(_bridge_positive_x_a, cells)
    out_b = evaluate(_bridge_positive_x_b, cells)
    assert out_a.extent_hash == out_b.extent_hash
    # Same scope but different bridge names → cluster key
    # is (source, target, extent_hash), so source+target also
    # matter; here they're identical for both bridges.
    assert out_a.source_name == out_b.source_name
    assert out_a.target_name == out_b.target_name


def test_extent_hash_distinct_when_disjoint_scopes() -> None:
    """Disjoint admitted-cell-sets produce distinct cluster
    keys. Validates the framework derives cluster identity
    structurally from the data the scope admits, not from
    scope-expression syntax."""
    cells = _cells('c0', 'c1', 'c2', 'c3', x=(1.0, 2.0, -1.0, -2.0))
    out_pos = evaluate(_bridge_positive_x_a, cells)
    out_neg = evaluate(_bridge_negative_x, cells)
    assert out_pos.extent_hash != out_neg.extent_hash
    assert out_pos.extent_hash == stable_extent_hash({'c0', 'c1'})
    assert out_neg.extent_hash == stable_extent_hash({'c2', 'c3'})
