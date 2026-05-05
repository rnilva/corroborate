"""Print the structural-measurable surface of a configured dqn as
a tree — no run, no JAX kernel compilation, no env build.

`walk(configured)` returns the recursive `ClaimSignature` tree the
runner walks at composition time to derive `RunRow.measurements`
column names. Each node carries a name + regime + default; nested
slot Claims (`bootstrap.greedification`), config bundles
(`replay`, `q_network`), and partial-bound factories
(`optimizer.inner`) are children.

Useful for:
- Eyeballing exactly what columns a sweep will produce before
  spending GPU minutes.
- Diffing the leaf set across two `partial(dqn, **intervention)`
  bindings to see which paths an intervention shifts.
- Verifying the endogeneity gate's leaf set
  (`corroborate.bridge.admission._claim_leaves`) — the union of
  paths printed here is exactly what `is_endogenous(name, dqn)`
  keys on.

Trajectory measurables (`jensen_gap` / `mc_return` / etc.) aren't
here — they're emitted by a running cell, not by the composition.
See `scripts/visualize_dqn_trace.py` for those.

Usage:
    uv run python scripts/inspect_dqn_structure.py
    uv run python scripts/inspect_dqn_structure.py --gamma 0.95 \\
        --total-steps 200000 --replay-capacity 50000
    uv run python scripts/inspect_dqn_structure.py --flat
"""
from __future__ import annotations

import argparse
import functools

from corroborate.core.claim import FnClaim
from corroborate.core.signature import (
    ClaimSignature,
    KwargInfo,
    walk,
    walk_paths,
)
from corroborate_rl.dqn.dqn import dqn


def _format_default(d: object) -> str:
    """Compact one-cell representation of a kwarg's default."""
    if isinstance(d, type):
        return '<required>'
    if isinstance(d, bool):
        return repr(d)
    if isinstance(d, (int, float, str)):
        return repr(d)
    if isinstance(d, FnClaim):
        return f'Claim:{d.name}'
    if isinstance(d, functools.partial):
        inner: object = d.func
        inner_name = inner.name if isinstance(inner, FnClaim) else type(inner).__name__
        return f'partial({inner_name})'
    if isinstance(d, tuple):
        return f'tuple[{len(d)}]'
    if d is None:
        return 'None'
    return f'<{type(d).__name__}>'


def _format_node(kw: KwargInfo) -> str:
    """One line: `name [regime] = default`. Regime is annotated
    only when exogenous to keep leaf nodes uncluttered."""
    parts = [kw.name]
    if kw.regime == 'exogenous':
        parts.append('[exogenous]')
    parts.append(f'= {_format_default(kw.default)}')
    return ' '.join(parts)


def _print_tree(sig: ClaimSignature, prefix: str = '') -> None:
    """Recurse over `sig.kwargs`; box-draw the parent/child
    structure with `├──` / `└──` connectors."""
    n = len(sig.kwargs)
    for i, kw in enumerate(sig.kwargs):
        is_last = (i == n - 1)
        connector = '└── ' if is_last else '├── '
        child_prefix = prefix + ('    ' if is_last else '│   ')
        print(prefix + connector + _format_node(kw))
        if kw.inner is not None and kw.inner.kwargs:
            _print_tree(kw.inner, child_prefix)


def _print_flat(sig: ClaimSignature) -> None:
    """Old-style dotted-path listing — useful for grep / diff."""
    leaves = walk_paths(sig, regime='leaf')
    exogenous = walk_paths(sig, regime='exogenous')
    print(f'regime=leaf ({len(leaves)} paths):')
    for k in sorted(leaves):
        print(f'  {k:48s}  default={_format_default(leaves[k].default)}')
    print(f'\nregime=exogenous ({len(exogenous)} paths):')
    for k in sorted(exogenous):
        print(f'  {k:48s}  default={_format_default(exogenous[k].default)}')


def main() -> None:
    p = argparse.ArgumentParser(prog='inspect_dqn_structure')
    p.add_argument('--gamma', type=float, default=None)
    p.add_argument('--sync-period', type=int, default=None)
    p.add_argument('--n-step', type=int, default=None)
    p.add_argument('--total-steps', type=int, default=None)
    p.add_argument('--replay-capacity', type=int, default=None)
    p.add_argument(
        '--flat',
        action='store_true',
        help='dotted-path listing instead of tree',
    )
    args = p.parse_args()

    overrides: dict[str, object] = {}
    if args.gamma is not None:
        overrides['gamma'] = args.gamma
    if args.sync_period is not None:
        overrides['sync_period'] = args.sync_period
    if args.n_step is not None:
        overrides['n_step'] = args.n_step
    if args.total_steps is not None:
        overrides['total_steps'] = args.total_steps
    if args.replay_capacity is not None:
        from corroborate_rl.dqn.claims.replay import Replay
        overrides['replay'] = Replay(capacity=args.replay_capacity)

    configured = functools.partial(dqn, **overrides) if overrides else dqn
    sig = walk(configured)

    header = (
        f'configured dqn (overrides: {overrides})'
        if overrides else 'bare dqn (no overrides)'
    )
    print(f'{header}')
    print(f'{"─" * len(header)}\n')

    if args.flat:
        _print_flat(sig)
    else:
        print(sig.name)
        _print_tree(sig)

    leaves = walk_paths(sig, regime='leaf')
    exogenous = walk_paths(sig, regime='exogenous')
    print(
        f'\nTotal: {len(leaves) + len(exogenous)} structural paths '
        f'(leaf={len(leaves)}, exogenous={len(exogenous)})',
    )


if __name__ == '__main__':
    main()
