"""One-shot: rewrite stale canonical-string columns to the new
default-elided unified form.

Background. Commit 2857dd8 changed `canonical_str` to ELIDE
default-equal dataclass fields and partial kwargs. Cells written
BEFORE that commit have stale canonical strings — e.g.,
`'dataclass:Adam(b1=0.9,b2=0.999,eps=1e-08,lr=0.0001)'` and
`'dataclass:Adam(b1=0.9,b2=0.999,eps=1e-08,lr=0.0001,weight_decay=0.0)'`
both represent Adam(lr=0.0001) but cache as different strings.

Layered on top of that: substrate refactored some claims from
dataclass-Module form to `@claim` factory form (e.g.,
`dataclass:Adam(...)` → `partial(Claim:adam;...)`). Cells from
both pre-refactor and post-refactor sweeps coexist in the cache.

This script normalises both axes by string-rewriting cached
canonical strings to the new unified form:

  dataclass:Adam(b1=0.9,b2=0.999,eps=1e-08,lr=X)
    → partial(Claim:adam;lr=X)               # if X != adam default 1e-3
  dataclass:Adam(b1=0.9,b2=0.999,eps=1e-08,lr=X,weight_decay=Y)
    → partial(Claim:adam;lr=X[,weight_decay=Y if Y != 0])
  dataclass:WarmedUpdate(inner=I,warmup_steps=W)
    → partial(Claim:warmed_update;inner=I[,warmup_steps=W if W != 1000])
  dataclass:Replay(batch_size=B,capacity=C,sample=Claim:uniform_sample
                   [,gamma=G,n_step=N])
    → dataclass:Replay([batch_size=B if B != 64][,capacity=C if C != 10000])
                                              # sample at default elided;
                                              # legacy gamma/n_step dropped
                                              # (no longer Replay fields)
  dataclass:EpsilonGreedy(schedule=Claim:linear_epsilon)
    → dataclass:EpsilonGreedy()              # schedule at default elided

Substrate-knowledge embedded in this script:
  Adam:           lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0
  warmed_update:  warmup_steps=1000
  Replay:         batch_size=64, capacity=10000, sample=uniform_sample
  EpsilonGreedy:  schedule=linear_epsilon

The script reads each parquet, rewrites in-place (after a
`.bak` backup), and reports per-column distinct-value counts
before/after.

Usage:
  PYTHONPATH=. uv run python scripts/sanitize_cache_canonical.py \\
      experiments/data/cache/dqn_bridges.parquet \\
      experiments/data/cache/ddqn.parquet
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import polars as pl


# ---- substrate-side defaults (mirrors the @claim function signatures) ----

_ADAM_DEFAULTS: dict[str, str] = {
    'b1': '0.9', 'b2': '0.999', 'eps': '1e-08',
    'lr': '0.001', 'weight_decay': '0.0',
}
_WARMED_UPDATE_DEFAULTS: dict[str, str] = {'warmup_steps': '1000'}
_REPLAY_DEFAULTS: dict[str, str] = {
    'batch_size': '64', 'capacity': '10000',
    'sample': 'Claim:uniform_sample',
}
_EPSILON_GREEDY_DEFAULTS: dict[str, str] = {
    'schedule': 'Claim:linear_epsilon',
}
_LINEAR_EPSILON_DEFAULTS: dict[str, str] = {
    'eps_init': '1.0', 'eps_final': '0.05', 'anneal_steps': '10000',
}


def _parse_kv_body(body: str) -> list[tuple[str, str]]:
    """Parse `key1=val1,key2=val2,...` where val may itself contain
    `=` and nested `()`. Splits at top-level commas only.

    Returns (key, value) tuples in input order. Doesn't validate;
    treats anything after the first `=` as the value."""
    out: list[tuple[str, str]] = []
    depth = 0
    start = 0
    parts: list[str] = []
    for i, ch in enumerate(body):
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        elif ch == ',' and depth == 0:
            parts.append(body[start:i])
            start = i + 1
    parts.append(body[start:])
    for part in parts:
        part = part.strip()
        if not part:
            continue
        eq = part.find('=')
        if eq < 0:
            continue
        out.append((part[:eq].strip(), part[eq + 1:].strip()))
    return out


def _emit_kv_body(kvs: list[tuple[str, str]]) -> str:
    """Inverse of _parse_kv_body. Produces sorted-by-key
    `k1=v1,k2=v2,...` for deterministic output."""
    sorted_kvs = sorted(kvs, key=lambda p: p[0])
    return ','.join(f'{k}={v}' for k, v in sorted_kvs)


def _parse_outer(s: str) -> tuple[str, str] | None:
    """Match `Wrapper(body)` or `Wrapper:Name(body)` — return
    (prefix, body) or None if no match. The prefix includes the
    trailing `(`; the body excludes the final `)`."""
    if not s.endswith(')'):
        return None
    paren = s.find('(')
    if paren < 0:
        return None
    return s[:paren + 1], s[paren + 1:-1]


def _strip_defaults(
    kvs: list[tuple[str, str]],
    defaults: dict[str, str],
    drop_unknown: bool = False,
) -> list[tuple[str, str]]:
    """Filter kvs: drop any (k, v) where v == defaults[k]; drop
    any unknown key when `drop_unknown=True` (used for legacy
    fields like Replay's gamma/n_step that no longer exist)."""
    out: list[tuple[str, str]] = []
    for k, v in kvs:
        if k in defaults and v == defaults[k]:
            continue
        if drop_unknown and k not in defaults and k != 'inner':
            # 'inner' is a non-default field on WarmedUpdate; never drop.
            continue
        out.append((k, v))
    return out


def _rewrite_adam(s: str) -> str:
    """`dataclass:Adam(...)` or `partial(Claim:adam;...)` →
    `partial(Claim:adam;...)` with defaults elided."""
    body = None
    if s.startswith('dataclass:Adam('):
        body = s[len('dataclass:Adam('):-1]
    elif s.startswith('partial(Claim:adam;') and s.endswith(')'):
        body = s[len('partial(Claim:adam;'):-1]
    elif s == 'Claim:adam':
        return 'Claim:adam'  # nothing to elide
    if body is None:
        return s  # unknown form, leave alone
    kvs = _parse_kv_body(body)
    kvs = _strip_defaults(kvs, _ADAM_DEFAULTS)
    if not kvs:
        return 'Claim:adam'  # all kwargs at default → bare Claim
    return f'partial(Claim:adam;{_emit_kv_body(kvs)})'


def _rewrite_warmed_update(s: str) -> str:
    """`dataclass:WarmedUpdate(inner=I,warmup_steps=W)` or
    `partial(Claim:warmed_update;...)` →
    `partial(Claim:warmed_update;inner=I[,warmup_steps=W])`,
    recursing into `inner`."""
    body = None
    if s.startswith('dataclass:WarmedUpdate('):
        body = s[len('dataclass:WarmedUpdate('):-1]
    elif s.startswith('partial(Claim:warmed_update;') and s.endswith(')'):
        body = s[len('partial(Claim:warmed_update;'):-1]
    if body is None:
        return s
    kvs = _parse_kv_body(body)
    # Recurse into the `inner` value (which is itself an optimizer
    # canonical string).
    kvs = [
        (k, _rewrite_optimizer(v) if k == 'inner' else v)
        for k, v in kvs
    ]
    kvs = _strip_defaults(kvs, _WARMED_UPDATE_DEFAULTS)
    if not kvs:
        return 'Claim:warmed_update'
    return f'partial(Claim:warmed_update;{_emit_kv_body(kvs)})'


def _rewrite_optimizer(s: str) -> str:
    """Top-level optimizer canonical string. Dispatches between
    WarmedUpdate (wrapper) and Adam (inner) based on prefix."""
    if 'WarmedUpdate' in s or 'warmed_update' in s:
        return _rewrite_warmed_update(s)
    if 'Adam' in s or 'Claim:adam' in s:
        return _rewrite_adam(s)
    return s


def _rewrite_replay(s: str) -> str:
    """`dataclass:Replay(batch_size=B,capacity=C,sample=...,[gamma=,
    n_step=])` → `dataclass:Replay([batch_size=B][,capacity=C])`.
    Legacy `gamma` and `n_step` fields (no longer on Replay) are
    dropped — they migrated to `dqn_step`-level kwargs."""
    parsed = _parse_outer(s)
    if parsed is None or not parsed[0].startswith('dataclass:Replay'):
        return s
    body = parsed[1]
    kvs = _parse_kv_body(body)
    kvs = _strip_defaults(kvs, _REPLAY_DEFAULTS, drop_unknown=True)
    if not kvs:
        return 'dataclass:Replay()'
    return f'dataclass:Replay({_emit_kv_body(kvs)})'


def _rewrite_action_select_schedule(s: str) -> str:
    """`partial(Claim:linear_epsilon;eps_init=...,eps_final=...,
    anneal_steps=...)` → strip default-equal kwargs."""
    if not s.startswith('partial(Claim:linear_epsilon;') or not s.endswith(')'):
        return s
    body = s[len('partial(Claim:linear_epsilon;'):-1]
    kvs = _parse_kv_body(body)
    kvs = _strip_defaults(kvs, _LINEAR_EPSILON_DEFAULTS)
    if not kvs:
        return 'Claim:linear_epsilon'
    return f'partial(Claim:linear_epsilon;{_emit_kv_body(kvs)})'


def _rewrite_action_select(s: str) -> str:
    """`dataclass:EpsilonGreedy(schedule=S)` → strip schedule if
    it's at default; recurse into the schedule value."""
    parsed = _parse_outer(s)
    if parsed is None:
        # Could be `partial(Claim:epsilon_greedy;schedule=S)` or
        # bare `Claim:epsilon_greedy`.
        if s.startswith('partial(Claim:epsilon_greedy;') and s.endswith(')'):
            body = s[len('partial(Claim:epsilon_greedy;'):-1]
            kvs = _parse_kv_body(body)
            kvs = [
                (k, _rewrite_action_select_schedule(v) if k == 'schedule' else v)
                for k, v in kvs
            ]
            kvs = _strip_defaults(kvs, _EPSILON_GREEDY_DEFAULTS)
            if not kvs:
                return 'Claim:epsilon_greedy'
            return f'partial(Claim:epsilon_greedy;{_emit_kv_body(kvs)})'
        return s
    prefix, body = parsed
    if not prefix.startswith('dataclass:EpsilonGreedy'):
        return s
    kvs = _parse_kv_body(body)
    kvs = [
        (k, _rewrite_action_select_schedule(v) if k == 'schedule' else v)
        for k, v in kvs
    ]
    kvs = _strip_defaults(kvs, _EPSILON_GREEDY_DEFAULTS)
    if not kvs:
        return 'dataclass:EpsilonGreedy()'
    return f'dataclass:EpsilonGreedy({_emit_kv_body(kvs)})'


# Per-column rewriters. Substrate-aware: each column has known
# canonical-string semantics.
_COLUMN_REWRITERS: dict[str, callable] = {
    'optimizer': _rewrite_optimizer,
    'optimizer.inner': _rewrite_optimizer,
    'replay': _rewrite_replay,
    'action_select': _rewrite_action_select,
    'action_select.schedule': _rewrite_action_select_schedule,
}


def sanitize_cell_value(col: str, val: object) -> object:
    """Apply the per-column rewriter if registered; pass through
    otherwise. Returns the new value (or original if no rewrite)."""
    if val is None or not isinstance(val, str):
        return val
    rewriter = _COLUMN_REWRITERS.get(col)
    if rewriter is None:
        return val
    return rewriter(val)


def sanitize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Apply rewrites to each registered column in `df`. Returns a
    NEW DataFrame; doesn't mutate the input."""
    for col in _COLUMN_REWRITERS:
        if col not in df.columns:
            continue
        if df.schema[col] != pl.String:
            continue
        df = df.with_columns(
            pl.col(col).map_elements(
                lambda v, c=col: sanitize_cell_value(c, v),
                return_dtype=pl.String,
            ),
        )
    return df


def main(paths: list[Path]) -> None:
    """For each cache parquet, sanitize and overwrite. Backs up
    the original to `<path>.bak` first."""
    for path in paths:
        if not path.exists():
            print(f'[skip] {path} does not exist')
            continue
        backup = path.with_suffix(path.suffix + '.bak')
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f'[backup] {path} -> {backup}')
        df = pl.read_parquet(path)
        print(f'[load] {path} ({df.shape[0]} cells)')
        # Pre-counts for the high-impact cols.
        for col in ('optimizer.inner', 'replay'):
            if col in df.columns:
                n = df[col].n_unique()
                print(f'  pre   {col}: {n} distinct')
        new_df = sanitize_dataframe(df)
        for col in ('optimizer.inner', 'replay'):
            if col in new_df.columns:
                n = new_df[col].n_unique()
                print(f'  post  {col}: {n} distinct')
        new_df.write_parquet(path)
        print(f'[write] {path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: sanitize_cache_canonical.py <parquet> [<parquet> ...]')
        sys.exit(1)
    paths = [Path(p) for p in sys.argv[1:]]
    main(paths)
