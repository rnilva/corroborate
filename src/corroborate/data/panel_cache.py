"""Persist the facts a DataFrame cannot carry beside a Panel cache.

This is cache metadata, not an experiment record format. Binding it to the
parquet bytes prevents an overwritten legacy cache from acquiring an old
configuration registry or scope by accident.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import polars as pl

from corroborate._internals.json import loads
from corroborate._internals.narrow import is_list_of_object, is_mapping_str_object
from corroborate.corpus.persistence import atomic_write_text
from corroborate.data.panel import Panel


def metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix('.panel.json')


def _digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_metadata(panel: Panel, cache_path: Path) -> None:
    """Write after parquet and sources, so interrupted writes fail closed."""
    atomic_write_text(metadata_path(cache_path), json.dumps({
        'version': 1,
        'parquet_sha256': _digest(cache_path),
        'sources_sha256': (_digest(cache_path.with_suffix('.sources.json'))
                           if cache_path.with_suffix('.sources.json').exists() else None),
        'leaves': None if panel.leaves is None else sorted(panel.leaves),
        'scope_chain': [expr.meta.serialize(format='json') for expr in panel.scope_chain],
        'stratify_by': list(panel.stratify_by),
        'required_measurables': sorted(panel.required_measurables),
    }, sort_keys=True, indent=2))


def _strings(value: object, field: str) -> tuple[str, ...]:
    # JSON is intrinsically polymorphic only at this boundary; callers
    # immediately recover the concrete tuple contract.
    if not is_list_of_object(value) or not all(isinstance(item, str) for item in value):
        raise ValueError(f'Invalid Panel cache metadata field: {field}')
    return tuple(item for item in value if isinstance(item, str))


def restore_metadata(panel: Panel, cache_path: Path) -> Panel:
    """Old caches retain their historical inference; explicit registries do not.

    In particular, a persisted unknown registry stays unknown even if the
    current process happens to have additional measurables registered.
    """
    path = metadata_path(cache_path)
    if not path.exists():
        return panel
    parsed = loads(path.read_text())
    if not is_mapping_str_object(parsed) or parsed.get('version') != 1:
        raise ValueError(f'Invalid Panel cache metadata: {path}')
    if parsed.get('parquet_sha256') != _digest(cache_path):
        raise ValueError(f'Panel cache metadata does not match {cache_path}; rebuild the cache')
    sources = cache_path.with_suffix('.sources.json')
    if parsed.get('sources_sha256') != (_digest(sources) if sources.exists() else None):
        raise ValueError(f'Panel cache provenance does not match {sources}; rebuild the cache')
    if 'leaves' not in parsed:
        raise ValueError(f'Panel cache metadata has no configuration registry: {path}')
    raw_leaves = parsed['leaves']
    leaves = None if raw_leaves is None else frozenset(_strings(raw_leaves, 'leaves'))
    scopes = _strings(parsed.get('scope_chain'), 'scope_chain')
    return replace(
        panel,
        leaves=leaves,
        scope_chain=tuple(pl.Expr.deserialize(io.StringIO(expr), format='json') for expr in scopes),
        stratify_by=_strings(parsed.get('stratify_by'), 'stratify_by'),
        required_measurables=frozenset(_strings(parsed.get('required_measurables'), 'required_measurables')),
    )
