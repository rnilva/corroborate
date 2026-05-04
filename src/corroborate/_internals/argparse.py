"""Argparse boundary — typed wrapper around argparse.Namespace.

`argparse.Namespace.__getattr__` is typed `Any` in typeshed
(Namespace's attribute set is defined dynamically by the parser
configuration; typeshed can't statically know the dest names).
This module narrows the parsed namespace's contents to a typed
`Mapping[str, object]` once at the boundary; downstream callers
narrow each dest via `_narrow.require_*` predicates without
touching `Any`.

Same shape as `_json_boundary.py` / `_polars_boundary.py`.

Module name is underscore-prefixed to signal **internal use
only**. External users should import argparse directly."""
from __future__ import annotations

import argparse
from collections.abc import Mapping


def to_mapping(ns: argparse.Namespace) -> Mapping[str, object]:
    """Decode an argparse.Namespace to a `Mapping[str, object]`.
    Single laundering point. Downstream callers narrow each dest
    via `_narrow.require_*` predicates without touching `Any`."""
    return vars(ns)
