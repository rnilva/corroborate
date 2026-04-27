"""JSON boundary — the framework's ONE Any-laundering point.

`json.loads` is typed as `Any` in typeshed (typeshed limitation;
JSON values are genuinely polymorphic — the closed union
None | bool | int | float | str | list | dict IS the framework's
`object`). basedpyright's `reportAny` flags the Any source.

This module wraps `json.loads` once, narrows the return to
`object` via two scoped `pyright: ignore[reportAny]` comments
(source line + return line). Both are justified per CLAUDE.md's
"last-resort with rationale" clause: polymorphism over JSON's
return type, sourced from stdlib whose stub annotation we cannot
tighten. Bounding the scope to this file means any Any leaks
elsewhere in the framework are still caught.

Module name is underscore-prefixed to signal **internal use
only**. External users should `import json` directly."""
from __future__ import annotations

import json


def loads(s: str) -> object:
    """Decode a JSON string to `object`. Downstream callers narrow
    further via TypeIs predicates / isinstance checks."""
    raw = json.loads(s)  # pyright: ignore[reportAny]
    return raw  # pyright: ignore[reportAny]
