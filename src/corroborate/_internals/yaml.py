"""YAML boundary — laundering point for `yaml.safe_load`.

`yaml.safe_load` is typed `Any` in PyYAML's stubs (the YAML spec
admits the closed union `None | bool | int | float | str | list |
dict`, which IS the framework's `object`). Same shape as
`_json_boundary` — single laundering point that narrows the
return to `object` so downstream `is_*`/`require_*` predicates do
the typed work.

PyYAML is an optional framework dep (`pip install corroborate[yaml]`).
The import is therefore lazy: `safe_load` raises a helpful
`ImportError` if the extra wasn't installed, and module-import
itself doesn't fail. Module name is underscore-prefixed to signal
**internal use only**. External users should `import yaml`
directly."""
from __future__ import annotations

from typing import IO


def safe_load(stream: IO[str] | str) -> object:
    """Decode a YAML document to `object`. Mirrors
    `_json_boundary.loads` — narrows PyYAML's `Any` return to
    `object` at the boundary; downstream callers narrow further via
    TypeIs predicates / isinstance checks."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "YAML config loading requires the optional `yaml` extra. "
            "Install it with `pip install 'corroborate[yaml]'` "
            "(or add `pyyaml` to your environment)."
        ) from e
    raw = yaml.safe_load(stream)  # pyright: ignore[reportAny]
    return raw  # pyright: ignore[reportAny]
