"""Substrate test bootstrap.

Adds the workspace root to `sys.path` so substrate tests can
import `experiments.*` modules. The `experiments/` directory is
shared infrastructure for ad-hoc analysis scripts and bridge
authoring; it lives at the repo root, not inside either package.
The framework tests get this for free because pytest's rootdir
is the repo root when running from there; the substrate tests'
rootdir is the substrate package, so the workspace root has to
be added explicitly."""
from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))
