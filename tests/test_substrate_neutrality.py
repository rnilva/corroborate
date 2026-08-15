"""Architectural invariant: framework imports never pull in a
implementation module.

The whole point of splitting `corroborate.rl` into a sibling
`corroborate_rl` package is that the framework wheel installs
without JAX/optax/gymnax/etc., and that no `from corroborate_rl
import ...` ever survives in `src/corroborate/`. Pyright catches
type leaks; this test catches the dependency-graph leak — if a
framework module gains a `from corroborate_rl import ...` (even
guarded behind a feature flag or runtime branch), this test
fails immediately.

The test runs `import corroborate` in a *fresh subprocess* so the
parent test runner's already-loaded modules don't pollute
`sys.modules`. The forbidden set covers every dep that ought to
land in the implementation's pyproject, not the framework's."""
from __future__ import annotations

import subprocess
import sys


_FORBIDDEN_TOP_LEVEL_MODULES = (
    'jax',
    'jaxlib',
    'optax',
    'gymnax',
    'corroborate_rl',
)


def test_framework_import_is_substrate_free() -> None:
    """`import corroborate` (and walking its public surface) must
    not load any implementation module. A regression here means somebody
    re-introduced a `from corroborate_rl import ...` (or `import
    jax`) inside `src/corroborate/`."""
    code = (
        'import sys\n'
        'import corroborate  # noqa: F401\n'
        'import corroborate.analyses  # noqa: F401\n'
        'import corroborate.bridge  # noqa: F401\n'
        'import corroborate.corpus  # noqa: F401\n'
        'import corroborate.measurables  # noqa: F401\n'
        'import corroborate.runner  # noqa: F401\n'
        'import corroborate.stats  # noqa: F401\n'
        'forbidden = ' + repr(_FORBIDDEN_TOP_LEVEL_MODULES) + '\n'
        'leaked = sorted(\n'
        '    m for m in sys.modules\n'
        '    if any(m == f or m.startswith(f + ".") for f in forbidden)\n'
        ')\n'
        'if leaked:\n'
        '    print("LEAKED:" + ",".join(leaked))\n'
        '    sys.exit(2)\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'framework import pulled substrate modules.\n'
        f'stdout: {result.stdout}\nstderr: {result.stderr}'
    )
