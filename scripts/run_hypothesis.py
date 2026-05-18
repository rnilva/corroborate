"""Back-compat shim — forwards to `corroborate.cli.hypothesis.main`.

The real CLI surface moved to `corroborate.cli.hypothesis` so it
can be invoked both as a subcommand of the top-level CLI
(`corroborate hypothesis ...`) and as a standalone module
(`python -m corroborate.cli.hypothesis ...`). This script is
retained for existing call sites of the form
`python scripts/run_hypothesis.py ...`.

Prefer the top-level CLI for new invocations:

    corroborate hypothesis experiments.findings.ddqn --ingest-all experiments/data/
"""
from __future__ import annotations

import sys

from corroborate.cli.hypothesis import main


if __name__ == '__main__':
    sys.exit(main())
