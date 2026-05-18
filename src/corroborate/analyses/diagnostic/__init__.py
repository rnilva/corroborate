"""Diagnostic analyses — audits + verdict-distribution summaries
that bridges call to introspect verdicts at the hypothesis level
rather than to test a substrate claim.
"""
from corroborate.analyses.diagnostic.tautology_audit import (  # noqa: F401
    tautology_audit as _audit,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.diagnostic.verdict_distribution import (  # noqa: F401
    verdict_distribution_per_env as _verdict_dist,  # pyright: ignore[reportUnusedImport]
)
