"""Static checks: existing Result classes satisfy the structural
protocols in `analyses/_result_protocols.py`.

Each `_check_paired_*` / `_check_stratified_*` function is a
*pyright-only* assertion: it does nothing at runtime, but pyright
fails if the concrete Result no longer structurally matches the
protocol (e.g., a field was renamed without updating the
protocol). The test functions are imported by `pytest` so they
also run; runtime they're no-ops.

Coverage is representative, not exhaustive — one or two members
per protocol family is enough to catch regressions; new Result
classes opt in by adding a `_check_*` here.
"""
from __future__ import annotations

from corroborate.analyses._result_protocols import (
    PairedEffectResult, StratifiedResult,
)
from corroborate.analyses.paired.paired_g import PairedGResult
from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)


def _check_paired_g(r: PairedGResult) -> PairedEffectResult:
    return r


def _check_stratified_arm_diff_pooled(
    r: StratifiedArmDiffPooledResult,
) -> StratifiedResult:
    return r


def test_protocols_importable() -> None:
    """Sanity test — protocol module imports without error and
    the static-check helpers above type-check (pyright enforces).
    """
    assert _check_paired_g is not None
    assert _check_stratified_arm_diff_pooled is not None
