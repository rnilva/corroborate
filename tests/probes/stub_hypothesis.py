"""Minimal on-disk Hypothesis for CLI-level tests.

CLI tests (`tests/test_cli_preflight_gating.py`) drive
`corroborate.cli.hypothesis.main` with a real dotted module path —
`importlib.import_module` needs an importable module, so a
`types.ModuleType` surrogate (the pattern unit tests use) doesn't
suffice. This module satisfies `_validate_hypothesis`'s Protocol
surface with the smallest possible shape: a synthetic two-arm
intervention, zero bridges, zero findings.
"""
from __future__ import annotations

from corroborate.core.claim import claim
from corroborate.core.intervention import DoEffect, Intervention


@claim
def _treatment_op(x: int) -> int:
    return x


@claim
def _baseline_op(x: int) -> int:
    return x


INTERVENTION = DoEffect(
    arms=(
        (Intervention(slot_path='op', replacement=_baseline_op),),
        (Intervention(slot_path='op', replacement=_treatment_op),),
    ),
)
BRIDGES: tuple[()] = ()
FINDINGS: tuple[()] = ()
