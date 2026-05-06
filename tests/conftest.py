"""pytest configuration shared across all tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_runner_report_warnings() -> None:
    """Clear `corroborate.runner.report._COERCE_WARNINGS_EMITTED`
    between tests so test order doesn't determine which warnings
    fire (a warning emitted by an earlier test would silence the
    same code path in a later test, breaking warning-coverage
    assertions like `test_property_that_raises_yields_null_and_warns`).

    Autouse: applies to every test. Cheap (set.clear() on a set
    that's almost always empty)."""
    from corroborate.runner.report import _reset_warnings
    _reset_warnings()
