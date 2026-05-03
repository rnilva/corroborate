"""Minimal statsmodels.stats.power stub — `TTestPower` is the only
surface corroborate reaches. Used by `statistics.py` for noncentral-
t aware MDE / power solving."""
from __future__ import annotations

from typing import Literal


_Alternative = Literal['two-sided', 'larger', 'smaller']


class TTestPower:
    """Solver for the paired-t / one-sample-t power equation.

    Real statsmodels has the four-parameter `solve_power` (any one
    of effect_size / nobs / alpha / power may be `None`, the rest
    must be supplied; the solver inverts on the missing parameter).
    We declare the loose union signature."""

    def solve_power(
        self,
        effect_size: float | None = ...,
        nobs: float | int | None = ...,
        alpha: float | None = ...,
        power: float | None = ...,
        alternative: _Alternative = ...,
    ) -> float: ...
