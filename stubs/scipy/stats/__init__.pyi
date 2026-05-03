"""Minimal scipy.stats stubs — typed facade for the surface
corroborate uses.

scipy ships no upstream type stubs; the framework's strict-Any
contract rejects the resulting `Unknown`/`Any` leaks. This stub
narrows just the calls the framework reaches: `spearmanr`,
`pearsonr`, frozen distributions `norm` and `t`. Real scipy.stats
returns `SignificanceResult` / `PearsonRResult` NamedTuples; we
declare them as `tuple[float, float]` since callers always
destructure to two floats and never read attributes.

Mirrors the gymnax / optax / statsmodels stubs in shape — narrow at
the boundary, only the surface that's actually used. Expand when a
new scipy.stats function is reached."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


# Real scipy accepts any array-like; narrowing to the two shapes
# the framework actually passes (a numpy float-array, or a Python
# `list[float]` / `Sequence[float]`).
_ArrayLikeFloat = Sequence[float] | npt.NDArray[np.float64]


def spearmanr(
    a: _ArrayLikeFloat,
    b: _ArrayLikeFloat, /,
) -> tuple[float, float]: ...


def pearsonr(
    x: _ArrayLikeFloat,
    y: _ArrayLikeFloat, /,
) -> tuple[float, float]: ...


class _Norm:
    """Frozen standard-normal distribution — `scipy.stats.norm` is
    an instance of `norm_gen` exposed at module scope. Callers use
    `cdf`, `pdf`, `ppf` (and rarely `sf` / `isf`); declare those."""
    def cdf(self, x: float, /) -> float: ...
    def pdf(self, x: float, /) -> float: ...
    def ppf(self, q: float, /) -> float: ...
    def sf(self, x: float, /) -> float: ...
    def isf(self, q: float, /) -> float: ...


norm: _Norm


class _T:
    """Frozen Student's-t distribution — `scipy.stats.t` exposed at
    module scope. Callers pass the degrees of freedom via `df=`."""
    def cdf(self, x: float, /, df: float | int) -> float: ...
    def pdf(self, x: float, /, df: float | int) -> float: ...
    def ppf(self, q: float, /, df: float | int) -> float: ...
    def sf(self, x: float, /, df: float | int) -> float: ...


t: _T
