"""Minimal scipy.stats stubs — typed facade for the surface
corroborate uses.

scipy ships no upstream type stubs; the framework's strict-Any
contract rejects the resulting `Unknown`/`Any` leaks. This stub
narrows just the calls the framework reaches: `spearmanr`,
`pearsonr`, frozen distributions `norm` and `t`. Real scipy.stats
returns `SignificanceResult` / `PearsonRResult` NamedTuples; we
declare them as `tuple[float, float]` since callers always
destructure to two floats and never read attributes.

Distribution methods (`cdf`, `pdf`, `ppf`, `sf`, `isf`) are
broadcasting in real scipy: scalar in → scalar out, array in →
array out. Overloads encode that so `ss.t.cdf(np.abs(t_stats),
df=...)` types as `npt.NDArray[np.float64]` while
`ss.t.cdf(scalar, df=...)` stays `float`.

Mirrors the gymnax / optax / statsmodels stubs in shape — narrow at
the boundary, only the surface that's actually used. Expand when a
new scipy.stats function is reached."""
from __future__ import annotations

from collections.abc import Sequence
from typing import overload

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
    an instance of `norm_gen` exposed at module scope. Methods
    broadcast: scalar in → scalar out, array in → array out."""
    @overload
    def cdf(self, x: float, /) -> float: ...
    @overload
    def cdf(
        self, x: npt.NDArray[np.float64], /,
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def pdf(self, x: float, /) -> float: ...
    @overload
    def pdf(
        self, x: npt.NDArray[np.float64], /,
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def ppf(self, q: float, /) -> float: ...
    @overload
    def ppf(
        self, q: npt.NDArray[np.float64], /,
    ) -> npt.NDArray[np.float64]: ...

    def sf(self, x: float, /) -> float: ...
    def isf(self, q: float, /) -> float: ...


norm: _Norm


class _T:
    """Frozen Student's-t distribution — `scipy.stats.t` exposed at
    module scope. Methods broadcast on x; `df` is scalar (int or
    float) for corroborate's call sites."""
    @overload
    def cdf(self, x: float, /, df: float | int) -> float: ...
    @overload
    def cdf(
        self, x: npt.NDArray[np.float64], /, df: float | int,
    ) -> npt.NDArray[np.float64]: ...

    def pdf(self, x: float, /, df: float | int) -> float: ...
    def ppf(self, q: float, /, df: float | int) -> float: ...
    def sf(self, x: float, /, df: float | int) -> float: ...


t: _T
