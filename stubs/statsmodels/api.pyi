"""Minimal statsmodels.api stubs — only what corroborate's
`causal_discovery` module needs (`OLS`, `add_constant`).

statsmodels has no upstream stubs. The framework's strict-Any
contract rejects the resulting `Any` leaks (`OLS().fit().resid`
returned `Any` before this stub).

Mirrors the gymnax / optax / scipy stubs — narrow at the boundary,
only the surface that's actually used. Expand when a new
statsmodels primitive is reached."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def add_constant(
    arr: npt.NDArray[np.float64], /,
    prepend: bool = ...,
    has_constant: str = ...,
) -> npt.NDArray[np.float64]: ...


class _RegressionResults:
    """Subset of `statsmodels.regression.linear_model.RegressionResults`
    that corroborate consumes. Only `.resid` is read — fitted-value
    residuals as a numpy array."""
    @property
    def resid(self) -> npt.NDArray[np.float64]: ...


class OLS:
    """`statsmodels.api.OLS(endog, exog)` — ordinary least squares.
    Constructor stores arrays; `fit()` returns the regression
    result. Only the residual attribute is read downstream."""
    def __init__(
        self,
        endog: npt.NDArray[np.float64],
        exog: npt.NDArray[np.float64],
        /,
    ) -> None: ...
    def fit(self) -> _RegressionResults: ...
