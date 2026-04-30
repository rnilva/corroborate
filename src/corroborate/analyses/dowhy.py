"""DoWhy backdoor + refutations as framework analyses.

Three analyses, all consuming a corpus of cells (each a flat
path-keyed mapping with scalar fields for the treatment,
outcome, and adjustment-set covariates) plus a DAG:

- `backdoor_ate(cells, *, treatment, outcome, dag, method_name)`
  → `BackdoorResult` carrying `(ate, identified, estimand_str)`.
- `placebo_refutation(cells, *, treatment, outcome, dag,
  method_name)` → `RefutationResult` carrying `(real_ate,
  refuted_ate, drift)`.
- `random_common_cause_refutation(...)` → same shape.

Each analysis is independent — re-builds the DoWhy CausalModel
internally — so a bridge that consumes only one doesn't pay the
cost of the other two. A bridge that asserts "ATE positive AND
robust to placebo AND robust to RCC" consumes all three as
separate fixtures, which the framework resolves by name and
injects into the bridge's `holds_when` body.

Reproduces FINDINGS.md fourth revision shape:
  ATE = +8.82 / SCV unit            HELD
  placebo ATE / real = 1.4%          HELD
  RCC drift = 0.0075                 HELD
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from corroborate.analysis import analysis
# Private helpers from bridges_dowhy are reused here while the
# legacy bridges module still exists; cross-module private use
# is intentional during the migration. Once bridges_dowhy retires
# these helpers move to a shared `_dowhy_internal` location.
from corroborate.bridges_dowhy import (  # noqa
    DAGLike,
    _build_causal_model,  # pyright: ignore[reportPrivateUsage]
    _record_keys_for,  # pyright: ignore[reportPrivateUsage]
    _refuter_effect,  # pyright: ignore[reportPrivateUsage]
)


if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class BackdoorResult:
    """Output of `backdoor_ate`: identified ATE + bookkeeping.

    `identified` is False when the DAG admits no backdoor /
    frontdoor / IV adjustment for the (treatment, outcome) pair —
    the bridge `holds_when` should map to POWER_INSUFFICIENT in
    that case rather than NO_EFFECT (Pearl's tier-2 verdict
    requires identification first)."""
    ate: float
    identified: bool
    estimand_str: str
    method_name: str
    treatment: str
    outcome: str
    n_rows: int


@dataclass(frozen=True, slots=True)
class RefutationResult:
    """Output of a refutation analysis (placebo or random-common-
    cause). `real_ate` is the unrefuted ATE; `refuted_ate` is the
    ATE under the refuter's perturbation; `drift` is `|refuted -
    real|`. A bridge typically asserts `drift < tolerance`."""
    real_ate: float
    refuted_ate: float
    drift: float
    method_name: str
    refuter_name: str
    treatment: str
    outcome: str
    n_rows: int


def _cells_to_dataframe(
    cells: Iterable[Mapping[str, object]],
    keys: list[str],
) -> 'pd.DataFrame':
    """Project the cell collection to a pandas DataFrame: one
    row per cell, columns = `keys`. Cells missing any required
    key are skipped (so partial corpora don't crash). Non-scalar
    values are skipped."""
    import pandas as pd

    rows: list[dict[str, float]] = []
    for cell in cells:
        row: dict[str, float] = {}
        complete = True
        for k in keys:
            v = cell.get(k)
            if isinstance(v, bool):
                row[k] = float(v)
            elif isinstance(v, (int, float)):
                row[k] = float(v)
            else:
                complete = False
                break
        if complete:
            rows.append(row)
    return pd.DataFrame(rows)


def _backdoor_estimate(
    cells: Iterable[Mapping[str, object]],
    treatment: str,
    outcome: str,
    dag: DAGLike,
    method_name: str,
) -> tuple[
    'pd.DataFrame',
    object,  # identified estimand
    object | None,  # estimate object (None if unidentified)
]:
    """Build DataFrame + CausalModel + run identification +
    (when identified) estimation. Helper shared by all three
    analyses so the model construction is consistent."""
    df = _cells_to_dataframe(cells, _record_keys_for(dag))
    model = _build_causal_model(df, treatment, outcome, dag)
    identified = model.identify_effect(
        proceed_when_unidentifiable=False,
    )
    if (
        getattr(identified, 'no_directed_path', False)
        or not getattr(identified, 'estimands', None)
    ):
        return df, identified, None
    estimate = model.estimate_effect(
        identified, method_name=method_name,
    )
    return df, identified, estimate


@analysis
def backdoor_ate(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment: str,
    outcome: str,
    dag: DAGLike,
    method_name: str = 'backdoor.linear_regression',
) -> BackdoorResult:
    """Identify + estimate the ATE of `treatment` on `outcome`
    under `dag`. Returns identified=False when the DAG admits
    no admissible adjustment."""
    cells_list = list(cells)
    df, identified, estimate = _backdoor_estimate(
        cells_list, treatment, outcome, dag, method_name,
    )
    if estimate is None:
        return BackdoorResult(
            ate=float('nan'),
            identified=False,
            estimand_str=str(identified),
            method_name=method_name,
            treatment=treatment,
            outcome=outcome,
            n_rows=len(df),
        )
    # `estimate.value` is `float | numpy scalar` per DoWhy stubs.
    ate_val = getattr(estimate, 'value')  # pyright: ignore[reportAny]
    return BackdoorResult(
        ate=float(ate_val),  # pyright: ignore[reportAny]
        identified=True,
        estimand_str=str(identified),
        method_name=method_name,
        treatment=treatment,
        outcome=outcome,
        n_rows=len(df),
    )


def _run_refuter(
    cells: Iterable[Mapping[str, object]],
    treatment: str,
    outcome: str,
    dag: DAGLike,
    method_name: str,
    refuter_method: str,
) -> RefutationResult:
    """Generic refutation runner: identifies, estimates the real
    ATE, then runs the named refuter. `refuter_method` is the
    DoWhy refuter name (e.g. `'placebo_treatment_refuter'`)."""
    cells_list = list(cells)
    df, identified, estimate = _backdoor_estimate(
        cells_list, treatment, outcome, dag, method_name,
    )
    if estimate is None:
        return RefutationResult(
            real_ate=float('nan'),
            refuted_ate=float('nan'),
            drift=float('nan'),
            method_name=method_name,
            refuter_name=refuter_method,
            treatment=treatment,
            outcome=outcome,
            n_rows=len(df),
        )
    real_ate = float(getattr(estimate, 'value'))  # pyright: ignore[reportAny]

    # `estimate` and `identified` are dynamic DoWhy objects; the
    # refute_estimate API takes both.
    model = _build_causal_model(df, treatment, outcome, dag)
    refuter = model.refute_estimate(
        identified, estimate, method_name=refuter_method,
    )
    refuted_ate = _refuter_effect(refuter)
    return RefutationResult(
        real_ate=real_ate,
        refuted_ate=refuted_ate,
        drift=abs(refuted_ate - real_ate),
        method_name=method_name,
        refuter_name=refuter_method,
        treatment=treatment,
        outcome=outcome,
        n_rows=len(df),
    )


@analysis
def placebo_refutation(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment: str,
    outcome: str,
    dag: DAGLike,
    method_name: str = 'backdoor.linear_regression',
) -> RefutationResult:
    """Refute the ATE by replacing the treatment with a placebo
    (random permutation). The real ATE should not survive — a
    HELD bridge requires placebo drift below tolerance."""
    return _run_refuter(
        cells, treatment, outcome, dag, method_name,
        refuter_method='placebo_treatment_refuter',
    )


@analysis
def random_common_cause_refutation(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment: str,
    outcome: str,
    dag: DAGLike,
    method_name: str = 'backdoor.linear_regression',
) -> RefutationResult:
    """Refute the ATE by adding a random synthetic common cause
    of treatment and outcome. The real ATE should be robust — a
    HELD bridge requires drift below tolerance."""
    return _run_refuter(
        cells, treatment, outcome, dag, method_name,
        refuter_method='random_common_cause',
    )


__all__ = [
    'BackdoorResult',
    'RefutationResult',
    'backdoor_ate',
    'placebo_refutation',
    'random_common_cause_refutation',
]
