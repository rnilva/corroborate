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

from corroborate.bridge.analysis import analysis
from corroborate.analyses._dowhy_internal import (
    DAGLike,
    _backdoor_estimate,
    _build_causal_model,
    _refuter_effect,
)


if TYPE_CHECKING:
    pass


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
    ATE under the refuter's perturbation; `drift` is
    `|refuted_ate - real_ate|`.

    **Field to gate on depends on the refuter** — `drift` is
    NOT a uniform robustness signal:
    - **Random-common-cause** (`random_common_cause_refutation`):
      a robust estimate is invariant to the synthetic confounder,
      so `refuted_ate ≈ real_ate` and `drift ≈ 0`. Bridge gate:
      `drift < tolerance` → robust.
    - **Placebo** (`placebo_refutation`): a randomized treatment
      should yield no effect, so `refuted_ate ≈ 0` and
      `drift ≈ |real_ate|`. Bridge gate: `abs(refuted_ate) <
      tolerance` → robust. The framework intentionally exposes
      both fields so each refuter is gated on the one that reads
      "robust" in its own semantics; do NOT gate placebo on
      `drift`."""
    real_ate: float
    refuted_ate: float
    drift: float
    method_name: str
    refuter_name: str
    treatment: str
    outcome: str
    n_rows: int


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
    ate_val = getattr(estimate, 'value')
    return BackdoorResult(
        ate=float(ate_val),
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
    real_ate = float(getattr(estimate, 'value'))

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
    (random permutation). A robust estimate yields
    `refuted_ate ≈ 0` (the placebo has no causal channel to the
    outcome). Bridge gate: `abs(result.refuted_ate) < tolerance`
    → HELD; do NOT gate on `drift` here (drift ≈ |real_ate|
    when the model is correct, so a `drift < tolerance` gate
    would only fire when the real ATE is itself near zero, which
    is the opposite of robustness). See `RefutationResult`'s
    docstring for the per-refuter gate convention."""
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
    of treatment and outcome. A robust estimate is invariant to
    the synthetic confounder: `refuted_ate ≈ real_ate` and
    `drift ≈ 0`. Bridge gate: `result.drift < tolerance` → HELD."""
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
