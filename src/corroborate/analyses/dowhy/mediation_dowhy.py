"""DoWhy mediation decomposition — typed linearity-broken diagnostic.

Two-stage approach (the v10 PoC `analysis_dowhy_mediation.py`
shape, ported forward with current framework conventions):

1. **Total ATE** — backdoor identification + estimation of
   `treatment → outcome` under `dag`. Reuses
   `corroborate.analyses.dowhy.backdoor_estimate`.
2. **Direct ATE** — OLS of `outcome` on `(treatment, *mediators)`;
   the coefficient on `treatment` is the controlled direct
   effect (what's left after holding mediators fixed). This is
   the natural-direct-effect under DAG linearity + no
   exposure-mediator interaction.
3. **Indirect ATE = total − direct**; **indirect proportion =
   indirect / total** (NaN-guarded when |total| < `eps`).

For multi-mediator chains, all mediators enter Stage 2's OLS
simultaneously; the result is the controlled direct effect
holding ALL mediators fixed (jointly).

## What this primitive is FOR

This primitive's role is **diagnostic, not magnitude estimator**.
CLAUDE.md §"Mediation recipe" prescribes `partial_spearman` as
the canonical mediation primitive (rank-based, multicollinearity-
robust, bounded). The empirical failure mode of reading
mediation_dowhy magnitudes WITHOUT prior gating is documented in
detail below.

This primitive's contribution is the typed `linearity_status`
field on the result. Bridges consume it to ask:

  > "Does the linear-mediation assumption HOLD on this corpus?"

The four-valued enum surfaces the failure modes that the v10
lesson identified:

- `RELIABLE`        — direct/total same sign + proportion ∈ [0, 1].
                      Linear decomposition gives a coherent answer
                      that AGREES with the rank-based form.
- `SIGN_FLIPPED`    — direct/total opposite signs. Classic
                      multicollinearity artifact — OLS coefficients
                      fight each other; direct ATE flips relative
                      to total. Linear assumption broken.
- `OUT_OF_BOUNDS`   — proportion < 0 or > 1 with same-signed
                      direct/total. Suppression: the mediator
                      transmits a same-sign-but-overshooting path.
                      Linear assumption broken.
- `UNIDENTIFIED`    — DAG admits no backdoor adjustment for
                      `(treatment, outcome)`.
- `POWER_INSUFFICIENT` — |total| < eps, or OLS rank-deficient.

A bridge asserting "mediation linearity holds at this scope"
checks `linearity_status == RELIABLE`. A bridge documenting the
methodological lesson ("on this corpus, linearity is broken")
checks `linearity_status in {SIGN_FLIPPED, OUT_OF_BOUNDS}`.

## EMPIRICAL EXAMPLE OF THE FAILURE MODE

The v10 lesson reproduced on a real corpus: FR × MLP × unshaped ×
baseline (n=120) on the corroborate-rl DDQN sweep, treatment =
γ ∈ {0.99, 0.999}, outcome = jensen_gap, mediators = {self_ref,
σ_action}:

    Total ATE    = +1023.36   (well-powered; refutations pass)
    Direct ATE   = −57.31     (NEGATIVE — sign-flipped vs total!)
    Indirect ATE = +1080.67
    Indirect %   = +105.6%    (> 100% — outside [0, 1])
    linearity_status = SIGN_FLIPPED

The three variables (γ, self_ref, σ_action) are pairwise
correlated 0.78–0.93 at this scope, so OLS coefficients fight
each other and γ's residual coefficient lands at −57 (regression
artifact, not mechanism). Rank-based partial Spearman on the
same data gives `ρ(γ, jens | self_ref, σ_action) = +0.06 NS` — a
clean "chain closes" result. The `linearity_status =
SIGN_FLIPPED` IS the diagnostic flagging that the magnitudes
should not be read at face value.

## Recipe

Bridges that DO want magnitudes should chain:
1. Power-gate the TOTAL ATE first via `dowhy.placebo_refutation`
   + `random_common_cause_refutation`. If POWER_INSUFFICIENT,
   stop.
2. PC-topology check via
   `corroborate.graph.discovery.discover_adjacency`. If PC
   doesn't support the mediator's position on the path, the
   posited DAG is suspect.
3. Run `mediation_dowhy`. Read `linearity_status` BEFORE reading
   magnitudes. RELIABLE → magnitudes are coherent. Anything else
   → use `partial_spearman` for the canonical mediation answer.
4. Refute the total ATE.

`proportion_mediated` (deleted 2026-05-18) was the v9 form of
this primitive; CLAUDE.md's statistical case for its deprecation
is documented in the canonical-analyses table notes.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from corroborate._internals.polars import as_rows
from corroborate.analyses._dowhy_internal import (
    DAGLike,
    backdoor_estimate,
    cells_to_dataframe,
)
from corroborate.bridge.analysis import analysis


class LinearityStatus(StrEnum):
    """Typed verdict on whether the linear-mediation assumption is
    defensible on this corpus. See module docstring §"What this
    primitive is FOR" for the per-status semantics + the FR γ-WHY
    empirical example of `SIGN_FLIPPED`."""
    RELIABLE = 'reliable'
    SIGN_FLIPPED = 'sign_flipped'
    OUT_OF_BOUNDS = 'out_of_bounds'
    UNIDENTIFIED = 'unidentified'
    POWER_INSUFFICIENT = 'power_insufficient'


@dataclass(frozen=True, slots=True)
class MediationResult:
    """Output of `mediation_dowhy`: total, direct, indirect ATE +
    indirect proportion + typed `linearity_status`.

    **Bridges should read `linearity_status` BEFORE reading
    magnitudes** (see module docstring §"What this primitive is
    FOR"). RELIABLE means direct/total agree on sign AND
    proportion ∈ [0, 1] — the linear decomposition's coherent
    range. SIGN_FLIPPED / OUT_OF_BOUNDS mean the linear
    assumption is broken on this corpus; use `partial_spearman`
    for the canonical mediation answer.

    `identified` mirrors `linearity_status == UNIDENTIFIED` for
    backward compatibility with consumers that read the bool
    directly. The enum is the modern surface."""
    total_ate: float
    direct_ate: float
    indirect_ate: float
    indirect_proportion: float
    identified: bool
    linearity_status: LinearityStatus
    treatment: str
    outcome: str
    mediators: tuple[str, ...]
    method_name: str
    n_rows: int


def _classify_linearity(
    *, total_ate: float, direct_ate: float,
    indirect_proportion: float,
    identified: bool, eps: float,
) -> LinearityStatus:
    """Status classifier — pure function of the computed
    magnitudes. The branches encode the v10 lesson's failure
    taxonomy (module docstring §"What this primitive is FOR")."""
    if not identified:
        return LinearityStatus.UNIDENTIFIED
    if math.isnan(direct_ate) or math.isnan(total_ate):
        return LinearityStatus.POWER_INSUFFICIENT
    if abs(total_ate) < eps:
        return LinearityStatus.POWER_INSUFFICIENT
    if math.isnan(indirect_proportion):
        return LinearityStatus.POWER_INSUFFICIENT
    sign_total = 1 if total_ate > 0 else (-1 if total_ate < 0 else 0)
    sign_direct = 1 if direct_ate > 0 else (-1 if direct_ate < 0 else 0)
    # Treat exactly-zero direct as same-sign (degenerate but in-range)
    if sign_direct != 0 and sign_direct != sign_total:
        return LinearityStatus.SIGN_FLIPPED
    if indirect_proportion < 0.0 or indirect_proportion > 1.0:
        return LinearityStatus.OUT_OF_BOUNDS
    return LinearityStatus.RELIABLE


@analysis
def mediation_dowhy(
    cells: pl.DataFrame | Iterable[Mapping[str, object]],
    *,
    treatment: str,
    outcome: str,
    mediators: tuple[str, ...],
    dag: DAGLike,
    method_name: str = 'backdoor.linear_regression',
    eps: float = 1e-8,
) -> MediationResult:
    """Two-stage mediation decomposition with typed
    `linearity_status` diagnostic.

    `mediators` is a non-empty tuple of column names; all enter
    Stage 2's OLS jointly. The DAG SHOULD include the mediators
    as nodes on the `treatment → mediator → outcome` path for
    Stage 1's backdoor identification to admit the path-aware
    adjustment set.

    Reads `cells` and projects (treatment, outcome, *mediators)
    columns; cells missing any required key are skipped.

    Returns `MediationResult` with `linearity_status` set to one
    of {RELIABLE, SIGN_FLIPPED, OUT_OF_BOUNDS, UNIDENTIFIED,
    POWER_INSUFFICIENT}; magnitudes are NaN where the DAG
    doesn't identify the total ATE or the DataFrame is too sparse
    for OLS. Pair with `dowhy.placebo_refutation` +
    `random_common_cause_refutation` on the same total-ATE
    arguments to refute the foundation."""
    cells = as_rows(cells)
    if not mediators:
        raise ValueError(
            'mediation_dowhy: `mediators` must be a non-empty tuple of '
            'column names; use `dowhy.backdoor_ate` for total-only.',
        )

    # Early empty-scope guard: bridge scopes that admit zero cells
    # would otherwise crash `build_causal_model`'s column-presence
    # check on an empty DataFrame. Return POWER_INSUFFICIENT
    # cleanly so the cluster Finding's verdict aggregates as
    # UNDERPOWERED rather than the bridge erroring.
    cells_list = list(cells)
    if not cells_list:
        return MediationResult(
            total_ate=float('nan'), direct_ate=float('nan'),
            indirect_ate=float('nan'), indirect_proportion=float('nan'),
            identified=False,
            linearity_status=LinearityStatus.POWER_INSUFFICIENT,
            treatment=treatment, outcome=outcome,
            mediators=mediators, method_name=method_name, n_rows=0,
        )

    # Stage 1: total ATE via backdoor
    df, _identified_estimand, estimate = backdoor_estimate(
        cells_list, treatment, outcome, dag, method_name,
    )
    n_rows = len(df)
    if estimate is None:
        return MediationResult(
            total_ate=float('nan'), direct_ate=float('nan'),
            indirect_ate=float('nan'), indirect_proportion=float('nan'),
            identified=False,
            linearity_status=LinearityStatus.UNIDENTIFIED,
            treatment=treatment, outcome=outcome,
            mediators=mediators, method_name=method_name, n_rows=n_rows,
        )
    total_ate = float(getattr(estimate, 'value'))

    # Stage 2: direct ATE via OLS with mediators as covariates.
    needed = [treatment, outcome, *mediators]
    df_med = cells_to_dataframe(cells_list, needed)
    if df_med.empty or len(df_med) < 3:
        return MediationResult(
            total_ate=total_ate, direct_ate=float('nan'),
            indirect_ate=float('nan'), indirect_proportion=float('nan'),
            identified=True,
            linearity_status=LinearityStatus.POWER_INSUFFICIENT,
            treatment=treatment, outcome=outcome,
            mediators=mediators, method_name=method_name, n_rows=n_rows,
        )
    import numpy as np
    from sklearn.linear_model import LinearRegression
    feature_cols = [treatment, *mediators]
    X = df_med[feature_cols].values
    y = df_med[outcome].values
    try:
        reg = LinearRegression().fit(X, y)
    except (ValueError, np.linalg.LinAlgError):
        return MediationResult(
            total_ate=total_ate, direct_ate=float('nan'),
            indirect_ate=float('nan'), indirect_proportion=float('nan'),
            identified=True,
            linearity_status=LinearityStatus.POWER_INSUFFICIENT,
            treatment=treatment, outcome=outcome,
            mediators=mediators, method_name=method_name, n_rows=n_rows,
        )
    direct_ate = float(reg.coef_[0])

    # Stage 3: indirect = total - direct; proportion guarded.
    indirect_ate = total_ate - direct_ate
    if abs(total_ate) > eps:
        indirect_proportion = indirect_ate / total_ate
    else:
        indirect_proportion = float('nan')

    status = _classify_linearity(
        total_ate=total_ate, direct_ate=direct_ate,
        indirect_proportion=indirect_proportion,
        identified=True, eps=eps,
    )

    return MediationResult(
        total_ate=total_ate,
        direct_ate=direct_ate,
        indirect_ate=indirect_ate,
        indirect_proportion=indirect_proportion,
        identified=True,
        linearity_status=status,
        treatment=treatment,
        outcome=outcome,
        mediators=mediators,
        method_name=method_name,
        n_rows=n_rows,
    )


__all__ = [
    'LinearityStatus',
    'MediationResult',
    'mediation_dowhy',
]
