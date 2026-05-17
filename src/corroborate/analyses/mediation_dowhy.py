"""DoWhy mediation decomposition — total, direct, indirect ATE.

Two-stage approach (the v10 PoC `analysis_dowhy_mediation.py`
shape, ported forward with current framework conventions):

1. **Total ATE** — backdoor identification + estimation of
   `treatment → outcome` under `dag`. Reuses
   `corroborate.analyses.dowhy._backdoor_estimate`.
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

**RELIABILITY GATING (load-bearing — see v10
CASE_STUDY_LESSONS §2.11)**: mediation magnitudes are wildly
unreliable without prior power + topology gates. A v10 CartPole
DDQN example produced total=+3.0, direct=+13.2, indirect=−10.3 —
DoWhy will silently emit such cancellations when not gated. The
canonical recipe per the lesson:

    Step 1 — Power-gate the TOTAL ATE first (e.g., via
             `corroborate.analyses.dowhy.placebo_refutation` +
             `random_common_cause_refutation`). If the total is
             POWER_INSUFFICIENT, mediation magnitudes are
             meaningless.
    Step 2 — PC-topology check. Discover the adjacency at α=0.05
             via `corroborate.graph.discovery.discover_adjacency`.
             If PC does NOT support the mediator's position on
             the path, the posited DAG is suspect.
    Step 3 — Only NOW run `mediation_dowhy`. Magnitudes are
             reliability-gated by the prior steps.
    Step 4 — Refute the total ATE with placebo + RCC (use the
             existing primitives; refutation directly on the
             two-stage decomposition is not exposed here).

Bridges using this primitive should chain power-gate +
PC-topology + mediation as a multi-stage cluster Finding, NOT
read magnitudes from this primitive in isolation.

**Why not `proportion_mediated`** (which the framework
deprecated): linear-mediation as ratio-of-noisy-means explodes
near zero; can land outside [0, 1] under suppression; uses
first-difference identification that doesn't recover population
slopes under nonlinearity. DoWhy's identification-theory path
sidesteps (1) and (3) — though (2) is still a presentation
concern handled at the consumer side.

Refutations are NOT bundled here. Use `dowhy.placebo_refutation`
+ `dowhy.random_common_cause_refutation` on the same
`(cells, treatment, outcome, dag)` triple to refute the total
ATE; the mediation decomposition inherits the total's
reliability since direct is a deterministic OLS reading and
indirect is the residual.

**EMPIRICAL EXAMPLE OF THE FAILURE MODE** (the v10 lesson
reproduced on a real corpus). At FR × MLP × unshaped × baseline
(n=120) on the corroborate-rl DDQN sweep, with treatment = γ ∈
{0.99, 0.999}, outcome = jensen_gap, mediators = {self_ref,
σ_action}:

    Total ATE    = +1023.36   (well-powered; refutations pass)
    Direct ATE   = −57.31     (NEGATIVE — sign-flipped vs total!)
    Indirect ATE = +1080.67
    Indirect %   = +105.6%    (> 100% — outside [0, 1])

This is the textbook multicollinearity failure: the three
variables (γ, self_ref, σ_action) are pairwise correlated 0.78-
0.93 at this scope, so OLS coefficients fight each other and γ's
residual coefficient lands at -57 (regression artifact, not
mechanism). Meanwhile rank-based partial Spearman on the same
data gives ρ(γ, jens | self_ref, σ_action) = +0.06 NS — a clean
"chain closes" result. Two methods, two answers. The
rank-based one is more trustworthy because it's robust to
linearity violations and multicollinearity-induced sign flips.

The takeaway: when mediation_dowhy returns proportion > 1 or
direct/total sign flip, the linear-mediation assumption is
broken on this corpus. Use it as a DIAGNOSTIC (chain-is-
nonlinear flag), not a magnitude estimator. Partial-Spearman
remains the framework's canonical mediation primitive."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from corroborate.bridge.analysis import analysis
from corroborate.analyses._dowhy_internal import (
    DAGLike,
    _backdoor_estimate,
    _cells_to_dataframe,
)


if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class MediationResult:
    """Output of `mediation_dowhy`: total, direct, indirect ATE +
    indirect proportion + identification flag.

    `identified` is False when the DAG admits no backdoor
    adjustment for `(treatment, outcome)` — the bridge should
    map this to POWER_INSUFFICIENT (Pearl's tier-2 verdict
    requires identification first), NOT NO_EFFECT.

    `indirect_proportion` is `indirect_ate / total_ate` and is
    NaN when `|total_ate| < eps`; consumers should NOT compare
    NaN to thresholds directly. Verdict gates should check
    `not math.isnan(indirect_proportion)` first.

    NB: this primitive is RELIABILITY-GATED — see the module
    docstring §"RELIABILITY GATING". Magnitudes are unreliable
    without prior power + topology gating."""
    total_ate: float
    direct_ate: float
    indirect_ate: float
    indirect_proportion: float
    identified: bool
    treatment: str
    outcome: str
    mediators: tuple[str, ...]
    method_name: str
    n_rows: int


@analysis
def mediation_dowhy(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment: str,
    outcome: str,
    mediators: tuple[str, ...],
    dag: DAGLike,
    method_name: str = 'backdoor.linear_regression',
    eps: float = 1e-8,
) -> MediationResult:
    """Two-stage mediation decomposition: total via backdoor,
    direct via OLS-with-mediators, indirect via subtraction.

    `mediators` is a non-empty tuple of column names; all enter
    Stage 2's OLS jointly. The DAG SHOULD include the mediators
    as nodes on the `treatment → mediator → outcome` path for
    Stage 1's backdoor identification to admit the path-aware
    adjustment set.

    Reads `cells` and projects (treatment, outcome, *mediators)
    columns; cells missing any required key are skipped.

    Returns `MediationResult` with NaN-everywhere values when
    the DAG doesn't identify the total ATE (e.g., unmeasured
    confounding) or the DataFrame is too sparse for OLS.

    Pair with `dowhy.placebo_refutation` + `random_common_cause_refutation`
    on the same total-ATE arguments to refute the foundation."""
    if not mediators:
        raise ValueError(
            'mediation_dowhy: `mediators` must be a non-empty tuple of '
            'column names; use `dowhy.backdoor_ate` for total-only.',
        )
    # Stage 1: total ATE via backdoor (reuse existing machinery)
    df, _identified_estimand, estimate = _backdoor_estimate(
        cells, treatment, outcome, dag, method_name,
    )
    n_rows = len(df)
    if estimate is None:
        return MediationResult(
            total_ate=float('nan'), direct_ate=float('nan'),
            indirect_ate=float('nan'), indirect_proportion=float('nan'),
            identified=False, treatment=treatment, outcome=outcome,
            mediators=mediators, method_name=method_name, n_rows=n_rows,
        )
    total_ate = float(getattr(estimate, 'value'))

    # Stage 2: direct ATE via OLS with mediators as covariates.
    # Build a fresh DataFrame projection that explicitly includes
    # the mediators (the DAG-derived `df` may not — e.g., if the
    # DAG has a node not on the treatment-outcome path that's
    # included anyway, vs a mediator off-DAG that we want to
    # condition on).
    needed = [treatment, outcome, *mediators]
    df_med = _cells_to_dataframe(cells, needed)
    if df_med.empty or len(df_med) < 3:
        return MediationResult(
            total_ate=total_ate, direct_ate=float('nan'),
            indirect_ate=float('nan'), indirect_proportion=float('nan'),
            identified=True, treatment=treatment, outcome=outcome,
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
            identified=True, treatment=treatment, outcome=outcome,
            mediators=mediators, method_name=method_name, n_rows=n_rows,
        )
    direct_ate = float(reg.coef_[0])  # coefficient on `treatment`

    # Stage 3: indirect = total - direct; proportion guarded.
    indirect_ate = total_ate - direct_ate
    if abs(total_ate) > eps:
        indirect_proportion = indirect_ate / total_ate
    else:
        indirect_proportion = float('nan')

    return MediationResult(
        total_ate=total_ate,
        direct_ate=direct_ate,
        indirect_ate=indirect_ate,
        indirect_proportion=indirect_proportion,
        identified=True,
        treatment=treatment,
        outcome=outcome,
        mediators=mediators,
        method_name=method_name,
        n_rows=n_rows,
    )


__all__ = [
    'MediationResult',
    'mediation_dowhy',
]
