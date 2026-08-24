"""`paired_continuous_do_dowhy` — paired-Δ outcome under continuous
exogenous treatment, with DoWhy backdoor ATE + placebo + RCC
refutations.

Where `paired_delta_link_dowhy` consumes per-burst Δ_predictor /
Δ_target Measurables (mediator → outcome link), this primitive
consumes a continuous **treatment** that's an HP / scalar swept
across rows (e.g. `target_sync.tau`, `gamma`, `sync_period`)
and tests its causal effect on the paired-Δ outcome.

Schematically:

    pair (treatment_arm, baseline_arm) on `pair_by` (which MUST
        include the swept HP, so each pair has a well-defined
        treatment value)
    Δ_outcome  =  outcome[treatment_arm] − outcome[baseline_arm]
    treatment  =  HP value at the pair (read from each row)

    DAG:        treatment_var → delta_outcome
    ATE:        backdoor_ate(treatment_var, delta_outcome, dag)
    Refute:     placebo_refutation, random_common_cause_refutation

Returns `PairedContinuousDoResult` carrying the three DoWhy
sub-results plus pair count and treatment provenance. The
canonical use is encoding rung-2 evidence from a do(HP) sweep
where the HP IS the intervention (Polyak τ in target_sync,
γ in discount, sync_period in target sync).
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from corroborate._internals.polars import to_dicts
from corroborate.analyses.dowhy import (
    BackdoorResult, RefutationResult,
)
from corroborate.bridge.analysis import analysis


@dataclass(frozen=True, slots=True)
class PairedContinuousDoResult:
    """Output of `paired_continuous_do_dowhy`. `backdoor` carries
    the linear-regression ATE; `placebo` and `random_common_cause`
    are the standard refutations. `n_pairs` is the number of
    (treatment, baseline) pairs whose treatment value contributed
    to the regression."""
    backdoor: BackdoorResult
    placebo: RefutationResult
    random_common_cause: RefutationResult
    n_pairs: int
    treatment: str
    outcome: str = 'delta_outcome'


def _pair_and_extract(
    rows: Sequence[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    treatment_var: str,
    outcome_field: str,
    pair_by: tuple[str, ...],
    arm_field: str,
    treatment_var_arm: str = 'baseline',
) -> list[dict[str, float]]:
    """Pair (treatment_arm, baseline_arm) on `pair_by`. Returns a
    list of dict rows with `{treatment_var, 'delta_outcome'}` per
    pair.

    `treatment_var_arm` selects which arm the per-pair treatment
    value is read from: `'baseline'` (default — for endogenous
    mediator treatments where baseline's value represents the
    per-pair pre-DDQN exposure level) or `'treatment'`. For
    HP-style treatments where both arms share the value, either
    works."""
    keyed: dict[
        tuple[object, ...],
        dict[str, Mapping[str, object]],
    ] = defaultdict(dict)
    for row in rows:
        arm = row.get(arm_field)
        if arm not in (treatment_arm, baseline_arm):
            continue
        key = tuple(row.get(k) for k in pair_by)
        if any(v is None for v in key):
            continue
        keyed[key][arm] = row

    pair_rows: list[dict[str, float]] = []
    for key, arms in keyed.items():
        del key
        t = arms.get(treatment_arm)
        b = arms.get(baseline_arm)
        if t is None or b is None:
            continue
        treat_source = b if treatment_var_arm == 'baseline' else t
        treat_val = treat_source.get(treatment_var)
        if treat_val is None:
            continue
        if not isinstance(treat_val, (int, float)):
            continue
        try:
            tv_f = float(treat_val)
        except (TypeError, ValueError):
            continue
        if math.isnan(tv_f):
            continue
        ot = t.get(outcome_field)
        ob = b.get(outcome_field)
        if not (isinstance(ot, (int, float)) and isinstance(ob, (int, float))):
            continue
        try:
            ot_f, ob_f = float(ot), float(ob)
        except (TypeError, ValueError):
            continue
        if math.isnan(ot_f) or math.isnan(ob_f):
            continue
        pair_rows.append({
            treatment_var: tv_f,
            'delta_outcome': ot_f - ob_f,
        })
    return pair_rows


def _nan_backdoor(
    treatment: str, outcome: str, method_name: str,
) -> BackdoorResult:
    return BackdoorResult(
        ate=float('nan'), identified=False,
        estimand_str='', method_name=method_name,
        treatment=treatment, outcome=outcome, n_rows=0,
    )


def _nan_refutation(
    treatment: str, outcome: str,
    method_name: str, refuter_name: str,
) -> RefutationResult:
    return RefutationResult(
        real_ate=float('nan'), refuted_ate=float('nan'),
        drift=float('nan'), method_name=method_name,
        refuter_name=refuter_name,
        treatment=treatment, outcome=outcome, n_rows=0,
    )


@analysis
def paired_continuous_do_dowhy(
    df: pl.DataFrame,
    *,
    treatment_arm: str,
    baseline_arm: str,
    treatment_var: str,
    outcome: str,
    pair_by: tuple[str, ...] = ('seed',),
    arm_field: str = 'arm_key',
    treatment_var_arm: str = 'baseline',
    method_name: str = 'backdoor.linear_regression',
    random_state: int = 0,
) -> PairedContinuousDoResult:
    """Test `do(treatment_var) → Δ_outcome` via DoWhy backdoor +
    refutations.

    `treatment_var` is the column name on each row carrying the
    swept HP (continuous, exogenous by sweep design — e.g.
    `'target_sync.tau'`). `pair_by` MUST include the swept HP so
    each pair has a well-defined treatment value (treatment and
    baseline rows pair within a single HP value).

    `outcome` is the per-row column name of the bridge target
    (e.g. `'eval_best_burst_mean'`). Δ_outcome is computed at
    pair-time as `outcome[treatment_arm] − outcome[baseline_arm]`.

    DAG: single edge `treatment_var → delta_outcome`. No
    confounders since the HP is intervened by sweep design;
    refutations validate that the regression isn't an artifact.

    Returns the three DoWhy sub-results. Bridges typically check
    `backdoor.identified`, `backdoor.ate` against a sign-thresh,
    and `placebo.refuted_ate ≈ 0` / `random_common_cause.drift`
    small for the refutation gates."""
    rows = _pair_and_extract(
        to_dicts(df),
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        treatment_var=treatment_var,
        outcome_field=outcome,
        pair_by=pair_by,
        arm_field=arm_field,
        treatment_var_arm=treatment_var_arm,
    )
    if not rows:
        return PairedContinuousDoResult(
            backdoor=_nan_backdoor(treatment_var, 'delta_outcome', method_name),
            placebo=_nan_refutation(
                treatment_var, 'delta_outcome', method_name,
                'placebo_treatment_refuter',
            ),
            random_common_cause=_nan_refutation(
                treatment_var, 'delta_outcome', method_name,
                'random_common_cause',
            ),
            n_pairs=0,
            treatment=treatment_var,
        )

    dag: list[tuple[str, str]] = [(treatment_var, 'delta_outcome')]
    # Rows-level throughout: the derived pair rows go straight to
    # the DoWhy cores — encoding a frame here would only be
    # decoded again by each of the three calls.
    from corroborate.analyses.dowhy.dowhy import (
        backdoor_ate_rows, run_refuter,
    )
    backdoor = backdoor_ate_rows(
        rows,
        treatment=treatment_var,
        outcome='delta_outcome',
        dag=dag,
        method_name=method_name,
    )
    placebo = run_refuter(
        rows, treatment_var, 'delta_outcome', dag, method_name,
        refuter_method='placebo_treatment_refuter',
        random_state=random_state,
    )
    rcc = run_refuter(
        rows, treatment_var, 'delta_outcome', dag, method_name,
        refuter_method='random_common_cause',
        random_state=random_state,
    )
    return PairedContinuousDoResult(
        backdoor=backdoor,
        placebo=placebo,
        random_common_cause=rcc,
        n_pairs=len(rows),
        treatment=treatment_var,
    )


__all__ = [
    'PairedContinuousDoResult',
    'paired_continuous_do_dowhy',
]
