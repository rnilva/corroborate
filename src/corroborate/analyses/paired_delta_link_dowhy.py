"""Paired-Δ link backdoor + refutations.

Tests the *continuous* mech→outcome link via dowhy backdoor on
per-(env, burst, seed) paired-Δ rows. For each pair at each
burst, computes Δ_predictor = treatment[seed,b] − baseline[seed,b]
and Δ_target = treatment[seed,b] − baseline[seed,b]; the panel
rows are (env_name, burst_index, seed, Δ_predictor, Δ_target).
DoWhy backdoor adjusts for burst (one-hot dummies) and runs
placebo + random-common-cause refutations.

Substrate-blind: the substrate names which columns play the
mechanism / outcome roles + their per-burst reductions. This
analysis composes `paired_link_per_burst`'s panel-build logic
(per-pair-per-burst Δ rows) with `backdoor_ate` /
`placebo_refutation` / `random_common_cause_refutation` into
one fused result.

Distinct from `link_attenuation_dowhy`: that one tests a
*binary attenuator* on per-stratum link strength scalars.
This one tests *continuous* Δ_predictor → Δ_target with burst
as the adjuster.

A bridge consuming this fixture authors mechanism / outcome
column names + reductions; downstream verdicts read sub-fields
of `PairedDeltaLinkDowhyResult` (real ATE, placebo drift, RCC
drift)."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from corroborate.analyses.dowhy import (
    BackdoorResult,
    RefutationResult,
    backdoor_ate,
    placebo_refutation,
    random_common_cause_refutation,
)
from corroborate.analyses.paired_g_per_burst import (
    evaluate_per_burst_source,
)
from corroborate.bridge.analysis import analysis
from corroborate.measurables import Measurable


@dataclass(frozen=True, slots=True)
class PairedDeltaLinkDowhyResult:
    """Backdoor + placebo + RCC refutations of `Δ_predictor →
    Δ_target` on per-(env, burst, seed) paired-Δ rows. Adjusts
    for burst (one-hot); env can be filtered to a single
    substrate (Acrobot at γ=0.999, etc.).

    `backdoor` carries the real ATE of Δ_predictor on Δ_target
    after burst adjustment. `placebo` and `random_common_cause`
    carry the corresponding refutation pairs. `n_pairs` is the
    panel-row count."""
    backdoor: BackdoorResult
    placebo: RefutationResult
    random_common_cause: RefutationResult
    n_pairs: int
    treatment_col: str
    outcome_col: str


def _build_panel(
    cells: Sequence[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...],
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    env_filter: tuple[str, ...],
    arm_field: str,
) -> tuple[
    list[Mapping[str, object]],
    list[tuple[str, str]],
    str,
    str,
]:
    """Build per-(env, burst, seed) Δ panel rows + dowhy DAG.
    Returns `(rows, dag, treatment_col, outcome_col)`.

    Each row carries `djens` (Δ_predictor), `dout` (Δ_target),
    `burst_index`, plus burst one-hot dummies. The DAG nodes
    are `djens`, `dout`, and one node per burst-dummy column."""
    treatment_col = 'djens'
    outcome_col = 'dout'

    by_env_arm: dict[tuple[str, str], dict[
        tuple[object, ...], tuple[np.ndarray, np.ndarray],
    ]] = {}
    env_filter_set = set(env_filter) if env_filter else None
    for cell in cells:
        env = cell.get('env_name')
        arm = cell.get(arm_field)
        if not isinstance(env, str) or not isinstance(arm, str):
            continue
        if env_filter_set is not None and env not in env_filter_set:
            continue
        if arm not in (treatment_arm, baseline_arm):
            continue
        target_v = evaluate_per_burst_source(link_target, cell)
        predictor_v = evaluate_per_burst_source(link_predictor, cell)
        if target_v.size == 0 or predictor_v.size == 0:
            continue
        if target_v.shape[0] != predictor_v.shape[0]:
            continue
        bucket = by_env_arm.setdefault((env, arm), {})
        try:
            key = tuple(cell[k] for k in pair_by)
        except KeyError:
            continue
        bucket[key] = (target_v, predictor_v)

    pair_rows: list[dict[str, object]] = []
    burst_indices_seen: set[int] = set()
    envs = sorted({e for (e, _) in by_env_arm})
    for env in envs:
        treat = by_env_arm.get((env, treatment_arm), {})
        base = by_env_arm.get((env, baseline_arm), {})
        paired_keys = sorted(set(treat) & set(base))
        if not paired_keys:
            continue
        # Per-key arm-shape match — the real invariant. Cross-key
        # uniformity is NOT required: multi-regime corpora can have
        # different burst counts per pair_by key.
        for k in paired_keys:
            if (
                treat[k][0].shape[0] != base[k][0].shape[0]
                or treat[k][1].shape[0] != base[k][1].shape[0]
                or treat[k][0].shape[0] != treat[k][1].shape[0]
            ):
                continue  # arm-shape mismatch — skip the key
            n_b = treat[k][0].shape[0]
            for b in range(n_b):
                d_t = float(treat[k][0][b] - base[k][0][b])
                d_p = float(treat[k][1][b] - base[k][1][b])
                if not (math.isfinite(d_t) and math.isfinite(d_p)):
                    continue
                burst_indices_seen.add(b)
                pair_rows.append({
                    'env_name': env,
                    'burst_index': b,
                    outcome_col: d_t,
                    treatment_col: d_p,
                })

    if not pair_rows:
        return [], [], treatment_col, outcome_col

    # One-hot encode burst index, drop-first to avoid collinearity.
    bursts_sorted = sorted(burst_indices_seen)
    burst_dummy_cols = [f'__burst__{b}' for b in bursts_sorted[1:]]
    rows: list[Mapping[str, object]] = []
    for r in pair_rows:
        row = dict(r)
        b = r['burst_index']
        for col in burst_dummy_cols:
            row[col] = 1.0 if col == f'__burst__{b}' else 0.0
        rows.append(row)

    dag: list[tuple[str, str]] = []
    for c in burst_dummy_cols:
        dag.append((c, treatment_col))
        dag.append((c, outcome_col))
    dag.append((treatment_col, outcome_col))
    return rows, dag, treatment_col, outcome_col


def _nan_backdoor(
    treatment_col: str, outcome_col: str, method_name: str,
) -> BackdoorResult:
    return BackdoorResult(
        ate=float('nan'),
        identified=False,
        estimand_str='',
        method_name=method_name,
        treatment=treatment_col,
        outcome=outcome_col,
        n_rows=0,
    )


def _nan_refutation(
    treatment_col: str, outcome_col: str, method_name: str,
    refuter_name: str,
) -> RefutationResult:
    return RefutationResult(
        real_ate=float('nan'),
        refuted_ate=float('nan'),
        drift=float('nan'),
        method_name=method_name,
        refuter_name=refuter_name,
        treatment=treatment_col,
        outcome=outcome_col,
        n_rows=0,
    )


@analysis
def paired_delta_link_dowhy(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    pair_by: tuple[str, ...] = ('seed',),
    env_filter: tuple[str, ...] = (),
    arm_field: str = 'arm_key',
    method_name: str = 'backdoor.linear_regression',
) -> PairedDeltaLinkDowhyResult:
    """Test `Δ_predictor → Δ_target` after burst adjustment via
    dowhy backdoor + refutations.

    Composes `paired_link_per_burst`-style panel construction
    (per-(env, burst, seed) Δ rows) with `backdoor_ate` +
    `placebo_refutation` + `random_common_cause_refutation`. The
    DAG one-hot encodes `burst_index` (drop-first), edges from
    each burst column to both treatment + outcome, plus the
    treatment → outcome edge under test.

    `link_target` and `link_predictor` are typed Measurables
    returning per-burst NDArrays. The canonical mech → outcome
    link uses per-burst-mean of `mc_return` for the target and
    per-burst-mean of `jensen_bias_per_eps` for the predictor.

    Empty panel (no env survives filter, or no paired seeds)
    yields a NaN-everywhere result. Bridges consuming this check
    `n_pairs` for power and `backdoor.identified` before reading
    ATEs."""
    cells_list = list(cells)
    rows, dag, treatment_col, outcome_col = _build_panel(
        cells_list,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        pair_by=pair_by,
        link_predictor=link_predictor,
        link_target=link_target,
        env_filter=env_filter,
        arm_field=arm_field,
    )
    if not rows:
        return PairedDeltaLinkDowhyResult(
            backdoor=_nan_backdoor(
                treatment_col, outcome_col, method_name,
            ),
            placebo=_nan_refutation(
                treatment_col, outcome_col, method_name,
                refuter_name='placebo_treatment_refuter',
            ),
            random_common_cause=_nan_refutation(
                treatment_col, outcome_col, method_name,
                refuter_name='random_common_cause',
            ),
            n_pairs=0,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
        )

    backdoor = backdoor_ate.fn(
        rows, treatment=treatment_col, outcome=outcome_col,
        dag=dag, method_name=method_name,
    )
    placebo = placebo_refutation.fn(
        rows, treatment=treatment_col, outcome=outcome_col,
        dag=dag, method_name=method_name,
    )
    rcc = random_common_cause_refutation.fn(
        rows, treatment=treatment_col, outcome=outcome_col,
        dag=dag, method_name=method_name,
    )
    return PairedDeltaLinkDowhyResult(
        backdoor=backdoor,
        placebo=placebo,
        random_common_cause=rcc,
        n_pairs=len(rows),
        treatment_col=treatment_col,
        outcome_col=outcome_col,
    )


__all__ = [
    'PairedDeltaLinkDowhyResult',
    'paired_delta_link_dowhy',
]
