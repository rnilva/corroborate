"""Link-attenuation backdoor + refutations.

A binary attenuator's effect on per-(env, burst) link strength,
estimated via dowhy backdoor with env-family one-hot adjustment.

The "link" is the paired-Δ correlation between a mechanism-side
column and an outcome-side column at the (env, burst) panel
level, computed by `paired_link_per_burst`. The attenuator is
any per-cell column averaged to env-level then binarised by
threshold. The confounder is `env_name` one-hot encoded
(categorical; per `findings_dowhy_three_probes.md`, dowhy
linear regression needs one-hot, not int).

Substrate-blind: the substrate names which columns play the
mechanism / outcome / attenuator roles. This analysis composes
`paired_link_per_burst`'s panel build + `backdoor_ate` +
`placebo_refutation` + `random_common_cause_refutation` into one
fused result; bridges that consume the result check the
relevant sub-field (real ATE, placebo drift, RCC drift) for
their verdict.

A bridge consuming this fixture authors `attenuator='X'` and
`binary_threshold=Y` (the moderator predicate) plus the link's
mechanism-side / outcome-side columns + their reductions for
the panel build."""
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
from corroborate.analyses.paired_link_per_burst import (
    paired_link_per_burst,
)
from corroborate.runner.analysis import analysis
from corroborate.measurables import Measurable


@dataclass(frozen=True, slots=True)
class LinkAttenuationDowhyResult:
    """Backdoor + placebo + RCC refutations of a binary attenuator
    on per-(env, burst) link-strength panel.

    `backdoor` carries the real ATE of the binary attenuator on
    link strength after backdoor adjustment for env family.
    `placebo` and `random_common_cause` carry the corresponding
    refutation pairs (real + refuted ATE). `n_strata`,
    `n_above`, `n_below` characterise the panel split. Bridges
    consume different sub-fields depending on which refutation
    criterion they assert."""
    backdoor: BackdoorResult
    placebo: RefutationResult
    random_common_cause: RefutationResult
    n_strata: int
    n_above: int
    n_below: int
    binary_threshold: float
    attenuator: str
    treatment_col: str
    outcome_col: str


def _env_means(
    cells: Sequence[Mapping[str, object]],
    *,
    confounder: str,
    attenuator: str,
) -> dict[str, float]:
    """Per-env mean of `attenuator`. Cells lacking the confounder,
    a numeric attenuator, or a non-NaN value are excluded."""
    bins: dict[str, list[float]] = {}
    for cell in cells:
        env = cell.get(confounder)
        if not isinstance(env, str):
            continue
        v = cell.get(attenuator)
        if not isinstance(v, (int, float)):
            continue
        f = float(v)
        if math.isnan(f):
            continue
        bins.setdefault(env, []).append(f)
    return {e: sum(vs) / len(vs) for e, vs in bins.items() if vs}


def _build_panel(
    cells: Sequence[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    pair_by: tuple[str, ...],
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    attenuator: str,
    binary_threshold: float,
    confounder: str,
    dedupe_strategy: str = 'raise',
) -> tuple[
    list[Mapping[str, object]],
    list[tuple[str, str]],
    str,
    str,
    int,
    int,
]:
    """Build per-(env, burst) panel rows + dowhy DAG.

    Returns (panel_rows, dag_edges, treatment_col, outcome_col,
    n_above, n_below). `panel_rows` is empty when no env has both
    valid link strata and a numeric attenuator value."""
    panel = paired_link_per_burst.fn(
        cells,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        pair_by=pair_by,
        target=link_target,
        predictor=link_predictor,
        dedupe_strategy=dedupe_strategy,
    )
    env_attenuator = _env_means(
        cells, confounder=confounder, attenuator=attenuator,
    )
    panel_envs = {s.env_name for s in panel.strata}
    envs_in_panel = sorted(panel_envs & set(env_attenuator))

    treatment_col = f'{attenuator}__above_threshold'
    outcome_col = 'g_link'
    if not envs_in_panel:
        return [], [], treatment_col, outcome_col, 0, 0

    # One-hot encode env, dropping the first to avoid collinearity.
    env_cols = [f'__env__{e}' for e in envs_in_panel[1:]]

    rows: list[Mapping[str, object]] = []
    n_above = 0
    n_below = 0
    for s in panel.strata:
        if s.env_name not in env_attenuator:
            continue
        if math.isnan(s.r):
            continue
        env_mean = env_attenuator[s.env_name]
        treated = float(env_mean > binary_threshold)
        if treated >= 0.5:
            n_above += 1
        else:
            n_below += 1
        row: dict[str, object] = {
            confounder: s.env_name,
            'burst_index': s.burst_index,
            outcome_col: float(s.r),
            treatment_col: treated,
        }
        for c in env_cols:
            row[c] = 1.0 if c == f'__env__{s.env_name}' else 0.0
        rows.append(row)

    dag: list[tuple[str, str]] = []
    for c in env_cols:
        dag.append((c, treatment_col))
        dag.append((c, outcome_col))
    dag.append((treatment_col, outcome_col))
    return rows, dag, treatment_col, outcome_col, n_above, n_below


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
def link_attenuation_dowhy(
    cells: Iterable[Mapping[str, object]],
    *,
    treatment_arm: str,
    baseline_arm: str,
    link_target: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    link_predictor: Measurable[
        Mapping[str, object], npt.NDArray[np.floating],
    ],
    attenuator: str,
    binary_threshold: float,
    pair_by: tuple[str, ...] = ('seed',),
    confounder: str = 'env_name',
    method_name: str = 'backdoor.linear_regression',
    dedupe_strategy: str = 'raise',
) -> LinkAttenuationDowhyResult:
    """Test whether `attenuator > binary_threshold` attenuates
    per-(env, burst) link strength via dowhy backdoor +
    refutations.

    Composes `paired_link_per_burst` (panel build) with
    `backdoor_ate` + `placebo_refutation` +
    `random_common_cause_refutation` (one each on the panel
    rows). The DAG one-hot encodes `confounder` (drop-first),
    edges from each env column to the binary treatment + the
    outcome, plus the treatment → outcome edge under test.

    Empty panel (no env has both link strata and a numeric
    attenuator value) yields NaN-everywhere result. Bridges
    that consume this typically check `n_strata` for power and
    `backdoor.identified` before reading ATEs."""
    cells_list = list(cells)
    rows, dag, treatment_col, outcome_col, n_above, n_below = _build_panel(
        cells_list,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        pair_by=pair_by,
        link_target=link_target,
        link_predictor=link_predictor,
        attenuator=attenuator,
        binary_threshold=binary_threshold,
        confounder=confounder,
        dedupe_strategy=dedupe_strategy,
    )
    if not rows:
        return LinkAttenuationDowhyResult(
            backdoor=_nan_backdoor(treatment_col, outcome_col, method_name),
            placebo=_nan_refutation(
                treatment_col, outcome_col, method_name,
                refuter_name='placebo_treatment_refuter',
            ),
            random_common_cause=_nan_refutation(
                treatment_col, outcome_col, method_name,
                refuter_name='random_common_cause',
            ),
            n_strata=0, n_above=0, n_below=0,
            binary_threshold=binary_threshold,
            attenuator=attenuator,
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
    return LinkAttenuationDowhyResult(
        backdoor=backdoor,
        placebo=placebo,
        random_common_cause=rcc,
        n_strata=len(rows),
        n_above=n_above,
        n_below=n_below,
        binary_threshold=binary_threshold,
        attenuator=attenuator,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
    )


__all__ = [
    'LinkAttenuationDowhyResult',
    'link_attenuation_dowhy',
]
