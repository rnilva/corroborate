"""DoWhy-backed bridges — `Bridge[R]` factories around DoWhy's
identify / estimate / refute pipeline.

Three bridges, mirroring v10:

- `backdoor_ate`: identification + estimation under a posited
  DAG. HELD iff sign(estimate) == sign(expected_sign) and
  |estimate| >= threshold. Carries `stats['tier']='interventional'`
  so `build_causal_graph` promotes the resulting edge from
  `Tier.ASSOCIATIONAL` to `Tier.INTERVENTIONAL`.
- `placebo_refutation`: re-estimates with a randomised treatment.
  HELD iff |placebo_effect| < tolerance — the real estimate
  isn't reproduced from a fake treatment.
- `random_common_cause_refutation`: re-estimates with a synthetic
  common cause. HELD iff |new_effect − real_effect| < tolerance
  — the estimate is robust to a confounder that, by construction,
  has no real influence.

Together: 1 estimate + 2 refuters on the same `(treatment, outcome)`
pair. When all three HELD, `promote_bridged_evidence` upgrades the
edge from `'causal_one_sided'` to `'causal_bridged'` — Pearl-ladder
do-calculus inference is corroborated by an INDEPENDENT bridge.

Design constraints:

- **Lazy imports.** DoWhy / pandas / networkx are imported inside
  the bridge bodies, not at module top. corroborate's spine
  imports cleanly without them. ImportError surfaces at call
  time. This module's symbols are importable even when DoWhy
  isn't installed; calling them raises.

- **DAG is an explicit input.** The bridge takes
  `graph: nx.DiGraph | CausalGraph | list[tuple[str, str]]` —
  the author posits the structural assumption, the bridge holds
  them to it. Do-calculus on observational data IS a
  Tier.INTERVENTIONAL claim *conditional on the DAG*, and the
  bridge surfaces that conditionality in
  `stats['identified_estimand']`.

- **Same `Bridge[R]` shape as the rest of the framework.** Each
  factory returns a `Bridge[R]` whose `__call__(record)` runs the
  identify/estimate/refute pipeline. The record is
  `Mapping[str, np.ndarray | jax.Array]` where each column is
  the population over which the regression runs (rows are cells
  / steps depending on the substrate's projection)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from corroborate.bridge import Bridge, BridgeResult
from corroborate.verdict import Verdict

if TYPE_CHECKING:
    import networkx as nx
    import pandas as pd
    from dowhy import CausalModel

    from corroborate.causal_graph import CausalGraph


# Type alias — DAG accepted shapes. Coerced to `nx.DiGraph[str]`
# by `_to_networkx` at call time. PEP 695 lazy alias, so the
# TYPE_CHECKING imports above are sufficient.
type DAGLike = (
    'nx.DiGraph[str] | CausalGraph | list[tuple[str, str]]'
)


# ============ Adapters ============

def _record_to_dataframe(
    record: Mapping[str, object], keys: list[str],
) -> 'pd.DataFrame':
    """Project the record into a pandas DataFrame on `keys`. Each
    key's value is mean-collapsed on non-row axes and cast to
    float64. Booleans cast to {0.0, 1.0}. Pandas imported lazily."""
    import pandas as pd

    cols: dict[str, np.ndarray] = {}
    for k in keys:
        v = record.get(k)
        if v is None:
            raise KeyError(
                f'_record_to_dataframe: key {k!r} not in record '
                f'(have {sorted(record.keys())!r})',
            )
        arr: np.ndarray = np.asarray(v)
        if arr.dtype == np.bool_:
            arr = arr.astype(np.float64)
        if arr.ndim > 1:
            # numpy's `mean(axis=...)` stub types loosely as Any.
            # At our shape (collapsing every non-row axis) it
            # always returns an ndarray.
            arr = np.asarray(
                arr.mean(  # pyright: ignore[reportAny]
                    axis=tuple(range(1, arr.ndim)),
                )
            )
        cols[k] = arr.astype(np.float64, copy=False)
    lengths = {k: len(v) for k, v in cols.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f'_record_to_dataframe: column lengths differ: '
            f'{lengths!r}',
        )
    return pd.DataFrame(cols)


def _to_networkx(graph: DAGLike) -> 'nx.DiGraph[str]':
    """Coerce a graph spec into `nx.DiGraph[str]`. Accepts an
    `nx.DiGraph` directly; corroborate's `CausalGraph`
    (`Graph[str, BridgeEdge]`); or a `list[(source, target)]` edge
    tuple list."""
    import networkx as nx

    if isinstance(graph, nx.DiGraph):
        return graph

    g: nx.DiGraph[str] = nx.DiGraph()
    if isinstance(graph, list):
        for src, tgt in graph:
            g.add_edge(src, tgt)
        return g

    # corroborate's CausalGraph is `Graph[str, BridgeEdge]` —
    # frozen dataclass with `nodes: frozenset[str]` and
    # `edges: tuple[Edge[str, BridgeEdge], ...]`. Pyright resolves
    # the field types correctly when CausalGraph is in scope.
    for n in graph.nodes:
        g.add_node(n)
    for e in graph.edges:
        g.add_edge(e.source, e.target)
    return g


def _build_causal_model(
    df: 'pd.DataFrame', treatment: str, outcome: str, graph: DAGLike,
) -> 'CausalModel':
    """Construct a DoWhy `CausalModel`. Validates that treatment
    and outcome appear as both DAG nodes and DataFrame columns."""
    from dowhy import CausalModel

    nx_graph = _to_networkx(graph)
    for nm, role in ((treatment, 'treatment'), (outcome, 'outcome')):
        if nm not in nx_graph.nodes:
            raise ValueError(
                f'{role} {nm!r} is not a node in the supplied DAG. '
                f'DAG nodes: {sorted(nx_graph.nodes)!r}.',
            )
        if nm not in df.columns:
            raise ValueError(
                f'{role} {nm!r} is not a column in the record. '
                f'Record columns: {sorted(df.columns)!r}.',
            )
    return CausalModel(
        data=df, treatment=treatment, outcome=outcome, graph=nx_graph,
    )


def _refuter_effect(refuter: object) -> float:
    """Extract the post-refutation effect. DoWhy renamed this
    attribute between versions: older expose `estimated_effect`,
    newer expose `new_effect`. Try both."""
    for attr in ('new_effect', 'estimated_effect'):
        if hasattr(refuter, attr):
            # `getattr(object, str)` returns Any. The runtime
            # invariant: dowhy refuters expose `new_effect` /
            # `estimated_effect` as a numeric scalar.
            return float(getattr(refuter, attr))  # pyright: ignore[reportAny]
    raise AttributeError(
        f'DoWhy refuter {type(refuter).__name__} has neither '
        f'`new_effect` nor `estimated_effect` — version mismatch?',
    )


def _record_keys_for(graph: DAGLike) -> list[str]:
    """Return the variable names in `graph` — the DataFrame columns
    we need to project from the record."""
    nx_graph = _to_networkx(graph)
    return list(nx_graph.nodes)


# ============ Bridge factories ============

def backdoor_ate[R: Mapping[str, object]](
    treatment: str,
    outcome: str,
    *,
    graph: DAGLike,
    expected_sign: int = 1,
    threshold: float = 0.0,
    method_name: str = 'backdoor.linear_regression',
    name: str | None = None,
) -> Bridge[R]:
    """Identify + estimate the ATE of `treatment` on `outcome`
    under the supplied DAG, via DoWhy.

    `expected_sign` ∈ {-1, +1} — the author's prior sign for the
    causal effect. HELD iff identification succeeds, sign matches,
    AND `|estimate| >= threshold`.

    Verdict mapping:
    - `Verdict.HELD` — identification + sign + magnitude all pass.
    - `Verdict.NO_EFFECT` — identification passes but sign / magnitude
      fail.
    - `Verdict.POWER_INSUFFICIENT` — identification fails (no
      backdoor / frontdoor / IV adjustment exists in the DAG).

    Carries `stats['tier']='interventional'` so the resulting
    edge promotes from ASSOCIATIONAL to INTERVENTIONAL when
    consumed by `build_causal_graph`."""
    bridge_name = (
        name if name is not None
        else f'backdoor_ate({treatment}->{outcome})'
    )

    def fn(record: R) -> BridgeResult:
        df = _record_to_dataframe(
            record, _record_keys_for(graph),
        )
        model = _build_causal_model(df, treatment, outcome, graph)
        identified = model.identify_effect(
            proceed_when_unidentifiable=False,
        )
        estimand_str = str(identified)
        if (
            getattr(identified, 'no_directed_path', False)
            or not getattr(identified, 'estimands', None)
        ):
            return BridgeResult(
                verdict=Verdict.POWER_INSUFFICIENT,
                reason=(
                    f'no identified estimand for {treatment} → '
                    f'{outcome} under DAG'
                ),
                stats={
                    'identified': 0,
                    'tier': 'interventional',
                    'role': 'estimate',
                },
                name=bridge_name,
                targets=(treatment, outcome),
            )

        estimate = model.estimate_effect(
            identified, method_name=method_name,
        )
        ate = float(estimate.value)
        # Plain comparison rather than np.sign — numpy returns
        # Any from sign(); a direct sign-product compare avoids
        # the type erasure.
        sign_ok = (
            (ate > 0) == (expected_sign > 0) if ate != 0 else False
        )
        mag_ok = abs(ate) >= threshold

        if sign_ok and mag_ok:
            verdict = Verdict.HELD
            reason = (
                f'ATE = {ate:+.4f} (sign={expected_sign:+d}, '
                f'|ATE| ≥ {threshold:.4f}, method={method_name})'
            )
        else:
            verdict = Verdict.NO_EFFECT
            failures: list[str] = []
            if not sign_ok:
                failures.append(
                    f'wrong sign (got {ate:+.4f}, want '
                    f'{expected_sign:+d})',
                )
            if not mag_ok:
                failures.append(
                    f'|ATE| = {abs(ate):.4f} < {threshold:.4f}',
                )
            reason = (
                '; '.join(failures) + f' (method={method_name})'
            )

        return BridgeResult(
            verdict=verdict, reason=reason,
            stats={
                'ate': ate,
                'identified': 1,
                'estimand': estimand_str,
                'tier': 'interventional',
                'role': 'estimate',
            },
            name=bridge_name,
            targets=(treatment, outcome),
        )

    return Bridge(fn=fn, name=bridge_name, targets=(treatment, outcome))


def placebo_refutation[R: Mapping[str, object]](
    treatment: str,
    outcome: str,
    *,
    graph: DAGLike,
    method_name: str = 'backdoor.linear_regression',
    tolerance: float = 0.05,
    placebo_type: str = 'permute',
    name: str | None = None,
) -> Bridge[R]:
    """Refute `treatment → outcome` by replacing the treatment
    with a randomised placebo. Under H0 (no causal effect), the
    placebo estimate matches the real estimate (both spurious).
    Under H1 (real effect), the placebo estimate drops near zero.

    HELD iff `|placebo_effect| < tolerance` — the real estimate
    is NOT reproducible from a randomised treatment.
    NO_EFFECT iff `|placebo_effect| >= tolerance` — the estimator
    returns a comparable effect for a fake treatment, so the
    original estimate is suspect.
    POWER_INSUFFICIENT — identification fails."""
    bridge_name = (
        name if name is not None
        else f'placebo_refutation({treatment}->{outcome})'
    )

    def fn(record: R) -> BridgeResult:
        df = _record_to_dataframe(
            record, _record_keys_for(graph),
        )
        model = _build_causal_model(df, treatment, outcome, graph)
        identified = model.identify_effect(
            proceed_when_unidentifiable=False,
        )
        if (
            getattr(identified, 'no_directed_path', False)
            or not getattr(identified, 'estimands', None)
        ):
            return BridgeResult(
                verdict=Verdict.POWER_INSUFFICIENT,
                reason=(
                    f'no identified estimand for {treatment} → '
                    f'{outcome} under DAG'
                ),
                stats={
                    'identified': 0,
                    'tier': 'interventional',
                    'role': 'refuter',
                },
                name=bridge_name,
                targets=(treatment, outcome),
            )
        estimate = model.estimate_effect(
            identified, method_name=method_name,
        )
        refuter = model.refute_estimate(
            identified, estimate,
            method_name='placebo_treatment_refuter',
            placebo_type=placebo_type,
        )
        placebo_effect = _refuter_effect(refuter)
        real_ate = float(estimate.value)

        if abs(placebo_effect) < tolerance:
            verdict = Verdict.HELD
            reason = (
                f'placebo ATE = {placebo_effect:+.4f}, |·| < '
                f'{tolerance:.4f} (real ATE = {real_ate:+.4f})'
            )
        else:
            verdict = Verdict.NO_EFFECT
            reason = (
                f'placebo ATE = {placebo_effect:+.4f}, |·| ≥ '
                f'{tolerance:.4f} — estimator returns non-trivial '
                f'effect for randomised treatment '
                f'(real ATE = {real_ate:+.4f})'
            )

        return BridgeResult(
            verdict=verdict, reason=reason,
            stats={
                'real_ate': real_ate,
                'placebo_ate': placebo_effect,
                'tier': 'interventional',
                'role': 'refuter',
            },
            name=bridge_name,
            targets=(treatment, outcome),
        )

    return Bridge(fn=fn, name=bridge_name, targets=(treatment, outcome))


def random_common_cause_refutation[R: Mapping[str, object]](
    treatment: str,
    outcome: str,
    *,
    graph: DAGLike,
    method_name: str = 'backdoor.linear_regression',
    tolerance: float = 0.05,
    name: str | None = None,
) -> Bridge[R]:
    """Refute by adding a random common cause and re-estimating.
    If the estimate is robust, the new effect is ≈ the original.

    HELD iff `|new_effect − real_effect| < tolerance`.
    NO_EFFECT otherwise — the estimate is sensitive to a synthetic
    common cause that, by construction, has no real influence.
    POWER_INSUFFICIENT — identification fails."""
    bridge_name = (
        name if name is not None
        else f'rcc_refutation({treatment}->{outcome})'
    )

    def fn(record: R) -> BridgeResult:
        df = _record_to_dataframe(
            record, _record_keys_for(graph),
        )
        model = _build_causal_model(df, treatment, outcome, graph)
        identified = model.identify_effect(
            proceed_when_unidentifiable=False,
        )
        if (
            getattr(identified, 'no_directed_path', False)
            or not getattr(identified, 'estimands', None)
        ):
            return BridgeResult(
                verdict=Verdict.POWER_INSUFFICIENT,
                reason=(
                    f'no identified estimand for {treatment} → '
                    f'{outcome} under DAG'
                ),
                stats={
                    'identified': 0,
                    'tier': 'interventional',
                    'role': 'refuter',
                },
                name=bridge_name,
                targets=(treatment, outcome),
            )
        estimate = model.estimate_effect(
            identified, method_name=method_name,
        )
        refuter = model.refute_estimate(
            identified, estimate,
            method_name='random_common_cause',
        )
        new_effect = _refuter_effect(refuter)
        real_ate = float(estimate.value)
        drift = abs(new_effect - real_ate)
        if drift < tolerance:
            verdict = Verdict.HELD
            reason = (
                f'real ATE = {real_ate:+.4f}, with-RCC ATE = '
                f'{new_effect:+.4f}, |drift| = {drift:.4f} < '
                f'{tolerance:.4f}'
            )
        else:
            verdict = Verdict.NO_EFFECT
            reason = (
                f'real ATE = {real_ate:+.4f}, with-RCC ATE = '
                f'{new_effect:+.4f}, |drift| = {drift:.4f} ≥ '
                f'{tolerance:.4f}'
            )

        return BridgeResult(
            verdict=verdict, reason=reason,
            stats={
                'real_ate': real_ate,
                'rcc_ate': new_effect,
                'drift': drift,
                'tier': 'interventional',
                'role': 'refuter',
            },
            name=bridge_name,
            targets=(treatment, outcome),
        )

    return Bridge(fn=fn, name=bridge_name, targets=(treatment, outcome))
