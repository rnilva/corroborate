"""Claim-bridge — typed authored edge declaration on the
measurable graph, with `holds_when` threshold body.

A *claim bridge* is the authoring unit of the measurable-graph
file protocol: a structural declaration (source, target,
direction, tier, plus claim-specific kwargs) paired with a
`holds_when` body that consumes registered `@analysis` results
and returns a `Verdict`. Like a pytest test that consumes
fixtures.

    @claim_bridge(
        name='ddqn_helps_outcome_acrobot',
        source='outcome.eval_best_burst_mean',
        target='outcome.eval_best_burst_mean',
        direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL,
        treatment_arm='ddqn',
        baseline_arm='vanilla_dqn',
        pair_by=('seed',),
        env_name='Acrobot-v1',
    )
    def claim(paired_g: PairedGResult) -> Verdict:
        if paired_g.g > 0.3 and paired_g.p < 0.05:
            return Verdict.HELD
        if paired_g.n_pairs < 30:
            return Verdict.POWER_INSUFFICIENT
        return Verdict.NO_EFFECT

`evaluate(bridge, cells)` resolves every analysis the
`holds_when` body references (by parameter name, against the
analysis registry), parameterises each from the bridge's
structural fields + kwargs, runs them on `cells`, injects the
results into `holds_when`, and returns `(verdict,
analysis_results)` — verdict is the bridge's authored claim;
analysis_results is the audit trail.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType

from corroborate.analysis import resolve_for_holds_when
from corroborate.verdict import Verdict


class Direction(Enum):
    """Sign of an edge's claimed coupling.

    `DIRECT`: source ↑ ⇒ target ↑.
    `INVERSE`: source ↑ ⇒ target ↓.

    Lifted from the legacy `corroborate.causal_graph` so the
    file-protocol bridges have direction structure without
    importing the deprecated module."""
    DIRECT = 'direct'
    INVERSE = 'inverse'


class Tier(IntEnum):
    """Pearl-ladder tier of an edge's evidentiary claim.

    `ASSOCIATIONAL`: observational coupling (rung 1).
    `INTERVENTIONAL`: confirmed do-operation effect (rung 2).
    """
    ASSOCIATIONAL = 1
    INTERVENTIONAL = 2


@dataclass(frozen=True, slots=True)
class Bridge:
    """Authored edge declaration on the measurable graph.

    Structural fields name the edge (`source`, `target`,
    `direction`, `tier`); `params` is the open kwarg bag the
    bridge author populates with claim-specific values
    (`treatment_arm`, `pair_by`, `env_name`, `dag`,
    `adjustment_set`, ...) that the framework forwards to each
    registered analysis the `holds_when` body consumes.

    `holds_when` is the body of the bridge: a typed callable
    whose parameter names match registered `@analysis` names.
    The framework injects analysis results at evaluation time;
    the body returns a `Verdict`.

    Two bridges with the same `(source, target, direction, tier,
    params)` and identical `holds_when` are structurally
    equivalent; the framework doesn't enforce uniqueness, but
    the smoke layer can canonicalise."""
    name: str
    source: str
    target: str
    holds_when: Callable[..., Verdict]
    direction: Direction = Direction.DIRECT
    tier: Tier = Tier.ASSOCIATIONAL
    params: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({}),
    )


@dataclass(frozen=True, slots=True)
class BridgeEvaluation:
    """The result of evaluating one bridge against one cell-set:
    the authored verdict + the analysis results that produced it.

    `analysis_results` is the audit trail; downstream tooling
    (FINDINGS reproduction, falsifiability diff) can inspect the
    raw fixtures. Each key matches a `holds_when` parameter
    name."""
    bridge_name: str
    verdict: Verdict
    analysis_results: Mapping[str, object]


def claim_bridge(
    *,
    name: str,
    source: str,
    target: str,
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    **params: object,
) -> Callable[[Callable[..., Verdict]], Bridge]:
    """Decorator factory: wrap a `holds_when` body into a
    `Bridge` declaration. All extra kwargs to the decorator
    land in `Bridge.params` for the resolver to forward to
    analyses.

    Usage:

        @claim_bridge(
            name='X',
            source='source_measurable',
            target='target_measurable',
            direction=Direction.DIRECT,
            tier=Tier.ASSOCIATIONAL,
            # claim-specific params (forwarded to analyses):
            treatment_arm='ddqn',
            baseline_arm='vanilla_dqn',
            pair_by=('seed',),
        )
        def claim(paired_g: PairedGResult) -> Verdict:
            ...
    """
    def decorator(fn: Callable[..., Verdict]) -> Bridge:
        return Bridge(
            name=name,
            source=source,
            target=target,
            direction=direction,
            tier=tier,
            params=MappingProxyType(dict(params)),
            holds_when=fn,
        )
    return decorator


def evaluate(
    bridge: Bridge,
    cells: Iterable[Mapping[str, object]],
) -> BridgeEvaluation:
    """Run a bridge against a cell-set: resolve each analysis the
    `holds_when` body references, parameterise from the bridge's
    structural fields + params, inject results, return the
    verdict + audit trail.

    The bridge's structural fields (`source`, `target`,
    `direction`, `tier`) are added to the parameter bag the
    analyses receive — analyses that take e.g. `source: str` as
    a kwarg pull it from there; analyses that don't accept it
    silently ignore it (filtered by `Analysis._kwargs_for`)."""
    bridge_params: dict[str, object] = {
        'source': bridge.source,
        'target': bridge.target,
        'direction': bridge.direction,
        'tier': bridge.tier,
        **dict(bridge.params),
    }
    analysis_results = resolve_for_holds_when(
        bridge.holds_when, cells, bridge_params,
    )
    verdict = bridge.holds_when(**analysis_results)
    return BridgeEvaluation(
        bridge_name=bridge.name,
        verdict=verdict,
        analysis_results=MappingProxyType(dict(analysis_results)),
    )


__all__ = [
    'Bridge',
    'BridgeEvaluation',
    'Direction',
    'Tier',
    'claim_bridge',
    'evaluate',
]
