"""Claim-bridge — typed authored edge declaration on the
measurable graph, with `holds_when` threshold body.

A claim bridge is the authoring unit of the measurable-graph
file protocol. The author writes a Python function whose
*signature* IS the declaration:

  - **Function name** = bridge name.
  - **Defaulted kwargs `source`, `target`, `direction`, `tier`** =
    structural fields of the edge.
  - **Other defaulted kwargs** = the params bag the framework
    forwards to analyses at evaluation time (`treatment_arm`,
    `pair_by`, `env_name`, `dag`, ...).
  - **Parameters WITHOUT defaults** = analysis fixtures the
    framework injects by name. The type annotation
    (`paired_g: PairedGResult`) is for the IDE/type-checker;
    runtime resolution is by parameter name against the
    `@analysis` registry.

  - **Function body** = the `holds_when` threshold — explicit
    sign / magnitude / power criterion, returning a `Verdict`.

The decorator is no-arg: `@claim_bridge` reads everything from
the function's signature. Like a pytest test that consumes
fixtures, but with the test's metadata sitting as keyword
defaults.

    @claim_bridge
    def ddqn_helps_outcome_acrobot(
        paired_g: PairedGResult,
        *,
        source: str = 'eval_best_burst_mean',
        target: str = 'eval_best_burst_mean',
        direction: Direction = Direction.DIRECT,
        tier: Tier = Tier.ASSOCIATIONAL,
        treatment_arm: str = 'ddqn',
        baseline_arm: str = 'vanilla_dqn',
        pair_by: tuple[str, ...] = ('seed',),
        env_name: str = 'Acrobot-v1',
    ) -> Verdict:
        if paired_g.g > 0.3 and paired_g.p_value < 0.05:
            return Verdict.HELD
        if paired_g.n_pairs < 30:
            return Verdict.POWER_INSUFFICIENT
        return Verdict.NO_EFFECT
"""
from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from corroborate._introspection_boundary import get_param_default
from corroborate.analysis import resolve_for_holds_when
from corroborate.hypothesis import PredictedDirection
from corroborate.intervention import DoEffect
from corroborate.measurable import Measurable, register
from corroborate.verdict import Verdict


# Direction and Tier are the canonical edge-metadata enums used
# across the framework. Re-imported from `causal_graph` rather
# than redefined to keep the two layers' types unified — the
# graph builder reads `bridge.tier` / `bridge.direction` directly
# without conversion.
from corroborate.causal_graph import Direction, Tier  # noqa: E402


# A bridge endpoint is one of:
# - `str`: raw column path or a name that resolves in the
#   @measurable registry
# - `Measurable`: instance passed by value (typically a value-
#   composed reduction like `mean_window(from_key('q_max'), 0.5,
#   1.0)`). The framework normalises to `.name`.
# - `DoEffect`: ONLY valid as `Bridge.source`, not target. Marks
#   the bridge as Pearl-rung-2 — the source IS the do-contrast,
#   not a measurable. Per `intervention.py:153`: "Pearl-rung-2
#   edges in the causal graph have an *intervention* as the
#   source node, NOT a measurable." Analyses that consume a
#   DoEffect-sourced bridge get the contrast's `treatment_arm` /
#   `baseline_arm` extracted into their kwargs by `evaluate()`.
type BridgeEndpoint = (
    str | Measurable[Mapping[str, object], object] | DoEffect
)


def endpoint_name(e: BridgeEndpoint) -> str:
    """Normalise a `BridgeEndpoint` to a column name (str). For
    str inputs the name passes through; for `Measurable` instances
    `.name` is returned. For `DoEffect`, returns
    `node_key()` — the `'do(treatment|vs=baseline)'` graph-render
    string. Analyses that consume a DoEffect-sourced bridge see
    `treatment_arm` / `baseline_arm` extracted into their kwargs
    by `evaluate()`; the do-string is for graph builders, not
    measurable-resolution.

    Single laundering point so the causal graph builder and the
    cache walker can treat source/target uniformly as strings."""
    if isinstance(e, str):
        return e
    if isinstance(e, DoEffect):
        return e.node_key()
    return e.name


_PREDICTED_DIRECTION_VALUES: frozenset[str] = frozenset(
    {'a_gt_b', 'a_lt_b', 'two_sided'},
)


@dataclass(frozen=True, slots=True)
class Bridge:
    """Authored edge declaration on the measurable graph.

    Built by `@claim_bridge` from the wrapped function's
    signature. Structural fields name the edge (`source`,
    `target`, `direction`, `tier`); `params` is the bag of
    claim-specific kwargs the bridge forwards to each registered
    analysis the `holds_when` body consumes.

    `source` / `target` are `BridgeEndpoint`s: either a string
    (raw column path or a `@measurable`-registered name) OR a
    `Measurable` instance passed by value (typically a value-
    composed reduction from `corroborate.reductions`). Passing a
    Measurable directly avoids the boilerplate of a top-level
    `@measurable` wrapper for every reduction variant; the
    framework auto-registers each by-value Measurable at
    decoration time so the cache walker finds it. The
    `endpoint_name` helper normalises both cases to a single
    column-name string for analyses + the causal graph.

    `intervention: DoEffect | None` is the Pearl-rung-2
    annotation: the do-contrast for analyses that need it
    (paired_g, dowhy, mundlak, etc.). Auto-resolved at
    decoration time from `module.INTERVENTION` — bridge authors
    declare it once at the top of the file rather than per-
    bridge. Per-bridge `source = DoEffect(...)` (in the decorator
    args) overrides the module-level default. The graph builder
    emits a `do(treatment|vs=baseline) → target` edge when
    either source-as-DoEffect or this field is set.

    `predicted_direction: PredictedDirection | None` is the
    author-declared *prior* sign of the predicted effect, used by
    paired/random-effects analyses to pick a one- vs two-sided
    test. Promoted out of `params` because it is shared structural
    metadata across most claims (consumed by `paired_g`,
    `random_effects`, etc.) — keeping it inside the params bag
    forced every analysis to fish it out by name.

    `holds_when: Callable[..., Verdict] | None` is optional. The
    file-protocol path (analyses on a corpus) sets it; the
    Hypothesis-side typed-edge path (verdict walks via
    `hypothesis_subgraph_verdict`) leaves it None — the verdict
    walk consumes the Bridge as metadata (source / target /
    intervention / tier / predicted_direction) and computes the
    verdict from runs directly, never invoking a body. `evaluate`
    raises `TypeError` if called on a body-less Bridge."""
    name: str
    source: BridgeEndpoint
    target: BridgeEndpoint
    direction: Direction = Direction.DIRECT
    tier: Tier = Tier.ASSOCIATIONAL
    params: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    intervention: DoEffect | None = None
    predicted_direction: PredictedDirection | None = None
    holds_when: Callable[..., Verdict] | None = None

    @property
    def source_name(self) -> str:
        """Column name normalised from `source` (str passes through;
        Measurable returns `.name`)."""
        return endpoint_name(self.source)

    @property
    def target_name(self) -> str:
        """Column name normalised from `target`."""
        return endpoint_name(self.target)


@dataclass(frozen=True, slots=True)
class BridgeEvaluation:
    """One bridge evaluated against one cell-set: the verdict the
    `holds_when` body returned + the analysis results that
    produced it (the audit trail)."""
    bridge_name: str
    verdict: Verdict
    analysis_results: Mapping[str, object]


def _require_endpoint(
    value: object, field_name: str, fn_name: str,
    *,
    allow_do_effect: bool = False,
) -> BridgeEndpoint:
    """Validate a `source` / `target` default. Accepts a str (raw
    column or registered measurable name), a `Measurable` instance
    (value-composed reduction), or — only when `allow_do_effect=True`
    (i.e., for `source`) — a `DoEffect` (Pearl-rung-2 do-contrast).
    Anything else is an authoring mistake — fail loudly at import
    time."""
    if isinstance(value, str):
        return value
    if isinstance(value, Measurable):
        return value
    if allow_do_effect and isinstance(value, DoEffect):
        return value
    if not allow_do_effect and isinstance(value, DoEffect):
        raise TypeError(
            f'@claim_bridge {fn_name!r}: `target` cannot be a '
            f'DoEffect — only `source` may carry the Pearl-rung-2 '
            f'do-contrast. `target` must be a measurable column.',
        )
    raise TypeError(
        f'@claim_bridge {fn_name!r}: default for {field_name!r} '
        f'must be a str or Measurable'
        f'{" or DoEffect" if allow_do_effect else ""}; '
        f'got {type(value).__name__}',
    )


def _require_direction(
    value: object, fn_name: str,
) -> Direction:
    if not isinstance(value, Direction):
        raise TypeError(
            f'@claim_bridge {fn_name!r}: default for `direction` '
            f'must be a Direction enum; got {type(value).__name__}',
        )
    return value


def _require_tier(value: object, fn_name: str) -> Tier:
    if not isinstance(value, Tier):
        raise TypeError(
            f'@claim_bridge {fn_name!r}: default for `tier` must '
            f'be a Tier enum; got {type(value).__name__}',
        )
    return value


def _require_predicted_direction(
    value: object, fn_name: str,
) -> PredictedDirection | None:
    # Literal-by-literal narrowing — pyright narrows on `==
    # 'literal'` comparisons. Same pattern as
    # `_narrow.optional_direction` and `require_verdict`. Avoids a
    # `cast` at the end (which would smuggle the runtime
    # set-membership check past the type system).
    if value is None:
        return None
    if value == 'a_gt_b':
        return 'a_gt_b'
    if value == 'a_lt_b':
        return 'a_lt_b'
    if value == 'two_sided':
        return 'two_sided'
    raise TypeError(
        f'@claim_bridge {fn_name!r}: default for '
        f'`predicted_direction` must be one of '
        f"{sorted(_PREDICTED_DIRECTION_VALUES)!r} (or None); "
        f'got {value!r}',
    )


def claim_bridge(
    *,
    source: BridgeEndpoint,
    target: BridgeEndpoint,
    direction: Direction = Direction.DIRECT,
    tier: Tier = Tier.ASSOCIATIONAL,
    predicted_direction: PredictedDirection | None = None,
) -> Callable[[Callable[..., Verdict]], Bridge]:
    """Decorator factory: wraps a function into a `Bridge`
    declaration. Bridge metadata lives in the decorator args; the
    function signature carries only fixture parameters (analyses)
    and analysis-tool kwargs.

        @claim_bridge(
            source='eval_best_burst_mean',
            target='eval_best_burst_mean',
            direction=Direction.DIRECT,
            tier=Tier.INTERVENTIONAL,
        )
        def some_bridge(
            paired_g: PairedGResult,           # fixture (analysis result)
            *,
            pair_by: tuple[str, ...] = ('seed',),
            extra_filters: Mapping[str, object] = MappingProxyType({...}),
        ) -> Verdict:
            ...

    Module-level `INTERVENTION = DoEffect(...)` provides the
    contrast for all bridges in the file unless overridden via
    `source = DoEffect(...)` in the decorator (per-bridge
    different intervention).

    `Bridge.intervention` is auto-resolved from
    `module.INTERVENTION` at decoration time and threaded into
    analysis kwargs by `evaluate()`. Bridge authors do NOT write
    `treatment_arm` / `baseline_arm` / `intervention=` as bridge
    params anywhere.
    """
    # Validate decorator args at module-import time (early failure).
    source_validated = _require_endpoint(
        source, 'source', '<claim_bridge decorator>',
        allow_do_effect=True,
    )
    target_validated = _require_endpoint(
        target, 'target', '<claim_bridge decorator>',
    )
    direction_validated = _require_direction(
        direction, '<claim_bridge decorator>',
    )
    tier_validated = _require_tier(
        tier, '<claim_bridge decorator>',
    )
    predicted_direction_validated = _require_predicted_direction(
        predicted_direction, '<claim_bridge decorator>',
    )

    def _decorator(fn: Callable[..., Verdict]) -> Bridge:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f'@claim_bridge {fn.__name__!r}: cannot inspect '
                f'signature: {exc}',
            ) from exc

        # Defaulted kwargs become `Bridge.params`; non-defaulted
        # parameters are fixtures resolved at evaluate time.
        params: dict[str, object] = {}
        for param_name, param in sig.parameters.items():
            default = get_param_default(param)
            if default is inspect.Parameter.empty:
                continue
            params[param_name] = default

        # Auto-register Measurable instances passed by value.
        if isinstance(source_validated, Measurable):
            register(source_validated)
        if isinstance(target_validated, Measurable):
            register(target_validated)
        for v in params.values():
            if isinstance(v, Measurable):
                register(v)

        # Module-level INTERVENTION resolution. The module declares
        # `INTERVENTION = DoEffect(...)` at top level; every bridge
        # in that file inherits the contrast as a default. Per-bridge
        # `source = DoEffect(...)` overrides it.
        module = sys.modules.get(fn.__module__)
        module_intervention: object = (
            getattr(module, 'INTERVENTION', None)
            if module is not None else None
        )
        if module_intervention is not None and not isinstance(
            module_intervention, DoEffect,
        ):
            raise TypeError(
                f'@claim_bridge {fn.__name__!r}: module '
                f'{fn.__module__!r} declares `INTERVENTION` as '
                f'{type(module_intervention).__name__}; must be a '
                f'DoEffect (or omitted).',
            )
        intervention = (
            module_intervention
            if isinstance(module_intervention, DoEffect)
            else None
        )

        return Bridge(
            name=fn.__name__,
            source=source_validated,
            target=target_validated,
            direction=direction_validated,
            tier=tier_validated,
            params=MappingProxyType(params),
            holds_when=fn,
            intervention=intervention,
            predicted_direction=predicted_direction_validated,
        )

    return _decorator


def evaluate(
    bridge: Bridge,
    cells: Iterable[Mapping[str, object]],
) -> BridgeEvaluation:
    """Run a bridge against a cell-set: resolve each fixture (a
    `holds_when` parameter without a default) by looking up the
    matching `@analysis`, parameterise from the bridge's
    structural fields + params, run on `cells`, inject results,
    return verdict + audit trail.

    `source` and `target` are normalised via `endpoint_name`
    before reaching analyses — analyses always see the column-name
    string regardless of whether the bridge declared the endpoint
    as a string or a `Measurable` instance.

    Raises `TypeError` if the Bridge has no `holds_when` body —
    such a Bridge is a typed-edge declaration only (the
    Hypothesis-side verdict-walk surface) and carries no
    threshold logic to invoke."""
    if bridge.holds_when is None:
        raise TypeError(
            f'evaluate({bridge.name!r}): Bridge has no holds_when '
            f'body. Body-less Bridges are typed-edge declarations '
            f'consumed by `hypothesis_subgraph_verdict`; they do '
            f'not carry a threshold to evaluate against a cell-set.',
        )
    # Contrast resolution precedence (decided here):
    #   1. `source = DoEffect(...)` (per-bridge explicit override)
    #   2. `Bridge.intervention` (module-level `INTERVENTION` or
    #      legacy per-bridge `intervention=` field, resolved at
    #      decoration time)
    #   3. None (correlation/correlation-like bridges with no
    #      arm contrast)
    #
    # When source is a DoEffect, the analysis's `source` slot maps
    # to the bridge's TARGET measurable (the column to compute on).
    # When source is a measurable, it stays as-is.
    contrast: DoEffect | None
    source_for_analysis: str
    if isinstance(bridge.source, DoEffect):
        contrast = bridge.source
        source_for_analysis = bridge.target_name
    elif bridge.intervention is not None:
        contrast = bridge.intervention
        source_for_analysis = bridge.source_name
    else:
        contrast = None
        source_for_analysis = bridge.source_name
    bridge_params: dict[str, object] = {
        'source': source_for_analysis,
        'target': bridge.target_name,
        'direction': bridge.direction,
        'tier': bridge.tier,
        'predicted_direction': bridge.predicted_direction,
        **dict(bridge.params),
    }
    # Inject contrast arms ONLY when the bridge hasn't already
    # supplied them via legacy `treatment_arm` / `baseline_arm`
    # params. Migrated bridges drop those params and inherit from
    # the resolved contrast; unmigrated bridges keep their explicit
    # arm strings during the transition.
    if contrast is not None:
        if 'treatment_arm' not in bridge_params:
            bridge_params['treatment_arm'] = contrast.treatment_arm
        if 'baseline_arm' not in bridge_params:
            bridge_params['baseline_arm'] = contrast.baseline_arm
    analysis_results = resolve_for_holds_when(
        bridge.holds_when, cells, bridge_params,
    )
    verdict = bridge.holds_when(**analysis_results)
    return BridgeEvaluation(
        bridge_name=bridge.name,
        verdict=verdict,
        analysis_results=MappingProxyType(dict(analysis_results)),
    )


def measurable_names_for_bridges(
    bridges: Iterable[Bridge],
) -> frozenset[str]:
    """Walk `bridges` → return every registered-measurable name
    they consume (transitively via the @measurable graph).

    A bridge declares measurable names through three channels:

    - `bridge.source` and `bridge.target` are the canonical
      measurable-name slots. They may be either a string
      (registered measurable name or raw column path) OR a
      `Measurable` instance passed by value (auto-registered at
      `@claim_bridge` decode time). Both shapes normalise to a
      column-name string via `bridge.source_name` /
      `bridge.target_name`.
    - `bridge.params[*]` may carry measurable names too — bridges
      authored with extra defaulted-string kwargs (`predictor_name`,
      `mediator`) that downstream analyses route through the registry.
      `Measurable` instances passed via params are also auto-
      registered + walked.

    For each declared name that's a registered measurable, expand
    via `transitive_measurables` to include every dep. Returns
    the union — exactly the set a cache builder must materialise
    so analyses on this bridge file find their scalars
    pre-computed.

    Names that aren't in the measurable registry are silently
    dropped. They're either claim-output field paths
    (`eval_best_burst_mean`) — already in the raw
    parquet, no recompute needed — or bridge-specific identifiers
    that aren't measurables (e.g. `'outcome_native'` IS a
    measurable, but `'eval_best_burst_mean'` as a raw field is
    not).
    """
    from corroborate.measurable import (
        get_registered, transitive_measurables,
    )
    out: set[str] = set()
    for b in bridges:
        candidates: list[str] = [b.source_name, b.target_name]
        for v in b.params.values():
            if isinstance(v, str):
                candidates.append(v)
            elif isinstance(v, Measurable):
                candidates.append(v.name)
            elif isinstance(v, (tuple, list)):
                # Tuple/list params (e.g. `covariates: tuple[str, ...]`)
                # carry column names too — bridges declaring a list
                # of measurable-derived covariates expect each name
                # in the cache.
                for item in v:
                    if isinstance(item, str):
                        candidates.append(item)
                    elif isinstance(item, Measurable):
                        candidates.append(item.name)
        for name in candidates:
            if get_registered(name) is None:
                continue
            out.update(transitive_measurables(name))
    return frozenset(out)


__all__ = [
    'Bridge',
    'BridgeEndpoint',
    'BridgeEvaluation',
    'Direction',
    'Tier',
    'claim_bridge',
    'endpoint_name',
    'evaluate',
    'measurable_names_for_bridges',
]
