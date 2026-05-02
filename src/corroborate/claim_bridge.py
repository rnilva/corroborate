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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

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


# A bridge endpoint is either a string (raw column path or a name
# that resolves in the @measurable registry) OR a `Measurable`
# instance (typically a value-composed reduction like
# `mean_window(from_key('q_max'), 0.5, 1.0)`). The framework
# normalises the latter to its `.name` before handing off to
# analyses, which only see strings.
type BridgeEndpoint = str | Measurable[Mapping[str, object], object]


def endpoint_name(e: BridgeEndpoint) -> str:
    """Normalise a `BridgeEndpoint` to a column name. For str
    inputs the name passes through unchanged; for `Measurable`
    instances the `.name` field is returned (the cache builder
    materialises that name as a parquet column).

    Single laundering point so analyses, the causal graph builder,
    and the cache walker can treat source/target uniformly as
    strings."""
    if isinstance(e, str):
        return e
    return e.name


# Reserved kwarg names the decorator extracts as the bridge's
# structural fields; everything else lands in `params`.
_STRUCTURAL_FIELDS: frozenset[str] = frozenset(
    {'source', 'target', 'direction', 'tier', 'intervention',
     'predicted_direction'},
)


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
    annotation: when set, the graph builder emits an
    `do(treatment|vs=baseline) → target` edge instead of (or in
    addition to) the measurable-to-measurable one. Required for
    bridges whose verdict comes from an intervention contrast
    (paired_g.mean_diff between treatment_arm and baseline_arm
    cells). Strictly stronger than burying the arm names in
    `params` — surfaces the do() relationship at the graph
    layer, where it belongs.

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
) -> BridgeEndpoint:
    """Validate a `source` / `target` default. Accepts either a
    str (raw column or registered measurable name) or a
    `Measurable` instance (value-composed reduction). Anything
    else is an authoring mistake — fail loudly at import time."""
    if isinstance(value, str):
        return value
    if isinstance(value, Measurable):
        return value
    raise TypeError(
        f'@claim_bridge {fn_name!r}: default for {field_name!r} '
        f'must be a str or Measurable; got {type(value).__name__}',
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
    if value is None:
        return None
    if (not isinstance(value, str)
            or value not in _PREDICTED_DIRECTION_VALUES):
        raise TypeError(
            f'@claim_bridge {fn_name!r}: default for '
            f'`predicted_direction` must be one of '
            f"{sorted(_PREDICTED_DIRECTION_VALUES)!r} (or None); "
            f'got {value!r}',
        )
    # `value` is now provably a member of the literal set.
    return cast(PredictedDirection, value)


def claim_bridge(fn: Callable[..., Verdict]) -> Bridge:
    """Wrap `fn` into a `Bridge` declaration. Reads metadata from
    the function's name + signature defaults:

    - bridge name = `fn.__name__`
    - structural fields (`source`, `target`, `direction`, `tier`)
      = the function's defaulted kwargs of those reserved names
    - other defaulted kwargs → `Bridge.params`
    - parameters without defaults → analysis fixtures, resolved
      at evaluate time

    Raises `TypeError` if `source`/`target` aren't both supplied
    as defaulted-string kwargs, or if `direction`/`tier` are
    supplied with the wrong type.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f'@claim_bridge {fn.__name__!r}: cannot inspect '
            f'signature: {exc}',
        ) from exc

    structural: dict[str, object] = {}
    params: dict[str, object] = {}
    for param_name, param in sig.parameters.items():
        # `inspect.Parameter.default` is `Any` per typeshed; cast
        # at the boundary. Tracked in FUTURE_WORKS.md.
        default = cast(object, param.default)
        if default is inspect.Parameter.empty:
            continue  # fixture; resolved at evaluate time
        if param_name in _STRUCTURAL_FIELDS:
            structural[param_name] = default
        else:
            params[param_name] = default

    if 'source' not in structural or 'target' not in structural:
        raise TypeError(
            f'@claim_bridge {fn.__name__!r}: must declare both '
            f'`source` and `target` as defaulted kwargs',
        )
    source = _require_endpoint(
        structural['source'], 'source', fn.__name__,
    )
    target = _require_endpoint(
        structural['target'], 'target', fn.__name__,
    )
    # Auto-register Measurable instances passed by value so the
    # cache walker (`measurable_names_for_bridges`) finds them via
    # the standard registry path. Idempotent on (name, identity).
    if isinstance(source, Measurable):
        register(source)
    if isinstance(target, Measurable):
        register(target)
    for v in params.values():
        if isinstance(v, Measurable):
            register(v)
    direction = _require_direction(
        structural.get('direction', Direction.DIRECT), fn.__name__,
    )
    tier = _require_tier(
        structural.get('tier', Tier.ASSOCIATIONAL), fn.__name__,
    )
    intervention_default = structural.get('intervention')
    if intervention_default is not None and not isinstance(
        intervention_default, DoEffect,
    ):
        raise TypeError(
            f'@claim_bridge {fn.__name__!r}: default for '
            f'`intervention` must be a DoEffect (or omitted); got '
            f'{type(intervention_default).__name__}',
        )
    predicted_direction = _require_predicted_direction(
        structural.get('predicted_direction'), fn.__name__,
    )

    return Bridge(
        name=fn.__name__,
        source=source,
        target=target,
        direction=direction,
        tier=tier,
        params=MappingProxyType(params),
        holds_when=fn,
        intervention=intervention_default,
        predicted_direction=predicted_direction,
    )


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
    bridge_params: dict[str, object] = {
        'source': bridge.source_name,
        'target': bridge.target_name,
        'direction': bridge.direction,
        'tier': bridge.tier,
        'predicted_direction': bridge.predicted_direction,
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
