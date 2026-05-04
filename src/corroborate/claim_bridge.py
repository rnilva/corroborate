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
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import polars as pl

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


def _build_invariant_measurable(
    *,
    name: str,
    source_name: str,
    threshold: float,
    direction: Direction,
) -> Measurable[Mapping[str, object], object]:
    """Construct a per-cell verdict Measurable from a threshold
    predicate. Hidden behind `Bridge.to_invariant_measurable()`;
    not a public entrypoint.

    The synthesized fn declares the source measurable as a
    dependency via `__signature__` (so the framework's
    parameter-name dep resolver injects the source's value at
    cache-build time), AND falls back to a record-direct read for
    post-hoc evaluations on a persisted parquet (where the source
    column is already present in the record dict)."""
    use_at_most = direction is Direction.AT_MOST

    def fn(record: Mapping[str, object], **kwargs: object) -> str:
        v: object = kwargs.get(source_name)
        if v is None:
            v = record.get(source_name)
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
            return 'power_insufficient'
        fv = float(v)
        if math.isnan(fv):
            return 'power_insufficient'
        if use_at_most:
            return 'held' if fv <= threshold else 'invariant_violation'
        return 'held' if fv >= threshold else 'invariant_violation'

    # Declare the dep via __signature__ so `_measurable_param_names`
    # picks `source_name` up and the resolver pre-computes it.
    fn.__signature__ = inspect.Signature(parameters=[  # type: ignore[attr-defined]
        inspect.Parameter('record', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter(
            source_name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None, annotation=float,
        ),
    ])
    return Measurable(fn=fn, name=name, reads=())


@dataclass(frozen=True, slots=True)
class Bridge:
    """Authored edge declaration on the measurable graph.

    Built by `@claim_bridge` from the wrapped function's
    signature. Structural fields name the edge (`source`,
    `target`, `direction`, `tier`); `params` is the bag of
    claim-specific kwargs the bridge forwards to each registered
    analysis the `holds_when` body consumes.

    `source` / `target` are `BridgeEndpoint`s: either a string
    (raw column path or a `@measurable`-registered name), a
    `Measurable` instance passed by value (typically a value-
    composed reduction from `corroborate.reductions`), OR — for
    `source` only — a `DoEffect`. `DoEffect` declares the
    Pearl-rung-2 contrast (treatment_arm / baseline_arm) AND
    routes the analysis: when `source` is a DoEffect, the
    analysis's `source` slot maps to `bridge.target_name` (the
    measurement column). When `source` is a string/measurable,
    that name flows directly as the analysis's `source`. The
    explicit per-bridge declaration replaces an earlier
    file-level `INTERVENTION = DoEffect(...)` auto-resolution
    that introduced source-vs-target routing ambiguity — bridges
    now state their contrast unambiguously.

    Passing a `Measurable` directly avoids the boilerplate of a
    top-level `@measurable` wrapper for every reduction variant;
    the framework auto-registers each by-value Measurable at
    decoration time so the cache walker finds it. The
    `endpoint_name` helper normalises str/Measurable cases to a
    single column-name string for analyses + the causal graph.

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
    walk consumes the Bridge as metadata (source / target / tier
    / predicted_direction) and computes the verdict from runs
    directly, never invoking a body. `evaluate` raises
    `TypeError` if called on a body-less Bridge.

    `threshold: float | None = None` is the predicate threshold
    for INVARIANT self-loop bridges (`Direction.AT_MOST` /
    `Direction.AT_LEAST`). When set together with `tier=INVARIANT`,
    `direction in {AT_MOST, AT_LEAST}`, and `source_name ==
    target_name`, the bridge is a substrate-axiom claim. Use
    `to_invariant_measurable()` to synthesize the per-cell verdict
    Measurable that the cache builder evaluates and persists.

    `scope: pl.Expr | None = None` filters the cell-set the
    framework hands to the analysis. The cache flows as a
    `pl.DataFrame`; `evaluate()` applies `df.filter(scope)` (with
    missing-column null-padding via `_filter_with_missing_cols`)
    before converting to dicts and forwarding. `None` means
    "match all". Replaces the legacy `env_name` /
    `extra_filters` / `extra_min_pairs` / `extra_max_pairs` /
    `cell_predicate` kwargs that used to live in the holds_when
    params bag — scope is structural metadata, not body argument.

    `pair_by: tuple[str, ...] = ('seed',)` is the pairing-axis
    tuple forwarded to analyses that compute paired contrasts.
    Typed Bridge field (rather than holds_when default) since
    81% of bridges use the same `('seed',)` value and never
    consume it in their body."""
    name: str
    source: BridgeEndpoint
    target: BridgeEndpoint
    direction: Direction = Direction.DIRECT
    tier: Tier = Tier.ASSOCIATIONAL
    pair_by: tuple[str, ...] = ('seed',)
    scope: pl.Expr | None = None
    params: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    predicted_direction: PredictedDirection | None = None
    holds_when: Callable[..., Verdict] | None = None
    threshold: float | None = None

    @property
    def source_name(self) -> str:
        """Column name normalised from `source` (str passes through;
        Measurable returns `.name`)."""
        return endpoint_name(self.source)

    @property
    def target_name(self) -> str:
        """Column name normalised from `target`."""
        return endpoint_name(self.target)

    def to_invariant_measurable(
        self,
    ) -> Measurable[Mapping[str, object], object]:
        """Synthesize the per-cell verdict Measurable for an
        INVARIANT self-loop bridge with a threshold predicate.

        Validity preconditions (raise `ValueError` otherwise):

        - `tier is Tier.INVARIANT`
        - `direction in {Direction.AT_MOST, Direction.AT_LEAST}`
        - `threshold is not None`
        - `source_name == target_name` (self-loop)
        - the source name matches a registered measurable (lazily
          checked: the synthesized fn looks up the source from the
          registry at evaluation time, so the dep-resolver can
          inject it before invoking)

        The synthesized Measurable is named after `bridge.name`
        and returns one of `'held'` / `'invariant_violation'` /
        `'power_insufficient'` per record:

        - `held` when the source value satisfies the predicate
          (`≤ threshold` for AT_MOST, `≥ threshold` for AT_LEAST).
        - `invariant_violation` when the source value violates it.
        - `power_insufficient` when the source is NaN, missing,
          or non-numeric.

        Caller must register the returned Measurable (typically
        as part of a substrate's default measurable panel)."""
        from corroborate.causal_graph import Direction as _D
        from corroborate.causal_graph import Tier as _T
        if self.tier is not _T.INVARIANT:
            raise ValueError(
                f'to_invariant_measurable: bridge {self.name!r} has '
                f'tier={self.tier.name}; required INVARIANT.',
            )
        if self.direction not in (_D.AT_MOST, _D.AT_LEAST):
            raise ValueError(
                f'to_invariant_measurable: bridge {self.name!r} has '
                f'direction={self.direction.name}; required AT_MOST '
                f'or AT_LEAST.',
            )
        if self.threshold is None:
            raise ValueError(
                f'to_invariant_measurable: bridge {self.name!r} has '
                f'no threshold; required for INVARIANT predicate.',
            )
        src_name = self.source_name
        tgt_name = self.target_name
        if src_name != tgt_name:
            raise ValueError(
                f'to_invariant_measurable: bridge {self.name!r} '
                f'declares source={src_name!r} != target={tgt_name!r}; '
                f'INVARIANT bridges are self-loops.',
            )

        return _build_invariant_measurable(
            name=self.name,
            source_name=src_name,
            threshold=self.threshold,
            direction=self.direction,
        )


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


def _require_scope(value: object, fn_name: str) -> pl.Expr | None:
    """Validate `scope` decorator arg. `None` → no filter; a
    `pl.Expr` is accepted as the polars-native cell predicate. Any
    other value is an authoring mistake — fail loudly at import."""
    if value is None:
        return None
    if isinstance(value, pl.Expr):
        return value
    raise TypeError(
        f'@claim_bridge {fn_name!r}: default for `scope` must be a '
        f'pl.Expr (e.g. `pl.col(\'env_name\') == \'X\'`) or None; '
        f'got {type(value).__name__}',
    )


def _require_pair_by(value: object, fn_name: str) -> tuple[str, ...]:
    """Validate `pair_by` decorator arg. Must be a tuple of strings.
    Empty tuple is allowed (analyses that don't pair ignore it)."""
    if not isinstance(value, tuple):
        raise TypeError(
            f'@claim_bridge {fn_name!r}: default for `pair_by` must '
            f'be a tuple[str, ...]; got {type(value).__name__}',
        )
    for k in cast(tuple[object, ...], value):
        if not isinstance(k, str):
            raise TypeError(
                f'@claim_bridge {fn_name!r}: pair_by entries must '
                f'be strings; got {type(k).__name__} ({k!r}).',
            )
    return cast(tuple[str, ...], value)


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
    pair_by: tuple[str, ...] = ('seed',),
    scope: pl.Expr | None = None,
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
            pair_by=('seed',),
            scope=(
                (pl.col('env_name') == 'Acrobot-v1')
                & (pl.col('reward_scale') == 0.1)
            ),
        )
        def some_bridge(
            paired_g: PairedGResult,           # fixture (analysis result)
        ) -> Verdict:
            ...

    Interventional bridges declare the do-contrast via
    `source = DoEffect(treatment_arm=..., baseline_arm=...)`.
    The framework extracts treatment/baseline arms from
    `bridge.source` at evaluate() time and threads them into the
    analysis's kwargs. When `source` is a string/Measurable, no
    contrast is set and the analysis runs without arm-pairing
    (correlation-style or pre-paired bridges).

    A common idiom is to define `INTERVENTION = DoEffect(...)`
    once at the top of the bridge file and reference it as
    `source=INTERVENTION` in each interventional bridge — the
    constant lives in the file's namespace, not in framework
    auto-resolution. Per-bridge variants (e.g. HP-encoded arms)
    declare their own DoEffect inline.
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
    pair_by_validated = _require_pair_by(
        pair_by, '<claim_bridge decorator>',
    )
    scope_validated = _require_scope(
        scope, '<claim_bridge decorator>',
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

        return Bridge(
            name=fn.__name__,
            source=source_validated,
            target=target_validated,
            direction=direction_validated,
            tier=tier_validated,
            pair_by=pair_by_validated,
            scope=scope_validated,
            params=MappingProxyType(params),
            holds_when=fn,
            predicted_direction=predicted_direction_validated,
        )

    return _decorator


def _filter_with_missing_cols(
    df: pl.DataFrame, expr: pl.Expr,
) -> pl.DataFrame:
    """Apply `expr` as a filter to `df`. For columns referenced
    by `expr` but absent from `df`:

    - If the name resolves in the `@measurable` registry, compute
      it per-cell (with shared dep memoisation via
      `evaluate_with_measurables`) and add it as a column. This
      is the "bridges-verify-against-raw-traces" path: when a
      bridge declares `scope = pl.col('jensen_dormancy_gap') >= 0`
      and the input DataFrame is a raw `runs.parquet` without
      that measurable yet, the framework computes it on the fly.
    - Otherwise, pre-fill as null. Universal-cache schema
      heterogeneity (corpus A has `reward_scale`, corpus B
      doesn't, neither corpus computed it) lands here — null
      rows fail the predicate naturally (polars filter excludes
      null-result rows), matching the legacy `_matches_filters`
      "missing key → False" semantics."""
    referenced = expr.meta.root_names()
    missing = [c for c in referenced if c not in df.columns]
    if not missing:
        return df.filter(expr)

    from corroborate.measurable import compute_missing_columns
    # `compute_missing_columns` resolves whichever names are
    # registered measurables and adds them as columns; names that
    # aren't registered remain absent and get null-padded so the
    # filter excludes them naturally (matching the legacy
    # `_matches_filters` "missing key → False" semantics).
    df = compute_missing_columns(df, missing)
    truly_missing = [c for c in missing if c not in df.columns]
    if truly_missing:
        df = df.with_columns(
            [pl.lit(None).alias(c) for c in truly_missing],
        )

    return df.filter(expr)


def evaluate(
    bridge: Bridge,
    cells: pl.DataFrame | Iterable[Mapping[str, object]],
) -> BridgeEvaluation:
    """Run a bridge against a cell-set: apply `bridge.scope` as a
    polars filter, resolve each fixture (a `holds_when` parameter
    without a default) by looking up the matching `@analysis`,
    parameterise from the bridge's structural fields + params,
    run on the filtered cells, inject results, return verdict +
    audit trail.

    Cell input may be either a `pl.DataFrame` (the canonical cache
    shape — fast, vectorised filter) or an `Iterable[Mapping]`
    (synthetic test cells, ad-hoc). Iterables are materialised
    into a DataFrame before filtering. Analyses receive the
    filtered cells as `list[dict]` after a single `to_dicts()`
    conversion — they don't see `pl.DataFrame` directly.

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
    filtered_cells: list[dict[str, object]]
    if bridge.scope is None:
        # No scope filter — skip the DataFrame round-trip. Convert
        # to list[dict] so the analysis fn can re-iterate.
        if isinstance(cells, pl.DataFrame):
            filtered_cells = cast(list[dict[str, object]], cells.to_dicts())
        else:
            filtered_cells = [dict(c) for c in cells]
    else:
        # Scope filter — materialise to DataFrame (if not already),
        # filter, convert back to list[dict] for the analysis.
        df: pl.DataFrame
        if isinstance(cells, pl.DataFrame):
            df = cells
        else:
            cells_list = list(cells)
            df = pl.from_dicts(cells_list) if cells_list else pl.DataFrame()
        if df.height > 0:
            df = _filter_with_missing_cols(df, bridge.scope)
        filtered_cells = (
            cast(list[dict[str, object]], df.to_dicts())
            if df.height > 0 else []
        )
    # Contrast resolution:
    #   - `source = DoEffect(...)` → contrast = the DoEffect, and
    #     the analysis's `source` slot maps to bridge.target_name
    #     (the measurement column).
    #   - `source = str | Measurable` → no contrast; the name flows
    #     directly as the analysis's source. Correlational bridges,
    #     or bridges where the author wants paired_g to compute on
    #     a non-outcome measurable.
    contrast: DoEffect | None
    source_for_analysis: str
    if isinstance(bridge.source, DoEffect):
        contrast = bridge.source
        source_for_analysis = bridge.target_name
    else:
        contrast = None
        source_for_analysis = bridge.source_name
    bridge_params: dict[str, object] = {
        'source': source_for_analysis,
        'target': bridge.target_name,
        'direction': bridge.direction,
        'tier': bridge.tier,
        'predicted_direction': bridge.predicted_direction,
        'pair_by': bridge.pair_by,
        **dict(bridge.params),
    }
    if contrast is not None:
        bridge_params['treatment_arm'] = contrast.treatment_arm
        bridge_params['baseline_arm'] = contrast.baseline_arm
    analysis_results = resolve_for_holds_when(
        bridge.holds_when, filtered_cells, bridge_params,
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
