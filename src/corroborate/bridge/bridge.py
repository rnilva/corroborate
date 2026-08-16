"""Claim bridges — executable test modules on the measurable graph.

A bridge is the atomic claim-authoring unit:

* decorator arguments declare the structural edge, scope, pairing axes,
  evidential tier, and predicted direction;
* function parameters without defaults are registered analysis fixtures,
  injected by name at evaluation time;
* defaulted function parameters configure those analyses and the verdict
  thresholds; and
* the function body maps the injected evidence to a ``Verdict``.

The resulting module is independent of any observed data. It resembles a
pytest test that consumes fixtures, with scientific metadata made explicit:

    @claim_bridge(
        source='<intervention_parameter>',
        target='<outcome_metric>',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
        pair_by=('seed',),
        predicted_direction='a_gt_b',
    )
    def treatment_helps_outcome(
        paired_g: PairedGResult,
        *,
        minimum_pairs: int = 30,
    ) -> Verdict:
        if paired_g.n_pairs < minimum_pairs:
            return Verdict.POWER_INSUFFICIENT
        if paired_g.g > 0.3 and paired_g.p_value < 0.05:
            return Verdict.HELD
        return Verdict.NO_EFFECT
"""
from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Protocol, cast

import polars as pl

from corroborate._internals.introspection import get_param_default
from corroborate.bridge._filter import filter_cells
from corroborate.bridge.deferred_scope import DeferredScope
from corroborate.bridge.admission import (
    AUTO_GATES,
    AdmissionGate,
    GateLevel,
    GateResult,
    distinct_units,
    exogenous_source,
)
from corroborate.bridge.analysis import resolve_for_holds_when
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.core.claim import Claim
from corroborate.core.hypothesis import PredictedDirection
from corroborate.core.intervention import ArmRole, DoEffect
from corroborate.measurables import Measurable, register
from corroborate.graph._extent import stable_extent_hash


# Direction and Tier are the canonical edge-metadata enums used
# across the framework. Re-imported from `causal_graph` rather
# than redefined to keep the two layers' types unified — the
# graph builder reads `bridge.tier` / `bridge.direction` directly
# without conversion.
from corroborate.graph.causal import Direction, Tier  # noqa: E402


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


class RecordedContrastBinding(Protocol):
    """Verified runtime binding for an externally executed contrast.

    The bridge layer depends only on this structural surface, not on
    ``corroborate.data.adapter``.  ``RecordedContrast`` satisfies it,
    while the authored bridge remains independent of any bundle or
    producer-specific arm labels.
    """

    @property
    def parameter_path(self) -> str: ...

    @property
    def baseline_key(self) -> str: ...

    @property
    def treatment_key(self) -> str: ...

    @property
    def baseline_value(self) -> float: ...

    @property
    def treatment_value(self) -> float: ...

    @property
    def bundle_digest(self) -> str: ...


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
    {'a_gt_b', 'a_lt_b', 'two_sided', 'null'},
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


@dataclass(frozen=True)
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

    `holds_when: Callable[..., 'Verdict | tuple[Verdict, RefutationClass | None]'] | None` carries the
    threshold body. `@claim_bridge` always populates it from the
    decorated function; constructing a body-less Bridge directly
    is a programming error. `evaluate` raises `TypeError` if
    called on a body-less Bridge as a defensive guard.

    `threshold: float | None = None` is the predicate threshold
    for INVARIANT self-loop bridges (`Direction.AT_MOST` /
    `Direction.AT_LEAST`). When set together with `tier=INVARIANT`,
    `direction in {AT_MOST, AT_LEAST}`, and `source_name ==
    target_name`, the bridge is a substrate-axiom claim. Use
    `to_invariant_measurable()` to synthesize the per-cell verdict
    Measurable that the cache builder evaluates and persists.

    `scope: pl.Expr | None = None` filters the cell-set
    the framework hands to the analysis. The cache flows as a
    `pl.DataFrame`; `evaluate()` applies `df.filter(scope)`
    (with missing-column null-padding via
    `filter_cells`) before converting to dicts and
    forwarding. `None` means "match all". Replaces the legacy
    `env_name` / `extra_filters` / `extra_min_pairs` /
    `extra_max_pairs` / `cell_predicate` kwargs that used to
    live in the holds_when params bag — `scope` is
    structural metadata, not body argument. (Renamed from
    `scope` to disambiguate from the analytical `Scope` claim
    in `bridge.scope`.)

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
    scope: 'pl.Expr | DeferredScope | None' = None
    predicted_direction: PredictedDirection | None = None
    holds_when: Callable[..., 'Verdict | tuple[Verdict, RefutationClass | None]'] | None = None
    threshold: float | None = None
    gates: tuple['AdmissionGate', ...] = ()

    @cached_property
    def params(self) -> Mapping[str, object]:
        """Defaulted kwargs of `holds_when`, derived from
        `inspect.signature` once on first access then cached.

        Bridge is frozen but not slotted — `cached_property`
        writes through `__dict__`, which `frozen=True` doesn't
        block (it only intercepts `__setattr__`). First access:
        ~7 µs (signature walk + dict comprehension). Subsequent
        accesses: ~0.04 µs (dict lookup).

        Trades a `slots=True` (~28 bytes saved per instance, ~30
        bridges per `runner.run` = ~840 bytes total — irrelevant)
        for the cached-on-first-access ergonomics that match what
        the old `Bridge.params` field stored without the
        decoration-time work."""
        if self.holds_when is None:
            return {}
        sig = inspect.signature(self.holds_when)
        return {
            name: get_param_default(p)
            for name, p in sig.parameters.items()
            if get_param_default(p) is not inspect.Parameter.empty
        }

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
        from corroborate.graph.causal import Direction as _D
        from corroborate.graph.causal import Tier as _T
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
    produced it (the audit trail).

    `warnings`: WARN-level / INFO-level admission-gate results
    that fired but didn't block. Bridge author can scan these to
    surface non-principled authoring (HP-envelope scope, missing
    predicted_direction, etc.). Per
    `ADMISSION_GATES_DESIGN.md`.

    `blocked_by`: set when `verdict == INADMISSIBLE` — the first
    BLOCK-level gate that vetoed the bridge. Body did not run.

    `n_cells_in_scope`: row count after `bridge.scope` was applied.
    Equal to the input cell-count when `bridge.scope is None`. Used
    by the post-run report to surface sample-size diagnostics
    without re-running the scope filter.

    `assumption_violations`: flat tuple of distributional /
    sample-size flags collected from each analysis result's
    `.assumption_violations` field (when present). Each string
    is prefixed with `<fixture_name>:` so the audit reader can
    trace which fixture surfaced which warning. Empty when no
    fixture's result carried any flags. Propagates through the
    runner report into the run.json audit trail.

    `source_name` / `target_name`: mirror of `Bridge.source_name` /
    `Bridge.target_name` carried on the evaluation result so
    downstream consumers (extent-cluster grouping, walks) can key
    by edge endpoints without re-loading the Bridge.

    `evidence_digest`: content identity of an externally adapted
    bundle when evaluation used a recorded contrast. ``None`` for
    native/ordinary cell sets. This makes cached evaluation records
    distinguish evidence from different sealed bundles.

    `extent_hash`: stable BLAKE2b identity of the set of cell IDs admitted by
    the bridge's effective scope (`bridge.scope ∧ module_scope`).
    Two bridges with the same `(source_name, target_name,
    extent_hash)` admit identical cell-sets on the current cache
    — the extent-based cluster identity proposed at the
    findings-walk layer. Empty-scope bridges share
    `stable_extent_hash(())`, honestly reflecting "framework cannot
    distinguish these on this cache." Cluster identity is
    therefore corpus-dependent by design (bridge verdicts already
    are; cluster identity inherits the dependency)."""
    bridge_name: str
    verdict: Verdict
    analysis_results: Mapping[str, object]
    warnings: tuple['GateResult', ...] = ()
    blocked_by: 'GateResult | None' = None
    n_cells_in_scope: int = -1
    assumption_violations: tuple[str, ...] = ()
    refutation_class: 'RefutationClass | None' = None
    source_name: str = ''
    target_name: str = ''
    evidence_digest: str | None = None
    extent_hash: int = 0
    """**Sub-classification of NO_EFFECT** (or, more rarely, of
    POWER_INSUFFICIENT). Bridge bodies that distinguish "predicted
    direction, observed opposite" (`SIGN_FLIP`) from "predicted
    direction, observed near-zero" (`NULL_EFFECT`) — or "predicted
    null, observed significant" (`SIGN_FLIP` again, by symmetry) —
    return `(Verdict, RefutationClass)` instead of bare `Verdict`.
    Default `None` means no class was author-attached.

    Surfacing this distinction matters because NO_EFFECT lumps two
    different scientific outcomes — same shape as the framework's
    POWER_INSUFFICIENT-vs-NO_EFFECT distinction (PAPER_NOTES.md
    §3.4): silently absorbing the difference smuggles a stronger
    refutation past the reader as merely 'no effect.'"""


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


def _require_scope(
    value: object, fn_name: str,
) -> 'pl.Expr | DeferredScope | None':
    """Validate `scope` decorator arg. `None` → no filter;
    a `pl.Expr` is accepted as the polars-native cell predicate,
    OR a `DeferredScope` for scopes that resolve at evaluation
    time (via `scope_from_panel`). Any other value is an
    authoring mistake — fail loudly at import."""
    if value is None:
        return None
    if isinstance(value, (pl.Expr, DeferredScope)):
        return value
    raise TypeError(
        f'@claim_bridge {fn_name!r}: default for `scope` '
        f'must be a pl.Expr, DeferredScope, or None; '
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
    if value == 'null':
        return 'null'
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
    scope: 'pl.Expr | DeferredScope | None' = None,
    predicted_direction: PredictedDirection | None = None,
    gates: tuple[AdmissionGate, ...] = (),
) -> Callable[[Callable[..., 'Verdict | tuple[Verdict, RefutationClass | None]']], Bridge]:
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
    `source = DoEffect(treatment=(Intervention(...),), baseline=())`.
    The framework derives `treatment_arm` / `baseline_arm` strings
    via `DoEffect.treatment_arm_key()` / `baseline_arm_key()` (the
    canonical_str fingerprints of the typed Intervention tuples)
    at evaluate() time and threads them into the analysis's kwargs.
    When `source` is a string/Measurable, no contrast is set and
    the analysis runs without arm-pairing (correlation-style or
    pre-paired bridges).

    A common idiom is to define `INTERVENTION = DoEffect(...)`
    once at the top of the bridge file and reference it as
    `source=INTERVENTION` in each interventional bridge — the
    constant lives in the file's namespace, not in framework
    auto-resolution. HP-cleaved variants of the same structural
    contrast (γ-stratified, n_step-stratified, etc.) reuse the
    file-level `INTERVENTION` and add an HP scope predicate via
    `scope=pl.col('gamma') == 0.999` on the per-bridge decorator.
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

    def _decorator(fn: Callable[..., 'Verdict | tuple[Verdict, RefutationClass | None]']) -> Bridge:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f'@claim_bridge {fn.__name__!r}: cannot inspect '
                f'signature: {exc}',
            ) from exc

        # Auto-register Measurable instances passed by value as
        # source / target / defaulted-kwargs. `Bridge.params` is a
        # `@property` derived from `inspect.signature` on demand,
        # but Measurable-registration happens at decoration time so
        # the registry is populated before the first bridge runs.
        if isinstance(source_validated, Measurable):
            register(source_validated)
        if isinstance(target_validated, Measurable):
            register(target_validated)
        for param in sig.parameters.values():
            default = get_param_default(param)
            if default is inspect.Parameter.empty:
                continue
            if isinstance(default, Measurable):
                register(default)

        return Bridge(
            name=fn.__name__,
            source=source_validated,
            target=target_validated,
            direction=direction_validated,
            tier=tier_validated,
            pair_by=pair_by_validated,
            scope=scope_validated,
            holds_when=fn,
            predicted_direction=predicted_direction_validated,
            gates=gates,
        )

    return _decorator


def evaluate(
    bridge: Bridge,
    cells: pl.DataFrame | Iterable[Mapping[str, object]],
    *,
    claim: Claim[..., object] | None = None,
    module_scope: pl.Expr | None = None,
    recorded_contrast: RecordedContrastBinding | None = None,
) -> BridgeEvaluation:
    """Run a bridge against a cell-set: apply `bridge.scope`
    as a polars filter, resolve each fixture (a `holds_when` parameter
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

    `claim` (kw-only) is the substrate's outermost @claim — the
    structural source of truth for endogeneity gating
    (`exogenous_source`, `exogenous_scope`). When None, those
    gates short-circuit with `gate-doesn't-apply` semantics; the
    framework remains usable in tests and synthetic-corpus
    contexts that have no substrate composition. Substrate
    runners (cell_runner, scripts/run_hypothesis.py) thread
    `claim=dqn` (or the substrate-level entry-point claim) so
    the gates fire.

    `module_scope` (kw-only) is the hypothesis-module-level scope
    filter (e.g., "every cross-env bridge in this file excludes
    bsuite diagnostic envs"). The runner reads it via
    `getattr(h, 'MODULE_SCOPE', None)` and threads it here. AND-
    combined with `bridge.scope` before filtering — so a bridge
    that already filters to a specific bsuite env will have its
    cell-set zeroed out by an exclusion module_scope, and a
    bridge with `scope=None` will pick up the module_scope alone.
    Bridges that intentionally violate the module-level filter
    must move to a different hypothesis module.

    `recorded_contrast` binds an externally executed, verified
    two-arm contrast at evaluation time.  The bridge still declares
    the data-independent edge (for example ``gamma -> return_mean``);
    the binding supplies the producer-specific arm keys and verifies
    its intervention path, arm values, and sealed bundle digest against
    the cells.
    Analyses then consume ``bridge.target`` as their measured source,
    exactly as they do for an executable ``DoEffect``.  Supplying both
    forms is an error: a recorded contrast is evidence about an action
    performed elsewhere, not an operation Corroborate can re-apply.

    Raises `TypeError` if the Bridge has no `holds_when` body —
    `@claim_bridge` always populates it; this guard catches direct
    `Bridge(...)` construction with `holds_when=None`."""
    if bridge.holds_when is None:
        raise TypeError(
            f'evaluate({bridge.name!r}): Bridge has no holds_when '
            f'body — @claim_bridge always populates it; constructing '
            f'a Bridge directly with holds_when=None is unsupported.',
        )
    # One-shot iterables must survive both deferred-scope resolution and
    # the eventual analysis pass. Materialise them once at the boundary.
    if not isinstance(cells, pl.DataFrame):
        cells = [dict(cell) for cell in cells]
    recorded_arm_keys: tuple[str, str] | None = None
    if recorded_contrast is not None:
        if isinstance(bridge.source, DoEffect):
            raise ValueError(
                f'evaluate({bridge.name!r}): cannot combine an executable '
                'DoEffect source with recorded_contrast',
            )
        if not isinstance(bridge.source, str):
            raise ValueError(
                f'evaluate({bridge.name!r}): recorded_contrast requires '
                'a string source naming its parameter path',
            )
        if bridge.source_name != recorded_contrast.parameter_path:
            raise ValueError(
                f'evaluate({bridge.name!r}): bridge source '
                f'{bridge.source_name!r} does not match recorded contrast '
                f'parameter path {recorded_contrast.parameter_path!r}',
            )
        recorded_arm_keys = (
            recorded_contrast.baseline_key,
            recorded_contrast.treatment_key,
        )
        if recorded_arm_keys[0] == recorded_arm_keys[1]:
            raise ValueError(
                f'evaluate({bridge.name!r}): recorded contrast arm '
                f'keys must be distinct; got {recorded_arm_keys[0]!r}',
            )
        recorded_rows = (
            cast(list[dict[str, object]], cells.to_dicts())
            if isinstance(cells, pl.DataFrame)
            else cells
        )
        expected_values = {
            recorded_contrast.baseline_key: recorded_contrast.baseline_value,
            recorded_contrast.treatment_key: recorded_contrast.treatment_value,
        }
        seen_arms: set[str] = set()
        for cell in recorded_rows:
            cell_digest = cell.get('bundle_digest')
            if cell_digest != recorded_contrast.bundle_digest:
                raise ValueError(
                    f'evaluate({bridge.name!r}): recorded contrast digest '
                    f'{recorded_contrast.bundle_digest!r} does not match '
                    f'cell bundle digest {cell_digest!r}',
                )
            arm = cell.get('arm_key')
            if not isinstance(arm, str) or arm not in expected_values:
                continue
            seen_arms.add(arm)
            source_value = cell.get(recorded_contrast.parameter_path)
            expected_value = expected_values[arm]
            if (
                isinstance(source_value, bool)
                or not isinstance(source_value, (int, float))
                or not math.isfinite(float(source_value))
                or float(source_value) != expected_value
            ):
                raise ValueError(
                    f'evaluate({bridge.name!r}): arm {arm!r} records '
                    f'{recorded_contrast.parameter_path}={source_value!r}; '
                    f'contrast binds it to {expected_value!r}',
                )
        missing_arms = set(recorded_arm_keys).difference(seen_arms)
        if missing_arms:
            raise ValueError(
                f'evaluate({bridge.name!r}): recorded contrast arm(s) '
                f'{sorted(missing_arms)!r} are absent from the cells',
            )
    # Effective scope = bridge.scope ∧ module_scope (either may
    # be None). Polars' `&` is the framework-honest composition.
    #
    # `bridge.scope` may be a `DeferredScope` whose resolution
    # depends on the cells themselves (e.g. scope_from_panel
    # filtering to strata where an upstream panel statistic
    # crosses a threshold). Resolve those FIRST: build the panel
    # from raw cells (with the deferred-scope's `static_scope`
    # applied if present), extract surviving strata, produce
    # a `pl.Expr` that's AND-combined with `module_scope` like
    # any other.
    bridge_scope_expr: pl.Expr | None
    if isinstance(bridge.scope, DeferredScope):
        # Pre-materialize cells for the panel build. The deferred
        # scope's `resolve` runs on raw cells; static_scope inside
        # the deferred scope is applied by the resolve method.
        if isinstance(cells, pl.DataFrame):
            cells_for_panel = cast(list[dict[str, object]], cells.to_dicts())
        else:
            cells_for_panel = [dict(c) for c in cells]
        bridge_scope_expr = bridge.scope.resolve(cells_for_panel)
    else:
        bridge_scope_expr = bridge.scope
    effective_scope: pl.Expr | None
    if bridge_scope_expr is None and module_scope is None:
        effective_scope = None
    elif bridge_scope_expr is None:
        effective_scope = module_scope
    elif module_scope is None:
        effective_scope = bridge_scope_expr
    else:
        effective_scope = bridge_scope_expr & module_scope

    filtered_cells: list[dict[str, object]]
    if effective_scope is None:
        # No cell filter — skip the DataFrame round-trip. Convert
        # to list[dict] so the analysis fn can re-iterate.
        if isinstance(cells, pl.DataFrame):
            filtered_cells = cast(list[dict[str, object]], cells.to_dicts())
        else:
            filtered_cells = [dict(c) for c in cells]
    else:
        # Cell filter — materialise to DataFrame (if not already),
        # filter, convert back to list[dict] for the analysis.
        df: pl.DataFrame
        if isinstance(cells, pl.DataFrame):
            df = cells
        else:
            cells_list = list(cells)
            df = pl.from_dicts(cells_list) if cells_list else pl.DataFrame()
        if df.height > 0:
            df = filter_cells(df, effective_scope)
        filtered_cells = (
            cast(list[dict[str, object]], df.to_dicts())
            if df.height > 0 else []
        )
    n_cells_in_scope = len(filtered_cells)
    # extent_hash: process-portable identity of the bridge's admitted
    # cell-set on the current cache. Set semantics mean two bridges
    # with identical admitted IDs get identical values irrespective of
    # row order. Empty extent maps to one deterministic constant.
    admitted_ids: list[str] = []
    for c in filtered_cells:
        cid = c.get('id')
        if isinstance(cid, str):
            admitted_ids.append(cid)
    extent_hash = stable_extent_hash(admitted_ids)
    # Run admission gates BEFORE the bridge body. Auto-gates
    # (typed-contract guards, exogenous-source/scope, etc.) are
    # always-on; per-bridge `gates=(...)` are appended. BLOCK-level
    # failures short-circuit the bridge with `Verdict.INADMISSIBLE`;
    # WARN/INFO results accumulate on `BridgeEvaluation.warnings`.
    all_gates: tuple[AdmissionGate, ...] = AUTO_GATES + bridge.gates
    warnings: list[GateResult] = []
    for gate in all_gates:
        # A recorded intervention repeats one condition value across
        # experimental units by construction, just like a DoEffect's arm
        # indicator. ``pair_by`` — not the parameter's value cardinality —
        # determines the independent units for its paired analysis. It is
        # also evidence of an executed contrast, so the native-substrate
        # exogenous-source gate does not reinterpret its parameter path as
        # an unexecuted author-controlled leaf.
        if recorded_contrast is not None and gate in (
            distinct_units, exogenous_source,
        ):
            continue
        result = gate(bridge, filtered_cells, claim=claim)
        if result is None or result.passed:
            continue
        if result.level is GateLevel.BLOCK:
            return BridgeEvaluation(
                bridge_name=bridge.name,
                verdict=Verdict.INADMISSIBLE,
                analysis_results=MappingProxyType({}),
                warnings=tuple(warnings),
                blocked_by=result,
                n_cells_in_scope=n_cells_in_scope,
                source_name=bridge.source_name,
                target_name=bridge.target_name,
                extent_hash=extent_hash,
                evidence_digest=(
                    recorded_contrast.bundle_digest
                    if recorded_contrast is not None else None
                ),
            )
        warnings.append(result)
    # Contrast resolution:
    #   - `source = DoEffect(...)` → contrast = the DoEffect, and
    #     the analysis's `source` slot maps to bridge.target_name
    #     (the measurement column).
    #   - `recorded_contrast=...` → bind the verified external arm
    #     labels without putting bundle-specific data in the bridge;
    #     the analysis source maps to bridge.target_name.
    #   - `source = str | Measurable` → no contrast; the name flows
    #     directly as the analysis's source. Correlational bridges,
    #     or bridges where the author wants paired_g to compute on
    #     a non-outcome measurable.
    contrast: DoEffect | RecordedContrastBinding | None
    source_for_analysis: str
    if isinstance(bridge.source, DoEffect):
        contrast = bridge.source
        source_for_analysis = bridge.target_name
    elif recorded_contrast is not None:
        contrast = recorded_contrast
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
        if isinstance(contrast, DoEffect):
            arm_keys = contrast.arm_keys()
        else:
            arm_keys = (contrast.baseline_key, contrast.treatment_key)
        bridge_params['arm_keys'] = arm_keys
        # Binary compat: when N=2, inject treatment_arm /
        # baseline_arm so existing bridges with default kwargs
        # `treatment_arm: str = ArmRole.TREATMENT` work unchanged.
        # Convention: arms[0] is baseline (typically empty-tuple /
        # control); arms[1] is treatment.
        if len(arm_keys) == 2:
            bridge_params['treatment_arm'] = arm_keys[1]
            bridge_params['baseline_arm'] = arm_keys[0]
        # Resolve any ArmRole-typed sentinels in the bridge's
        # defaulted kwargs to the canonical arm_key string.
        # ArmRole is binary-only by design — N-arm bridges access
        # `arm_keys[i]` directly or use explicit string keys.
        for k, v in bridge_params.items():
            if isinstance(v, ArmRole):
                if len(arm_keys) != 2:
                    raise ValueError(
                        f'ArmRole sentinel {v!r} used in '
                        f'{len(arm_keys)}-arm DoEffect on bridge '
                        f'{bridge.name!r}; ArmRole is binary-only. '
                        f'Use `arm_keys[i]` indexing or explicit '
                        f'string keys for multi-arm bridges.',
                    )
                bridge_params[k] = (
                    arm_keys[1] if v is ArmRole.TREATMENT
                    else arm_keys[0]
                )
    analysis_results = resolve_for_holds_when(
        bridge.holds_when, filtered_cells, bridge_params,
    )
    holds_result = bridge.holds_when(**analysis_results)
    # Bridges may return either bare `Verdict` (legacy) or
    # `(Verdict, RefutationClass | None)` (new path, lets bridges
    # mark sign-flip vs null-effect refutations explicitly).
    # Runtime isinstance guards remain even though pyright sees
    # the unioned return type — the `Bridge.holds_when` field is
    # typed as a callable but Python wouldn't enforce the contract
    # without these checks.
    verdict, refutation_class = _unpack_holds_result(
        holds_result, bridge_name=bridge.name,
    )
    # Collect assumption_violations from each fixture's result.
    # Analyses author opt into this by exposing an
    # `assumption_violations: tuple[str, ...]` attribute on their
    # Result dataclass (paired_g, random_effects_summary do today).
    # The bridge layer prefixes each string with `<fixture>:` so
    # the audit trail tells substrate authors WHICH fixture
    # surfaced which flag.
    assumption_flags: list[str] = []
    for fixture_name, result in analysis_results.items():
        flags = getattr(result, 'assumption_violations', None)
        if isinstance(flags, tuple):
            for flag in flags:
                if isinstance(flag, str):
                    assumption_flags.append(f'{fixture_name}: {flag}')
    return BridgeEvaluation(
        bridge_name=bridge.name,
        verdict=verdict,
        analysis_results=MappingProxyType(dict(analysis_results)),
        warnings=tuple(warnings),
        n_cells_in_scope=n_cells_in_scope,
        assumption_violations=tuple(assumption_flags),
        refutation_class=refutation_class,
        source_name=bridge.source_name,
        target_name=bridge.target_name,
        extent_hash=extent_hash,
        evidence_digest=(
            recorded_contrast.bundle_digest
            if recorded_contrast is not None else None
        ),
    )


def _unpack_holds_result(
    result: object, *, bridge_name: str,
) -> tuple[Verdict, RefutationClass | None]:
    """Validate + unpack a bridge body's return value into the
    (Verdict, RefutationClass | None) shape `BridgeEvaluation`
    expects. Bridges may return bare `Verdict` (legacy path) or
    `(Verdict, RefutationClass | None)` (new path)."""
    if isinstance(result, Verdict):
        return result, None
    if isinstance(result, tuple):
        if len(result) != 2:
            raise TypeError(
                f'Bridge {bridge_name!r} `holds_when` returned a '
                f'{len(result)}-tuple; expected '
                f'(Verdict, RefutationClass | None).',
            )
        verdict_obj, refutation_obj = result
        if not isinstance(verdict_obj, Verdict):
            raise TypeError(
                f'Bridge {bridge_name!r} `holds_when` returned '
                f'{type(verdict_obj).__name__} as the first tuple '
                f'element; expected Verdict.',
            )
        if refutation_obj is not None and not isinstance(
            refutation_obj, RefutationClass,
        ):
            raise TypeError(
                f'Bridge {bridge_name!r} `holds_when` returned '
                f'{type(refutation_obj).__name__} as the second '
                f'tuple element; expected RefutationClass | None.',
            )
        return verdict_obj, refutation_obj
    raise TypeError(
        f'Bridge {bridge_name!r} `holds_when` returned '
        f'{type(result).__name__}; expected Verdict '
        f'or (Verdict, RefutationClass | None).',
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
    from corroborate.measurables import (
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
        # `bridge.scope` may reference measurable columns that
        # aren't named in source/target/params (e.g. an endogenous
        # selector like `finite_ge('effective_horizon', 50)`).
        # `expr.meta.root_names()` extracts every column referenced
        # by the polars expression — exactly the set we need to
        # ensure those measurables are computed at ingest.
        if b.scope is not None:
            # DeferredScope: pull the static_scope (its polars-Expr
            # half) if any; the dynamic part references the
            # stratify column which is also a regular cell column.
            if isinstance(b.scope, DeferredScope):
                candidates.append(b.scope.stratify_column)
                static_expr = b.scope.static_scope
                if static_expr is not None:
                    try:
                        candidates.extend(static_expr.meta.root_names())
                    except Exception:  # noqa: BLE001
                        pass
            else:
                try:
                    candidates.extend(b.scope.meta.root_names())
                except Exception:  # noqa: BLE001
                    # If polars metadata extraction fails, fall through
                    # — the missing-column path in
                    # `filter_cells` will surface the error
                    # at evaluate-time instead.
                    pass
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
    'RecordedContrastBinding',
    'Tier',
    'claim_bridge',
    'endpoint_name',
    'evaluate',
    'measurable_names_for_bridges',
]
