"""Analyses — the framework's typed statistical primitives.

An *analysis* takes a corpus (iterable of records) plus
parameters and produces a typed result. Examples: paired-g
across seeds, meta-regression coefficients, DoWhy backdoor ATE.
Each analysis is reusable across many bridges; bridges consume
analysis results by typed parameter (pytest-fixture style).

The bridge file authors *claims* (declarations of edges +
thresholds); the analyses are the framework-supplied fixtures
the claim's `holds_when` body consumes:

    @analysis
    def paired_g(
        cells, *,
        treatment_arm, baseline_arm, pair_by, source, ...,
    ) -> PairedGResult: ...

    @claim_bridge
    def treatment_helps_outcome(
        paired_g: PairedGResult,
        *,
        source: str = '<outcome_metric>',
        target: str = '<outcome_metric>',
        direction: Direction = Direction.DIRECT,
        tier: Tier = Tier.ASSOCIATIONAL,
        treatment_arm: str = 'treatment',
        baseline_arm: str = 'baseline',
    ) -> Verdict:
        if paired_g.g > 0.3 and paired_g.p_value < 0.05:
            return Verdict.HELD
        ...

The framework reads `holds_when`'s signature, finds the
parameter named `paired_g` (no default → fixture), looks up the
registered analysis with that name, parameterises it from the
bridge's structural fields + the defaulted-kwarg `params` bag,
runs it on the cells, and injects the result.

Analyses are registered globally by `fn.__name__`. The naming is
the consumption protocol: a bridge's parameter name must match a
registered analysis name.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import cast, overload

from corroborate._internals.introspection import get_param_default
from corroborate._internals.registry import Registry


@dataclass(frozen=True, slots=True)
class Analysis[R: Mapping[str, object], O]:
    """Typed wrapper for a registered analysis. `fn` takes the
    corpus as its first positional argument plus keyword
    parameters that the bridge populates at run time. `O` is the
    typed result the bridge `holds_when` consumes; `name` is the
    lookup key (= `fn.__name__`).

    `reads` declares record-key columns the analysis touches off
    the cell record directly (i.e. `cell['<key>']`-style),
    BYPASSING the @measurable resolver. The runner unions these
    with bridge measurables' transitive_reads so the trace-column
    join + the no-drop set know what to bring in. Scalar fields
    already present in `runs.parquet` (`env_name`, `seed`, …) do
    NOT need to be declared — they're loaded for free."""
    fn: Callable[..., O]
    name: str
    reads: tuple[str, ...] = field(default=())


_REGISTRY: Registry[Analysis[Mapping[str, object], object]] = Registry()


def get_registered(
    name: str,
) -> Analysis[Mapping[str, object], object] | None:
    """Look up an analysis by name. Returns None if not registered."""
    return _REGISTRY.get(name)


def registered_names() -> tuple[str, ...]:
    """Sorted tuple of registered analysis names — for diagnostics."""
    return _REGISTRY.names()


@overload
def analysis[R: Mapping[str, object] = Mapping[str, object], O = object](
    fn: Callable[..., O], /,
) -> Analysis[R, O]: ...


@overload
def analysis[R: Mapping[str, object] = Mapping[str, object], O = object](
    *, reads: tuple[str, ...] = (),
) -> Callable[[Callable[..., O]], Analysis[R, O]]: ...


def analysis[R: Mapping[str, object] = Mapping[str, object], O = object](
    fn: Callable[..., O] | None = None,
    /,
    *,
    reads: tuple[str, ...] = (),
) -> Analysis[R, O] | Callable[[Callable[..., O]], Analysis[R, O]]:
    """Register `fn` as a framework analysis. Name is taken from
    `fn.__name__`; rename the function to rename the analysis.

    The wrapped function's first positional arg is the corpus
    (`Iterable[R]` or whatever shape the analysis consumes); the
    rest are keyword parameters supplied by the bridge's
    structural fields + params bag at run time. The type-param
    defaults (PEP 696) keep `paired_g`, `paired_g_per_burst`, etc.
    callable from standalone analysis scripts as
    `Analysis[Mapping[str, object], <Result>]` rather than
    `Analysis[Unknown, <Result>]`.

    Two decorator forms:

        @analysis
        def paired_g(cells, *, ...) -> PairedGResult: ...

        @analysis(reads=('<key_a>', '<key_b>'))
        def paired_link_per_burst(cells, *, ...) -> ...: ...

    `reads` declares trace-store record-keys the analysis touches
    directly (without going through the @measurable resolver). The
    runner uses this to decide which trace columns to load from
    `traces.parquet` and to KEEP after the per-corpus measurable
    compute step (otherwise the trace data goes away before the
    analysis sees it)."""

    def _build(fn_inner: Callable[..., O]) -> Analysis[R, O]:
        name = fn_inner.__name__
        wrapper: Analysis[R, O] = Analysis(fn=fn_inner, name=name, reads=reads)
        # `Registry[Analysis[Mapping[str, object], object]]` accepts
        # this generic-parameter narrowing at the storage boundary;
        # the `cast` lifts `Analysis[R, O]` through Python's
        # invariant-generic constraint without a `# type: ignore`.
        _REGISTRY.register(
            name,
            cast('Analysis[Mapping[str, object], object]', wrapper),
        )
        return wrapper

    if fn is None:
        return _build
    return _build(fn)


def _kwargs_for(
    analysis_obj: Analysis[Mapping[str, object], object],
    bridge_params: Mapping[str, object],
) -> dict[str, object]:
    """Pick the subset of `bridge_params` whose keys are accepted
    as keyword parameters by `analysis_obj.fn`. The bridge passes
    its full structural-field + params bag; each analysis filters
    to what its signature names. First positional (the corpus
    argument) is excluded — the framework supplies it
    explicitly."""
    try:
        sig = inspect.signature(analysis_obj.fn)
    except (ValueError, TypeError):
        return {}
    accepted: set[str] = set()
    for i, p in enumerate(sig.parameters.values()):
        if i == 0:
            continue
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted.add(p.name)
    return {
        k: v for k, v in bridge_params.items() if k in accepted
    }


def run_for(
    analysis_obj: Analysis[Mapping[str, object], object],
    cells: Iterable[Mapping[str, object]],
    bridge_params: Mapping[str, object],
) -> object:
    """Invoke `analysis_obj.fn(cells, **filtered_kwargs)` where
    `filtered_kwargs` is the subset of `bridge_params` matching
    the analysis's signature."""
    kwargs = _kwargs_for(analysis_obj, bridge_params)
    return analysis_obj.fn(cells, **kwargs)


def resolve_for_holds_when(
    holds_when: Callable[..., object],
    cells: Iterable[Mapping[str, object]],
    bridge_params: Mapping[str, object],
) -> dict[str, object]:
    """Walk `holds_when`'s signature; for each parameter WITHOUT a
    default (the fixtures), look it up as a registered analysis,
    run it, accumulate results into a `{param_name: result}` dict
    ready to splat as `**kwargs` into `holds_when`.

    Defaulted kwargs (the bridge's metadata: `source`, `target`,
    `direction`, `tier`, plus claim-specific params) are NOT
    re-passed at evaluate time — they're already bound as Python
    defaults; the framework reads them once at decoration time.

    Unknown fixture names raise `KeyError` with the registry's
    known set — typo / missing-analysis fails loudly at evaluation
    rather than producing a silent NaN verdict."""
    try:
        sig = inspect.signature(holds_when)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f'cannot inspect holds_when signature: {exc}',
        ) from exc
    out: dict[str, object] = {}
    cells_list = list(cells)
    for param_name, param in sig.parameters.items():
        default = get_param_default(param)
        if default is not inspect.Parameter.empty:
            # Metadata kwarg — already bound as the function's
            # default; no fixture to resolve.
            continue
        analysis_obj = _REGISTRY.get(param_name)
        if analysis_obj is None:
            raise KeyError(
                f'holds_when parameter {param_name!r} has no '
                f'default and is not a registered analysis; '
                f'known analyses: {_REGISTRY.names()}',
            )
        out[param_name] = run_for(
            analysis_obj, cells_list, bridge_params,
        )
    return out


__all__ = [
    'Analysis',
    'analysis',
    'get_registered',
    'registered_names',
    'resolve_for_holds_when',
    'run_for',
]
