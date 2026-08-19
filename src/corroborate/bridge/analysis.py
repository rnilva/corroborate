"""Analyses — the framework's typed statistical primitives.

An *analysis* takes a corpus (a polars DataFrame or iterable of
records) plus parameters and produces a typed result. Examples:
paired-g across seeds, meta-regression coefficients, DoWhy
backdoor ATE. Each analysis is reusable across many bridges;
bridges consume analysis results by typed parameter
(pytest-fixture style).

The bridge file authors *claims* (declarations of edges +
thresholds); the analyses are the framework-supplied fixtures
the claim's `holds_when` body consumes:

    @analysis
    def paired_g(
        cells, *,
        treatment_arm, baseline_arm, pair_by, source, ...,
    ) -> PairedGResult: ...

    @claim_bridge(
        source='<intervention_parameter>',
        target='<outcome_metric>',
        direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL,
    )
    def treatment_helps_outcome(
        paired_g: PairedGResult,
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

**Canonical cells input.** Every analysis accepts
`pl.DataFrame | Iterable[Mapping[str, object]]` and normalises
once at its own entry — `as_rows` for row-consuming bodies,
`data.kernel.cells_to_dataframe` for DataFrame-consuming bodies
(the two directions of the conversion boundary). The wrapper
therefore adds no hidden conversion: `__call__` is pure typed
delegation, and `panel.cells` works against the whole registry
by contract rather than by dispatch. The contract is enforced at
registration — `@analysis` rejects a first parameter that does
not spell the union — and a registry-wide test proves the whole
shipped surface passed it.

`Analysis[C, O, **P]` preserves the wrapped fn's full surface —
cells shape `C`, result `O`, keyword surface `P` — through the
wrapper (CLAUDE.md: ParamSpec preserves caller signature through
generic wrappers), so direct exploration calls are checked
against the analysis's real signature. Keeping `C` and `O` first
also preserves the original public `Analysis[Cells, Result]`
annotation spelling; callers that need to spell the captured
surface explicitly can supply `P` as the optional third argument.
The registry stores the erased upper bound
`Analysis[object, object, ...]` — the generic
upper bound lives at the container boundary only; the runner's
dynamically-filtered kwarg injection (`run_for`) rides the
gradual `...` form.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Concatenate, Protocol, cast, overload

import polars as pl

from corroborate._internals.introspection import (
    get_param_annotation,
    get_param_default,
)
from corroborate._internals.registry import Registry

_CANONICAL_CELLS = 'pl.DataFrame | Iterable[Mapping[str, object]]'


def _require_canonical_cells(fn: Callable[..., object]) -> None:
    """Registration gate for the canonical cells contract.

    Every analysis declares the union — spelled literally, starting
    ``pl.DataFrame | `` — on its first parameter, and normalises at
    its own entry (module docstring). The check is textual by
    design: the contract is a spelling convention, like the
    framework's other load-bearing identifiers, so validation needs
    no annotation resolution machinery, and a violation is an
    import-time TypeError naming the fix — never a silent
    mis-dispatch at call time."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        # Signature-less callables (C builtins) can't be checked;
        # nothing in-tree registers one.
        return
    params = list(sig.parameters.values())
    annotation: object = inspect.Parameter.empty
    if params:
        annotation = get_param_annotation(params[0])
    if (
        annotation is inspect.Parameter.empty
        or not str(annotation).startswith('pl.DataFrame | ')
    ):
        raise TypeError(
            f'@analysis {fn.__name__!r}: the first (cells) parameter '
            f'must declare the canonical union '
            f'"{_CANONICAL_CELLS}" (spelled literally) and normalise '
            f'at entry — `as_rows(cells)` for row-consuming bodies, '
            f'`cells_to_dataframe(cells)` for DataFrame-consuming '
            f'ones; got {str(annotation)!r}',
        )


@dataclass(frozen=True, slots=True)
class Analysis[
    C = pl.DataFrame | Iterable[Mapping[str, object]],
    O = object,
    **P = ...,
]:
    """Typed wrapper for a registered analysis.

    Generic over the wrapped fn's full surface: `C` is the cells
    shape the fn declares for its first positional parameter, `P`
    captures the remaining (keyword) parameters, and `O` is the
    typed result the bridge `holds_when` consumes. A direct call
    through `__call__` is therefore checked against the analysis's
    real signature — a mistyped kwarg is a pyright error at the
    call site, not a runtime TypeError. `name` is the lookup key
    (= `fn.__name__`).

    `reads` declares record-key columns the analysis touches off
    the cell record directly (i.e. `cell['<key>']`-style),
    BYPASSING the @measurable resolver. The runner unions these
    with bridge measurables' transitive_reads so the trace-column
    join + the no-drop set know what to bring in. Scalar fields
    already present in `runs.parquet` (`env_name`, `seed`, …) do
    NOT need to be declared — they're loaded for free."""
    # `Callable[Concatenate[C, P], O]` would make the first
    # parameter positional-only even when the wrapped function
    # declares a normal positional-or-keyword `cells` parameter.
    # Keep `.fn` gradual; `__call__` below is the surface whose
    # arguments are preserved precisely by `C` and `P`.
    fn: Callable[..., O]
    name: str
    reads: tuple[str, ...] = field(default=())

    def __call__(
        self,
        cells: C,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> O:
        """Run the analysis directly on a corpus or a Panel's cells.

        Bridges never need this — the runner resolves analyses by
        parameter name and injects the typed result (`run_for`).
        It exists because exploration code and test fixtures
        otherwise fail with ``TypeError: 'Analysis' object is not
        callable``, which sends the reader looking for `.fn`.

        Pure delegation: every analysis accepts the canonical
        cells union and normalises at its own entry (see the
        module docstring), so the wrapper performs no conversion
        and `panel.cells` flows through untouched. Unlike
        `run_for`, kwargs are passed through unfiltered — and,
        via `P`, statically checked."""
        return self.fn(cells, *args, **kwargs)


type _StoredAnalysis = Analysis[object, object, ...]
"""Registry-side erasure — the generic upper bound at the
container boundary, never at element use sites. `P = ...` keeps
the runner's dynamically-filtered kwarg injection callable;
`C = object` admits any cells shape; registry consumers read
`.fn` / `.name` / `.reads`, they don't re-narrow elements."""


_REGISTRY: Registry[_StoredAnalysis] = Registry()


def get_registered(name: str) -> _StoredAnalysis | None:
    """Look up an analysis by name. Returns None if not registered."""
    return _REGISTRY.get(name)


def registered_names() -> tuple[str, ...]:
    """Sorted tuple of registered analysis names — for diagnostics."""
    return _REGISTRY.names()


class _AnalysisDecorator(Protocol):
    """Return type of the `@analysis(reads=...)` factory form —
    generic at APPLICATION time, so `P` / `C` / `O` solve against
    the decorated fn rather than collapsing to the class type-param
    defaults at the factory call (the pre-ParamSpec behaviour,
    which erased every reads-declaring analysis's result to
    `object`)."""

    def __call__[**P, C, O](
        self, fn: Callable[Concatenate[C, P], O], /,
    ) -> Analysis[C, O, P]: ...


@overload
def analysis[**P, C, O](
    fn: Callable[Concatenate[C, P], O], /,
) -> Analysis[C, O, P]: ...


@overload
def analysis(*, reads: tuple[str, ...] = ()) -> _AnalysisDecorator: ...


def analysis(
    fn: Callable[..., object] | None = None,
    /,
    *,
    reads: tuple[str, ...] = (),
) -> object:
    """Register `fn` as a framework analysis. Name is taken from
    `fn.__name__`; rename the function to rename the analysis.

    The wrapped function's first positional arg is the corpus —
    `C` in the wrapper's type, and by convention the canonical
    union `pl.DataFrame | Iterable[Mapping[str, object]]`,
    normalised in the fn's own first statement. The remaining
    parameters are captured as `P` (PEP 612 `Concatenate`) and the
    result as `O`, so a direct exploration call like
    `paired_g(panel.cells, source=...)` is checked against the
    analysis's real signature. The bridge path is unaffected:
    `run_for` supplies kwargs dynamically through the registry's
    erased view.

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

    def _build[**P, C, O](
        fn_inner: Callable[Concatenate[C, P], O],
    ) -> Analysis[C, O, P]:
        _require_canonical_cells(fn_inner)
        name = fn_inner.__name__
        wrapper: Analysis[C, O, P] = Analysis(
            fn=fn_inner,
            name=name,
            reads=reads,
        )
        # `Registry[_StoredAnalysis]` accepts this generic-parameter
        # erasure at the storage boundary; the `cast` lifts
        # `Analysis[C, O, P]` through Python's invariant-generic
        # constraint without a `# type: ignore`.
        _REGISTRY.register(name, cast('_StoredAnalysis', wrapper))
        return wrapper

    if fn is None:
        return _build
    return _build(fn)


def _kwargs_for(
    analysis_obj: _StoredAnalysis,
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
    analysis_obj: _StoredAnalysis,
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
