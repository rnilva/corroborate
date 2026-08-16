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

`Analysis[**P, C, O]` preserves the wrapped fn's full surface —
cells shape `C`, keyword surface `P`, result `O` — through the
wrapper (CLAUDE.md: ParamSpec preserves caller signature through
generic wrappers), so direct exploration calls are checked
against the analysis's real signature. The registry stores the
erased upper bound `Analysis[..., object, object]` — the generic
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
from corroborate._internals.polars import to_dicts
from corroborate._internals.registry import Registry


@dataclass(frozen=True, slots=True)
class Analysis[
    **P = ...,
    C = pl.DataFrame | Iterable[Mapping[str, object]],
    O = object,
]:
    """Typed wrapper for a registered analysis.

    Generic over the wrapped fn's full surface: `C` is the cells
    shape the fn declares for its first positional parameter, `P`
    captures the remaining (keyword) parameters, and `O` is the
    typed result the bridge `holds_when` consumes. A direct call
    through `__call__` is therefore checked against the analysis's
    real signature — a mistyped kwarg, or an iterable handed to a
    DataFrame-native analysis, is a pyright error at the call
    site, not a runtime TypeError. `name` is the lookup key
    (= `fn.__name__`).

    `reads` declares record-key columns the analysis touches off
    the cell record directly (i.e. `cell['<key>']`-style),
    BYPASSING the @measurable resolver. The runner unions these
    with bridge measurables' transitive_reads so the trace-column
    join + the no-drop set know what to bring in. Scalar fields
    already present in `runs.parquet` (`env_name`, `seed`, …) do
    NOT need to be declared — they're loaded for free.

    `accepts_dataframe` records whether `fn`'s cells parameter
    admits a `pl.DataFrame` — detected once at registration from
    the declared signature, consumed by `__call__`'s runtime input
    dispatch (static types erase, so the conversion decision needs
    a runtime witness). Never set by authors; the signature is the
    truth."""
    fn: Callable[Concatenate[C, P], O]
    name: str
    reads: tuple[str, ...] = field(default=())
    accepts_dataframe: bool = field(default=False)

    def __call__(
        self,
        cells: C | pl.DataFrame,
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

        A `pl.DataFrame` (typically `panel.cells`) works against
        every registered analysis: analyses whose own signature
        admits a DataFrame receive it untouched (their canonical
        fast path — no conversion round-trip), and iterable-only
        analyses get it materialised to per-row mappings once,
        here. Iterable input always delegates untouched. Unlike
        `run_for`, kwargs are passed through unfiltered — and,
        via `P`, statically checked."""
        if isinstance(cells, pl.DataFrame) and not self.accepts_dataframe:
            # cast: `accepts_dataframe=False` ⇔ the fn's declared
            # cells shape is iterable-of-mappings (registration-time
            # signature detection), so the materialised rows satisfy
            # `C` at runtime — a coupling the static system can't see.
            return self.fn(cast('C', to_dicts(cells)), *args, **kwargs)
        # cast: either `cells` is not a DataFrame (the `C` arm of
        # the input union), or it is one AND `accepts_dataframe`
        # witnesses that the fn's declared `C` admits DataFrames.
        return self.fn(cast('C', cells), *args, **kwargs)


def _first_param_accepts_dataframe(fn: Callable[..., object]) -> bool:
    """Registration-time detection of `fn`'s cells shape.

    Under PEP 563 (every analysis module uses `from __future__
    import annotations`) the annotation is a string; textual
    containment of ``'DataFrame'`` is the deliberate check —
    resolving names instead (`typing.get_type_hints`) would
    evaluate EVERY parameter's annotation and raise on any
    `TYPE_CHECKING`-only name anywhere in the signature. The
    failure modes are asymmetric in the right direction: a false
    positive hands the fn the DataFrame it would have received
    before `__call__` normalised at all (the pre-existing status
    quo), never a new shape."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return False
    for param in sig.parameters.values():
        # First parameter only — the cells argument by contract.
        annotation = get_param_annotation(param)
        if annotation is inspect.Parameter.empty:
            return False
        # `str()` covers both PEP 563 string annotations and live
        # class objects (whose repr names the class).
        return 'DataFrame' in str(annotation)
    return False


type _StoredAnalysis = Analysis[..., object, object]
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
    ) -> Analysis[P, C, O]: ...


@overload
def analysis[**P, C, O](
    fn: Callable[Concatenate[C, P], O], /,
) -> Analysis[P, C, O]: ...


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
    `C` in the wrapper's type. The remaining parameters are
    captured as `P` (PEP 612 `Concatenate`) and the result as `O`,
    so a direct exploration call like `paired_g(panel.cells,
    source=...)` is checked against the analysis's real signature.
    The bridge path is unaffected: `run_for` supplies kwargs
    dynamically through the registry's erased view.

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
    ) -> Analysis[P, C, O]:
        name = fn_inner.__name__
        wrapper: Analysis[P, C, O] = Analysis(
            fn=fn_inner,
            name=name,
            reads=reads,
            accepts_dataframe=_first_param_accepts_dataframe(fn_inner),
        )
        # `Registry[_StoredAnalysis]` accepts this generic-parameter
        # erasure at the storage boundary; the `cast` lifts
        # `Analysis[P, C, O]` through Python's invariant-generic
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
