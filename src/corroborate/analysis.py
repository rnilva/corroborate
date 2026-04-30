"""Analyses — the framework's typed statistical primitives.

An *analysis* takes a corpus (iterable of records) plus parameters
and produces a typed result. Examples: paired-g across seeds,
meta-regression coefficients, DoWhy backdoor ATE, three-check
audit. Each analysis is reusable across many bridges; bridges
consume analysis results by typed parameter (pytest-fixture
style).

The bridge file authors *claims* (declarations of
edges + thresholds); the analyses are the framework-supplied
fixtures the claim's `holds_when` body consumes:

    @analysis(name='paired_g')
    def paired_g(
        cells, *,
        treatment_arm, baseline_arm, pair_by, measurable, ...,
    ) -> PairedGResult: ...

    @bridge(
        name='ddqn_helps_outcome',
        source='outcome.eval_best_burst_mean',
        treatment_arm='ddqn', baseline_arm='vanilla_dqn',
        pair_by=('seed',), env='Acrobot-v1',
    )
    def claim(paired_g: PairedGResult) -> Verdict:
        if paired_g.g > 0.3 and paired_g.p < 0.05:
            return Verdict.HELD
        ...

The framework reads `claim`'s signature, finds the parameter
named `paired_g`, looks up the registered analysis with that
name, parameterises it from the bridge's structural fields, runs
it on the cells, and injects the result.

Analyses are registered globally in a name-keyed registry. The
naming is the consumption protocol: a bridge's parameter name
must match a registered analysis name. Output type is
incidental (the bridge author can annotate for clarity).
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Analysis[R: Mapping[str, object], O]:
    """Typed wrapper for a registered analysis.

    `fn` takes the corpus as its first positional argument plus
    keyword parameters that the bridge's structural fields
    populate at run time. `O` is the typed result the bridge
    `holds_when` consumes.

    `name` is the lookup key — a bridge consumes this analysis
    by declaring a parameter of the same name in its
    `holds_when` signature."""
    fn: Callable[..., O]
    name: str


_REGISTRY: dict[str, Analysis[Mapping[str, object], object]] = {}


def get_registered(
    name: str,
) -> Analysis[Mapping[str, object], object] | None:
    """Look up an analysis by name. Returns None if not registered."""
    return _REGISTRY.get(name)


def registered_names() -> tuple[str, ...]:
    """Sorted tuple of registered analysis names — for diagnostics."""
    return tuple(sorted(_REGISTRY))


def analysis[R: Mapping[str, object], O](
    *, name: str | None = None,
) -> Callable[[Callable[..., O]], Analysis[R, O]]:
    """Register a function as a framework analysis. Decorator
    factory; `name` defaults to the function's `__name__`.

    The wrapped function's first positional arg is the corpus
    (`Iterable[R]` or whatever shape the analysis consumes); the
    rest are keyword parameters supplied by the bridge's
    structural fields at run time. Return type `O` is the typed
    result a bridge `holds_when` parameter annotates."""
    def decorator(fn: Callable[..., O]) -> Analysis[R, O]:
        resolved = name if name is not None else fn.__name__
        wrapper: Analysis[R, O] = Analysis(fn=fn, name=resolved)
        existing = _REGISTRY.get(resolved)
        if existing is not None and existing.fn is not fn:
            raise ValueError(
                f'analysis {resolved!r} already registered to a '
                f'different function',
            )
        _REGISTRY[resolved] = wrapper  # type: ignore[assignment]
        return wrapper
    return decorator


def _kwargs_for(
    analysis_obj: Analysis[Mapping[str, object], object],
    bridge_params: Mapping[str, object],
) -> dict[str, object]:
    """Pick the subset of `bridge_params` whose keys are accepted
    by `analysis_obj.fn` as keyword parameters. The bridge passes
    its full structural-field bag; each analysis filters to what
    its signature names.

    First positional parameter (the corpus) is excluded — the
    framework always supplies it explicitly."""
    try:
        sig = inspect.signature(analysis_obj.fn)
    except (ValueError, TypeError):
        return {}
    accepted: set[str] = set()
    for i, p in enumerate(sig.parameters.values()):
        if i == 0:
            # Skip the first positional (the corpus argument).
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
    the analysis's signature. The result is whatever the analysis
    declares — typically a typed dataclass."""
    kwargs = _kwargs_for(analysis_obj, bridge_params)
    return analysis_obj.fn(cells, **kwargs)


def resolve_for_holds_when(
    holds_when: Callable[..., object],
    cells: Iterable[Mapping[str, object]],
    bridge_params: Mapping[str, object],
) -> dict[str, object]:
    """Walk a bridge's `holds_when` signature, look up each
    parameter as a registered analysis, run it, accumulate
    results into a `{param_name: result}` dict ready to splat
    as `**kwargs` into `holds_when`.

    Unknown parameter names raise `KeyError` with the registry's
    known set — the typo / missing-analysis case fails loudly at
    bridge invocation time rather than producing a silent NaN
    verdict."""
    try:
        sig = inspect.signature(holds_when)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f'cannot inspect holds_when signature: {exc}',
        ) from exc
    out: dict[str, object] = {}
    cells_list = list(cells)  # iterate once; pass to each analysis
    for param_name in sig.parameters:
        analysis_obj = _REGISTRY.get(param_name)
        if analysis_obj is None:
            raise KeyError(
                f'holds_when parameter {param_name!r} is not a '
                f'registered analysis; known: '
                f'{sorted(_REGISTRY)}',
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
