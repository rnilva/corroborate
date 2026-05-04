"""DQN-substrate YAML → `run_hypotheses` dispatch.

`DQNSweep` is the typed shape of a configured sweep loaded from
YAML — one dataclass for both arm shapes. The dispatch
distinction lives in `arms_shape: 'chunked' | 'paired'` and the
`{from_env: <attr>}` placeholders inside `hypothesis_templates`,
not in the dataclass type.

- `arms_shape: 'chunked'` — hypotheses are env-generic. The
  templates resolve once (no env_attrs) and pair Cartesianly with
  envs via `chunked_arms`.
- `arms_shape: 'paired'` — each (template × env) builds one
  concrete Hypothesis after `{from_env: <attr>}` substitution
  against `EnvSpec.public_attrs()`. The substrate's `paired_arms`
  zips them with one env_config per arm.

The split between *shape* (the dataclass) and *dispatch* (the
function) keeps tests cheap: they load a `DQNSweep` and inspect
without spinning up the runner."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeIs

import yaml

from corroborate.runner.config_loader import (
    build_hypothesis_from_mapping,
    is_str_keyed_mapping,
)
from corroborate.core.hypothesis import LegacyHypothesis as Hypothesis
from corroborate.runner.registry import Registry
from corroborate_rl.dqn.collect import EnvConfig
from corroborate_rl.env_catalogue import EnvWrapper
from corroborate_rl.dqn.invariants import DQNTrajectoryRecord
from corroborate_rl.env_catalogue import EnvSpec


type ArmsShape = Literal['chunked', 'paired']


def _is_arms_shape(v: object) -> TypeIs[ArmsShape]:
    return isinstance(v, str) and v in ('chunked', 'paired')


@dataclass(frozen=True, slots=True)
class DQNSweep:
    """A configured DQN sweep. `hypothesis_templates` are raw
    string-keyed mappings (pre-resolution); call
    `build_hypotheses` with the appropriate env context to get
    concrete `Hypothesis` instances.

    The dataclass is shape-uniform between chunked and paired
    modes. The dispatch routine reads `arms_shape` to decide
    whether to resolve once (chunked) or per-env (paired)."""
    name: str
    out_dir: Path
    envs: tuple[EnvConfig, ...]
    hypothesis_templates: tuple[Mapping[str, object], ...]
    arms_shape: ArmsShape
    archive_remote: str | None = None

    def build_hypotheses(
        self,
        *,
        reg: Registry,
        env_attrs: Mapping[str, object] | None = None,
    ) -> tuple[Hypothesis[Mapping[str, object]], ...]:
        """Resolve every template against `reg` and return the
        built Hypothesis tuple. Pass `env_attrs=None` for chunked
        mode (any `{from_env: <attr>}` placeholder raises);
        provide an env's `public_attrs()` map for paired mode."""
        return tuple(
            build_hypothesis_from_mapping(
                t, reg=reg, env_attrs=env_attrs,
            )
            for t in self.hypothesis_templates
        )


def env_attrs_from_spec(spec: EnvSpec) -> Mapping[str, object]:
    """Adapter to the catalogue's whitelist. Kept here so the
    YAML loader doesn't import `EnvSpec.public_attrs` directly —
    swapping the env catalogue for a substrate-specific one is
    one function change."""
    return spec.public_attrs()


def load_sweep(path: Path, *, reg: Registry) -> DQNSweep:
    """Parse a YAML sweep config. Raises `TypeError` /
    `ValueError` / `KeyError` on schema violations with messages
    naming the offending field."""
    with path.open() as f:
        # `yaml.safe_load` returns the closed YAML union (None /
        # bool / int / float / str / list / dict), all of which are
        # `object` — narrow at this boundary so downstream `is_*`
        # predicates do typed work.
        raw: object = yaml.safe_load(f)
    if not is_str_keyed_mapping(raw):
        raise TypeError(
            f'top-level YAML must be a string-keyed mapping; got '
            f'{type(raw).__name__}',
        )
    return _build_sweep(raw)


def _build_sweep(node: Mapping[str, object]) -> DQNSweep:
    name = _require_str(node, 'name')
    out_dir = Path(_require_str(node, 'out_dir'))
    envs = _build_envs(node)
    arms_shape = _require_arms_shape(node)
    archive_remote = _build_archive_remote(node)
    base_intervention = _build_base_intervention(node)
    hypotheses_raw = node.get('hypotheses')
    if not isinstance(hypotheses_raw, list):
        raise TypeError(
            f'sweep.hypotheses must be a list; got '
            f'{type(hypotheses_raw).__name__}',
        )
    hypotheses_typed: list[object] = list(hypotheses_raw)
    templates = tuple(
        _merge_with_base(h, base_intervention)
        for h in hypotheses_typed
    )
    return DQNSweep(
        name=name, out_dir=out_dir, envs=envs,
        hypothesis_templates=templates,
        arms_shape=arms_shape, archive_remote=archive_remote,
    )


def _require_str(node: Mapping[str, object], key: str) -> str:
    v = node.get(key)
    if not isinstance(v, str):
        raise TypeError(
            f'sweep.{key} must be a string; got '
            f'{type(v).__name__}',
        )
    return v


def _build_envs(node: Mapping[str, object]) -> tuple[EnvConfig, ...]:
    envs_raw = node.get('envs')
    if not isinstance(envs_raw, list):
        raise TypeError(
            f'sweep.envs must be a list; got '
            f'{type(envs_raw).__name__}',
        )
    envs_typed: list[object] = list(envs_raw)
    return tuple(_build_env(e) for e in envs_typed)


def _require_arms_shape(node: Mapping[str, object]) -> ArmsShape:
    v = node.get('arms_shape', 'chunked')
    if not _is_arms_shape(v):
        raise ValueError(
            f'sweep.arms_shape must be chunked|paired; got {v!r}',
        )
    return v


def _build_archive_remote(node: Mapping[str, object]) -> str | None:
    v = node.get('archive_remote')
    if v is None:
        return None
    if isinstance(v, str):
        return v
    raise TypeError(
        f'sweep.archive_remote must be string|null; got '
        f'{type(v).__name__}',
    )


def _build_base_intervention(
    node: Mapping[str, object],
) -> Mapping[str, object]:
    v = node.get('base_intervention', {})
    if not is_str_keyed_mapping(v):
        raise TypeError(
            f'sweep.base_intervention must be a mapping; got '
            f'{type(v).__name__}',
        )
    return v


def _merge_with_base(
    h_node: object, base: Mapping[str, object],
) -> Mapping[str, object]:
    """Shallow-merge `base` under the hypothesis's own
    `intervention` (own keys override). Returns the merged
    template (still raw — not yet resolved)."""
    if not is_str_keyed_mapping(h_node):
        raise TypeError(
            f'hypothesis must be a mapping; got '
            f'{type(h_node).__name__}',
        )
    own_intervention = h_node.get('intervention', {})
    if not is_str_keyed_mapping(own_intervention):
        raise TypeError(
            f'hypothesis.intervention must be a mapping; got '
            f'{type(own_intervention).__name__}',
        )
    return {
        **h_node,
        'intervention': {**base, **own_intervention},
    }


def _build_env(node: object) -> EnvConfig:
    if not is_str_keyed_mapping(node):
        raise TypeError(
            f'env entry must be a mapping; got {type(node).__name__}',
        )
    name = node.get('name')
    if not isinstance(name, str):
        raise TypeError(
            f'env.name must be a string; got {type(name).__name__}',
        )
    n_seeds = node.get('n_seeds', 30)
    if not isinstance(n_seeds, int) or isinstance(n_seeds, bool):
        raise TypeError(
            f'env.n_seeds must be int; got {type(n_seeds).__name__}',
        )
    chunk_size = node.get('chunk_size', n_seeds)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(
            f'env.chunk_size must be int; got '
            f'{type(chunk_size).__name__}',
        )
    wrappers = _build_wrappers(node)
    return EnvConfig(
        env_name=name, n_seeds=n_seeds, chunk_size=chunk_size,
        wrappers=wrappers,
    )


def _build_wrappers(node: Mapping[str, object]) -> tuple['EnvWrapper', ...]:
    """Parse `wrappers: [{type: <name>, ...}]` into an
    `EnvWrapper` tuple. Each entry's `type` field looks up the
    wrapper class in the registry; remaining kwargs initialize
    the dataclass.

    Strict — no legacy `reward_scale: ...` sugar. If you see
    that on an old YAML, rewrite to
    `wrappers: [{type: reward_scale, scale: 0.1}]`. Single
    canonical form keeps future readers from wondering which
    is authoritative."""
    from corroborate_rl.env_catalogue import get_wrapper_class
    # Catch the legacy sugar fields with a loud error rather than
    # silently ignoring them — sweeps that relied on them would
    # otherwise produce a corpus without the wrapper applied.
    for legacy_key in ('reward_scale', 'reward_clip_min', 'reward_clip_max'):
        if legacy_key in node:
            raise ValueError(
                f"env.{legacy_key!r} is no longer accepted as a top-level "
                f"YAML key; use `wrappers: [{{type: ..., ...}}]`. "
                f"Existing YAMLs sanitized in-tree.",
            )
    raw = node.get('wrappers')
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError(
            f'env.wrappers must be a list; got {type(raw).__name__}',
        )
    out: list[EnvWrapper] = []
    for entry in raw:
        if not is_str_keyed_mapping(entry):
            raise TypeError(
                f'env.wrappers entry must be a mapping; got '
                f'{type(entry).__name__}',
            )
        type_v = entry.get('type')
        if not isinstance(type_v, str):
            raise TypeError(
                f"env.wrappers entry must have 'type: <name>'; "
                f"got {entry!r}",
            )
        cls = get_wrapper_class(type_v)
        kwargs = {k: v for k, v in entry.items() if k != 'type'}
        out.append(cls(**kwargs))
    return tuple(out)


def default_dqn_registry() -> Registry:
    """Pre-populated Registry covering the DQN substrate's claim
    namespace. `add_modules` auto-discovers `@claim` free
    functions and frozen-dataclass config bundles (`Replay`,
    `MLP`, `CNN`); authors of one-off sweeps rarely need to
    extend this."""
    reg = Registry()
    reg.add_modules((
        'corroborate_rl.dqn.claims.bootstrap',
        'corroborate_rl.dqn.claims.action_select',
        'corroborate_rl.dqn.claims.replay',
        'corroborate_rl.dqn.claims.q_network',
        'corroborate_rl.dqn.claims.optimizer',
        'corroborate_rl.dqn.claims.target_sync',
        'corroborate_rl.dqn.claims.loss',
    ))
    return reg


def build_paired(
    sweep: DQNSweep, *, reg: Registry,
) -> tuple[
    tuple[Hypothesis[Mapping[str, object]], ...],
    tuple[EnvConfig, ...],
]:
    """Resolve a paired sweep's templates against each env's
    `EnvSpec.public_attrs()`. Returns `(hypotheses,
    envs_aligned)` suitable for `paired_arms`: each env appears
    once per template, in env-major order.

    Standalone so tests can verify per-env resolution without
    dispatching the whole sweep."""
    if sweep.arms_shape != 'paired':
        raise ValueError(
            f"build_paired requires arms_shape='paired'; got "
            f'{sweep.arms_shape!r}',
        )
    from corroborate_rl.env_catalogue import get as get_env_spec

    hypotheses: list[Hypothesis[Mapping[str, object]]] = []
    envs_aligned: list[EnvConfig] = []
    for ec in sweep.envs:
        spec = get_env_spec(ec.env_name)
        env_attrs = env_attrs_from_spec(spec)
        for built in sweep.build_hypotheses(reg=reg, env_attrs=env_attrs):
            hypotheses.append(built)
            envs_aligned.append(ec)
    return tuple(hypotheses), tuple(envs_aligned)


def dispatch_sweep(sweep: DQNSweep) -> tuple[Path, Path]:
    """Run the sweep end-to-end: build arms + env_specs + runner,
    forward to `run_hypotheses`. Returns the merged
    `(runs.parquet, traces.parquet)` paths.

    Substrate-coupled by design (knows about `DQNRunner`,
    `Q_TRACE_REDUCTIONS`, env catalogue)."""
    from corroborate_rl.dqn.collect import (
        chunked_arms, env_arm_tag, paired_arms,
    )
    from corroborate_rl.dqn.trace_reductions import (
        Q_TRACE_DROPS, Q_TRACE_REDUCTIONS,
    )
    from corroborate_rl.env_catalogue import get as get_env_spec
    from corroborate_rl.sweep import DQNRunner
    from corroborate.runner.sweep import run_hypotheses

    reg = default_dqn_registry()
    if sweep.arms_shape == 'chunked':
        # `build_hypotheses` returns `Hypothesis[Mapping[str,
        # object]]` (framework-generic); the substrate slot wants
        # `Hypothesis[DQNTrajectoryRecord]`. Hypothesis.R is
        # contravariant (regular-class + @property form), so the
        # wider-R framework-generic IS assignable to the narrower-R
        # substrate type without a `cast`.
        hypotheses_dqn: list[Hypothesis[DQNTrajectoryRecord]] = list(
            sweep.build_hypotheses(reg=reg),
        )
        arms = chunked_arms(hypotheses_dqn, sweep.envs)
    else:
        built_paired, envs_aligned = build_paired(sweep, reg=reg)
        hypotheses_dqn = list(built_paired)
        arms = paired_arms(hypotheses_dqn, envs_aligned)

    env_specs = {
        ec.env_name: get_env_spec(ec.env_name) for ec in sweep.envs
    }
    return run_hypotheses(
        arms,
        runner=DQNRunner(env_specs),
        out_dir=sweep.out_dir,
        archive_remote=sweep.archive_remote,
        arm_tag=env_arm_tag,
        trace_reductions=Q_TRACE_REDUCTIONS,
        trace_drops=Q_TRACE_DROPS,
    )


__all__ = [
    'ArmsShape',
    'DQNSweep',
    'build_paired',
    'default_dqn_registry',
    'dispatch_sweep',
    'env_attrs_from_spec',
    'load_sweep',
]
