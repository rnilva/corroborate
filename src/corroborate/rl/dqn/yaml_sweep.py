"""DQN-substrate YAML manifest → `run_hypotheses` dispatch.

Two arm-shape modes, each with its own manifest dataclass for
typed discrimination at use-sites:

- `ChunkedManifest` — `arms_shape: chunked`. Hypotheses are
  env-generic and built once at load time. The dispatcher pairs
  each with every env Cartesianly via `chunked_arms`.
- `PairedManifest` — `arms_shape: paired`. Each Hypothesis is
  bound to one env (e.g. CNN configured per-env obs_shape). The
  YAML carries hypothesis *templates* with `{from_env: <attr>}`
  placeholders; the dispatcher iterates envs, substitutes
  placeholders against `EnvSpec` attributes, builds one concrete
  Hypothesis per (template, env) pair, and runs `paired_arms`.

The split between *manifest* and *dispatch* keeps tests cheap:
they load manifests without spinning up the runner so signature
parity can be checked against a Python-authored Hypothesis tuple
before any sweep launches."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Literal, TypeIs, cast

import yaml

from corroborate.config_loader import (
    build_hypothesis_from_mapping,
    is_str_keyed_mapping,
)
from corroborate.hypothesis import Hypothesis
from corroborate.registry import Registry
from corroborate.rl.dqn.collect import EnvConfig
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import EnvSpec


type ArmsShape = Literal['chunked', 'paired']


def _is_arms_shape(v: object) -> TypeIs[ArmsShape]:
    return isinstance(v, str) and v in ('chunked', 'paired')


@dataclass(frozen=True, slots=True)
class ChunkedManifest:
    """Manifest for chunked dispatch: hypotheses built eagerly and
    paired with envs Cartesianly at dispatch."""
    name: str
    out_dir: Path
    envs: tuple[EnvConfig, ...]
    hypotheses: tuple[Hypothesis[Mapping[str, object]], ...]
    archive_remote: str | None = None


@dataclass(frozen=True, slots=True)
class PairedManifest:
    """Manifest for paired dispatch: hypothesis templates are
    deferred — each env-spec resolves the templates per-env via
    `{from_env: <attr>}` substitution, then `paired_arms` zips
    one (built-from-template) Hypothesis with each env_config."""
    name: str
    out_dir: Path
    envs: tuple[EnvConfig, ...]
    hypothesis_templates: tuple[Mapping[str, object], ...]
    archive_remote: str | None = None


type DQNExperimentManifest = ChunkedManifest | PairedManifest


def env_attrs_from_spec(spec: EnvSpec) -> dict[str, object]:
    """Project an `EnvSpec` to the attribute map consumed by the
    `from_env` resolver. Each public field becomes a key; the
    YAML's `{from_env: observation_shape}` resolves directly to
    `spec.observation_shape`."""
    return {f.name: getattr(spec, f.name) for f in dc_fields(spec)}


def load_manifest(
    path: Path, *, reg: Registry,
) -> DQNExperimentManifest:
    """Parse a YAML manifest into the typed manifest variant
    matching its `arms_shape`. Raises `TypeError` /
    `ValueError` / `KeyError` on schema violations with messages
    naming the offending field."""
    with path.open() as f:
        raw = cast(object, yaml.safe_load(f))
    if not is_str_keyed_mapping(raw):
        raise TypeError(
            f'top-level YAML must be a string-keyed mapping; got '
            f'{type(raw).__name__}',
        )
    return _build_manifest(raw, reg=reg)


def _build_manifest(
    node: Mapping[str, object], *, reg: Registry,
) -> DQNExperimentManifest:
    name = _require_str(node, 'name')
    out_dir = Path(_require_str(node, 'out_dir'))
    envs = _build_envs(node)
    arms_shape = _require_arms_shape(node)
    archive_remote = _build_archive_remote(node)
    base_intervention = _build_base_intervention(node)
    hypotheses_raw = node.get('hypotheses')
    if not isinstance(hypotheses_raw, list):
        raise TypeError(
            f'manifest.hypotheses must be a list; got '
            f'{type(hypotheses_raw).__name__}',
        )
    hypotheses_typed: list[object] = list(hypotheses_raw)

    merged_templates = tuple(
        _merge_with_base(h, base_intervention)
        for h in hypotheses_typed
    )

    if arms_shape == 'chunked':
        # Eager build — chunked has no env-binding.
        hypotheses = tuple(
            build_hypothesis_from_mapping(t, reg=reg)
            for t in merged_templates
        )
        return ChunkedManifest(
            name=name, out_dir=out_dir, envs=envs,
            hypotheses=hypotheses,
            archive_remote=archive_remote,
        )
    return PairedManifest(
        name=name, out_dir=out_dir, envs=envs,
        hypothesis_templates=merged_templates,
        archive_remote=archive_remote,
    )


def _require_str(node: Mapping[str, object], key: str) -> str:
    v = node.get(key)
    if not isinstance(v, str):
        raise TypeError(
            f'manifest.{key} must be a string; got '
            f'{type(v).__name__}',
        )
    return v


def _build_envs(node: Mapping[str, object]) -> tuple[EnvConfig, ...]:
    envs_raw = node.get('envs')
    if not isinstance(envs_raw, list):
        raise TypeError(
            f'manifest.envs must be a list; got '
            f'{type(envs_raw).__name__}',
        )
    envs_typed: list[object] = list(envs_raw)
    return tuple(_build_env(e) for e in envs_typed)


def _require_arms_shape(node: Mapping[str, object]) -> ArmsShape:
    v = node.get('arms_shape', 'chunked')
    if not _is_arms_shape(v):
        raise ValueError(
            f'manifest.arms_shape must be chunked|paired; got {v!r}',
        )
    return v


def _build_archive_remote(node: Mapping[str, object]) -> str | None:
    v = node.get('archive_remote')
    if v is None:
        return None
    if isinstance(v, str):
        return v
    raise TypeError(
        f'manifest.archive_remote must be string|null; got '
        f'{type(v).__name__}',
    )


def _build_base_intervention(
    node: Mapping[str, object],
) -> Mapping[str, object]:
    v = node.get('base_intervention', {})
    if not is_str_keyed_mapping(v):
        raise TypeError(
            f'manifest.base_intervention must be a mapping; got '
            f'{type(v).__name__}',
        )
    return v


def _merge_with_base(
    h_node: object, base: Mapping[str, object],
) -> Mapping[str, object]:
    """Shallow-merge `base` under the hypothesis's own
    `intervention` (own keys override). Returns the merged
    template (still raw — not yet `build_hypothesis_from_mapping`-
    resolved). Used for both chunked and paired modes."""
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
    return EnvConfig(
        env_name=name, n_seeds=n_seeds, chunk_size=chunk_size,
    )


def default_dqn_registry() -> Registry:
    """Pre-populated Registry covering the DQN substrate's claim
    namespace + the `Replay` config bundle. Authors of one-off
    experiments rarely need to extend this; substrates with extra
    Module Claims call `add_modules` / `add_container` after."""
    from corroborate.rl.dqn.claims.replay import Replay
    reg = Registry()
    reg.add_modules((
        'corroborate.rl.dqn.claims.bootstrap',
        'corroborate.rl.dqn.claims.action_select',
        'corroborate.rl.dqn.claims.replay',
        'corroborate.rl.dqn.claims.q_network',
        'corroborate.rl.dqn.claims.optimizer',
        'corroborate.rl.dqn.claims.target_sync',
        'corroborate.rl.dqn.claims.loss',
    ))
    reg.add_container(Replay)
    return reg


def build_paired_hypotheses(
    manifest: PairedManifest, *, reg: Registry,
) -> tuple[
    tuple[Hypothesis[Mapping[str, object]], ...],
    tuple[EnvConfig, ...],
]:
    """Resolve a `PairedManifest`'s templates against each env's
    `EnvSpec` attributes. Returns `(hypotheses, envs_aligned)`
    suitable for `paired_arms`: each env appears once per template,
    and the hypothesis tuple is the (env, template) Cartesian
    expansion in env-major order.

    Pulled out as a standalone function so tests can verify
    per-env resolution without dispatching the whole sweep."""
    from corroborate.rl.env_catalogue import get as get_env_spec

    hypotheses: list[Hypothesis[Mapping[str, object]]] = []
    envs_aligned: list[EnvConfig] = []
    for ec in manifest.envs:
        spec = get_env_spec(ec.env_name)
        env_attrs = env_attrs_from_spec(spec)
        for template in manifest.hypothesis_templates:
            hypotheses.append(
                build_hypothesis_from_mapping(
                    template, reg=reg, env_attrs=env_attrs,
                ),
            )
            envs_aligned.append(ec)
    return tuple(hypotheses), tuple(envs_aligned)


def dispatch_manifest(
    manifest: DQNExperimentManifest,
) -> tuple[Path, Path]:
    """Run the manifest end-to-end: build arms + env_specs +
    runner, forward to `run_hypotheses`. Returns the merged
    `(runs.parquet, traces.parquet)` paths.

    Substrate-coupled by design (knows about `DQNRunner`,
    `Q_TRACE_REDUCTIONS`, env catalogue). Replaces what the
    per-experiment Python script's `main()` used to do."""
    from corroborate.rl.dqn.collect import (
        chunked_arms, env_arm_tag, paired_arms,
    )
    from corroborate.rl.dqn.trace_reductions import (
        Q_TRACE_DROPS, Q_TRACE_REDUCTIONS,
    )
    from corroborate.rl.env_catalogue import get as get_env_spec
    from corroborate.rl.sweep import DQNRunner
    from corroborate.sweep import run_hypotheses

    if isinstance(manifest, ChunkedManifest):
        hypotheses_dqn: list[Hypothesis[DQNTrajectoryRecord]] = [
            cast(Hypothesis[DQNTrajectoryRecord], h)
            for h in manifest.hypotheses
        ]
        arms = chunked_arms(hypotheses_dqn, manifest.envs)
        envs_for_specs = manifest.envs
    else:
        reg = default_dqn_registry()
        built, envs_aligned = build_paired_hypotheses(
            manifest, reg=reg,
        )
        hypotheses_dqn_paired: list[
            Hypothesis[DQNTrajectoryRecord]
        ] = [
            cast(Hypothesis[DQNTrajectoryRecord], h) for h in built
        ]
        arms = paired_arms(hypotheses_dqn_paired, envs_aligned)
        envs_for_specs = manifest.envs

    env_specs = {
        ec.env_name: get_env_spec(ec.env_name)
        for ec in envs_for_specs
    }
    return run_hypotheses(
        arms,
        runner=DQNRunner(env_specs),
        out_dir=manifest.out_dir,
        archive_remote=manifest.archive_remote,
        arm_tag=env_arm_tag,
        trace_reductions=Q_TRACE_REDUCTIONS,
        trace_drops=Q_TRACE_DROPS,
    )


__all__ = [
    'ArmsShape',
    'ChunkedManifest',
    'DQNExperimentManifest',
    'PairedManifest',
    'build_paired_hypotheses',
    'default_dqn_registry',
    'dispatch_manifest',
    'env_attrs_from_spec',
    'load_manifest',
]
