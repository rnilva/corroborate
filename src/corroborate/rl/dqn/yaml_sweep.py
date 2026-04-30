"""DQN-substrate YAML manifest → `run_hypotheses` dispatch.

A manifest YAML carries everything the per-experiment Python
script used to construct in code:

- `name`, `out_dir`, optional `archive_remote`
- `envs` — list of `{name, n_seeds, chunk_size}` triples
- `arms_shape` — `chunked` (Cartesian h×env) or `paired` (zip
  h-with-env, e.g. when each Hypothesis carries an env-specific
  CNN).
- `base_intervention` — shared kwargs each Hypothesis inherits via
  shallow merge (per-hypothesis `intervention` keys override).
- `hypotheses` — list of Hypothesis nodes (same schema as
  `corroborate.config_loader.load_hypothesis`).

`load_manifest(path, reg=...)` parses the file into a typed
`DQNExperimentManifest`. `dispatch_manifest(manifest)` builds the
arms list (chunked or paired), the env_specs map, and forwards to
`run_hypotheses` with the canonical `Q_TRACE_REDUCTIONS` /
`env_arm_tag` for the DQN substrate.

The split between *manifest* and *dispatch* is intentional: tests
load manifests without spinning up the runner, so signature
parity can be checked against a Python-authored Hypothesis tuple
before any sweep launches."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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


type ArmsShape = Literal['chunked', 'paired']


def _is_arms_shape(v: object) -> TypeIs[ArmsShape]:
    return isinstance(v, str) and v in ('chunked', 'paired')


@dataclass(frozen=True, slots=True)
class DQNExperimentManifest:
    """Parsed YAML manifest for a DQN sweep. Immutable; the
    dispatcher reads but does not mutate.

    `hypotheses` is generic in `Mapping[str, object]` rather than
    `DQNTrajectoryRecord` to keep the loader substrate-loose; the
    runtime narrows when binding to `DQNRunner` at dispatch."""
    name: str
    out_dir: Path
    envs: tuple[EnvConfig, ...]
    hypotheses: tuple[Hypothesis[Mapping[str, object]], ...]
    arms_shape: ArmsShape
    archive_remote: str | None = None


def load_manifest(
    path: Path, *, reg: Registry,
) -> DQNExperimentManifest:
    """Parse a YAML manifest into a typed `DQNExperimentManifest`.
    Raises `TypeError` / `ValueError` / `KeyError` on schema
    violations with messages naming the offending field."""
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
    name = node.get('name')
    if not isinstance(name, str):
        raise TypeError(
            f'manifest.name must be a string; got '
            f'{type(name).__name__}',
        )

    out_dir_raw = node.get('out_dir')
    if not isinstance(out_dir_raw, str):
        raise TypeError(
            f'manifest.out_dir must be a string; got '
            f'{type(out_dir_raw).__name__}',
        )
    out_dir = Path(out_dir_raw)

    envs_raw = node.get('envs')
    if not isinstance(envs_raw, list):
        raise TypeError(
            f'manifest.envs must be a list; got '
            f'{type(envs_raw).__name__}',
        )
    envs_typed: list[object] = list(envs_raw)
    envs = tuple(_build_env(e) for e in envs_typed)

    arms_shape_raw = node.get('arms_shape', 'chunked')
    if not _is_arms_shape(arms_shape_raw):
        raise ValueError(
            f'manifest.arms_shape must be chunked|paired; got '
            f'{arms_shape_raw!r}',
        )

    archive_remote_raw = node.get('archive_remote')
    archive_remote: str | None
    if archive_remote_raw is None:
        archive_remote = None
    elif isinstance(archive_remote_raw, str):
        archive_remote = archive_remote_raw
    else:
        raise TypeError(
            f'manifest.archive_remote must be string|null; got '
            f'{type(archive_remote_raw).__name__}',
        )

    base_intervention_raw = node.get('base_intervention', {})
    if not is_str_keyed_mapping(base_intervention_raw):
        raise TypeError(
            f'manifest.base_intervention must be a mapping; got '
            f'{type(base_intervention_raw).__name__}',
        )

    hypotheses_raw = node.get('hypotheses')
    if not isinstance(hypotheses_raw, list):
        raise TypeError(
            f'manifest.hypotheses must be a list; got '
            f'{type(hypotheses_raw).__name__}',
        )
    hypotheses_typed: list[object] = list(hypotheses_raw)
    hypotheses = tuple(
        _merge_and_build(h, base_intervention_raw, reg=reg)
        for h in hypotheses_typed
    )

    return DQNExperimentManifest(
        name=name,
        out_dir=out_dir,
        envs=envs,
        hypotheses=hypotheses,
        arms_shape=arms_shape_raw,
        archive_remote=archive_remote,
    )


def _merge_and_build(
    h_node: object,
    base: Mapping[str, object],
    *,
    reg: Registry,
) -> Hypothesis[Mapping[str, object]]:
    """Shallow-merge `base` under the hypothesis's own
    `intervention` (own keys override) before delegating to
    `build_hypothesis_from_mapping`. Avoids per-hypothesis
    duplication of the shared HP block."""
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
    merged_intervention: dict[str, object] = {
        **base, **own_intervention,
    }
    h_with_merged: dict[str, object] = {
        **h_node, 'intervention': merged_intervention,
    }
    return build_hypothesis_from_mapping(h_with_merged, reg=reg)


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


def dispatch_manifest(
    manifest: DQNExperimentManifest,
) -> tuple[Path, Path]:
    """Run the manifest end-to-end: build arms, env specs, runner;
    forward to `run_hypotheses`. Returns the merged
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

    hypotheses_dqn: list[Hypothesis[DQNTrajectoryRecord]] = [
        cast(Hypothesis[DQNTrajectoryRecord], h)
        for h in manifest.hypotheses
    ]
    arms = (
        chunked_arms(hypotheses_dqn, manifest.envs)
        if manifest.arms_shape == 'chunked'
        else paired_arms(hypotheses_dqn, manifest.envs)
    )
    env_specs = {
        ec.env_name: get_env_spec(ec.env_name)
        for ec in manifest.envs
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
    'DQNExperimentManifest',
    'default_dqn_registry',
    'dispatch_manifest',
    'load_manifest',
]
