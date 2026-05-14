"""DQN-substrate YAML → `run_intervention` dispatch.

`DQNSweep` is the typed shape of a configured sweep loaded from
YAML — one dataclass for both arm shapes. The dispatch
distinction lives in `env_binding: 'shared' | 'per_env'` and the
`{from_env: <attr>}` placeholders inside `intervention_templates`,
not in the dataclass type.

- `env_binding: 'shared'` — intervention templates are env-generic.
  Each resolves once (no env_attrs) and pairs Cartesianly with
  envs at dispatch time.
- `env_binding: 'per_env'` — each (template × env) builds one
  concrete InterventionConfig after `{from_env: <attr>}`
  substitution against `EnvSpec.public_attrs()`. The substrate
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

from corroborate.runner.registry import Registry
from corroborate_rl.dqn.config_loader import (
    InterventionConfig,
    build_intervention_from_mapping,
    is_str_keyed_mapping,
)
from corroborate_rl.dqn.collect import EnvConfig
from corroborate_rl.env_catalogue import EnvWrapper
from corroborate_rl.dqn.invariants import DQNTrajectoryRecord
from corroborate_rl.env_catalogue import EnvSpec


type EnvBinding = Literal['shared', 'per_env']


def _is_env_binding(v: object) -> TypeIs[EnvBinding]:
    return isinstance(v, str) and v in ('shared', 'per_env')


@dataclass(frozen=True, slots=True)
class DQNSweep:
    """A configured DQN sweep. `intervention_templates` are raw
    string-keyed mappings (pre-resolution); call
    `build_interventions` with the appropriate env context to get
    concrete `InterventionConfig` instances.

    The dataclass is shape-uniform between shared and per-env
    modes. The dispatch routine reads `env_binding` to decide
    whether to resolve once (shared) or per-env (per_env)."""
    name: str
    out_dir: Path
    envs: tuple[EnvConfig, ...]
    intervention_templates: tuple[Mapping[str, object], ...]
    env_binding: EnvBinding
    archive_remote: str | None = None

    def build_interventions(
        self,
        *,
        reg: Registry,
        env_attrs: Mapping[str, object] | None = None,
    ) -> tuple[InterventionConfig, ...]:
        """Resolve every template against `reg` and return the
        built InterventionConfig tuple. Pass `env_attrs=None` for
        shared mode (any `{from_env: <attr>}` placeholder raises);
        provide an env's `public_attrs()` map for per-env mode."""
        return tuple(
            build_intervention_from_mapping(
                t, reg=reg, env_attrs=env_attrs,
            )
            for t in self.intervention_templates
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
    env_binding = _require_env_binding(node)
    archive_remote = _build_archive_remote(node)
    defaults = _build_defaults(node)
    interventions_raw = node.get('interventions')
    if not isinstance(interventions_raw, list):
        raise TypeError(
            f'sweep.interventions must be a list; got '
            f'{type(interventions_raw).__name__}',
        )
    interventions_typed: list[object] = list(interventions_raw)
    templates = tuple(
        _merge_with_defaults(h, defaults)
        for h in interventions_typed
    )
    return DQNSweep(
        name=name, out_dir=out_dir, envs=envs,
        intervention_templates=templates,
        env_binding=env_binding, archive_remote=archive_remote,
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


def _require_env_binding(node: Mapping[str, object]) -> EnvBinding:
    v = node.get('env_binding', 'shared')
    if not _is_env_binding(v):
        raise ValueError(
            f'sweep.env_binding must be shared|per_env; got {v!r}',
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


def _build_defaults(
    node: Mapping[str, object],
) -> Mapping[str, object]:
    v = node.get('defaults', {})
    if not is_str_keyed_mapping(v):
        raise TypeError(
            f'sweep.defaults must be a mapping; got '
            f'{type(v).__name__}',
        )
    return v


def _merge_with_defaults(
    h_node: object, defaults: Mapping[str, object],
) -> Mapping[str, object]:
    """Shallow-merge `defaults` under the intervention's own
    `base` (own keys override). Returns the merged template
    (still raw — not yet resolved)."""
    if not is_str_keyed_mapping(h_node):
        raise TypeError(
            f'intervention must be a mapping; got '
            f'{type(h_node).__name__}',
        )
    own_base = h_node.get('base', {})
    if not is_str_keyed_mapping(own_base):
        raise TypeError(
            f'intervention.base must be a mapping; got '
            f'{type(own_base).__name__}',
        )
    return {
        **h_node,
        'base': {**defaults, **own_base},
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
    seed_offset = node.get('seed_offset', 0)
    if not isinstance(seed_offset, int) or isinstance(seed_offset, bool):
        raise TypeError(
            f'env.seed_offset must be int; got '
            f'{type(seed_offset).__name__}',
        )
    wrappers = _build_wrappers(node)
    return EnvConfig(
        env_name=name, n_seeds=n_seeds, chunk_size=chunk_size,
        wrappers=wrappers, seed_offset=seed_offset,
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


def build_per_env(
    sweep: DQNSweep, *, reg: Registry,
) -> tuple[
    tuple[InterventionConfig, ...],
    tuple[EnvConfig, ...],
]:
    """Resolve a per-env sweep's templates against each env's
    `EnvSpec.public_attrs()`. Returns `(interventions,
    envs_aligned)` suitable for paired dispatch: each env appears
    once per template, in env-major order.

    Standalone so tests can verify per-env resolution without
    dispatching the whole sweep."""
    if sweep.env_binding != 'per_env':
        raise ValueError(
            f"build_per_env requires env_binding='per_env'; got "
            f'{sweep.env_binding!r}',
        )
    from corroborate_rl.env_catalogue import get as get_env_spec

    interventions: list[InterventionConfig] = []
    envs_aligned: list[EnvConfig] = []
    for ec in sweep.envs:
        spec = get_env_spec(ec.env_name)
        env_attrs = env_attrs_from_spec(spec)
        for built in sweep.build_interventions(reg=reg, env_attrs=env_attrs):
            interventions.append(built)
            envs_aligned.append(ec)
    return tuple(interventions), tuple(envs_aligned)


def dispatch_sweep(sweep: DQNSweep) -> tuple[Path, Path]:
    """Run the sweep end-to-end. For each YAML-loaded
    `InterventionConfig`, decompose into a Hypothesis Protocol-
    conformer + `base` Callable (`partial(dqn, **HPs)`), build a
    discrete grid_points list (env × seed_chunk × wrappers), and
    dispatch to the framework's `run_intervention` paired-sweep
    primitive. Each InterventionConfig produces its own per-arm
    parquet pair under `<out_dir>/<name>/`; the per-config corpora
    are concatenated to `<out_dir>/runs.parquet` /
    `<out_dir>/traces.parquet`.

    Substrate-coupled by design (knows about `DQNRunner`,
    `Q_TRACE_REDUCTIONS`, env catalogue, `dqn` theory)."""
    from collections.abc import Callable, Sequence
    from functools import partial

    from corroborate.corpus.persistence import stream_concat_parquets
    from corroborate.runner.sweep import run_intervention
    from corroborate_rl.dqn.collect import _chunks
    from corroborate_rl.dqn.dqn import dqn
    from corroborate_rl.dqn.measurables import dqn_default_measurables
    from corroborate_rl.dqn.trace_reductions import (
        Q_TRACE_DROPS, Q_TRACE_REDUCTIONS,
    )
    from corroborate_rl.env_catalogue import (
        get as get_env_spec, wrappers_canonical_str,
    )
    from corroborate_rl.sweep import DQNRunner

    reg = default_dqn_registry()
    if sweep.env_binding == 'shared':
        configs: list[InterventionConfig] = list(
            sweep.build_interventions(reg=reg),
        )
        envs_per_h: list[Sequence[EnvConfig]] = [
            list(sweep.envs)
        ] * len(configs)
    else:
        built_per_env, envs_aligned = build_per_env(sweep, reg=reg)
        configs = list(built_per_env)
        envs_per_h = [[ec] for ec in envs_aligned]

    # Each config writes to `sweep.out_dir / cfg.name`; two configs
    # sharing a name silently overwrite each other's runs.parquet
    # at merge time and the final corpus loses all but one config's
    # data. Refuse to dispatch on collision. With `env_binding:
    # per_env`, this fires when the intervention template's `name`
    # field omits an env-attribute substitution (e.g.,
    # `name: ddqn_vs_{from_env: env_name}`); fix the template or
    # switch to `env_binding: shared`.
    seen_names: dict[str, int] = {}
    for cfg in configs:
        seen_names[cfg.name] = seen_names.get(cfg.name, 0) + 1
    collisions = {n: c for n, c in seen_names.items() if c > 1}
    if collisions:
        raise ValueError(
            f'dispatch_sweep: configs share output paths — '
            f'{collisions!r} would overwrite each other at '
            f'`<out_dir>/<cfg.name>/runs.parquet`. '
            f'Templating the intervention `name` with '
            f"`{{from_env: env_name}}` (env_binding='per_env') or "
            f"switching to env_binding='shared' resolves this. "
            f'Sweep aborted before any data is written.',
        )

    env_specs = {
        ec.env_name: get_env_spec(ec.env_name) for ec in sweep.envs
    }
    runner = DQNRunner(env_specs)

    def _arm_tag(arm_key: str, gp: Mapping[str, object]) -> str:
        env_name = gp.get('env_name', '')
        wrappers = gp.get('wrappers', ())
        suffix = (
            f'__wrap[{wrappers_canonical_str(wrappers)}]'
            if isinstance(wrappers, tuple) and wrappers
            else ''
        )
        return f'{env_name}__{arm_key}{suffix}'

    # **CORPUS_INTEGRITY.md CI1 + sentinel discipline**: write
    # `.in_progress` at sweep start so an `--ingest-all` walk in
    # parallel skips the half-built corpus. Removed in the
    # `try/finally` after `dispatch_sweep` returns / errors so the
    # sentinel matches actual sweep state.
    from corroborate.corpus.integrity import IN_PROGRESS_SENTINEL
    sweep.out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = sweep.out_dir / IN_PROGRESS_SENTINEL
    sentinel.touch()
    sub_runs: list[Path] = []
    sub_traces: list[Path] = []
    sub_arm_dirs: list[Path] = []
    for cfg, env_configs in zip(configs, envs_per_h, strict=True):
        # `base` IS the SCM kwargs map; each arm's interventions
        # override slot values via partial precedence in
        # `apply_interventions`. Empty-tuple arm = "use base".
        base: Callable[..., object] = partial(dqn, **cfg.base)
        intervention = cfg.do_effect
        # Flat grid_points: env × chunk × wrappers.
        grid_points: list[Mapping[str, object]] = [
            {
                'env_name': ec.env_name,
                'seeds': chunk,
                'wrappers': ec.wrappers,
            }
            for ec in env_configs
            for chunk in _chunks(ec)
        ]
        h_out_dir = sweep.out_dir / cfg.name
        # Mirror the local out_dir composition on the remote
        # (invariant I1 in SWEEP_PERSISTENCY.md). The local path is
        # already arm-config-namespaced via `sweep.out_dir / cfg.name`;
        # without mirroring, two arm-configs share the same
        # `<archive_remote>/tmp/cell{NNN}__{tag}` URI on S3 and the
        # second upload silently overwrites the first. Per-arm-config
        # remote prefix closes the collision.
        h_archive_remote: str | None = (
            f'{sweep.archive_remote.rstrip("/")}/{cfg.name}'
            if sweep.archive_remote is not None else None
        )
        rp, tp = run_intervention(
            intervention,
            base=base,
            measurables=dqn_default_measurables(),
            grid_points=grid_points,
            runner=runner,
            out_dir=h_out_dir,
            archive_remote=h_archive_remote,
            arm_tag=_arm_tag,
            trace_reductions=Q_TRACE_REDUCTIONS,
            trace_drops=Q_TRACE_DROPS,
        )
        sub_runs.append(rp)
        sub_traces.append(tp)
        sub_arm_dirs.append(h_out_dir)

    final_runs = sweep.out_dir / 'runs.parquet'
    final_traces = sweep.out_dir / 'traces.parquet'
    # If either merge raises, the sentinel stays — subsequent
    # `--ingest-all` walks see "still in progress" and skip the
    # corpus rather than ingest a half-merged parent.
    stream_concat_parquets(sub_runs, final_runs)
    # **Disk-full graceful fallback**: traces.parquet merge is the
    # large one (per-cell shards can sum to 30 GB on 60-seed × 1M
    # sweeps). If the pre-flight check in stream_concat_parquets
    # raises ENOSPC, leave the per-arm sub-corpora intact instead
    # of crashing the sweep. The sub-corpora are independently
    # usable for analysis and downstream --ingest is shard-aware.
    # Pre-fix this triggered a ~20 GB orphan `.partial` mid-write
    # and burned 30+ min of GPU compute on an unrecoverable merge.
    import errno
    import sys
    try:
        stream_concat_parquets(sub_traces, final_traces)
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise
        sys.stderr.write(
            f'run_sweep: WARNING — top-level traces.parquet merge '
            f'skipped (insufficient disk in '
            f'{sweep.out_dir.parent}). Per-intervention sub-corpora '
            f'under {sweep.out_dir} are intact and usable directly '
            f'for analysis / ingest. To finish the top-level merge '
            f'later: archive sub-corpora, free disk, then concat '
            f'their traces.parquet via '
            f'`stream_concat_parquets`. The sweep '
            f'`.in_progress` sentinel stays UP — `--ingest-all` '
            f'will skip the parent dir until merged or removed.\n',
        )
        # Don't delete sub_arm_dirs (they're the salvage path).
        # Don't remove sentinel (signals incomplete merge state).
        # Return the per-arm paths so callers know what landed.
        return final_runs, sweep.out_dir
    # **Scratch cleanup**: per-arm sub-corpora are scratch — the
    # parent runs.parquet + traces.parquet now have everything.
    # Pre-fix this was documented as a manual `rm -rf` step,
    # which created CORPUS_INTEGRITY.md CI1 nested-corpus
    # violations on every subsequent `--ingest-all` walk. Auto-
    # clean now: each per-arm `<out_dir>/<arm>/` directory
    # (containing the unconcatenated runs/traces used as merge
    # inputs) gets removed once the parent merge is durable.
    import shutil
    for arm_dir in sub_arm_dirs:
        if arm_dir.exists() and arm_dir.is_dir():
            shutil.rmtree(arm_dir)
    # Sentinel removed only on successful completion (atomicity:
    # crash → sentinel stays → ingest skips).
    if sentinel.exists():
        try:
            sentinel.unlink()
        except OSError:
            pass
    return final_runs, final_traces


__all__ = [
    'DQNSweep',
    'EnvBinding',
    'build_per_env',
    'default_dqn_registry',
    'dispatch_sweep',
    'env_attrs_from_spec',
    'load_sweep',
]
