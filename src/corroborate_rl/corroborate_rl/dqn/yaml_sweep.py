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
without spinning up the runner.

The substrate-agnostic YAML primitives (`Sweep` Protocol, scalar
parsers, manifest writer) live in `corroborate.runner.yaml_sweep`;
this module composes them with DQN-specific env / intervention
parsing + dispatch."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeIs

import yaml

from corroborate.runner.registry import Registry
from corroborate.runner.yaml_sweep import (
    BridgeCommitmentInput,
    assert_unique_cfg_names,
    build_archive_remote,
    build_merge_top_level,
    build_pre_registered_bridges,
    require_sweep_str,
    write_pre_registration_manifest_for_sweep,
)
from corroborate_rl.dqn.config_loader import (
    InterventionConfig,
    build_intervention_from_mapping,
    is_str_keyed_mapping,
)
from corroborate_rl.dqn.collect import EnvConfig
from corroborate_rl.dqn.q_checkpoint import CheckpointRole
from corroborate_rl.dqn.q_checkpoint_bundle import QCheckpointBundle
from corroborate_rl.env_catalogue import EnvSpec, EnvWrapper


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
    whether to resolve once (shared) or per-env (per_env).

    Structurally satisfies `corroborate.runner.yaml_sweep.Sweep`
    (frozen-dataclass fields match the Protocol's read-only
    `@property` shape — name, out_dir, archive_remote,
    merge_top_level, pre_registered_bridges)."""
    name: str
    out_dir: Path
    envs: tuple[EnvConfig, ...]
    intervention_templates: tuple[Mapping[str, object], ...]
    env_binding: EnvBinding
    archive_remote: str | None = None
    # Jacobian-based intra/inter-state α probes in `train_phase`.
    # Default True (backwards compat). YAML field
    # `gradient_probes: false` disables; `dispatch_sweep` mutates
    # `corroborate_rl.dqn.phases._GRADIENT_PROBES_ENABLED` before
    # the sweep loop. Disabling drops ~2× wall-clock on |A|≥12.
    gradient_probes: bool = True
    # Retain the full per-step per-action Q vectors
    # (`online_q_per_action`, `target_q_per_action`) in the per-cell
    # traces.parquet. Default False — `Q_TRACE_DROPS` strips them
    # after `Q_TRACE_REDUCTIONS` has computed scalar summaries
    # (max/min/mean/std/argmax-per-step), since the full vectors
    # blow up parquet size to ~O(n_actions × total_steps × seeds)
    # floats per cell. Set `keep_q_per_action: true` in the YAML
    # for analyses that need the raw Q distribution post-hoc.
    keep_q_per_action: bool = False
    # Persist the final online + target Q-network params per cell
    # as a msgpack sidecar under
    # `<out_dir>/<cfg.name>/q_checkpoints/cell<NNN>_<seed>_final.msgpack`.
    # Default False — substrate doesn't pay the disk cost (~25 KB
    # MLP / ~80 KB CNN per cell) unless a post-hoc analysis needs
    # to re-evaluate Q at arbitrary observations after training
    # ends. Independent of `keep_q_checkpoint_per_burst` — both
    # flags can co-exist (final + 50 per-burst snapshots) or stand
    # alone (final-only is the cheapest option).
    keep_q_checkpoint_final: bool = False
    # Persist a Q-network snapshot at the end of EVERY eval burst
    # as msgpack sidecars under
    # `<out_dir>/<cfg.name>/q_checkpoints/cell<NNN>_<seed>_burst<BB>.msgpack`.
    # Default False — total disk per sweep is
    # `n_super_steps × n_cells × param_bytes` (~240 MB for a 50-
    # burst × 60-cell CNN sweep, ~75 MB for the MLP equivalent),
    # manageable but only worth it for analyses that track the Q
    # surface's evolution across training.
    keep_q_checkpoint_per_burst: bool = False
    # Whether to merge per-intervention parquets into top-level
    # `<out_dir>/{runs,traces}.parquet`. Default True for backwards
    # compat. When False, per-intervention sub-corpora remain as
    # the canonical local artifacts (matching cloud's per-corpus
    # archive shape) and no merged top-level file is produced.
    # YAML field `merge_top_level: false` opts out.
    #
    # When False:
    # - Per-intervention sub-dirs `<out_dir>/<cfg.name>/` persist
    #   locally (instead of being rm'd post-merge).
    # - Top-level `<out_dir>/runs.parquet` and `traces.parquet`
    #   are NOT created.
    # - Downstream `--ingest <out_dir>` walks the sub-corpora.
    # - Saves up to ~tens of GB of disk on trace-heavy sweeps.
    merge_top_level: bool = True
    # Pre-registration commitments. Each entry names a bridge by
    # fully-qualified import path + the author's predicted
    # direction + the author's predicted verdict. At sweep launch
    # `dispatch_sweep` resolves each bridge, hashes its source,
    # and writes `<out_dir>/pre_registration.json` BEFORE any
    # cell runs. Empty tuple = no pre-registration (existing
    # sweeps unaffected; manifest is opt-in).
    #
    # YAML form:
    #   pre_registered_bridges:
    #     - bridge: pkg.mod.fn_name
    #       predicted_direction: a_lt_b
    #       predicted_verdict: held
    pre_registered_bridges: tuple[BridgeCommitmentInput, ...] = ()
    # Path template for "init from saved Q-checkpoint" interventions.
    # When set, every cell loads one msgpack-serialised `QCheckpoint`
    # per seed, stacks the `online_params` pytrees along a leading
    # seed axis, and injects the batched pytree as `init_override`
    # (via `init_override_batched` on the grid point) into the
    # vmap'd `dqn` call. The dqn claim's `init_state` replaces the
    # freshly-initialised online params with the loaded ones.
    #
    # `{seed}` is the placeholder substituted per cell. Paths
    # resolve relative to the sweep's CWD when not absolute. Loads
    # are eager (per cell, before vmap): a missing file raises
    # FileNotFoundError before any compute. None = no init override
    # (existing sweeps unaffected; freshly-init params as usual).
    #
    # Sweep-wide by design: when set, every arm in the sweep
    # consumes the same per-seed checkpoint family. Three-arm
    # studies that need "two arms init-from-ckpt + one arm
    # from-scratch" express the from-scratch baseline as a
    # SEPARATE sweep (e.g., the canonical sweep that produced
    # the checkpoints is itself the from-scratch baseline).
    #
    # YAML form:
    #   init_q_checkpoint_path_template: experiments/data/.../q_checkpoints/canonical_n_eps20/cell000_{seed}_burst25.msgpack
    init_q_checkpoint_path_template: str | None = None
    # When True AND `init_q_checkpoint_path_template` is set, the
    # loaded `InitOverride` populates BOTH `online_params` AND
    # `target_params` from the same msgpack file (the QCheckpoint
    # record carries both). The resumed cell starts with the source-
    # trajectory's actual (online, target) pair — preserving the
    # τ-step staleness DDQN's bias-reduction premise depends on.
    #
    # Default False = legacy behaviour (target mirrors online, same
    # as the from-scratch path; matches the running asterix_g0999
    # _init_v_burst25_continue_ddqn sweep). Set True for experiments
    # that probe the steady-state operator's effect on a paired
    # (online, target) attractor — see CHECKPOINT_RESUME_DESIGN.md
    # §7 step 5 for the experiment-design implication.
    init_q_checkpoint_load_target: bool = False
    # Bundle-format analog of `init_q_checkpoint_path_template`.
    # When set, the dispatch loader reads ONE per-cell bundle
    # (`q_checkpoints/cell{NNN}.msgpack`) and slices out the
    # requested seeds — the new write path (post-bundle migration)
    # emits this layout instead of the per-(seed, burst) sidecars.
    #
    # `init_q_checkpoint_bundle_path` is a direct path to the
    # bundle file. `init_q_checkpoint_bundle_burst` is either an
    # int (0-indexed burst → `role='per_burst'`) or the literal
    # string `"final"` (→ `role='final'`).
    #
    # Set EITHER `init_q_checkpoint_path_template` (legacy per-file)
    # OR these two fields, NOT both — the sweep validator rejects
    # the conflict at construction. `init_q_checkpoint_load_target`
    # applies symmetrically to both.
    init_q_checkpoint_bundle_path: str | None = None
    init_q_checkpoint_bundle_burst: int | str | None = None

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
    name = require_sweep_str(node, 'name')
    out_dir = Path(require_sweep_str(node, 'out_dir'))
    envs = _build_envs(node)
    env_binding = _require_env_binding(node)
    archive_remote = build_archive_remote(node)
    defaults = _build_defaults(node)
    gradient_probes = _build_gradient_probes(node)
    keep_q_per_action = _build_keep_q_per_action(node)
    keep_q_checkpoint_final = _build_keep_q_checkpoint_final(node)
    keep_q_checkpoint_per_burst = _build_keep_q_checkpoint_per_burst(node)
    merge_top_level = build_merge_top_level(node)
    pre_registered_bridges = build_pre_registered_bridges(node)
    init_q_checkpoint_path_template = (
        _build_init_q_checkpoint_path_template(node)
    )
    init_q_checkpoint_load_target = (
        _build_init_q_checkpoint_load_target(node)
    )
    init_q_checkpoint_bundle_path, init_q_checkpoint_bundle_burst = (
        _build_init_q_checkpoint_bundle(node)
    )
    if (
        init_q_checkpoint_path_template is not None
        and init_q_checkpoint_bundle_path is not None
    ):
        raise ValueError(
            'sweep config: set EITHER init_q_checkpoint_path_template '
            '(legacy per-file resume) OR init_q_checkpoint_bundle_path '
            '(bundle-format resume) — not both. The two formats are '
            "mutually exclusive at the loader's dispatch level.",
        )
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
        gradient_probes=gradient_probes,
        keep_q_per_action=keep_q_per_action,
        keep_q_checkpoint_final=keep_q_checkpoint_final,
        keep_q_checkpoint_per_burst=keep_q_checkpoint_per_burst,
        merge_top_level=merge_top_level,
        pre_registered_bridges=pre_registered_bridges,
        init_q_checkpoint_path_template=init_q_checkpoint_path_template,
        init_q_checkpoint_load_target=init_q_checkpoint_load_target,
        init_q_checkpoint_bundle_path=init_q_checkpoint_bundle_path,
        init_q_checkpoint_bundle_burst=init_q_checkpoint_bundle_burst,
    )


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


def _build_gradient_probes(node: Mapping[str, object]) -> bool:
    v = node.get('gradient_probes', True)
    # `isinstance(v, bool)` accepts True/False; `int` would let 0/1
    # through (Python bool subclasses int) — separately bool-check
    # to keep the schema strict.
    if not isinstance(v, bool):
        raise TypeError(
            f'sweep.gradient_probes must be bool; got '
            f'{type(v).__name__}',
        )
    return v


def _build_keep_q_per_action(node: Mapping[str, object]) -> bool:
    v = node.get('keep_q_per_action', False)
    if not isinstance(v, bool):
        raise TypeError(
            f'sweep.keep_q_per_action must be bool; got '
            f'{type(v).__name__}',
        )
    return v


def _build_keep_q_checkpoint_final(node: Mapping[str, object]) -> bool:
    v = node.get('keep_q_checkpoint_final', False)
    if not isinstance(v, bool):
        raise TypeError(
            f'sweep.keep_q_checkpoint_final must be bool; got '
            f'{type(v).__name__}',
        )
    return v


def _build_keep_q_checkpoint_per_burst(node: Mapping[str, object]) -> bool:
    v = node.get('keep_q_checkpoint_per_burst', False)
    if not isinstance(v, bool):
        raise TypeError(
            f'sweep.keep_q_checkpoint_per_burst must be bool; got '
            f'{type(v).__name__}',
        )
    return v


def _build_init_q_checkpoint_path_template(
    node: Mapping[str, object],
) -> str | None:
    v = node.get('init_q_checkpoint_path_template')
    if v is None:
        return None
    if not isinstance(v, str):
        raise TypeError(
            f'sweep.init_q_checkpoint_path_template must be str or '
            f'absent; got {type(v).__name__}',
        )
    if '{seed}' not in v:
        raise ValueError(
            f'sweep.init_q_checkpoint_path_template must contain '
            f"'{{seed}}' placeholder; got {v!r}",
        )
    return v


def _build_init_q_checkpoint_bundle(
    node: Mapping[str, object],
) -> tuple[str | None, int | str | None]:
    """Parse `init_q_checkpoint_bundle_path` + `_bundle_burst` pair.

    Returns `(None, None)` when neither is set (the default / no-op).
    Requires BOTH when EITHER is set — a bundle path without a burst
    spec is ambiguous (the bundle holds 50+1 snapshots per seed); a
    burst spec without a path has nothing to resolve against."""
    path_raw = node.get('init_q_checkpoint_bundle_path')
    burst_raw = node.get('init_q_checkpoint_bundle_burst')
    if path_raw is None and burst_raw is None:
        return None, None
    if path_raw is None or burst_raw is None:
        raise ValueError(
            'sweep.init_q_checkpoint_bundle_path and '
            'init_q_checkpoint_bundle_burst must be set together '
            '(or both absent). The bundle path identifies WHICH '
            "bundle to load; the burst spec identifies WHICH snapshot "
            'inside it. Got '
            f'path={path_raw!r}, burst={burst_raw!r}.',
        )
    if not isinstance(path_raw, str):
        raise TypeError(
            f'sweep.init_q_checkpoint_bundle_path must be str; got '
            f'{type(path_raw).__name__}',
        )
    # Burst spec: int (0-indexed per_burst) or the literal "final".
    # Reject booleans up-front — `isinstance(True, int) is True` in
    # Python so a YAML `true` would otherwise sneak through.
    if isinstance(burst_raw, bool) or not isinstance(burst_raw, (int, str)):
        raise TypeError(
            f'sweep.init_q_checkpoint_bundle_burst must be int or '
            f'"final"; got {type(burst_raw).__name__}',
        )
    if isinstance(burst_raw, str) and burst_raw != 'final':
        raise ValueError(
            f'sweep.init_q_checkpoint_bundle_burst string value must '
            f'be "final"; got {burst_raw!r}',
        )
    if isinstance(burst_raw, int) and burst_raw < 0:
        raise ValueError(
            f'sweep.init_q_checkpoint_bundle_burst int must be >= 0; '
            f'got {burst_raw}',
        )
    return path_raw, burst_raw


def _build_init_q_checkpoint_load_target(
    node: Mapping[str, object],
) -> bool:
    v = node.get('init_q_checkpoint_load_target', False)
    if not isinstance(v, bool):
        raise TypeError(
            f'sweep.init_q_checkpoint_load_target must be bool; '
            f'got {type(v).__name__}',
        )
    return v


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


# Single source of truth for the DQN substrate's registered claim
# namespace — the slot-claim modules PLUS the two root programs
# (`dqn`, `paired_dqn`). Root programs are registered so a sweep's
# `program:` field resolves through the SAME `reg.fn(name)` path as
# every other claim token (`add_module` keys each `FnClaim` by its
# own `.name`; re-registering the imported slot claims these program
# modules pull in is a no-op on the memoised instances). Adding a
# new root program = add its module here; ZERO edits to dispatch or
# config validation. Tests import this so the registry surface is
# defined once.
DQN_REGISTRY_MODULES: tuple[str, ...] = (
    'corroborate_rl.dqn.claims.bootstrap',
    'corroborate_rl.dqn.claims.action_select',
    'corroborate_rl.dqn.claims.replay',
    'corroborate_rl.dqn.claims.q_network',
    'corroborate_rl.dqn.claims.optimizer',
    'corroborate_rl.dqn.claims.target_sync',
    'corroborate_rl.dqn.claims.loss',
    # Root programs — resolvable as `program:` values.
    'corroborate_rl.dqn.dqn',
    'corroborate_rl.dqn.dqn_paired',
)


def default_dqn_registry() -> Registry:
    """Pre-populated Registry covering the DQN substrate's claim
    namespace. `add_modules` auto-discovers `@claim` free
    functions and frozen-dataclass config bundles (`Replay`,
    `MLP`, `CNN`); authors of one-off sweeps rarely need to
    extend this. Root programs (`dqn`, `paired_dqn`) are included
    so `cfg.program` resolves via `reg.fn(...)`."""
    reg = Registry()
    reg.add_modules(DQN_REGISTRY_MODULES)
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


def expand_sweep(
    sweep: DQNSweep, *, reg: Registry,
) -> tuple[InterventionConfig, ...]:
    """The dispatch-time intervention list, without dispatching.

    For `env_binding: shared`, returns `sweep.build_interventions()`.
    For `env_binding: per_env`, returns the per-env expansion.
    Asserts post-expansion `cfg.name` uniqueness (the same check
    `dispatch_sweep` performs before any cell runs).

    Substrate-shared helper for tests that want to validate a
    YAML would dispatch cleanly without actually running cells."""
    if sweep.env_binding == 'shared':
        configs = sweep.build_interventions(reg=reg)
    else:
        configs, _ = build_per_env(sweep, reg=reg)
    assert_unique_cfg_names(configs)
    return configs


def _resolve_measurables(
    extras: tuple[str, ...],
) -> 'tuple[Measurable[Mapping[str, object], object], ...]':
    """Resolve a tuple of measurable names to `Measurable`
    instances and concat with `dqn_default_measurables()`.

    Names are pre-validated by the YAML loader; this just looks
    them up. Defaults come first so `--ingest` consumers see a
    stable canonical column order; extras append in declaration
    order. Duplicates between defaults and extras dedupe by
    identity (the framework's measurable registry returns
    singletons per name)."""
    from corroborate.measurables.measurable import (
        Measurable, get_registered,
    )
    from corroborate_rl.dqn.measurables import dqn_default_measurables
    defaults = dqn_default_measurables()
    if not extras:
        return defaults
    seen: set[Measurable[Mapping[str, object], object]] = set(defaults)
    out: list[Measurable[Mapping[str, object], object]] = list(defaults)
    for name in extras:
        m = get_registered(name)
        if m is None:
            # Loader validates names; should not reach here.
            raise KeyError(f'unknown measurable {name!r}')
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return tuple(out)


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
    from collections.abc import Callable
    from functools import partial

    from corroborate.corpus.persistence import stream_concat_parquets
    from corroborate.runner.sweep import run_intervention
    import inspect

    from corroborate_rl.dqn import phases
    from corroborate_rl.dqn.collect import _chunks
    from corroborate_rl.dqn.measurables import dqn_default_measurables
    from corroborate_rl.dqn.trace_reductions import (
        Q_TRACE_DROPS, Q_TRACE_REDUCTIONS,
    )
    from corroborate_rl.env_catalogue import (
        get as get_env_spec, wrappers_canonical_str,
    )
    from corroborate_rl.sweep import DQNRunner

    # Wire YAML `gradient_probes:` field into the module flag
    # `train_phase` reads. Must happen BEFORE any cell runs — JAX
    # jit-compiles `train_phase` on first call and the conditional
    # branch is baked into the traced graph. Mutating after start
    # is racy and only affects fresh-jit calls.
    phases._GRADIENT_PROBES_ENABLED = sweep.gradient_probes

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
    assert_unique_cfg_names(configs)

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

    # **corpus-integrity invariant CI1 + sentinel discipline**: write
    # `.in_progress` at sweep start so an `--ingest-all` walk in
    # parallel skips the half-built corpus. Removed in the
    # `try/finally` after `dispatch_sweep` returns / errors so the
    # sentinel matches actual sweep state.
    from corroborate.corpus.integrity import IN_PROGRESS_SENTINEL
    sweep.out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = sweep.out_dir / IN_PROGRESS_SENTINEL
    sentinel.touch()
    # Write the pre-registration manifest BEFORE any cell runs.
    # Manifests are immutable per spec §5 — `write_pre_registration_
    # manifest_for_sweep` raises FileExistsError on a second
    # invocation against the same out_dir. Empty
    # `pre_registered_bridges` returns None without touching disk
    # (existing sweeps unaffected).
    _ = write_pre_registration_manifest_for_sweep(sweep)
    sub_runs: list[Path] = []
    sub_traces: list[Path] = []
    sub_arm_dirs: list[Path] = []
    for cfg, env_configs in zip(configs, envs_per_h, strict=True):
        # Per-arm-config base: bake the Q-checkpoint persistence
        # flags into `cfg.base` so `dqn` sees them as Exogenous
        # kwargs at composition time. Authors can override per-
        # arm via the standard YAML mechanism, but the sweep-wide
        # flags are the canonical opt-in.
        base_overrides: dict[str, object] = {**cfg.base}
        if sweep.keep_q_checkpoint_final:
            base_overrides['keep_q_checkpoint_final'] = True
        if sweep.keep_q_checkpoint_per_burst:
            base_overrides['keep_q_checkpoint_per_burst'] = True
        # `base` IS the SCM kwargs map; each arm's interventions
        # override slot values via partial precedence in
        # `apply_interventions`. Empty-tuple arm = "use base".
        #
        # The base PROGRAM (root claim) is `cfg.program`-resolved
        # through the registry — `'dqn'`, `'paired_dqn'`, or any
        # future `@claim` root program registered in
        # `DQN_REGISTRY_MODULES`. A program is a DISTINCT claim, not
        # a slot config of `dqn`; its identity is stamped on
        # `RunRow.program` by the cell runner (NOT smuggled into
        # `arm_key`, which stays the pure intervention fingerprint).
        program_fn = reg.fn(cfg.program)
        # Resume support is capability-as-property: a program admits
        # checkpoint resume iff its signature exposes `init_override`.
        # A name check would drift from what the program accepts.
        supports_resume = (
            'init_override' in inspect.signature(program_fn.fn).parameters
        )
        if not supports_resume and (
            sweep.init_q_checkpoint_path_template is not None
            or sweep.init_q_checkpoint_bundle_path is not None
        ):
            raise ValueError(
                f'intervention {cfg.name!r} uses program={cfg.program!r}, '
                'which does not support checkpoint resume — its signature '
                'has no init_override parameter. Resume requires a program '
                'that exposes init_override (e.g. dqn).',
            )
        base: Callable[..., object] = partial(program_fn, **base_overrides)
        intervention = cfg.do_effect
        # Flat grid_points: env × chunk × wrappers. When the
        # sweep's `init_q_checkpoint_path_template` is set, each
        # grid point carries `init_override_batched`: an InitOverride
        # whose online_params (and target_params when load_target=
        # True) carry per-seed pytrees stacked along axis 0. Loaded
        # eagerly (before dispatch) so a missing checkpoint raises
        # here, not mid-vmap.
        from corroborate_rl.dqn.q_checkpoint import (
            load_batched_init_override,
        )
        from corroborate_rl.dqn.q_checkpoint_bundle import (
            extract_batched_init_override,
            load_bundle,
        )
        # Resolve bundle resume EAGERLY ONCE — load_bundle of a
        # multi-GB msgpack inside the per-(env, chunk) loop would
        # re-read + msgpack-decode the same file N times for a
        # sweep-wide resume target. Bundle role/burst dispatch
        # happens here too, so the per-chunk loop body only does
        # the seed-slice work.
        bundle_resume: tuple[QCheckpointBundle, CheckpointRole, int | None] | None
        bundle_resume = None
        if sweep.init_q_checkpoint_bundle_path is not None:
            burst_spec = sweep.init_q_checkpoint_bundle_burst
            if burst_spec == 'final':
                bundle_role: CheckpointRole = 'final'
                bundle_burst: int | None = None
            else:
                bundle_role = 'per_burst'
                # `isinstance(True, int) is True` in Python — mirror
                # the loader's explicit bool rejection (yaml_sweep
                # _build_init_q_checkpoint_bundle line ~417). A direct
                # DQNSweep(...) construction bypasses the YAML
                # validator, so we re-guard here.
                if (
                    not isinstance(burst_spec, int)
                    or isinstance(burst_spec, bool)
                ):
                    raise TypeError(
                        'sweep.init_q_checkpoint_bundle_burst '
                        'must be int or "final" at dispatch; '
                        f'got {type(burst_spec).__name__}',
                    )
                bundle_burst = burst_spec
            bundle_resume = (
                load_bundle(Path(sweep.init_q_checkpoint_bundle_path)),
                bundle_role,
                bundle_burst,
            )
        grid_points: list[Mapping[str, object]] = []
        for ec in env_configs:
            for chunk in _chunks(ec):
                gp: dict[str, object] = {
                    'env_name': ec.env_name,
                    'seeds': chunk,
                    'wrappers': ec.wrappers,
                }
                if sweep.init_q_checkpoint_path_template is not None:
                    gp['init_override_batched'] = (
                        load_batched_init_override(
                            sweep.init_q_checkpoint_path_template,
                            chunk,
                            load_target=(
                                sweep.init_q_checkpoint_load_target
                            ),
                        )
                    )
                elif bundle_resume is not None:
                    bundle_obj, br_role, br_burst = bundle_resume
                    gp['init_override_batched'] = (
                        extract_batched_init_override(
                            bundle_obj,
                            chunk,
                            role=br_role,
                            burst=br_burst,
                            load_target=(
                                sweep.init_q_checkpoint_load_target
                            ),
                        )
                    )
                grid_points.append(gp)
        h_out_dir = sweep.out_dir / cfg.name
        # Re-arm the runner per arm-config: each arm's checkpoint
        # files live under its own `<h_out_dir>/q_checkpoints/`,
        # and the cell_idx counter restarts at 0 to mirror the
        # framework's per-`run_intervention` cell numbering.
        q_ckpt_enabled = (
            sweep.keep_q_checkpoint_final
            or sweep.keep_q_checkpoint_per_burst
        )
        runner.reset_for_intervention(
            q_checkpoint_dir=(
                h_out_dir / 'q_checkpoints' if q_ckpt_enabled else None
            ),
        )
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
        # Substrate defaults + YAML-requested extras. Extras are
        # validated by the loader (`config_loader._build_required_
        # measurables`) so by the time we reach here every name
        # resolves; duplicates are de-duped by identity since
        # `Measurable` instances are registry-cached singletons.
        measurables = _resolve_measurables(cfg.required_measurables)
        rp, tp = run_intervention(
            intervention,
            base=base,
            measurables=measurables,
            grid_points=grid_points,
            runner=runner,
            out_dir=h_out_dir,
            archive_remote=h_archive_remote,
            arm_tag=_arm_tag,
            trace_reductions=Q_TRACE_REDUCTIONS,
            trace_drops=() if sweep.keep_q_per_action else Q_TRACE_DROPS,
        )
        sub_runs.append(rp)
        sub_traces.append(tp)
        sub_arm_dirs.append(h_out_dir)

    final_runs = sweep.out_dir / 'runs.parquet'
    final_traces = sweep.out_dir / 'traces.parquet'

    # Skip top-level merge when YAML config opts out. Per-intervention
    # sub-corpora persist locally as canonical artifacts, matching
    # the cloud's per-corpus shape; downstream `--ingest <out_dir>`
    # transparently walks the sub-corpora via the `.sub_corpora_only`
    # sentinel below. Saves up to ~tens of GB of disk on trace-heavy
    # sweeps where the merged top-level is only used for one-shot
    # analysis before hypothesis-cache ingest evicts it anyway.
    if not sweep.merge_top_level:
        # Sentinel removed; sub-dirs intact.
        if sentinel.exists():
            try:
                sentinel.unlink()
            except OSError:
                pass
        # Mark this dir as "intentionally a container of
        # sub-corpora, no top-level merged parquet" — CI1 skips it,
        # `--ingest <out_dir>` transparently expands to sub-corpora.
        from corroborate.corpus.integrity import SUB_CORPORA_ONLY_SENTINEL
        (sweep.out_dir / SUB_CORPORA_ONLY_SENTINEL).touch()
        # Return parent out_dir for both — downstream consumers
        # (e.g., `run_sweep.py` printing the path) treat this as
        # "walk this directory for sub-corpora".
        return sweep.out_dir, sweep.out_dir

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
    # which created corpus-integrity invariant CI1 nested-corpus
    # violations on every subsequent `--ingest-all` walk. Auto-
    # clean now: each per-arm `<out_dir>/<arm>/` directory
    # (containing the unconcatenated runs/traces used as merge
    # inputs) gets removed once the parent merge is durable.
    #
    # **q_checkpoints/ preservation**: when the sweep opts into
    # keep_q_checkpoint_{final,per_burst}, the per-cell runners
    # write msgpack files to `<arm_dir>/q_checkpoints/`. These
    # are NOT in the merged parquets and are the explicit
    # data product the user requested — must be lifted to a
    # preserved location before rmtree wipes the arm dir.
    # Destination: `<out_dir>/q_checkpoints/<arm_name>/` keeps
    # the per-intervention namespace so multi-arm sweeps don't
    # collide.
    import shutil
    for arm_dir in sub_arm_dirs:
        if arm_dir.exists() and arm_dir.is_dir():
            q_ckpt_src = arm_dir / 'q_checkpoints'
            if q_ckpt_src.exists() and q_ckpt_src.is_dir():
                q_ckpt_dst = sweep.out_dir / 'q_checkpoints' / arm_dir.name
                q_ckpt_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(q_ckpt_src), str(q_ckpt_dst))
            shutil.rmtree(arm_dir)

    # **Top-level archive**: the merged top-level parquets are the
    # canonical artifact, but the per-arm cleanup above wiped the
    # local sub-corpus dirs that held the per-arm `_remote.json`.
    # Without a top-level archive, the sweep_dir has data files but
    # no manifest — `corroborate purge` refuses to delete, forcing
    # callers into the cloud-fallback path. Archive the merged
    # parquets directly so the sweep has its own self-contained
    # local + cloud manifest.
    #
    # File selection uses cloud._default_files (None) which picks
    # up: top-level *.parquet + pre_registration.json (if any) +
    # the entire SIDECAR_DIRS tree (currently q_checkpoints/,
    # recursed for the nested-by-arm layout). This makes the
    # top-level manifest self-contained: subsequent
    # `corroborate purge <sweep_dir>` deletes BOTH the merged
    # parquets AND the Q-checkpoint msgpacks the sweep produced.
    #
    # Best-effort: failure is warned (cloud might be transiently
    # down) but doesn't crash the sweep. Sub-corpus archives at
    # `<remote>/<arm>/` are intact and provide a recovery path.
    if sweep.archive_remote is not None:
        from corroborate.corpus.cloud import archive as _cloud_archive
        # validate=False — the manifest includes msgpack sidecars
        # which are below the CI5 1 KiB floor and have no PAR1
        # footer; CI5 check is parquet-shaped only.
        try:
            _ = _cloud_archive(
                sweep.out_dir, sweep.archive_remote, validate=False,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            import sys
            sys.stderr.write(
                f'run_sweep: WARNING — top-level archive failed: '
                f'{exc}\n'
                f'  Sub-corpora at {sweep.archive_remote}/<arm>/ '
                f'are intact; use\n'
                f'  `corroborate purge --remote-prefix <prefix>` '
                f'for cloud-fallback purge.\n',
            )

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
    'expand_sweep',
    'load_sweep',
    'write_pre_registration_manifest_for_sweep',
]
