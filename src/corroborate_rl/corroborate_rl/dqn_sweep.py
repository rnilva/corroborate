"""DQN substrate's lightweight entry point for `corroborate sweep run`.

Top-level under `corroborate_rl` (NOT under `corroborate_rl.dqn`)
so importing this module does NOT trigger
`corroborate_rl.dqn.__init__`'s eager
`from corroborate_rl.dqn import measurables` side-effect — which
in turn would `import jax.numpy as jnp` and latch JAX onto the
default platform before the framework's `pre_import_setup` hook
ever runs.

The module exposes two attributes the framework reads via
`--substrate corroborate_rl.dqn_sweep`:

- `SWEEP_CLI_EXTENSIONS: SweepCliExtensions` — registers
  `--device cpu|gpu` and stamps `JAX_PLATFORMS` / `XLA_FLAGS`
  env vars via `pre_import_setup` BEFORE any heavy import.

- `SWEEP_ENTRY_POINTS: SweepEntryPoints[DQNSweep]` — Protocol-
  Callable fields are LAZY: each closure imports
  `corroborate_rl.dqn.yaml_sweep` (the heavy implementation) on
  first invocation, by which time `pre_import_setup` has stamped
  the env vars JAX needs.

Why lazy proxies rather than direct references: the framework's
`_resolve_substrate(substrate)` does
`importlib.import_module(substrate)`, which is THIS module —
its module-level evaluation must not pull JAX. Direct references
like `load_sweep=load_sweep` would force a module-level
`from corroborate_rl.dqn.yaml_sweep import load_sweep`, which
loads JAX before `pre_import_setup` can stamp anything.

`set_jax_env` is exported at module top-level for callers that
need the env-stamp logic outside the CLI path (tests, ad-hoc
scripts) without re-implementing it."""
from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

from corroborate._internals.argparse import to_mapping
from corroborate._internals.narrow import require_bool, require_str
from corroborate.runner.registry import Registry
from corroborate.runner.yaml_sweep import (
    ConfigName,
    Sweep,
    SweepCliExtensions,
    SweepEntryPoints,
)


type Device = Literal['cpu', 'gpu']


_DEFAULT_DEVICE: Final[Device] = 'cpu'


# ============ CLI extensions (substrate-specific argparse args) ============


def _add_args(parser: argparse.ArgumentParser) -> None:
    """Register DQN-substrate-specific CLI options. The framework
    calls this AFTER its own args are registered on the same
    `run` subparser, so `--device` appears in
    `corroborate sweep run --help` alongside `--substrate`,
    `--dry-run`, etc."""
    _ = parser.add_argument(
        '--device', choices=['cpu', 'gpu'], default=_DEFAULT_DEVICE,
        help='JAX platform. CPU is the safe default for small envs '
             '(FourRooms-class); flip to GPU if it exists + matters. '
             'Translates to `JAX_PLATFORMS=cuda` (gpu) or `cpu` plus '
             'deterministic XLA flags — stamped before the heavy '
             '`corroborate_rl.dqn.yaml_sweep` module imports JAX.',
    )
    _ = parser.add_argument(
        '--no-deterministic', action='store_true', default=False,
        help='Skip the `--xla_gpu_deterministic_ops=true` XLA stamp. '
             'Default is deterministic ON (preserves the '
             '~1e-7-per-matmul reproducibility floor on Q-explosion-'
             'prone envs; see `set_jax_env` docstring). Flip OFF when '
             'the perf cost is significant — measured 10-30× on '
             'PacMan-jumanji (tiny-op-heavy step disables CUDA Graphs '
             'under determinism) vs negligible on MinAtar. Sweep YAML '
             'may also set `deterministic: false` at the top level; '
             'this CLI flag overrides the YAML when present. '
             'Operators who set `XLA_FLAGS=--xla_gpu_deterministic_'
             'ops=...` in the shell already get their explicit value '
             '(both this flag and the YAML field are no-ops then).',
    )


def _pre_import_setup(args: argparse.Namespace) -> None:
    """Stamp JAX env vars before the heavy
    `corroborate_rl.dqn.yaml_sweep` module is imported. JAX
    latches the backend on first init; setting `JAX_PLATFORMS`
    afterwards has no effect.

    Reads `args.device` + `args.no_deterministic` (registered by
    `_add_args`). argparse's `choices=['cpu', 'gpu']` validates
    `--device` at parse time; the narrow is the type-level
    dispatch.

    Determinism resolution order: `--no-deterministic` CLI flag
    forces OFF (always wins). Otherwise peek the YAML's
    `deterministic: true|false` top-level field via a lightweight
    plain-yaml read that does NOT trigger the substrate's lazy
    loader (which would import JAX before we've stamped the
    flags). Default ON when neither is set."""
    device = _require_device(args, 'device')
    args_map = to_mapping(args)
    deterministic = _resolve_deterministic(
        no_det_cli=require_bool(args_map, 'no_deterministic'),
        yaml_path=require_str(args_map, 'config'),
    )
    set_jax_env(device, deterministic=deterministic)


def _resolve_deterministic(
    *, no_det_cli: bool, yaml_path: str,
) -> bool:
    """Resolve the effective determinism setting from the CLI
    flag + YAML's optional `deterministic:` top-level field.

    CLI > YAML > default-True. The YAML peek uses plain
    `yaml.safe_load` — no JAX, no substrate-specific dataclass
    parsing — so it's safe to call before
    `corroborate_rl.dqn.yaml_sweep` is imported."""
    if no_det_cli:
        return False
    return _peek_yaml_bool(yaml_path, 'deterministic', default=True)


def _peek_yaml_bool(path: str, key: str, *, default: bool) -> bool:
    """Lightweight pre-import peek at a top-level bool field in a
    YAML file. Returns `default` if the file is missing, isn't a
    mapping at the top level, or the key is absent / non-bool.

    Intentionally narrow: no schema validation, no error on
    malformed YAML beyond what `yaml.safe_load` raises. Bad YAML
    surfaces later when the substrate's typed loader runs; this
    function exists ONLY to extract env-stamping hints before
    JAX is imported."""
    import yaml
    p = Path(path)
    if not p.is_file():
        return default
    with p.open() as f:
        raw: object = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        return default
    val = raw.get(key)
    if isinstance(val, bool):
        return val
    return default


def _require_device(
    args: argparse.Namespace, key: str,
) -> Device:
    """Narrow `args.<key>` to the typed `Device` literal. argparse's
    `choices=['cpu', 'gpu']` validation feeds this; the narrow is
    the type-level dispatch."""
    m = to_mapping(args)
    raw = require_str(m, key)
    if raw == 'cpu':
        return 'cpu'
    if raw == 'gpu':
        return 'gpu'
    raise ValueError(
        f'{key!r} must be \'cpu\' or \'gpu\'; got {raw!r}',
    )


def set_jax_env(device: Device, *, deterministic: bool = True) -> None:
    """Set `JAX_PLATFORMS` + XLA flags. The `--device` CLI flag
    is treated as explicit user intent: it OVERRIDES any
    pre-existing `JAX_PLATFORMS` env var. Callers that want
    shell-env precedence set `JAX_PLATFORMS=` before invoking
    the CLI and pass no `--device`.

    `gpu` maps to `cuda` — the JAX backend installed on this
    machine. Operators wanting ROCm / TPU set `JAX_PLATFORMS=`
    explicitly AND pass no `--device`; this function then is a
    no-op for `JAX_PLATFORMS` (still sets the deterministic XLA
    flags when `deterministic=True`).

    Sets:
    - `JAX_PLATFORMS=cuda` (or `cpu`) — overrides any prior value
      (treated as explicit `--device` intent).
    - `XLA_PYTHON_CLIENT_PREALLOCATE=false` (setdefault) — avoid
      JAX's default ~80% prealloc.
    - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (setdefault) — leave
      headroom.
    - `XLA_FLAGS+=--xla_gpu_deterministic_ops=true` when
      `deterministic=True` and the user hasn't already set the
      flag in `XLA_FLAGS`. When `deterministic=False`, the
      framework does NOT append the flag; XLA's defaults apply
      (CUDA Graphs / command-buffer capture enabled for the
      training loop).

    The reproducibility / perf trade-off: with
    `--xla_gpu_deterministic_ops`, GPU thread-scheduling jitter
    introduces per-matmul ~1e-7 noise that compounds chaotically
    over 1M-step training on Q-explosion-prone vanilla DQN
    (~8σ cross-realisation drift on Asterix / SI canonical-
    verify; memory `findings_substrate_realization_variance`).

    Perf overhead depends sharply on the env's per-step op
    profile:
    - **MinAtar scale** (small (10,10,4) obs, minimal env logic,
      CNN[16]/FC[128]) — negligible: 271s deterministic ≈ 273s
      non-deterministic on 1M-step Asterix.
    - **PacMan-jumanji** (31×28×5 obs, 4 scatter ops in
      `obs_extract`, dynamic_slice replay) — **10-30× slower**.
      Determinism blocks XLA from capturing the inner loop as a
      CUDA Graph, leaving 20M+ `cuLaunchKernel` calls per chunk
      (host-bound at 33K launches/sec). Profiled via nsys on
      RTX 5090 — see provenance notes in
      `experiments/configs/pacman_g0999_n20.yaml`.

    Operators on Jumanji-class envs should pass
    `--no-deterministic` (or set `XLA_FLAGS=
    --xla_gpu_deterministic_ops=false` in the shell, which this
    function respects) and budget for the reproducibility loss
    in the analysis.

    Exposed at module top-level (not just inside
    `_pre_import_setup`) so callers outside the CLI path (tests,
    ad-hoc scripts) can stamp the same env without duplicating
    the logic."""
    platform = 'cuda' if device == 'gpu' else device
    # `--device` is explicit user intent; override any prior env
    # var. Reviewer flagged the prior `setdefault` semantics as a
    # silent-override footgun (`--device cpu` could be ignored
    # because `JAX_PLATFORMS=cuda` was set in the shell).
    os.environ['JAX_PLATFORMS'] = platform
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    if not deterministic:
        # XLA's defaults apply. If the operator pre-set
        # `XLA_FLAGS`, leave it untouched — their explicit value
        # (which may include `--xla_gpu_deterministic_ops=false`
        # or other flags) wins.
        return
    if 'XLA_FLAGS' not in os.environ:
        os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
    elif '--xla_gpu_deterministic_ops' not in os.environ['XLA_FLAGS']:
        os.environ['XLA_FLAGS'] = (
            os.environ['XLA_FLAGS'].rstrip()
            + ' --xla_gpu_deterministic_ops=true'
        )


# ============ Lazy entry-point proxies (lazy => no JAX at import) ============
#
# Each closure imports `corroborate_rl.dqn.yaml_sweep` on first
# invocation. That import triggers `corroborate_rl.dqn.__init__`
# which does the eager `import measurables` (which pulls JAX).
# Because these are LAZY, the framework's `pre_import_setup` has
# already stamped `JAX_PLATFORMS` before any of them runs.


def _load_sweep_lazy(path: Path, *, reg: Registry) -> Sweep:
    """Lazy proxy: imports `corroborate_rl.dqn.yaml_sweep` on
    first call (so JAX initialises now, with whatever
    `pre_import_setup` already stamped). Returns a `DQNSweep`,
    which structurally satisfies the framework's `Sweep`
    Protocol."""
    from corroborate_rl.dqn.yaml_sweep import load_sweep
    return load_sweep(path, reg=reg)


def _dispatch_sweep_lazy(sweep: Sweep) -> tuple[Path, Path]:
    """Lazy proxy for the heavy `dispatch_sweep`. The runtime
    invariant: callers thread `Sweep`-typed values that are
    actually `DQNSweep` instances (the framework's CLI loads
    them via `_load_sweep_lazy` which produces `DQNSweep`)."""
    from corroborate_rl.dqn.yaml_sweep import DQNSweep, dispatch_sweep
    if not isinstance(sweep, DQNSweep):
        # Belt-and-braces: framework should only feed `DQNSweep`
        # produced by `_load_sweep_lazy` back through here. A
        # framework that synthesised a different `Sweep` and fed
        # it through would land here.
        raise TypeError(
            f'_dispatch_sweep_lazy: expected DQNSweep, got '
            f'{type(sweep).__name__}',
        )
    return dispatch_sweep(sweep)


def _default_registry_lazy() -> Registry:
    """Lazy proxy: pre-populated Registry covering the DQN
    substrate's claim namespace."""
    from corroborate_rl.dqn.yaml_sweep import default_dqn_registry
    return default_dqn_registry()


def _expand_sweep_lazy(
    sweep: Sweep, *, reg: Registry,
) -> tuple[ConfigName, ...]:
    """Lazy proxy: resolves intervention templates against the
    registry. Returns substrate-specific `InterventionConfig`
    instances, which structurally satisfy `ConfigName` via
    `.name: str`."""
    from corroborate_rl.dqn.yaml_sweep import DQNSweep, expand_sweep
    if not isinstance(sweep, DQNSweep):
        raise TypeError(
            f'_expand_sweep_lazy: expected DQNSweep, got '
            f'{type(sweep).__name__}',
        )
    return expand_sweep(sweep, reg=reg)


def _format_dry_run_summary_lazy(
    sweep: Sweep,
    configs: Sequence[ConfigName],
) -> str:
    """Lazy proxy: substrate-specific dry-run summary matching
    `scripts/run_sweep.py:_dry_run` output. Surfaces
    `env_binding` + envs (count + per-env n_seeds / chunk_size)
    + intervention list with arm counts + measurables.

    Imports `InterventionConfig` lazily for the arm-count narrow
    (substrate-specific attribute beyond the framework's
    `ConfigName` Protocol)."""
    from corroborate_rl.dqn.config_loader import InterventionConfig
    from corroborate_rl.dqn.yaml_sweep import DQNSweep
    if not isinstance(sweep, DQNSweep):
        raise TypeError(
            f'_format_dry_run_summary_lazy: expected DQNSweep, '
            f'got {type(sweep).__name__}',
        )
    lines = [
        f'  env_binding   : {sweep.env_binding}',
        f'  envs          : {len(sweep.envs)}',
    ]
    for ec in sweep.envs:
        lines.append(
            f'    - {ec.env_name} (n_seeds={ec.n_seeds}, '
            f'chunk={ec.chunk_size})',
        )
    lines.append(f'  interventions (expanded): {len(configs)}')
    for cfg in configs:
        # The framework only sees `ConfigName.name`; the
        # substrate's expand_sweep returns `InterventionConfig`
        # which has additional attributes. Narrow at the
        # consumer site.
        if not isinstance(cfg, InterventionConfig):
            lines.append(f'    - {cfg.name}')
            continue
        arm_keys = cfg.do_effect.arm_keys()
        extras = (
            f' +measurables={list(cfg.required_measurables)}'
            if cfg.required_measurables else ''
        )
        lines.append(
            f'    - {cfg.name} ({len(arm_keys)} arms){extras}',
        )
    return '\n'.join(lines)


# ============ Exposed registration attributes ============


SWEEP_CLI_EXTENSIONS: SweepCliExtensions = SweepCliExtensions(
    add_args=_add_args,
    pre_import_setup=_pre_import_setup,
)
"""Substrate registration for `corroborate sweep run`'s CLI
extension surface. Framework reads this via the typed-shape
discovery in `corroborate.cli.sweep`."""


SWEEP_ENTRY_POINTS: SweepEntryPoints[Sweep] = SweepEntryPoints[Sweep](
    load_sweep=_load_sweep_lazy,
    dispatch_sweep=_dispatch_sweep_lazy,
    default_registry=_default_registry_lazy,
    expand_sweep=_expand_sweep_lazy,
    format_dry_run_summary=_format_dry_run_summary_lazy,
)
"""Substrate registration for `corroborate sweep run`'s dispatch
surface. The Callable fields are LAZY — each closure imports
the heavy `corroborate_rl.dqn.yaml_sweep` on first invocation,
so this module's import does NOT pull JAX. The framework's
`pre_import_setup` runs BEFORE any of these closures fire."""


__all__ = [
    'Device',
    'SWEEP_CLI_EXTENSIONS',
    'SWEEP_ENTRY_POINTS',
    'set_jax_env',
]
