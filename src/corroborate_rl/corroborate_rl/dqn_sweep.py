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

`set_jax_env` is also re-exported at module top-level so
`scripts/run_sweep.py` (the back-compat substrate-coupled
script) can call the same helper without re-implementing the
env-stamp logic."""
from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

from corroborate._internals.argparse import to_mapping
from corroborate._internals.narrow import require_str
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


def _pre_import_setup(args: argparse.Namespace) -> None:
    """Stamp JAX env vars before the heavy
    `corroborate_rl.dqn.yaml_sweep` module is imported. JAX
    latches the backend on first init; setting `JAX_PLATFORMS`
    afterwards has no effect.

    Reads `args.device` (registered by `_add_args`). argparse's
    `choices=['cpu', 'gpu']` validates at parse time; this is the
    type-level narrow + the actual env stamp."""
    device = _require_device(args, 'device')
    set_jax_env(device)


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


def set_jax_env(device: Device) -> None:
    """Set `JAX_PLATFORMS` + XLA flags. The `--device` CLI flag
    is treated as explicit user intent: it OVERRIDES any
    pre-existing `JAX_PLATFORMS` env var. Callers that want
    shell-env precedence set `JAX_PLATFORMS=` before invoking
    the CLI and pass no `--device`.

    `gpu` maps to `cuda` — the JAX backend installed on this
    machine. Operators wanting ROCm / TPU set `JAX_PLATFORMS=`
    explicitly AND pass no `--device`; this function then is a
    no-op for `JAX_PLATFORMS` (still sets the deterministic XLA
    flags).

    Sets:
    - `JAX_PLATFORMS=cuda` (or `cpu`) — overrides any prior value
      (treated as explicit `--device` intent).
    - `XLA_PYTHON_CLIENT_PREALLOCATE=false` (setdefault) — avoid
      JAX's default ~80% prealloc.
    - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (setdefault) — leave
      headroom.
    - `XLA_FLAGS+=--xla_gpu_deterministic_ops=true` — append if
      `XLA_FLAGS` already set, else set fresh.

    Without `--xla_gpu_deterministic_ops`, GPU thread-scheduling
    jitter introduces per-matmul ~1e-7 noise that compounds
    chaotically over 1M-step training on Q-explosion-prone
    vanilla DQN (~8σ cross-realisation drift on Asterix / SI
    canonical-verify; memory
    `findings_substrate_realization_variance`). Measured
    negligible perf overhead at MinAtar scale (CNN[16]/FC[128]
    1M-step Asterix: 271s deterministic ≈ 273s non-deterministic).

    Exposed at module top-level (not just inside
    `_pre_import_setup`) so `scripts/run_sweep.py` — the
    substrate-coupled back-compat script — can call the same
    helper without duplicating the env logic."""
    platform = 'cuda' if device == 'gpu' else device
    # `--device` is explicit user intent; override any prior env
    # var. Reviewer flagged the prior `setdefault` semantics as a
    # silent-override footgun (`--device cpu` could be ignored
    # because `JAX_PLATFORMS=cuda` was set in the shell).
    os.environ['JAX_PLATFORMS'] = platform
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
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
