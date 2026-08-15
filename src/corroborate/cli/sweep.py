"""CLI surface for `corroborate sweep` subcommands.

Currently one subcommand: `corroborate sweep run <yaml>
--substrate <module>`. The flow:

1. Framework parser registers implementation-agnostic args
   (`config`, `--substrate`, `--dry-run`, `--profile`,
   `--skip-preflight`).
2. Before `parser.parse_args(argv)`, the framework PEEKS argv
   for `--substrate <name>`. The implementation is expected to be a
   LIGHTWEIGHT module that exports `SWEEP_CLI_EXTENSIONS:
   SweepCliExtensions` (optional) AND
   `SWEEP_ENTRY_POINTS: SweepEntryPoints` (required) — both read
   from a SINGLE `importlib.import_module(substrate)`.
3. If `SWEEP_CLI_EXTENSIONS` is present, the framework calls
   `ext.add_args(p_run)` so the implementation's own argparse options
   (e.g. `--device cpu|gpu` for the JAX-using RL implementation)
   appear in `--help` and get validated alongside framework args.
4. `parser.parse_args(argv)` produces the full Namespace.
5. Framework calls `ext.pre_import_setup(args)` to stamp env
   vars (e.g. `JAX_PLATFORMS`) BEFORE any of `SWEEP_ENTRY_POINTS`'
   lazy callables fire. Critical for JAX-using implementations —
   `import jax` latches the backend on first init.
6. Framework calls `ep.default_registry()` /
   `ep.load_sweep(...)` etc. — these are LAZY proxies on the
   implementation side; the heavy implementation module (which may pull
   JAX) is imported on first invocation, AFTER env vars are
   stamped.

The framework's implementation module path resolution single-imports
— it does NOT use a `<substrate>_cli` sibling-suffix convention.
The implementation is responsible for keeping its `--substrate`-named
module lightweight (no JAX-pulling imports at module-load time);
the production DQN implementation at `corroborate_rl.dqn_sweep` is a
deliberate lightweight wrapper around the heavy
`corroborate_rl.dqn.yaml_sweep`.

**Why a substrate-registered entry-point shape, not a framework-
level abstract `dispatch_sweep`.** The framework can't run a
implementation's cells (no per-cell-runner Protocol exists at this
level — see `IMPLEMENTATION_SPEC_yaml_sweep_lift.md` §1
non-goals). The CLI here is a thin import + delegate layer; the
real dispatch logic stays in the implementation's `dispatch_sweep`."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Final

from corroborate._internals.argparse import to_mapping
from corroborate._internals.narrow import (
    optional_str,
    require_bool,
    require_str,
)
from corroborate.runner.yaml_sweep import (
    ConfigName,
    Sweep,
    SweepCliExtensions,
    SweepEntryPoints,
)


_DEFAULT_SUBSTRATE: Final[str] = 'corroborate_rl.dqn_sweep'


# Sentinel for `getattr` missing-attribute case. Using `None`
# would collapse with an implementation that accidentally wrote
# `SWEEP_ENTRY_POINTS = None` — the sentinel separates the two
# failure modes for the error message.
_MISSING: Final[object] = object()


# Module-level cache: implementation import path → (entry_points,
# cli_extensions_or_None). Populated by `load_substrate` during
# `add_args` and reused by `dispatch` (one import per process).
# Tests clear this via the `_clear_substrate_cache` fixture.
_substrate_cache: dict[
    str, tuple[SweepEntryPoints[Sweep], SweepCliExtensions | None],
] = {}


def peek_substrate(argv: Sequence[str]) -> str:
    """Inspect argv for `--substrate <name>` or
    `--substrate=<name>`, returning the resolved implementation
    import path. Returns the LAST occurrence to match argparse's
    later-wins semantics (an earlier reviewer caught the
    first-wins mismatch — argparse takes the last duplicate, so
    the peek must too).

    Tolerant: returns the resolved default
    (`$CORROBORATE_SWEEP_SUBSTRATE` then `_DEFAULT_SUBSTRATE`)
    when argv carries no `--substrate` flag.

    Used by `add_args` BEFORE `parser.parse_args(argv)` so the
    implementation's CLI extensions can be loaded + registered onto
    the parser in time for argparse to validate substrate-
    specific args alongside framework args."""
    found: str | None = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == '--substrate' and i + 1 < len(argv):
            found = argv[i + 1]
            i += 2
            continue
        if tok.startswith('--substrate='):
            found = tok.split('=', 1)[1]
        i += 1
    if found is not None:
        return found
    return _resolve_substrate_name(None)


def _resolve_substrate_name(flag: str | None) -> str:
    """Three-level precedence: explicit `--substrate` flag wins,
    then `$CORROBORATE_SWEEP_SUBSTRATE`, then the documented
    default. Distinct from comparing `args.substrate` to the
    default string — that path silently overrode an explicit
    flag that happened to equal the default."""
    if flag is not None:
        return flag
    env_override = os.environ.get('CORROBORATE_SWEEP_SUBSTRATE')
    if env_override:
        return env_override
    return _DEFAULT_SUBSTRATE


def _import_substrate_module(module_path: str) -> ModuleType:
    """Import the implementation module, raising `SystemExit` with a
    typo-hint on ImportError. The hint surfaces the
    framework-side convention so a typo'd `--substrate` flag
    doesn't dead-end at argparse's generic 'unrecognized
    arguments' (which it would, since the implementation's `--device`
    etc. never get registered when the lightweight module fails
    to import)."""
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        # Try to import the implementation's parent package so the
        # error message can disambiguate "typo in --substrate"
        # from "substrate package not installed".
        parent_path, _, leaf = module_path.rpartition('.')
        hint = ''
        if parent_path:
            try:
                _ = importlib.import_module(parent_path)
                hint = (
                    f' (parent {parent_path!r} imports cleanly — '
                    f'check the leaf name {leaf!r})'
                )
            except ImportError:
                hint = (
                    f' (parent {parent_path!r} also missing — '
                    f'check that the substrate package is '
                    f'installed)'
                )
        raise SystemExit(
            f'corroborate sweep: could not import substrate module '
            f'{module_path!r}: {exc}{hint}',
        ) from exc


def load_substrate(
    substrate: str,
) -> tuple[SweepEntryPoints[Sweep], SweepCliExtensions | None]:
    """Import the implementation module ONCE and read both attributes.

    `SWEEP_ENTRY_POINTS` is REQUIRED; missing or wrong-typed
    raises `SystemExit` with a typed shape error. The implementation
    might keep the heavy bits behind lazy proxies in its
    `SWEEP_ENTRY_POINTS` Callables (the DQN implementation does this
    in `corroborate_rl/dqn_sweep.py`); the framework doesn't
    care HOW the implementation stays lightweight, only that this
    single import is cheap enough to run BEFORE
    `pre_import_setup`.

    `SWEEP_CLI_EXTENSIONS` is OPTIONAL. Missing → returns None;
    the framework registers only its own args. `=None`
    intentionally → also returns None (separate from missing —
    distinguished only in the wrong-type error path). Wrong-typed
    → typed shape error.

    Results are cached in the module-level `_substrate_cache`
    (one import per process); tests clear via the
    `_clear_substrate_cache` fixture in `tests/conftest.py`."""
    cached = _substrate_cache.get(substrate)
    if cached is not None:
        return cached

    module = _import_substrate_module(substrate)
    ep_raw: object = getattr(module, 'SWEEP_ENTRY_POINTS', _MISSING)
    if ep_raw is _MISSING:
        raise SystemExit(
            f'corroborate sweep: substrate {substrate!r} does '
            f'not export `SWEEP_ENTRY_POINTS`. The substrate must '
            f'expose a module-level `SweepEntryPoints` instance — '
            f'see `corroborate.runner.yaml_sweep.SweepEntryPoints`.',
        )
    if not isinstance(ep_raw, SweepEntryPoints):
        raise SystemExit(
            f'corroborate sweep: substrate {substrate!r}\'s '
            f'`SWEEP_ENTRY_POINTS` is a {type(ep_raw).__name__}, '
            f'not a `SweepEntryPoints`.',
        )

    ext_raw: object = getattr(
        module, 'SWEEP_CLI_EXTENSIONS', _MISSING,
    )
    ext: SweepCliExtensions | None
    if ext_raw is _MISSING or ext_raw is None:
        ext = None
    elif isinstance(ext_raw, SweepCliExtensions):
        ext = ext_raw
    else:
        raise SystemExit(
            f'corroborate sweep: substrate {substrate!r}\'s '
            f'`SWEEP_CLI_EXTENSIONS` is a {type(ext_raw).__name__}, '
            f'not a `SweepCliExtensions`.',
        )

    _substrate_cache[substrate] = (ep_raw, ext)
    return ep_raw, ext


def _print_dry_run(
    sweep: Sweep,
    configs: Sequence[ConfigName],
    substrate_summary: str | None,
) -> None:
    """Echo the resolved sweep summary.

    When `substrate_summary` is non-None (implementation provides
    `format_dry_run_summary`), the implementation controls the entire
    implementation-specific block — env layout AND intervention list
    AND arm counts. The framework's default intervention loop
    runs only when the implementation doesn't provide a summary
    function (typed Protocol allows None)."""
    print(
        f'sweep: {sweep.name!r}\n'
        f'  out_dir       : {sweep.out_dir}\n'
        f'  archive_remote: {sweep.archive_remote}\n'
        f'  pre_registered_bridges: '
        f'{len(sweep.pre_registered_bridges)}',
    )
    if substrate_summary is not None:
        print(substrate_summary)
    else:
        print(f'  interventions (expanded): {len(configs)}')
        for cfg in configs:
            print(f'    - {cfg.name}')
    print('dry-run: OK')


def add_args(
    parser: argparse.ArgumentParser,
    *,
    argv: Sequence[str] | None = None,
) -> None:
    """Register `sweep run` arguments onto `parser`.

    `argv`, when provided, is the post-`corroborate` argv slice
    used to peek at `--substrate <name>` so the implementation's
    `SWEEP_CLI_EXTENSIONS.add_args(p_run)` gets called BEFORE
    `parser.parse_args(argv)` — implementation-specific args then
    appear in `--help` and validate alongside framework args.
    When `argv` is None (programmatic API callers), only
    framework args register; implementation extensions are skipped."""
    sub = parser.add_subparsers(
        dest='sweep_subcmd', required=True,
        title='sweep subcommands',
    )
    p_run = sub.add_parser(
        'run',
        help='run a YAML-configured sweep through a substrate',
        description=(
            'Load a YAML sweep config + dispatch to the substrate\'s '
            '`dispatch_sweep`. Substrate is selected via '
            '`--substrate <module>` (defaults to '
            f'`{_DEFAULT_SUBSTRATE}` — the in-tree RL substrate). '
            'The substrate module exports `SWEEP_ENTRY_POINTS: '
            'SweepEntryPoints` and may optionally export '
            '`SWEEP_CLI_EXTENSIONS: SweepCliExtensions` to '
            'register substrate-specific CLI options + pre-import '
            'env setup. Substrates that need to stay lightweight '
            'at module-load (e.g. JAX-using substrates that must '
            'set `JAX_PLATFORMS` before any heavy import) keep '
            'their entry-point Callables lazy.'
        ),
    )
    _ = p_run.add_argument(
        'config',
        help='path to the YAML sweep config to run.',
    )
    # `default=None` (NOT the documented default string) so
    # `dispatch` can distinguish "user passed flag explicitly" from
    # "argparse filled in the default" — env var precedence is
    # then exact, not approximated.
    _ = p_run.add_argument(
        '--substrate', default=None,
        help='import path of the substrate module exposing '
             '`SWEEP_ENTRY_POINTS` (and optionally '
             '`SWEEP_CLI_EXTENSIONS`). Falls back to '
             f'`$CORROBORATE_SWEEP_SUBSTRATE` then to '
             f'`{_DEFAULT_SUBSTRATE}` (the in-tree RL substrate).',
    )
    _ = p_run.add_argument(
        '--dry-run', action='store_true',
        help='Validate the YAML through the substrate\'s '
             '`expand_sweep` and print the resolved configs '
             'without running cells. Requires the substrate to '
             'provide `expand_sweep` on its `SweepEntryPoints`.',
    )
    _ = p_run.add_argument(
        '--profile', dest='profile', default=None,
        help='AWS profile name for the cloud preflight when the '
             'sweep config has `archive_remote` set. Falls back to '
             '$AWS_PROFILE, then the default credential chain.',
    )
    _ = p_run.add_argument(
        '--skip-preflight', action='store_true',
        help='Skip the cloud-auth check that fires before the '
             'sweep loop. Off by default — preflight protects '
             'against wasting hours of compute then failing at '
             'the archive step.',
    )

    # Implementation extension discovery + registration. The argv peek
    # is what makes this two-phase parsing work: we resolve which
    # implementation the user picked BEFORE argparse runs so the
    # implementation's `add_args(p_run)` can append implementation-specific
    # args (which argparse then validates as part of the same
    # parse pass).
    if argv is not None:
        substrate = peek_substrate(argv)
        _, ext = load_substrate(substrate)
        if ext is not None:
            ext.add_args(p_run)


def dispatch(args: argparse.Namespace) -> int:
    """Argparse dispatch for `corroborate sweep run`. Returns
    0 on success, non-zero on preflight failure / config error.

    Calls the implementation's `pre_import_setup(args)` BEFORE any
    `SweepEntryPoints` Callable is invoked — this is when env
    vars like `JAX_PLATFORMS` get stamped, so the implementation's
    lazy proxies see the correct config when they fire."""
    m = to_mapping(args)
    sub = require_str(m, 'sweep_subcmd')
    if sub != 'run':
        raise ValueError(f'unknown sweep subcommand: {sub!r}')

    cfg_path = Path(require_str(m, 'config'))
    if not cfg_path.exists():
        sys.stderr.write(
            f'corroborate sweep: config not found: {cfg_path}\n',
        )
        return 1

    substrate = _resolve_substrate_name(optional_str(m, 'substrate'))
    dry_run = require_bool(m, 'dry_run')

    ep, ext = load_substrate(substrate)
    if ext is not None:
        # Implementation CLI extensions: pre_import_setup runs BEFORE
        # any `SWEEP_ENTRY_POINTS` Callable. For JAX-using
        # implementations this is where `JAX_PLATFORMS` gets stamped;
        # the framework knows nothing about JAX.
        ext.pre_import_setup(args)

    if dry_run:
        if ep.expand_sweep is None:
            sys.stderr.write(
                f'corroborate sweep: substrate {substrate!r} '
                f'does not provide `expand_sweep` on its '
                f'`SweepEntryPoints` — --dry-run is unavailable.\n',
            )
            return 1
        reg = ep.default_registry()
        sweep = ep.load_sweep(cfg_path, reg=reg)
        configs = ep.expand_sweep(sweep, reg=reg)
        substrate_summary = (
            ep.format_dry_run_summary(sweep, configs)
            if ep.format_dry_run_summary is not None else None
        )
        _print_dry_run(sweep, configs, substrate_summary)
        return 0

    # `default_registry` / `load_sweep` fire the implementation's
    # lazy proxies (if any) AFTER pre_import_setup; JAX (or
    # whatever the implementation's heavy deps need) picks the
    # platform up on first init.
    reg = ep.default_registry()
    sweep = ep.load_sweep(cfg_path, reg=reg)
    sys.stderr.write(
        f'corroborate sweep: loaded {sweep.name!r} → '
        f'out_dir={sweep.out_dir} '
        f'({len(sweep.pre_registered_bridges)} pre-registered '
        f'bridges)\n',
    )

    profile_arg = optional_str(m, 'profile')
    skip_preflight = require_bool(m, 'skip_preflight')
    if profile_arg is not None:
        os.environ['AWS_PROFILE'] = profile_arg

    if sweep.archive_remote is not None and not skip_preflight:
        from corroborate._internals.cloud_auth import (
            CloudAuthError, preflight,
        )
        try:
            preflight(sweep.archive_remote, profile=profile_arg)
        except CloudAuthError as e:
            sys.stderr.write(
                f'corroborate sweep: cloud preflight FAILED — '
                f'aborting before sweep loop.\n  {e}\n',
            )
            return 1

    runs_path, traces_path = ep.dispatch_sweep(sweep)
    sys.stderr.write(
        f'corroborate sweep: done → {runs_path}, {traces_path}\n',
    )
    return 0


__all__ = [
    'add_args',
    'dispatch',
    'load_substrate',
    'peek_substrate',
]
