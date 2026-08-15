"""Framework-side YAML sweep machinery.

Substrate-agnostic primitives for loading YAML-configured sweeps:
the `Sweep` Protocol (read-only shape that implementation sweep
dataclasses satisfy structurally), YAML scalar parsers
(`require_sweep_str`, `build_archive_remote`, `build_merge_top_level`,
`build_pre_registered_bridges`, `require_predicted_direction`,
`require_predicted_verdict`), the post-expansion cfg-name
uniqueness check (`assert_unique_cfg_names`), and the
pre-registration manifest writer
(`write_pre_registration_manifest_for_sweep`).

Implementation sweep modules (`corroborate_rl.dqn.yaml_sweep`) compose
these for their own YAML loader + dispatch path. The substrate's
sweep dataclass adds implementation-specific fields (envs, agent HPs)
and satisfies the framework `Sweep` Protocol structurally — no
inheritance friction.

Why Protocol instead of dataclass inheritance: frozen dataclasses
with `slots=True` don't compose via inheritance cleanly (slot-
layout conflicts, immutability layering). Structural typing via
`runtime_checkable` Protocol gives the same compile-time check +
no inheritance friction. The substrate's `DQNSweep` remains a flat
frozen dataclass that *happens to* satisfy the Protocol."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from corroborate._internals.narrow import is_mapping_str_object
from corroborate.bridge.verdict import Verdict
from corroborate.core.hypothesis import PredictedDirection
from corroborate.core.pre_registration import BridgeCommitmentInput
from corroborate.runner.registry import Registry


@runtime_checkable
class Sweep(Protocol):
    """Framework-visible shape of a configured sweep.

    Implementation sweep dataclasses (e.g.,
    `corroborate_rl.dqn.yaml_sweep.DQNSweep`) satisfy this Protocol
    structurally — no inheritance. The framework's primitives
    (manifest write, archive-remote handling, top-level merge
    gating) read these fields; the implementation adds its own fields
    (envs, agent HPs) without inheritance friction.

    Read-only `@property` declarations match frozen-dataclass
    fields under pyright's structural type check; writable
    Protocol fields would NOT match immutable concrete fields
    (see CLAUDE.md typing discipline section)."""

    @property
    def name(self) -> str: ...

    @property
    def out_dir(self) -> Path: ...

    @property
    def archive_remote(self) -> str | None: ...

    @property
    def merge_top_level(self) -> bool: ...

    @property
    def pre_registered_bridges(self) -> tuple[BridgeCommitmentInput, ...]: ...


class ConfigName(Protocol):
    """Structural shape for `assert_unique_cfg_names` and the CLI
    dry-run print loop — anything with a `.name: str` attribute
    that the implementation uses as a per-config output subdirectory.

    Public so `corroborate.cli.sweep._print_dry_run` can be typed
    against it without cross-module private import (CLAUDE.md's
    typing discipline forbids the `getattr`-then-isinstance pattern
    `_Named` would otherwise force on consumers).

    Deliberately narrower than `Sweep` despite both exposing
    `.name: str` — a `Sweep` instance also satisfies `ConfigName`
    structurally, but the intent of `ConfigName` is "per-cfg
    output-dir tag", not "sweep name". Reviewers caught the
    semantic overlap; the rename makes the role explicit."""

    @property
    def name(self) -> str: ...


def require_sweep_str(node: Mapping[str, object], key: str) -> str:
    v = node.get(key)
    if not isinstance(v, str):
        raise TypeError(
            f'sweep.{key} must be a string; got '
            f'{type(v).__name__}',
        )
    return v


def build_archive_remote(node: Mapping[str, object]) -> str | None:
    v = node.get('archive_remote')
    if v is None:
        return None
    if isinstance(v, str):
        return v
    raise TypeError(
        f'sweep.archive_remote must be string|null; got '
        f'{type(v).__name__}',
    )


def build_merge_top_level(node: Mapping[str, object]) -> bool:
    v = node.get('merge_top_level', True)
    if not isinstance(v, bool):
        raise TypeError(
            f'sweep.merge_top_level must be bool; got '
            f'{type(v).__name__}',
        )
    return v


def build_pre_registered_bridges(
    node: Mapping[str, object],
) -> tuple[BridgeCommitmentInput, ...]:
    """Parse `pre_registered_bridges:` from YAML.

    Empty/absent → empty tuple (sweep is not pre-registered).
    Otherwise each entry must declare `bridge` (import path),
    `predicted_direction`, and `predicted_verdict`. Unknown
    verdict strings or directions raise loudly at load time —
    we won't burn sweep compute on a typo'd commitment."""
    raw = node.get('pre_registered_bridges')
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError(
            f'sweep.pre_registered_bridges must be a list; got '
            f'{type(raw).__name__}',
        )
    raw_typed: list[object] = list(raw)
    out: list[BridgeCommitmentInput] = []
    for entry in raw_typed:
        if not is_mapping_str_object(entry):
            raise TypeError(
                f'pre_registered_bridges entry must be a mapping; '
                f'got {type(entry).__name__}',
            )
        bridge_name = require_sweep_str(entry, 'bridge')
        out.append(BridgeCommitmentInput(
            bridge_name=bridge_name,
            predicted_direction=require_predicted_direction(entry),
            predicted_verdict=require_predicted_verdict(entry),
        ))
    return tuple(out)


def require_predicted_direction(
    entry: Mapping[str, object],
) -> PredictedDirection:
    """Narrow `predicted_direction` to the typed Literal. Match
    the four allowed strings; anything else is a typo'd
    commitment that should fail loud at YAML load."""
    v = entry.get('predicted_direction')
    if v == 'a_gt_b':
        return 'a_gt_b'
    if v == 'a_lt_b':
        return 'a_lt_b'
    if v == 'two_sided':
        return 'two_sided'
    if v == 'null':
        return 'null'
    raise ValueError(
        f"pre_registered_bridges entry: 'predicted_direction' must "
        f"be one of ('a_gt_b', 'a_lt_b', 'two_sided', 'null'); "
        f'got {v!r}',
    )


def require_predicted_verdict(entry: Mapping[str, object]) -> Verdict:
    """Narrow `predicted_verdict` to the typed `Verdict` enum.
    Matches against the same string values as parquet
    persistence (`Verdict.value`) so the YAML form is
    'held' / 'no_effect' / 'power_insufficient' / ..."""
    v = entry.get('predicted_verdict')
    for verdict in Verdict:
        if v == verdict.value:
            return verdict
    raise ValueError(
        f"pre_registered_bridges entry: 'predicted_verdict' must "
        f'be a Verdict value '
        f'({[v.value for v in Verdict]!r}); got {v!r}',
    )


def assert_unique_cfg_names(configs: Sequence[ConfigName]) -> None:
    """Raise `ValueError` if any two configs share `cfg.name`.

    `dispatch_sweep` writes each config to `<out_dir>/<cfg.name>/`;
    a shared name silently overwrites at merge time. Fires for
    `env_binding: per_env` when the template's `name` field lacks
    `{from_env: ...}` substitution and produces post-expansion
    duplicates across envs. Exposed for the cross-config lint
    (`tests/test_configs_lint.py`) so the check runs at test
    time on every YAML, not only at dispatch."""
    seen: dict[str, int] = {}
    for cfg in configs:
        seen[cfg.name] = seen.get(cfg.name, 0) + 1
    collisions = {n: c for n, c in seen.items() if c > 1}
    if collisions:
        raise ValueError(
            f'configs share output paths — {collisions!r} would '
            f'overwrite each other at '
            f'`<out_dir>/<cfg.name>/runs.parquet`. '
            f'Templating the intervention `name` with '
            f"`{{from_env: env_name}}` (env_binding='per_env') or "
            f"switching to env_binding='shared' resolves this. "
            f'Sweep aborted before any data is written.',
        )


def write_pre_registration_manifest_for_sweep(
    sweep: Sweep,
) -> Path | None:
    """Resolve each bridge in `sweep.pre_registered_bridges`,
    compute its source hash, and write the manifest to
    `<sweep.out_dir>/pre_registration.json`.

    Empty `pre_registered_bridges` → returns None without
    touching disk (manifest is opt-in; sweeps without explicit
    commitments behave identically to the pre-feature baseline).

    Manifests are immutable per spec §5: a second invocation
    against the same `out_dir` raises `FileExistsError` rather
    than silently overwriting. Callers that legitimately need to
    re-commit must delete the corpus and re-run.

    The git HEAD is read via `git rev-parse HEAD` in the
    framework's repo (`Path.cwd()`); the audit later verifies the
    SHA exists in `git log --all` and exits with the dedicated
    `EXIT_GIT_HASH_NOT_FOUND` code if missing. The
    `sweep_config_hash` is a sha256 of the canonicalised sweep
    dict — a re-run from the same YAML produces a matching hash."""
    from corroborate.core.pre_registration import (
        PreRegistrationManifest, asdict_for_hash, build_commitments,
        compute_sweep_config_hash, get_git_head_sha, now_utc,
        write_manifest,
    )
    if not sweep.pre_registered_bridges:
        return None
    commitments = build_commitments(sweep.pre_registered_bridges)
    sweep_dict = asdict_for_hash(sweep)
    cfg_hash = compute_sweep_config_hash(sweep_dict)
    manifest = PreRegistrationManifest(
        sweep_launched_at=now_utc(),
        git_commit_hash=get_git_head_sha(),
        sweep_config_hash=cfg_hash,
        bridge_commitments=commitments,
    )
    return write_manifest(sweep.out_dir, manifest)


class _LoadSweep[S: Sweep](Protocol):
    """Callable shape `(path, *, reg) -> S`. Defined as a Protocol
    (not plain `Callable[...]`) so the keyword-only `reg`
    parameter is part of the typed contract; implementations that
    accept positional `reg` would not match.

    PEP 695 generic over `S: Sweep` so the substrate's
    `load_sweep(...) -> DQNSweep` matches under contravariant
    parameter / covariant return rules — `(Path, Registry) -> S` is
    covariant in `S`, so the substrate's narrower return is
    assignable into `_LoadSweep[DQNSweep]`."""

    def __call__(self, path: Path, *, reg: Registry) -> S: ...


class _DispatchSweep[S: Sweep](Protocol):
    """Callable shape `(sweep) -> (runs_path, traces_path)`.
    Generic over `S: Sweep` because function PARAMETERS are
    contravariant — `(sweep: DQNSweep) -> ...` is NOT assignable
    to `(sweep: Sweep) -> ...`. Without the generic, the
    substrate's typed dispatch_sweep would fail pyright when
    constructed into `SweepEntryPoints`."""

    def __call__(self, sweep: S) -> tuple[Path, Path]: ...


class _DefaultRegistry(Protocol):
    """Substrate-provided pre-populated `Registry` factory.
    Called once per CLI invocation; the result is threaded through
    `load_sweep` + `expand_sweep`."""

    def __call__(self) -> Registry: ...


class _ExpandSweep[S: Sweep](Protocol):
    """Optional dry-run helper. Returns the substrate's resolved
    config tuple — element type satisfies `ConfigName` for the
    CLI's print loop. Implementation may name elements anything (DQN's
    `InterventionConfig`), so the framework only sees `ConfigName`.

    Generic over `S: Sweep` for the same contravariance reason as
    `_DispatchSweep`."""

    def __call__(
        self, sweep: S, *, reg: Registry,
    ) -> tuple[ConfigName, ...]: ...


class _FormatDryRunSummary[S: Sweep](Protocol):
    """Optional implementation-shaped dry-run summary printer.

    `cli.sweep._print_dry_run` prints framework-visible fields
    (`name`, `out_dir`, `archive_remote`, `pre_registered_bridges`
    count). When the implementation provides this callback, the CLI
    routes the implementation-specific block through it — receiving
    BOTH the sweep AND the resolved configs — so the implementation
    controls per-env summary AND the intervention list (with arm
    counts, measurable extras, etc.). The framework's default
    intervention loop runs only when this callback is `None`.

    DQN's `_dqn_dry_run_summary` uses this to match
    `scripts/run_sweep.py:_dry_run`'s output byte-for-byte —
    operators swapping the back-compat script for
    `corroborate sweep run` see no per-line drift."""

    def __call__(
        self, sweep: S, configs: Sequence[ConfigName],
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class SweepEntryPoints[S: Sweep]:
    """Substrate registration for `corroborate sweep run`.

    A implementation module exposes a module-level
    `SWEEP_ENTRY_POINTS: SweepEntryPoints[<SubstrateSweep>] =
    SweepEntryPoints(...)` pointing at its own `load_sweep`,
    `dispatch_sweep`, `default_registry`, and `expand_sweep`
    functions. The CLI imports the implementation module by
    `--substrate <module>` and reads this attribute — failure is
    a typed error pointing at the missing attribute or wrong
    shape.

    Why a dataclass + module attribute, not a Protocol on the
    module itself: Python modules aren't instances, so a Protocol
    can't structurally match them. A frozen dataclass instance
    with typed Protocol-Callable fields gives pyright something
    concrete to check against and the CLI a single `getattr` call
    to retrieve.

    PEP 695 generic over `S: Sweep` so implementation-specific sweep
    types (e.g. `DQNSweep`) flow through `dispatch_sweep` /
    `expand_sweep` / `format_dry_run_summary` without contravariance
    bugs. The CLI's `_resolve_substrate` returns
    `SweepEntryPoints[Sweep]` — fine because the framework only
    feeds `Sweep`-typed values back through those Callables.

    `expand_sweep` and `format_dry_run_summary` are optional —
    only `--dry-run` calls them. A implementation that doesn't support
    dry-run sets `expand_sweep=None`; the CLI emits a clear error
    if dry-run is requested against such a substrate. implementations
    that don't add to the framework summary set
    `format_dry_run_summary=None`."""

    load_sweep: _LoadSweep[S]
    dispatch_sweep: _DispatchSweep[S]
    default_registry: _DefaultRegistry
    expand_sweep: _ExpandSweep[S] | None = None
    format_dry_run_summary: _FormatDryRunSummary[S] | None = None


@runtime_checkable
class _AddCliArgs(Protocol):
    """Substrate hook to add implementation-specific argparse arguments
    to `corroborate sweep run`. Called by the framework BEFORE
    `parser.parse_args()` so the substrate's options appear in
    `--help` output and are validated by argparse alongside the
    framework's own args.

    `runtime_checkable` so `SweepCliExtensions.__post_init__`
    can fail-loud on a non-callable field value at construction
    time, not at first invocation."""

    def __call__(self, parser: argparse.ArgumentParser) -> None: ...


@runtime_checkable
class _PreImportSetup(Protocol):
    """Substrate hook to stamp environment variables before the
    `SweepEntryPoints` Callables fire. Called by the framework
    AFTER `parser.parse_args()` and BEFORE any
    `ep.default_registry()` / `ep.load_sweep(...)` invocation.

    Critical use case: `JAX_PLATFORMS` must be set before any
    `import jax` because JAX latches the backend on first init.
    The framework's CLI doesn't know about JAX; the substrate's
    pre-import hook does, and it runs at the right moment in the
    import sequence. implementations with heavy deps keep their
    `SWEEP_ENTRY_POINTS` Callables lazy so the actual heavy
    imports happen on first invocation — AFTER this hook ran."""

    def __call__(self, args: argparse.Namespace) -> None: ...


@dataclass(frozen=True, slots=True)
class SweepCliExtensions:
    """Substrate registration for `corroborate sweep run` CLI
    extensions — implementation-specific argparse args + pre-import
    environment setup.

    A implementation exposes a module-level
    `SWEEP_CLI_EXTENSIONS: SweepCliExtensions` on the SAME
    module the framework reads `SWEEP_ENTRY_POINTS` from. The
    framework's single `importlib.import_module(<implementation>)`
    reads both attributes; implementations that need to stay
    lightweight at import-time (e.g. JAX-using implementations that
    must set `JAX_PLATFORMS` before any heavy import) keep
    their `SWEEP_ENTRY_POINTS` Callables lazy.

    The framework's `corroborate.cli.sweep`:

    1. Calls `add_args(p_run)` to register implementation-specific
       options (e.g. `--device cpu|gpu` for the JAX-using RL
       implementation). These appear in `corroborate sweep run --help`.
    2. After `parser.parse_args(argv)` produces the Namespace,
       calls `pre_import_setup(args)` to stamp env vars (e.g.
       `JAX_PLATFORMS`, `XLA_FLAGS`).
    3. Invokes `SWEEP_ENTRY_POINTS` Callables — the substrate's
       lazy proxies now import heavy deps with correct env.

    Construction enforces a runtime shape check on both fields
    (the dataclass is constructed once per implementation, at module
    load) — non-Callable values fail loud here rather than at
    first invocation."""

    add_args: _AddCliArgs
    pre_import_setup: _PreImportSetup

    def __post_init__(self) -> None:
        """Fail-loud shape check on the typed Callable fields.

        pyright statically narrows the field types via the
        Protocol annotations, so `isinstance(self.add_args,
        _AddCliArgs)` looks redundant to it
        (`reportUnnecessaryIsInstance`). But the RUNTIME check
        is the whole point — Python doesn't validate Protocol
        membership at attribute assignment. Use `callable()`
        instead of `isinstance()` here: the `runtime_checkable`
        Protocol with one `__call__` member reduces to
        `callable(...)` plus a check that `__call__` exists,
        which `callable` already covers."""
        if not callable(self.add_args):
            raise TypeError(
                f'SweepCliExtensions.add_args must be callable; '
                f'got {type(self.add_args).__name__}',
            )
        if not callable(self.pre_import_setup):
            raise TypeError(
                f'SweepCliExtensions.pre_import_setup must be '
                f'callable; got {type(self.pre_import_setup).__name__}',
            )


__all__ = [
    'BridgeCommitmentInput',
    'ConfigName',
    'Sweep',
    'SweepCliExtensions',
    'SweepEntryPoints',
    'assert_unique_cfg_names',
    'build_archive_remote',
    'build_merge_top_level',
    'build_pre_registered_bridges',
    'require_predicted_direction',
    'require_predicted_verdict',
    'require_sweep_str',
    'write_pre_registration_manifest_for_sweep',
]
