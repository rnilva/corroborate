"""Runner — sweep dispatch + corpus runner.

Three sub-modules:

- `runner.runner` — `run` evaluates authored claims over records;
  it consumes the `Hypothesis` Protocol
  (`corroborate.core.hypothesis.Hypothesis`).
- `runner.sweep` — `Runner[R]` Protocol, `run_intervention`
  paired-sweep driver, `SweepResult`, `SweepCellResult`,
  `RowFailure`.
- `runner.registry` — substrate-facing string→handle
  `Registry` (FnClaim + frozen-dataclass auto-discovery).

`@analysis` + fixture injection live in
`corroborate.bridge.analysis` — they're the glue for
`claim_bridge`, not a runner concern.

YAML-loaded `InterventionConfig` lives substrate-side and supplies
an explicit `DoEffect` plus a `base` callable to the sweep runner.
It does not define the hypothesis evaluated over the resulting records."""
from corroborate.runner.registry import Registry
from corroborate.runner.runner import (
    SourceDrift,
    check,
    check_cache_sources,
    collect_bridges,
    evict,
    run,
)
from corroborate.runner.sweep import (
    RowFailure,
    Runner,
    SweepCellResult,
    SweepResult,
    run_intervention,
)
from corroborate.runner.yaml_sweep import (
    BridgeCommitmentInput,
    ConfigName,
    Sweep,
    SweepCliExtensions,
    SweepEntryPoints,
    assert_unique_cfg_names,
    build_archive_remote,
    build_merge_top_level,
    build_pre_registered_bridges,
    require_predicted_direction,
    require_predicted_verdict,
    require_sweep_str,
    write_pre_registration_manifest_for_sweep,
)

__all__ = [
    'BridgeCommitmentInput',
    'RowFailure',
    'ConfigName',
    'Registry',
    'Runner',
    'SourceDrift',
    'Sweep',
    'SweepCellResult',
    'SweepCliExtensions',
    'SweepEntryPoints',
    'SweepResult',
    'assert_unique_cfg_names',
    'build_archive_remote',
    'build_merge_top_level',
    'build_pre_registered_bridges',
    'check',
    'check_cache_sources',
    'collect_bridges',
    'evict',
    'require_predicted_direction',
    'require_predicted_verdict',
    'require_sweep_str',
    'run',
    'run_intervention',
    'write_pre_registration_manifest_for_sweep',
]
