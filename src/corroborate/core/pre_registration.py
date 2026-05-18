"""Pre-registration manifest — sweep-launch commitment to bridges.

Authors who pre-register a sweep declare, BEFORE any cell runs,
which bridges they intend to evaluate against the corpus and what
verdict they predict. The framework writes a `pre_registration.json`
sidecar at sweep launch capturing:

- the git HEAD at launch (so post-launch refactors can be detected
  against a known anchor),
- a canonical hash of the `DQNSweep` config (so a re-run from the
  same YAML matches),
- per-bridge entries with the bridge's fully-qualified import path,
  a hash of the resolved bridge's source, the predicted direction,
  and the predicted verdict.

`corroborate audit pre-registration <corpus>` later resolves each
committed bridge, re-computes the source hash, runs the bridge
against the corpus, and reports drift (source-hash mismatch or
empirical-vs-predicted verdict mismatch).

**Scope (honest disclosure).** This module catches post-launch
rewrites and verdict drift. It does NOT catch pilot-corpus HARKing
(running a pilot, observing results, editing the bridge, then
relaunching with a 'fresh' git hash) or git-history rewriting.
Those failure modes require the priority-1 `--pre-data-repin` lint
and external anchors (priority 5) — see
`docs/FALSIFIABILITY_AND_PRE_REGISTRATION.md` §1.3 / §6.

`BridgeCommitment` carries the materialised commitment that lives
on disk; `PreRegistrationManifest` is the typed sidecar shape.
Both are immutable, JSON-round-trippable, and pyright-strict-
clean."""
from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TypeIs

from corroborate._internals.json import loads as _json_loads
from corroborate._internals.narrow import (
    is_list_of_object,
    is_mapping_str_object,
    optional_direction,
    require_str,
    require_verdict,
)
from corroborate.bridge.bridge import Bridge
from corroborate.bridge.verdict import Verdict
from corroborate.core.hypothesis import PredictedDirection


MANIFEST_NAME: Final[str] = 'pre_registration.json'
SCHEMA_VERSION: Final[int] = 1


# ============ Dataclasses ============


@dataclass(frozen=True, slots=True)
class BridgeCommitment:
    """Sweep-launch commitment to a single bridge's prediction.

    `bridge_name`: fully-qualified import path
    (`pkg.module.fn_name`). The audit resolves the bridge by this
    path and refuses to fall back to function-name-only search —
    if a bridge moves, the audit fails loud rather than silently
    binding to a renamed sibling.

    `source_hash`: sha256 of the resolved bridge's source, computed
    via `compute_bridge_source_hash`. Canonicalises whitespace /
    comments / formatting (AST-of-source) and includes the bridge's
    structural metadata (predicted_direction, source/target,
    direction/tier, scope) so semantic edits flip the hash while
    cosmetic edits do not.

    `predicted_direction` / `predicted_verdict`: what the author
    committed to BEFORE seeing the corpus. The audit compares
    `predicted_verdict` against the verdict the bridge produces
    when run against the corpus."""
    bridge_name: str
    source_hash: str
    predicted_direction: PredictedDirection
    predicted_verdict: Verdict

    def as_dict(self) -> Mapping[str, object]:
        return {
            'bridge_name': self.bridge_name,
            'source_hash': self.source_hash,
            'predicted_direction': self.predicted_direction,
            'predicted_verdict': self.predicted_verdict.value,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> 'BridgeCommitment':
        pd = optional_direction(d, 'predicted_direction')
        if pd is None:
            raise ValueError(
                "BridgeCommitment.from_dict: 'predicted_direction' "
                'is required (got null)',
            )
        return cls(
            bridge_name=require_str(d, 'bridge_name'),
            source_hash=require_str(d, 'source_hash'),
            predicted_direction=pd,
            predicted_verdict=require_verdict(d, 'predicted_verdict'),
        )


@dataclass(frozen=True, slots=True)
class PreRegistrationManifest:
    """Sweep-launch manifest. Mirrored to remote alongside cells.

    Immutable once written; no append after sweep starts (see
    `write_manifest`'s `FileExistsError` behaviour). The on-disk
    JSON shape carries an explicit `schema_version` field so a
    future migration can bump the schema without silent breakage.

    `sweep_launched_at` is UTC ISO8601; `git_commit_hash` is the
    full 40-char HEAD SHA at launch; `sweep_config_hash` is the
    sha256 of the canonicalised `DQNSweep` dict (so a sweep
    re-run from the same YAML matches at the byte level)."""
    sweep_launched_at: datetime
    git_commit_hash: str
    sweep_config_hash: str
    bridge_commitments: tuple[BridgeCommitment, ...]

    def as_dict(self) -> Mapping[str, object]:
        """JSON-serialisable form. Stable key order — same order
        as the dataclass declaration."""
        return {
            'schema_version': SCHEMA_VERSION,
            'sweep_launched_at': self.sweep_launched_at.isoformat(),
            'git_commit_hash': self.git_commit_hash,
            'sweep_config_hash': self.sweep_config_hash,
            'bridge_commitments': [
                dict(c.as_dict()) for c in self.bridge_commitments
            ],
        }

    @classmethod
    def from_dict(
        cls, d: Mapping[str, object],
    ) -> 'PreRegistrationManifest':
        # `schema_version` is read but not currently routed through
        # version-conditional parsing — only one version exists. A
        # future schema bump would route here.
        sv_raw = d.get('schema_version')
        if not isinstance(sv_raw, int) or sv_raw != SCHEMA_VERSION:
            raise ValueError(
                f'PreRegistrationManifest.from_dict: unsupported '
                f'schema_version={sv_raw!r}; expected {SCHEMA_VERSION}',
            )
        commitments_raw = d.get('bridge_commitments')
        if not is_list_of_object(commitments_raw):
            raise TypeError(
                "'bridge_commitments' must be a list of objects; "
                f'got {type(commitments_raw).__name__}',
            )
        commitments: list[BridgeCommitment] = []
        for entry in commitments_raw:
            if not is_mapping_str_object(entry):
                raise TypeError(
                    "'bridge_commitments' entry must be a mapping; "
                    f'got {type(entry).__name__}',
                )
            commitments.append(BridgeCommitment.from_dict(entry))
        launched_raw = require_str(d, 'sweep_launched_at')
        return cls(
            sweep_launched_at=datetime.fromisoformat(launched_raw),
            git_commit_hash=require_str(d, 'git_commit_hash'),
            sweep_config_hash=require_str(d, 'sweep_config_hash'),
            bridge_commitments=tuple(commitments),
        )


# ============ Source-hash algorithm (PINNED) ============


def _is_bridge(v: object) -> TypeIs[Bridge]:
    return isinstance(v, Bridge)


def compute_bridge_source_hash(bridge: Bridge) -> str:
    """sha256 of (AST of holds_when source) + (decorator kwargs).

    AST-of-source (via `ast.dump(..., include_attributes=False)`)
    canonicalises whitespace, comments, blank lines, and line
    numbers — a black / ruff reformat doesn't bust the hash. A
    semantic change (e.g. `harm_floor=0.3` → `0.5`) shows up as a
    different AST literal so the hash flips.

    Decorator kwargs (`predicted_direction`, `source` / `target`
    names, `direction` / `tier`, `scope`) are canonicalised via
    `json.dumps(..., sort_keys=True)` and concatenated with the
    AST-dump payload. A change to the polars scope expression
    shows up via `str(bridge.scope)`.

    **Polars-version coupling.** `str(pl.Expr)` is not formally
    stable across polars versions. The framework's pyproject pins
    polars; consumers who upgrade polars MUST re-write their
    manifest (the audit's source-hash check will otherwise drift
    on a no-op upgrade).
    """
    if bridge.holds_when is None:
        raise ValueError(
            f'compute_bridge_source_hash: bridge {bridge.name!r} '
            f'has no holds_when body; bridges constructed via '
            f'`@claim_bridge` always carry one. Refusing to hash '
            f'a body-less Bridge.',
        )
    src = inspect.getsource(bridge.holds_when)
    parsed = ast.parse(src)
    # Strip docstrings from every FunctionDef / ClassDef / Module
    # body before dumping — docstrings are documentation, not
    # behaviour; the spec §3 contract says cosmetic edits
    # (whitespace, comments, docstrings) must not bust the hash
    # while semantic edits (literal values, expressions, control
    # flow) must.
    _strip_docstrings(parsed)
    ast_repr = ast.dump(
        parsed, annotate_fields=True, include_attributes=False,
    )
    # Endpoint serialisation: strings pass through; Measurables
    # expose `.name`; DoEffect surfaces a canonical node-key string.
    source_repr = _endpoint_serialise(bridge.source)
    target_repr = _endpoint_serialise(bridge.target)
    pd = bridge.predicted_direction
    decorator_kwargs: dict[str, str] = {
        'predicted_direction': pd if pd is not None else 'null',
        'source': source_repr,
        'target': target_repr,
        'direction': bridge.direction.value,
        # Tier is an IntEnum — `.value` returns int. Use `.name`
        # for a stable string ('INVARIANT' / 'ASSOCIATIONAL' /
        # 'INTERVENTIONAL') so the json.dumps payload's dict type
        # stays `dict[str, str]`.
        'tier': bridge.tier.name,
        # Polars Expr / DeferredScope / None — str() is the only
        # stable serialisation available without polars-version
        # coupling we don't already inherit.
        'scope': str(bridge.scope),
    }
    payload = ast_repr + '\n' + json.dumps(decorator_kwargs, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _strip_docstrings(tree: ast.AST) -> None:
    """Walk an AST and drop the leading docstring node from every
    `Module`, `FunctionDef`, `AsyncFunctionDef`, and `ClassDef`
    body. Mutates in place — caller passes the parsed tree
    directly, then calls `ast.dump`.

    The contract: a docstring edit is a documentation change and
    must NOT flip the source hash. A code change (literal,
    expression, control flow) does flip it. This walk is the
    canonicalisation step that makes both true."""
    for node in ast.walk(tree):
        if isinstance(node, (
            ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        )):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:]


def _endpoint_serialise(endpoint: object) -> str:
    """Stable string for a Bridge endpoint.

    Strings pass through; Measurables surface `.name`; DoEffect
    surfaces `node_key()` (the canonical `do(treatment|vs=baseline)`
    graph-render string). Routes through the framework's
    `endpoint_name` so the serialised form stays in lockstep with
    every other consumer of the BridgeEndpoint union."""
    from corroborate.bridge.bridge import endpoint_name
    from corroborate.core.intervention import DoEffect
    from corroborate.measurables import Measurable
    if isinstance(endpoint, str):
        return endpoint_name(endpoint)
    if isinstance(endpoint, Measurable):
        return endpoint_name(endpoint)
    if isinstance(endpoint, DoEffect):
        return endpoint_name(endpoint)
    # Defensive: an unexpected endpoint subclass would land here.
    # `repr()` keeps the hash stable rather than raising.
    return repr(endpoint)


# ============ Bridge resolution ============


def resolve_bridge_by_name(bridge_name: str) -> Bridge:
    """Resolve 'pkg.module.fn_name' to the Bridge object.

    Refuses to fall back to function-name-only search: if the
    bridge has moved (refactor, rename, file move), the audit
    surfaces an explicit error telling the user to update the
    manifest or restore the bridge. Silent fallback would let
    post-data refactors hide commitment violations."""
    module_path, _, fn_name = bridge_name.rpartition('.')
    if not module_path or not fn_name:
        raise ValueError(
            f'bridge_name {bridge_name!r}: expected a fully-qualified '
            f"import path like 'pkg.module.fn_name'",
        )
    module = importlib.import_module(module_path)
    # `getattr` returns `Any` per typeshed; narrow to `object` at
    # this single boundary so the `_is_bridge` TypeIs can then
    # narrow downward to `Bridge`. This is the only Any-laundering
    # point in this module (per the framework's typing-discipline
    # heuristics in CLAUDE.md).
    obj: object = getattr(module, fn_name, None)
    if obj is None:
        raise ValueError(
            f'bridge {bridge_name!r}: function {fn_name!r} not found '
            f'in module {module_path!r}',
        )
    if not _is_bridge(obj):
        raise ValueError(
            f'bridge {bridge_name!r}: object at this path is a '
            f'{type(obj).__name__}, not a Bridge — did you forget '
            f'to apply `@claim_bridge`, or did the symbol move?',
        )
    return obj


# ============ Sweep-config hash (PINNED) ============


def compute_sweep_config_hash(sweep_dict: Mapping[str, object]) -> str:
    """sha256 of the canonicalised sweep config dict.

    Caller passes a `dataclasses.asdict(sweep)` result (or
    equivalent JSON-serialisable mapping). We `json.dumps(...,
    sort_keys=True, default=str)` to canonicalise: `Path` and
    other non-JSON types serialise via `str()` rather than
    raising. The hash is reproducible across runs from the same
    YAML."""
    payload = json.dumps(sweep_dict, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def asdict_for_hash(sweep: object) -> Mapping[str, object]:
    """Adapter that calls `dataclasses.asdict` on a `DQNSweep`-
    shaped frozen dataclass. Kept here so the runner doesn't have
    to import dataclasses just to thread the hash; also so a
    future schema change (e.g. excluding `pre_registered_bridges`
    from the hash so the manifest's own list doesn't self-include)
    can land in one place."""
    # `asdict` requires a dataclass instance — the caller's
    # `DQNSweep` IS one, but pyright sees `object` and refuses.
    # The runtime invariant (caller passes a DQNSweep) justifies
    # the suppression; redesigning to type `sweep` more strictly
    # would force this module to import DQNSweep from the
    # substrate package (corroborate_rl), which would invert the
    # framework / substrate dependency direction.
    return asdict(sweep)  # pyright: ignore[reportArgumentType]


# ============ Git HEAD ============


def get_git_head_sha(repo_root: Path | None = None) -> str:
    """Read HEAD via `git rev-parse HEAD`. Subprocess; fail loudly
    if not in a git repo (or `git` not on PATH).

    `repo_root`: optional cwd override for the subprocess. Tests
    parameterise this against a tmp-dir git repo so the hash
    isn't pinned to the framework's own HEAD."""
    cwd = repo_root if repo_root is not None else Path.cwd()
    out = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return out.stdout.strip()


# ============ Disk I/O ============


def manifest_path_for(corpus_dir: Path) -> Path:
    return corpus_dir / MANIFEST_NAME


def write_manifest(
    corpus_dir: Path,
    manifest: PreRegistrationManifest,
) -> Path:
    """Write `<corpus_dir>/pre_registration.json` atomically.

    Refuses to overwrite — manifests are immutable per spec §5. If
    the file already exists, raises `FileExistsError` with a
    pointed message. Callers that legitimately need to re-commit
    (after deleting the corpus) get a clear actionable error."""
    p = manifest_path_for(corpus_dir)
    if p.exists():
        raise FileExistsError(
            f'pre_registration.json already exists at {p}; '
            f'manifests are immutable. Delete the corpus and re-run '
            f'if you mean to re-commit.',
        )
    payload = json.dumps(manifest.as_dict(), indent=2)
    tmp = p.with_suffix(p.suffix + '.tmp')
    _ = tmp.write_text(payload, encoding='utf-8')
    tmp.replace(p)
    return p


def read_manifest(corpus_dir: Path) -> PreRegistrationManifest | None:
    """Return the parsed manifest, or None if not present.

    Raises on parse error (malformed JSON, schema-version
    mismatch, missing required field). 'Not present' is a
    legitimate state — corpora swept before this feature landed
    have no manifest."""
    p = manifest_path_for(corpus_dir)
    if not p.exists():
        return None
    raw_text = p.read_text(encoding='utf-8')
    raw_json = _json_loads(raw_text)
    if not is_mapping_str_object(raw_json):
        raise TypeError(
            f'{p}: expected JSON object at top level; got '
            f'{type(raw_json).__name__}',
        )
    return PreRegistrationManifest.from_dict(raw_json)


# ============ Manifest factory at sweep launch ============


def build_commitments(
    bridge_names_and_predictions: 'tuple[BridgeCommitmentInput, ...]',
) -> tuple[BridgeCommitment, ...]:
    """Resolve each declared bridge, compute its source_hash, and
    return the materialised commitments. Caller threads the
    result into `PreRegistrationManifest(bridge_commitments=...)`.

    Raises `ValueError` if any bridge name fails to resolve — see
    `resolve_bridge_by_name`. We DO NOT swallow resolution errors
    at this layer: a typo or moved bridge at sweep launch is a
    pre-flight authoring mistake the operator should fix before
    burning sweep compute."""
    out: list[BridgeCommitment] = []
    for entry in bridge_names_and_predictions:
        bridge = resolve_bridge_by_name(entry.bridge_name)
        out.append(BridgeCommitment(
            bridge_name=entry.bridge_name,
            source_hash=compute_bridge_source_hash(bridge),
            predicted_direction=entry.predicted_direction,
            predicted_verdict=entry.predicted_verdict,
        ))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class BridgeCommitmentInput:
    """User-facing commitment spec (no source_hash — that's
    computed by `build_commitments` from the resolved bridge).

    Authored either programmatically (test fixtures) or as a YAML
    entry under `DQNSweep.pre_registered_bridges`. Distinct from
    `BridgeCommitment` (which is the materialised on-disk shape)
    because the input doesn't carry the bridge's source-hash —
    only the bridge name + author's prediction. The framework
    computes the hash at sweep launch from the resolved bridge."""
    bridge_name: str
    predicted_direction: PredictedDirection
    predicted_verdict: Verdict


def now_utc() -> datetime:
    """`datetime.now(UTC)` factored out so tests can monkey-patch
    in a deterministic clock."""
    return datetime.now(UTC)


__all__ = [
    'MANIFEST_NAME',
    'SCHEMA_VERSION',
    'BridgeCommitment',
    'BridgeCommitmentInput',
    'PreRegistrationManifest',
    'asdict_for_hash',
    'build_commitments',
    'compute_bridge_source_hash',
    'compute_sweep_config_hash',
    'get_git_head_sha',
    'manifest_path_for',
    'now_utc',
    'read_manifest',
    'resolve_bridge_by_name',
    'write_manifest',
]
