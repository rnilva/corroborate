"""External study ingestion — a verifier and normaliser at the
scientific boundary, not a permissive file reader.

The user keeps the implementation that produced the experiment.
Corroborate owns the boundary at which its record becomes a
scientifically typed panel: units, contrast, pairing, scope,
provenance, and admissibility are made explicit there. Statements
the files can prove are `VERIFIED`; statements only the producer
can make stay `ATTESTED` / `UNVERIFIABLE` in the receipt; any
broken obligation fails closed with a typed
`BundleValidationError` carrying the partial receipt.

One call takes a sealed bundle directory to a `Panel` plus an
admissibility receipt::

    study = adapt_study('path/to/bundle')
    panel = study.to_panel()          # corroborate.data.Panel
    study.receipt.admissible          # True — else adapt_study raised
    study.contrast.treatment_key      # condition labels for analyses

Bundle format v1 — one directory, sealed by
`corroborate.data.seal_bundle`:

- ``manifest.json`` — content-addressed seal: per-file SHA-256 +
  size and an aggregate bundle digest.
- ``contract.json`` — the compact study description (the design
  note's StudySpec): ``contract_version`` (currently ``1``),
  ``study_id``, ``pair_by`` (the run-record
  field naming the pairing unit, integer-valued), the two-condition
  ``contrast`` (``parameter_path`` into the resolved config,
  ``baseline_arm`` / ``treatment_arm`` names, their intervention
  values, optional ``logical_arm_keys`` relabelling — defaults to
  the producer's own condition names), ``scope`` (field → value
  every seeded run must match), ``evaluation`` (``checkpoints`` ×
  ``seeds`` extent + ``outcomes`` naming the numeric fields each
  evaluation record carries), optional ``run_measurements``
  (producer-computed per-run scalars admitted as attested),
  optional ``assignment`` attestation, optional
  ``prospective_protocol`` commitment.
- ``runs.jsonl`` — one record per seeded run: ``run_id``,
  ``physical_arm``, the ``pair_by`` value, ``config_path`` to the
  resolved configuration actually used, ``complete: true``, the
  scope fields, any declared run measurements.
- ``evaluations.jsonl`` — one record per (seeded run, evaluation
  checkpoint, evaluation seed) with the declared outcome fields.
- ``provenance.json`` — producer identity + invocation record.
- the referenced resolved-config JSON files (and optional
  prospective-protocol document).

Mechanically verified: seal integrity, pair completeness (one
seeded run per condition per pairing unit), configuration
isolation (every resolved config identical outside the declared
intervention path and pairing path), exact evaluation extent, and
scope consistency. Attested, never silently upgraded: assignment
procedure, producer invocation, producer-computed measurements.

Rows are one scalar cell per seeded run: identity + condition
columns (``arm_key``, ``arm_is_baseline``), the scope fields, the
intervention value at its dotted ``parameter_path`` (the
framework's leaf-column convention), and per declared outcome a
final-checkpoint mean (``<outcome>_mean``) and a
checkpoint-normalised area under the curve (``<outcome>_auc``).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, override

from corroborate._internals.narrow import is_mapping_str_object
from corroborate.corpus.schema import MeasurementLeaf
from corroborate.data._bundle_io import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    ManifestEntry,
    bundle_digest,
    read_json,
    read_jsonl,
    safe_bundle_path,
    seal_bundle,
    sha256_file,
)
from corroborate.data.kernel import cells_to_dataframe
from corroborate.data.panel import CorpusSource, Panel

__all__ = [
    'AdaptedStudy',
    'AdapterCheck',
    'AdapterReceipt',
    'BundleValidationError',
    'CheckStatus',
    'RecordedContrast',
    'adapt_study',
    'seal_bundle',
]

ADAPTER_VERSION = '1.0.0'

_CONTRACT_VERSION = 1


class CheckStatus(StrEnum):
    """Epistemic status of one adapter assertion — the receipt's
    closed vocabulary. Keeping VERIFIED / ATTESTED / UNVERIFIABLE
    distinct is the point: a statement the files cannot prove is
    never silently upgraded to one they can."""

    VERIFIED = 'verified'
    ATTESTED = 'attested'
    UNVERIFIABLE = 'unverifiable'
    FAILED = 'failed'


@dataclass(frozen=True, slots=True)
class AdapterCheck:
    """One validation or assurance statement in an adapter
    receipt — the unit a reader audits, one obligation per line."""

    code: str
    status: CheckStatus
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            'code': self.code,
            'status': self.status.value,
            'message': self.message,
        }

    @override
    def __str__(self) -> str:
        return f'[{self.status.value}] {self.code}: {self.message}'


@dataclass(frozen=True, slots=True)
class AdapterReceipt:
    """Audit record emitted only from the files actually consumed
    — the serialisable artifact that travels with the panel so a
    later reader can see what was proven vs merely attested."""

    adapter_version: str
    study_id: str
    bundle_root: Path
    bundle_digest: str
    n_runs: int
    n_pairs: int
    checks: tuple[AdapterCheck, ...]

    @property
    def admissible(self) -> bool:
        """Whether every mechanically required check passed.
        Attested / unverifiable statements do not block admission;
        they stay visible as what they are."""
        return all(
            check.status is not CheckStatus.FAILED for check in self.checks
        )

    def as_dict(self) -> dict[str, object]:
        return {
            'adapter_version': self.adapter_version,
            'study_id': self.study_id,
            'bundle_root': str(self.bundle_root),
            'bundle_digest': self.bundle_digest,
            'n_runs': self.n_runs,
            'n_pairs': self.n_pairs,
            'admissible': self.admissible,
            'checks': [check.as_dict() for check in self.checks],
        }


class BundleValidationError(ValueError):
    """Fail-closed bundle rejection carrying the partial receipt —
    a broken bundle produces a typed audit trail, never a
    half-parsed row table."""

    def __init__(
        self,
        message: str,
        *,
        checks: tuple[AdapterCheck, ...],
    ) -> None:
        super().__init__(message)
        self.checks = checks


@dataclass(frozen=True, slots=True)
class RecordedContrast:
    """Typed identity of the external study's two-condition
    contrast. Deliberately NOT a `DoEffect`: an intervention
    executed elsewhere is *evidence*, not an executable operation
    Corroborate can re-apply — analyses take the condition labels
    and intervention values from the verified record instead of
    guessing strings, and `bundle_digest` binds the contrast to
    the exact sealed record it came from."""

    parameter_path: str
    baseline_key: str
    treatment_key: str
    baseline_value: float
    treatment_value: float
    bundle_digest: str
    assurance: CheckStatus

    @property
    def arm_keys(self) -> tuple[str, str]:
        """(baseline, treatment) — the `arm_key` values stamped on
        the adapted rows, in analysis-argument order."""
        return (self.baseline_key, self.treatment_key)


@dataclass(frozen=True, slots=True)
class AdaptedStudy:
    """The adapter's single result: validated per-seeded-run rows,
    the recorded contrast, and the receipt — bound together so the
    rows are never evaluated apart from the assurance record that
    admitted them."""

    rows: tuple[Mapping[str, MeasurementLeaf], ...]
    contrast: RecordedContrast
    receipt: AdapterReceipt

    def to_panel(
        self,
        *,
        stratify_by: tuple[str, ...] = ('arm_key',),
    ) -> Panel:
        """Hand the validated rows to the framework's canonical
        exploration/analysis surface. Built on
        `Panel.from_dataframe` — the adapted study is an ordinary
        panel whose provenance entry names the sealed bundle."""
        return Panel.from_dataframe(
            cells_to_dataframe(self.rows),
            stratify_by=stratify_by,
            sources=(
                CorpusSource(
                    corpus=self.receipt.study_id,
                    data_root=self.receipt.bundle_root,
                ),
            ),
        )


# ============ label-carrying narrowing helpers ============
# Sibling to `_internals.narrow`'s key-based accessors: these
# carry a dotted *label* (``contrast.baseline_value``,
# ``run.run_id``) so a failed check names the exact bundle field,
# including nested-path and list-element positions the key-based
# accessors can't express.


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if is_mapping_str_object(value):
        return value
    raise TypeError(f'{label} must be an object with string keys')


def _as_list(value: object, label: str) -> list[object]:
    if isinstance(value, list):
        items: list[object] = value
        return items
    raise TypeError(f'{label} must be a list')


def _as_str(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f'{label} must be a non-empty string')
    return value


def _as_int(value: object, label: str) -> int:
    # bool is a subclass of int — reject it so True/False can't
    # slip through as pairing keys or checkpoint indices.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{label} must be an integer')
    return value


def _as_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{label} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{label} must be finite')
    return result


def _as_scalar(value: object, label: str) -> MeasurementLeaf:
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'{label} must be finite')
        return value
    raise TypeError(f'{label} must be a scalar (str / int / float / bool)')


def _get_path(value: Mapping[str, object], dotted_path: str) -> object:
    current: object = value
    for part in dotted_path.split('.'):
        mapping = _as_mapping(current, f'parent of {dotted_path!r}')
        if part not in mapping:
            raise KeyError(f'configuration lacks {dotted_path!r}')
        current = mapping[part]
    return current


def _without_path(
    value: Mapping[str, object],
    dotted_path: str,
) -> dict[str, object]:
    projected = deepcopy(dict(value))
    current: dict[str, object] = projected
    parts = dotted_path.split('.')
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise TypeError(f'parent of {dotted_path!r} must be an object')
        # Runtime invariant: JSON-decoded dicts carry string keys.
        narrowed: dict[str, object] = {
            str(key): item for key, item in child.items()
        }
        current[part] = narrowed
        current = narrowed
    if parts[-1] not in current:
        raise KeyError(f'configuration lacks {dotted_path!r}')
    del current[parts[-1]]
    return projected


def _normalised_auc(
    checkpoints: tuple[int, ...],
    means: tuple[float, ...],
) -> float:
    """Trapezoid area over the checkpoint axis, normalised by its
    span — reduces to the single checkpoint mean when the study
    evaluated only once."""
    if len(checkpoints) == 1:
        return means[0]
    area = 0.0
    for index in range(len(checkpoints) - 1):
        step = float(checkpoints[index + 1] - checkpoints[index])
        area += (means[index] + means[index + 1]) / 2.0 * step
    return area / float(checkpoints[-1] - checkpoints[0])


# ============ typed contract (internal parse target) ============


@dataclass(frozen=True, slots=True)
class _Contrast:
    parameter_path: str
    baseline_arm: str
    treatment_arm: str
    baseline_value: float
    treatment_value: float
    # producer's condition name → arm_key stamped on rows
    arm_keys: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _Contract:
    study_id: str
    pair_by: str
    pair_by_config_path: str | None
    contrast: _Contrast
    scope: Mapping[str, MeasurementLeaf]
    checkpoints: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    outcomes: tuple[str, ...]
    run_measurements: tuple[str, ...]
    assignment_statement: str | None
    protocol_path: str | None
    protocol_sha256: str | None


def _parse_contract(raw: Mapping[str, object]) -> _Contract:
    study_id = _as_str(raw.get('study_id'), 'study_id')
    pair_by = _as_str(raw.get('pair_by'), 'pair_by')
    pair_by_config_path_raw = raw.get('pair_by_config_path')
    pair_by_config_path = (
        None
        if pair_by_config_path_raw is None
        else _as_str(pair_by_config_path_raw, 'pair_by_config_path')
    )

    contrast_raw = _as_mapping(raw.get('contrast'), 'contrast')
    baseline_arm = _as_str(
        contrast_raw.get('baseline_arm'), 'contrast.baseline_arm',
    )
    treatment_arm = _as_str(
        contrast_raw.get('treatment_arm'), 'contrast.treatment_arm',
    )
    arm_keys_raw = contrast_raw.get('logical_arm_keys')
    if arm_keys_raw is None:
        # Default: the producer's condition names ARE the arm keys.
        arm_keys: dict[str, str] = {
            baseline_arm: baseline_arm,
            treatment_arm: treatment_arm,
        }
    else:
        arm_keys = {
            key: _as_str(value, f'contrast.logical_arm_keys[{key!r}]')
            for key, value in _as_mapping(
                arm_keys_raw, 'contrast.logical_arm_keys',
            ).items()
        }
    contrast = _Contrast(
        parameter_path=_as_str(
            contrast_raw.get('parameter_path'), 'contrast.parameter_path',
        ),
        baseline_arm=baseline_arm,
        treatment_arm=treatment_arm,
        baseline_value=_as_finite_number(
            contrast_raw.get('baseline_value'), 'contrast.baseline_value',
        ),
        treatment_value=_as_finite_number(
            contrast_raw.get('treatment_value'), 'contrast.treatment_value',
        ),
        arm_keys=arm_keys,
    )

    scope = {
        key: _as_scalar(value, f'scope.{key}')
        for key, value in _as_mapping(raw.get('scope'), 'scope').items()
    }

    evaluation = _as_mapping(raw.get('evaluation'), 'evaluation')
    checkpoints = tuple(
        _as_int(value, 'evaluation checkpoint')
        for value in _as_list(
            evaluation.get('checkpoints'), 'evaluation.checkpoints',
        )
    )
    eval_seeds = tuple(
        _as_int(value, 'evaluation seed')
        for value in _as_list(evaluation.get('seeds'), 'evaluation.seeds')
    )
    outcomes = tuple(
        _as_str(value, 'evaluation outcome')
        for value in _as_list(
            evaluation.get('outcomes'), 'evaluation.outcomes',
        )
    )
    if not checkpoints or not eval_seeds or not outcomes:
        raise ValueError(
            'evaluation.checkpoints, evaluation.seeds, and '
            'evaluation.outcomes must be non-empty',
        )

    run_measurements_raw = raw.get('run_measurements')
    run_measurements = (
        ()
        if run_measurements_raw is None
        else tuple(
            _as_str(value, 'run measurement name')
            for value in _as_list(run_measurements_raw, 'run_measurements')
        )
    )

    assignment_raw = raw.get('assignment')
    if assignment_raw is None:
        assignment_statement = None
    else:
        assignment = _as_mapping(assignment_raw, 'assignment')
        if assignment.get('assurance') != 'attested':
            raise ValueError(
                "assignment.assurance must be 'attested' when an "
                'assignment note is committed',
            )
        assignment_statement = _as_str(
            assignment.get('statement'), 'assignment.statement',
        )

    protocol_raw = raw.get('prospective_protocol')
    if protocol_raw is None:
        protocol_path = None
        protocol_sha256 = None
    else:
        protocol = _as_mapping(protocol_raw, 'prospective_protocol')
        protocol_path = _as_str(
            protocol.get('path'), 'prospective_protocol.path',
        )
        protocol_sha256 = _as_str(
            protocol.get('sha256'), 'prospective_protocol.sha256',
        )

    return _Contract(
        study_id=study_id,
        pair_by=pair_by,
        pair_by_config_path=pair_by_config_path,
        contrast=contrast,
        scope=scope,
        checkpoints=checkpoints,
        eval_seeds=eval_seeds,
        outcomes=outcomes,
        run_measurements=run_measurements,
        assignment_statement=assignment_statement,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha256,
    )


@dataclass(frozen=True, slots=True)
class _Run:
    run_id: str
    physical_arm: str
    pair_value: int
    config_path: str
    measurements: Mapping[str, float]


_REQUIRED_FILES = (
    'contract.json',
    'runs.jsonl',
    'evaluations.jsonl',
    'provenance.json',
)

_SCHEMA_ERRORS = (TypeError, ValueError, KeyError)


class _Adaptation:
    """One fail-closed pass over a sealed bundle. Transient check
    state only — the public entry point is `adapt_study`."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._checks: list[AdapterCheck] = []

    # ---- receipt plumbing ----

    def _pass(self, code: str, message: str) -> None:
        self._checks.append(
            AdapterCheck(code, CheckStatus.VERIFIED, message),
        )

    def _note(self, code: str, status: CheckStatus, message: str) -> None:
        self._checks.append(AdapterCheck(code, status, message))

    def _fail(self, code: str, message: str) -> NoReturn:
        self._checks.append(AdapterCheck(code, CheckStatus.FAILED, message))
        raise BundleValidationError(message, checks=tuple(self._checks))

    def _require(self, condition: bool, code: str, message: str) -> None:
        if not condition:
            self._fail(code, message)

    def _load_object(self, relative: str, label: str) -> Mapping[str, object]:
        try:
            path = safe_bundle_path(self._root, relative)
            return _as_mapping(read_json(path), label)
        except (OSError, *_SCHEMA_ERRORS) as exc:
            self._fail(f'{label}_readable', f'{label}: {exc}')

    # ---- phases ----

    def _verify_seal(self) -> tuple[dict[str, ManifestEntry], str]:
        manifest = self._load_object(MANIFEST_NAME, 'manifest')
        self._require(
            manifest.get('manifest_version') == MANIFEST_VERSION,
            'manifest_version',
            'unsupported manifest version',
        )
        try:
            entries = {
                relative: ManifestEntry(
                    sha256=_as_str(
                        _as_mapping(
                            entry, f'manifest.files[{relative!r}]',
                        ).get('sha256'),
                        f'{relative}.sha256',
                    ),
                    size=_as_int(
                        _as_mapping(
                            entry, f'manifest.files[{relative!r}]',
                        ).get('size'),
                        f'{relative}.size',
                    ),
                )
                for relative, entry in _as_mapping(
                    manifest.get('files'), 'manifest.files',
                ).items()
            }
        except _SCHEMA_ERRORS as exc:
            self._fail('manifest_schema', str(exc))
        for relative, entry in entries.items():
            try:
                path = safe_bundle_path(self._root, relative)
            except ValueError as exc:
                self._fail('manifest_path_safe', str(exc))
            self._require(
                path.is_file(),
                'manifest_file_exists',
                f'missing file: {relative}',
            )
            self._require(
                path.stat().st_size == entry.size,
                'manifest_size',
                f'size mismatch: {relative}',
            )
            self._require(
                sha256_file(path) == entry.sha256,
                'manifest_sha256',
                f'SHA-256 mismatch: {relative}',
            )
        self._pass('manifest_files', f'verified {len(entries)} sealed files')
        computed = bundle_digest(entries)
        self._require(
            manifest.get('bundle_digest') == computed,
            'bundle_digest',
            'bundle digest does not match manifest entries',
        )
        self._pass('bundle_digest', f'bundle digest {computed[:12]}…')
        for required in _REQUIRED_FILES:
            self._require(
                required in entries,
                'required_file',
                f'manifest does not include {required}',
            )
        return entries, computed

    def _read_contract(self) -> _Contract:
        contract_raw = self._load_object('contract.json', 'contract')
        self._require(
            contract_raw.get('contract_version') == _CONTRACT_VERSION,
            'contract_version',
            'unsupported contract version',
        )
        try:
            contract = _parse_contract(contract_raw)
        except _SCHEMA_ERRORS as exc:
            self._fail('contract_schema', str(exc))
        c = contract.contrast
        self._require(
            c.baseline_arm != c.treatment_arm,
            'contrast_arms_distinct',
            'baseline and treatment conditions must be distinct',
        )
        self._require(
            c.baseline_value != c.treatment_value,
            'contrast_values_distinct',
            'baseline and treatment intervention values must be distinct',
        )
        self._require(
            set(c.arm_keys) == {c.baseline_arm, c.treatment_arm},
            'arm_key_mapping',
            'logical arm mapping must cover exactly the two declared '
            'conditions',
        )
        self._require(
            len(set(c.arm_keys.values())) == 2,
            'arm_keys_distinct',
            'logical arm keys must be distinct',
        )
        self._require(
            contract.checkpoints
            == tuple(sorted(set(contract.checkpoints))),
            'evaluation_checkpoints',
            'evaluation checkpoints must be unique and strictly increasing',
        )
        self._require(
            len(set(contract.eval_seeds)) == len(contract.eval_seeds),
            'evaluation_seeds',
            'evaluation seeds must be unique',
        )
        self._require(
            len(set(contract.outcomes)) == len(contract.outcomes),
            'evaluation_outcomes',
            'evaluation outcome names must be unique',
        )
        return contract

    def _read_provenance(self) -> str:
        provenance = self._load_object('provenance.json', 'provenance')
        try:
            producer = _as_str(
                provenance.get('producer'), 'provenance.producer',
            )
            _as_str(provenance.get('command'), 'provenance.command')
        except _SCHEMA_ERRORS as exc:
            self._fail('provenance_recorded', str(exc))
        self._pass(
            'provenance_recorded',
            f'execution provenance recorded for producer {producer!r}',
        )
        self._note(
            'provenance_attested',
            CheckStatus.ATTESTED,
            'producer identity and invocation are attested by the record, '
            'not mechanically verified',
        )
        return producer

    def _read_runs(
        self,
        contract: _Contract,
        entries: Mapping[str, ManifestEntry],
    ) -> dict[str, _Run]:
        try:
            raw_runs = read_jsonl(
                safe_bundle_path(self._root, 'runs.jsonl'),
            )
        except (OSError, ValueError) as exc:
            self._fail('runs_readable', str(exc))
        self._require(bool(raw_runs), 'runs_nonempty', 'runs.jsonl is empty')
        known_arms = {
            contract.contrast.baseline_arm,
            contract.contrast.treatment_arm,
        }
        runs: dict[str, _Run] = {}
        for raw in raw_runs:
            try:
                run_id = _as_str(raw.get('run_id'), 'run.run_id')
                physical_arm = _as_str(
                    raw.get('physical_arm'), f'{run_id}.physical_arm',
                )
                pair_value = _as_int(
                    raw.get(contract.pair_by),
                    f'{run_id}.{contract.pair_by}',
                )
                config_path = _as_str(
                    raw.get('config_path'), f'{run_id}.config_path',
                )
                measurements = {
                    name: _as_finite_number(
                        raw.get(name), f'{run_id}.{name}',
                    )
                    for name in contract.run_measurements
                }
            except _SCHEMA_ERRORS as exc:
                self._fail('run_schema', str(exc))
            self._require(
                run_id not in runs,
                'run_id_unique',
                f'duplicate run_id: {run_id}',
            )
            self._require(
                raw.get('complete') is True,
                'run_complete',
                f'incomplete run: {run_id}',
            )
            self._require(
                physical_arm in known_arms,
                'arm_known',
                f'unknown declared condition {physical_arm!r} in {run_id}',
            )
            for field_name, expected in contract.scope.items():
                self._require(
                    raw.get(field_name) == expected,
                    'run_scope_consistent',
                    f'{run_id}: {field_name} differs from the contract '
                    'scope',
                )
            self._require(
                config_path in entries,
                'referenced_file_manifested',
                f'{run_id} references unmanifested {config_path}',
            )
            runs[run_id] = _Run(
                run_id=run_id,
                physical_arm=physical_arm,
                pair_value=pair_value,
                config_path=config_path,
                measurements=measurements,
            )
        return runs

    def _verify_pairs(
        self,
        contract: _Contract,
        runs: Mapping[str, _Run],
    ) -> dict[int, tuple[_Run, ...]]:
        pairs: dict[int, tuple[_Run, ...]] = {}
        for run in runs.values():
            pairs[run.pair_value] = (
                *pairs.get(run.pair_value, ()), run,
            )
        expected_arms = {
            contract.contrast.baseline_arm,
            contract.contrast.treatment_arm,
        }
        for pair_value, members in pairs.items():
            self._require(
                len(members) == 2
                and {m.physical_arm for m in members} == expected_arms,
                'pair_complete',
                f'pair {pair_value!r} does not contain exactly one seeded '
                'run per condition',
            )
        self._pass('pairs_complete', f'verified {len(pairs)} complete pairs')
        return pairs

    def _verify_config_isolation(
        self,
        contract: _Contract,
        runs: Mapping[str, _Run],
    ) -> dict[str, float]:
        c = contract.contrast
        isolation_scope = (
            c.parameter_path
            if contract.pair_by_config_path is None
            else f'{c.parameter_path} and {contract.pair_by_config_path}'
        )
        intervention_by_run: dict[str, float] = {}
        template: dict[str, object] | None = None
        template_run_id: str | None = None
        for run_id, run in runs.items():
            config = self._load_object(run.config_path, f'config[{run_id}]')
            try:
                observed = _as_finite_number(
                    _get_path(config, c.parameter_path),
                    f'{run_id}.{c.parameter_path}',
                )
                projected = _without_path(config, c.parameter_path)
                observed_pair: int | None = None
                if contract.pair_by_config_path is not None:
                    observed_pair = _as_int(
                        _get_path(config, contract.pair_by_config_path),
                        f'{run_id}.{contract.pair_by_config_path}',
                    )
                    projected = _without_path(
                        projected, contract.pair_by_config_path,
                    )
            except _SCHEMA_ERRORS as exc:
                self._fail('config_parameter', str(exc))
            expected_value = (
                c.baseline_value
                if run.physical_arm == c.baseline_arm
                else c.treatment_value
            )
            self._require(
                observed == expected_value,
                'intervention_value',
                f'{run_id}: intervention value differs from the declared '
                'condition',
            )
            if observed_pair is not None:
                self._require(
                    observed_pair == run.pair_value,
                    'config_pair_key_consistent',
                    f'{run_id}: {contract.pair_by_config_path} differs '
                    f'from run field {contract.pair_by}',
                )
            if template is None:
                template = projected
                template_run_id = run_id
            else:
                self._require(
                    projected == template,
                    'config_isolation',
                    f'{run_id}: configuration differs from '
                    f'{template_run_id} outside {isolation_scope}',
                )
            intervention_by_run[run_id] = observed
        self._pass(
            'config_isolation',
            'all resolved configs share one template after removing only '
            f'{isolation_scope}',
        )
        return intervention_by_run

    def _read_evaluations(
        self,
        contract: _Contract,
        runs: Mapping[str, _Run],
    ) -> dict[tuple[str, int, int], Mapping[str, float]]:
        try:
            raw_evaluations = read_jsonl(
                safe_bundle_path(self._root, 'evaluations.jsonl'),
            )
        except (OSError, ValueError) as exc:
            self._fail('evaluations_readable', str(exc))
        expected_keys = {
            (run_id, checkpoint, eval_seed)
            for run_id in runs
            for checkpoint in contract.checkpoints
            for eval_seed in contract.eval_seeds
        }
        index: dict[tuple[str, int, int], Mapping[str, float]] = {}
        for raw in raw_evaluations:
            try:
                run_id = _as_str(raw.get('run_id'), 'evaluation.run_id')
                checkpoint = _as_int(
                    raw.get('checkpoint'), 'evaluation.checkpoint',
                )
                eval_seed = _as_int(
                    raw.get('eval_seed'), 'evaluation.eval_seed',
                )
                values = {
                    name: _as_finite_number(
                        raw.get(name), f'evaluation.{name}',
                    )
                    for name in contract.outcomes
                }
            except _SCHEMA_ERRORS as exc:
                self._fail('evaluation_schema', str(exc))
            self._require(
                run_id in runs,
                'evaluation_run_known',
                f'unknown evaluated run: {run_id}',
            )
            key = (run_id, checkpoint, eval_seed)
            self._require(
                key in expected_keys,
                'evaluation_extent_exact',
                f'unexpected evaluation record: {key}',
            )
            self._require(
                key not in index,
                'evaluation_unique',
                f'duplicate evaluation record: {key}',
            )
            index[key] = values
        missing = expected_keys.difference(index)
        self._require(
            not missing,
            'evaluation_complete',
            'missing evaluation record '
            f'{next(iter(sorted(missing)), None)!r}',
        )
        self._pass(
            'evaluation_complete',
            f'verified {len(runs)} × {len(contract.checkpoints)} × '
            f'{len(contract.eval_seeds)} evaluation extent',
        )
        return index

    def _verify_protocol(
        self,
        contract: _Contract,
        entries: Mapping[str, ManifestEntry],
        pairs: Mapping[int, tuple[_Run, ...]],
    ) -> None:
        if contract.protocol_path is None:
            self._note(
                'protocol',
                CheckStatus.UNVERIFIABLE,
                'no prospective protocol committed; the design is admitted '
                'retrospectively',
            )
            return
        c = contract.contrast
        try:
            protocol_file = safe_bundle_path(
                self._root, contract.protocol_path,
            )
        except ValueError as exc:
            self._fail('protocol_committed', str(exc))
        self._require(
            contract.protocol_path in entries
            and sha256_file(protocol_file) == contract.protocol_sha256,
            'protocol_committed',
            'prospective protocol is absent or differs from its committed '
            'digest',
        )
        self._pass(
            'protocol_committed', 'verified prospective protocol digest',
        )
        document = self._load_object(
            contract.protocol_path, 'prospective_protocol',
        )
        try:
            protocol_pair_keys = tuple(
                _as_int(value, 'protocol pair key')
                for value in _as_list(
                    document.get('confirmatory_pair_keys'),
                    'prospective_protocol.confirmatory_pair_keys',
                )
            )
            arms_document = _as_mapping(
                document.get('paired_arms'),
                'prospective_protocol.paired_arms',
            )
            baseline_key = c.arm_keys[c.baseline_arm]
            treatment_key = c.arm_keys[c.treatment_arm]
            protocol_baseline_value = _as_finite_number(
                _get_path(
                    _as_mapping(
                        arms_document.get(baseline_key),
                        f'prospective_protocol.paired_arms[{baseline_key!r}]',
                    ),
                    c.parameter_path,
                ),
                'protocol baseline value',
            )
            protocol_treatment_value = _as_finite_number(
                _get_path(
                    _as_mapping(
                        arms_document.get(treatment_key),
                        f'prospective_protocol.paired_arms'
                        f'[{treatment_key!r}]',
                    ),
                    c.parameter_path,
                ),
                'protocol treatment value',
            )
            evaluation_document = _as_mapping(
                document.get('evaluation'),
                'prospective_protocol.evaluation',
            )
        except _SCHEMA_ERRORS as exc:
            self._fail('protocol_schema', str(exc))
        self._require(
            protocol_pair_keys == tuple(sorted(pairs))
            and protocol_baseline_value == c.baseline_value
            and protocol_treatment_value == c.treatment_value
            and evaluation_document.get('checkpoints')
            == list(contract.checkpoints)
            and evaluation_document.get('seeds')
            == list(contract.eval_seeds),
            'protocol_design_match',
            'executed pair keys, conditions, or evaluation extent differ '
            'from the protocol',
        )
        self._pass(
            'protocol_design_match', 'executed design matches the protocol',
        )

    def _put(
        self,
        row: dict[str, MeasurementLeaf],
        key: str,
        value: MeasurementLeaf,
    ) -> None:
        self._require(
            key not in row,
            'row_key_collision',
            f'derived row column {key!r} collides with an existing column',
        )
        row[key] = value

    def _derive_rows(
        self,
        contract: _Contract,
        runs: Mapping[str, _Run],
        intervention_by_run: Mapping[str, float],
        evaluations: Mapping[tuple[str, int, int], Mapping[str, float]],
        producer: str,
    ) -> tuple[Mapping[str, MeasurementLeaf], ...]:
        c = contract.contrast
        rows: list[Mapping[str, MeasurementLeaf]] = []
        for run_id, run in runs.items():
            row: dict[str, MeasurementLeaf] = {}
            self._put(row, 'id', run_id)
            self._put(row, 'corpus', contract.study_id)
            self._put(row, 'program', f'external:{producer}')
            self._put(row, 'pair_id', str(run.pair_value))
            self._put(row, contract.pair_by, run.pair_value)
            self._put(row, 'physical_arm', run.physical_arm)
            self._put(row, 'arm_key', c.arm_keys[run.physical_arm])
            self._put(
                row, 'arm_is_baseline', run.physical_arm == c.baseline_arm,
            )
            for field_name, value in contract.scope.items():
                self._put(row, field_name, value)
            self._put(row, c.parameter_path, intervention_by_run[run_id])
            for outcome in contract.outcomes:
                per_checkpoint_means = tuple(
                    sum(
                        evaluations[(run_id, checkpoint, eval_seed)][outcome]
                        for eval_seed in contract.eval_seeds
                    )
                    / len(contract.eval_seeds)
                    for checkpoint in contract.checkpoints
                )
                self._put(row, f'{outcome}_mean', per_checkpoint_means[-1])
                self._put(
                    row,
                    f'{outcome}_auc',
                    _normalised_auc(
                        contract.checkpoints, per_checkpoint_means,
                    ),
                )
            for name, value in run.measurements.items():
                self._put(row, name, value)
            rows.append(row)
        self._pass('rows_derived', f'derived {len(rows)} seeded-run rows')
        if contract.run_measurements:
            self._note(
                'run_measurements',
                CheckStatus.ATTESTED,
                f'admitted {len(contract.run_measurements)} '
                'producer-computed measurement column(s) without '
                f'recomputation: {", ".join(contract.run_measurements)}',
            )
        return tuple(rows)

    # ---- entry point ----

    def run(self) -> AdaptedStudy:
        self._require(
            self._root.is_dir(), 'bundle_root',
            'bundle root is not a directory',
        )
        entries, computed_digest = self._verify_seal()
        contract = self._read_contract()
        producer = self._read_provenance()
        runs = self._read_runs(contract, entries)
        pairs = self._verify_pairs(contract, runs)
        intervention_by_run = self._verify_config_isolation(contract, runs)
        evaluations = self._read_evaluations(contract, runs)
        self._verify_protocol(contract, entries, pairs)
        if contract.assignment_statement is not None:
            self._note(
                'assignment',
                CheckStatus.ATTESTED,
                contract.assignment_statement,
            )
            assurance = CheckStatus.ATTESTED
        else:
            self._note(
                'assignment',
                CheckStatus.UNVERIFIABLE,
                'assignment process was not mechanically recorded',
            )
            assurance = CheckStatus.UNVERIFIABLE
        rows = self._derive_rows(
            contract, runs, intervention_by_run, evaluations, producer,
        )
        c = contract.contrast
        contrast = RecordedContrast(
            parameter_path=c.parameter_path,
            baseline_key=c.arm_keys[c.baseline_arm],
            treatment_key=c.arm_keys[c.treatment_arm],
            baseline_value=c.baseline_value,
            treatment_value=c.treatment_value,
            bundle_digest=computed_digest,
            assurance=assurance,
        )
        receipt = AdapterReceipt(
            adapter_version=ADAPTER_VERSION,
            study_id=contract.study_id,
            bundle_root=self._root.resolve(),
            bundle_digest=computed_digest,
            n_runs=len(rows),
            n_pairs=len(pairs),
            checks=tuple(self._checks),
        )
        return AdaptedStudy(rows=rows, contrast=contrast, receipt=receipt)


def adapt_study(bundle_root: Path | str) -> AdaptedStudy:
    """Verify and normalise a sealed external study bundle.

    The one-call entry point: trusts neither filenames nor run
    declarations until the seal, pair completeness, held-fixed
    configuration, and evaluation extent have been checked; then
    returns the validated rows, the recorded contrast, and the
    receipt bound together as an `AdaptedStudy`. Raises
    `BundleValidationError` (carrying the partial receipt) on the
    first broken obligation."""
    return _Adaptation(Path(bundle_root)).run()
