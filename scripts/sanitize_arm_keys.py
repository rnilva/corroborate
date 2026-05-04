"""One-shot arm_key sanitisation: rewrite per-corpus runs.parquet so
`arm_key` carries the canonical_str fingerprint instead of the
substrate-chosen short name (`'ddqn'`, `'ddqn_g0999'`, etc.).

Pre-Phase-6 substrate's cell_runner stamped `arm_key = config.name`
(the YAML hypothesis `name:` field, e.g. `'ddqn_g0999'`). Post-
Phase-6, `run_intervention` stamps `arm_key =
combined_arm_key(intervention_arms)` — e.g.
`'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'`.

For each corpus, this script:

1. Locates the YAML config that produced it (matched by
   `out_dir: experiments/data/<corpus>` field).
2. Resolves each hypothesis template's `intervention_arms`
   through the substrate's claim registry to compute the
   canonical_str arm_key (HP block is skipped — `gamma`,
   `n_step` etc. don't enter arm_key; they survive as their own
   columns).
3. Rewrites the parquet's `arm_key` column from substrate-chosen
   short names to canonical fingerprints. HP cleavages remain
   in their own columns.

Idempotent: re-running on a parquet whose `arm_key` is already
canonical (heuristic: contains `=` or equals `'baseline'`) is a
no-op.

Some corpora pre-date even the `arm_key` column and carry the
substrate-chosen short name in `intervention_name`. The script
reads `intervention_name` in that case, writes `arm_key`, and
drops `intervention_name`.

Atomic-rename means an already-archived dir's manifest sha256
will MISMATCH after sanitisation. The caller must either
re-archive the sanitised file or accept the local-clean / s3-dirty
divergence.

Usage:
    PYTHONPATH=. uv run python scripts/sanitize_arm_keys.py
        # Sanitise every locally-present experiments/data/<corpus>/runs.parquet
        # whose YAML config exists. Idempotent.

    PYTHONPATH=. uv run python scripts/sanitize_arm_keys.py \\
        action_dim_at_low_rs gamma_sweep_metamaze_high
        # Sanitise only the named corpora.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import polars as pl  # noqa: E402

from corroborate._internals.yaml import safe_load as _yaml_load  # noqa: E402
from corroborate.core.intervention import (  # noqa: E402
    Intervention, combined_arm_key, is_replacement,
)
from corroborate_rl.dqn.config_loader import (  # noqa: E402
    is_str_keyed_mapping, resolve,
)
from corroborate_rl.dqn.yaml_sweep import default_dqn_registry  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / 'experiments' / 'data'
CONFIGS_ROOT = REPO_ROOT / 'experiments' / 'configs'


def discover_config_for_corpus(corpus_dir: str) -> Path | None:
    """Find the YAML config whose `out_dir:` matches this corpus dir."""
    for yaml_path in sorted(CONFIGS_ROOT.glob('*.yaml')):
        with yaml_path.open() as f:
            raw = _yaml_load(f)
        if not is_str_keyed_mapping(raw):
            continue
        out_dir = raw.get('out_dir')
        if not isinstance(out_dir, str):
            continue
        if Path(out_dir).name == corpus_dir:
            return yaml_path
    return None


def derive_name_to_canonical(yaml_path: Path) -> dict[str, str]:
    """Build `hypothesis.name → canonical_str arm_key` mapping
    from a YAML config.

    Resolves only `intervention_arms` (skipping the `intervention`
    HP block), so paired-mode YAMLs whose HP block contains
    `{from_env: ...}` placeholders are processed cleanly: HPs
    don't enter arm_key. If a YAML's `intervention_arms` itself
    references `from_env`, the canonical fingerprint would be
    env-dependent and this script raises.
    """
    reg = default_dqn_registry()
    with yaml_path.open() as f:
        raw = _yaml_load(f)
    if not is_str_keyed_mapping(raw):
        raise ValueError(f'{yaml_path}: top-level not a mapping')

    hypotheses_raw = raw.get('hypotheses')
    if not isinstance(hypotheses_raw, list):
        raise ValueError(
            f'{yaml_path}: missing/non-list `hypotheses` field',
        )
    hypotheses_typed: list[object] = list(hypotheses_raw)

    out: dict[str, str] = {}
    for h_raw in hypotheses_typed:
        if not is_str_keyed_mapping(h_raw):
            raise ValueError(f'{yaml_path}: hypothesis not a mapping')
        name = h_raw.get('name')
        if not isinstance(name, str):
            raise ValueError(
                f'{yaml_path}: hypothesis missing or non-str `name`',
            )

        arms_raw = h_raw.get('intervention_arms', [])
        if not isinstance(arms_raw, list):
            raise ValueError(
                f'{yaml_path}:{name}: intervention_arms not a list',
            )
        arms_typed: list[object] = list(arms_raw)

        arms: list[Intervention] = []
        for a in arms_typed:
            if not is_str_keyed_mapping(a):
                raise ValueError(
                    f'{yaml_path}:{name}: arm not a mapping',
                )
            slot_path = a.get('slot_path')
            if not isinstance(slot_path, str):
                raise ValueError(
                    f'{yaml_path}:{name}: arm.slot_path not str',
                )
            if 'replacement' not in a:
                raise ValueError(
                    f'{yaml_path}:{name}: arm missing `replacement`',
                )
            # env_attrs=None — raises if intervention_arms touches
            # `from_env` (would mean the canonical_str is env-
            # dependent and this script can't safely resolve).
            raw_repl = resolve(a['replacement'], reg=reg, env_attrs=None)
            if not is_replacement(raw_repl):
                raise TypeError(
                    f'{yaml_path}:{name}: arm.replacement not callable',
                )
            arms.append(
                Intervention(slot_path=slot_path, replacement=raw_repl),
            )

        canonical = combined_arm_key(tuple(arms))
        if name in out and out[name] != canonical:
            raise ValueError(
                f'{yaml_path}: name {name!r} resolves to two distinct '
                f'canonical arm_keys: {out[name]!r} vs {canonical!r}',
            )
        out[name] = canonical
    return out


def needs_sanitisation(values: pl.Series) -> bool:
    """Heuristic: a canonical arm_key contains `=` (slot=replacement
    format) or is exactly `'baseline'`. Substrate-chosen tokens
    (`'ddqn'`, `'ddqn_g0999'`) match neither."""
    sample = values.unique().drop_nulls().to_list()
    for v in sample:
        if not isinstance(v, str):
            # Unknown shape — let the rewrite pass produce a clear error.
            return True
        if v == 'baseline':
            continue
        if '=' in v:
            continue
        return True
    return False


def sanitise_one(
    runs_path: Path, name_to_canonical: Mapping[str, str],
) -> bool:
    """Sanitise `runs_path` in-place. Returns `True` if rewritten,
    `False` if already canonical. Atomic via tmp + rename.

    Raises `ValueError` if the source column carries values not in
    `name_to_canonical` — fail loud rather than silently dropping
    cells."""
    df = pl.read_parquet(runs_path)
    cols = set(df.columns)

    if 'arm_key' in cols:
        if not needs_sanitisation(df['arm_key']):
            return False
        source_col = 'arm_key'
    elif 'intervention_name' in cols:
        source_col = 'intervention_name'
    else:
        raise ValueError(
            f'{runs_path}: neither `arm_key` nor `intervention_name` '
            f'column present; cannot sanitise.',
        )

    source_values = df[source_col].unique().drop_nulls().to_list()
    unmapped = [v for v in source_values if v not in name_to_canonical]
    if unmapped:
        raise ValueError(
            f'{runs_path}: column {source_col!r} contains values not '
            f'in YAML name→canonical mapping: {sorted(unmapped)!r}. '
            f'Mapping has: {sorted(name_to_canonical)!r}',
        )

    rewritten = df.with_columns(
        pl.col(source_col)
        .replace_strict(name_to_canonical)
        .alias('arm_key'),
    )
    if source_col == 'intervention_name':
        rewritten = rewritten.drop('intervention_name')

    tmp = runs_path.with_suffix(runs_path.suffix + '.armkey.tmp')
    rewritten.write_parquet(tmp)
    tmp.replace(runs_path)  # atomic on POSIX
    return True


def main(argv: list[str]) -> None:
    only = tuple(argv[1:])

    targets: list[tuple[str, Path]] = []
    if only:
        for name in only:
            runs = DATA_ROOT / name / 'runs.parquet'
            if not runs.exists():
                raise SystemExit(
                    f'{name}: no local runs.parquet at {runs}',
                )
            targets.append((name, runs))
    else:
        for d in sorted(DATA_ROOT.iterdir()):
            if not d.is_dir():
                continue
            runs = d / 'runs.parquet'
            if runs.exists():
                targets.append((d.name, runs))

    print(f'scanning {len(targets)} locally-present corpora.')
    n_changed = n_clean = n_skipped = n_error = 0
    for corpus_name, runs_path in targets:
        config = discover_config_for_corpus(corpus_name)
        if config is None:
            print(f'  {corpus_name}: SKIP (no matching YAML config)')
            n_skipped += 1
            continue
        try:
            mapping = derive_name_to_canonical(config)
            rewrote = sanitise_one(runs_path, mapping)
        except Exception as e:  # noqa: BLE001
            print(f'  {corpus_name}: ERROR {type(e).__name__}: {e}')
            n_error += 1
            continue
        if rewrote:
            print(
                f'  {corpus_name}: rewrote arm_key '
                f'(via {config.name})',
            )
            n_changed += 1
        else:
            print(f'  {corpus_name}: clean (already canonical).')
            n_clean += 1

    print(
        f'\nsanitised {n_changed}, clean {n_clean}, '
        f'skipped {n_skipped}, errors {n_error}.',
    )


if __name__ == '__main__':
    main(sys.argv)
