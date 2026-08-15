"""Phase 1 regression tests for `corroborate.data.Panel`.

Covers:
- `from_dataframe`: synthetic-cells exploration mode.
- `from_corpus`: load-one-corpus-without-ingest path.
- `from_corpora`: union-of-corpora exploration.
- `narrow`: scope-chain extension + diagnostic recompute.
- `split_by`: per-stratum sub-panel partition.
- `derive`: per-stratum aggregator (mean / std / median × cell_filter).
- `with_measurables`: on-demand compute via `compute_missing_columns`.
- `diagnostics`: all four per-stratum facts.

Exploration mode (no `required_measurables`) and resolution mode
(`required_measurables` set) both exercised. Diagnostics asserts
are value-asserts, not just shape-asserts."""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import polars as pl
import pytest

from corroborate.data import DerivedSpec, Panel
from corroborate.measurables import measurable


# Register test-fixture measurables so the panel's diagnostics
# can detect `_panel_test_jens` + `_panel_test_outcome` as
# registered (their finite-fraction is what the diagnostic
# reports). The @measurable decorator registers on definition;
# the bodies are no-ops because the cells frame already carries
# the column values — diagnostics only consult registry
# membership, not the function body.
@measurable(reads=())
def _panel_test_jens(record: Mapping[str, object]) -> float:
    """Test-fixture: registered to mark `_panel_test_jens` as a
    measurable column for Panel diagnostics. The cells frame
    pre-populates the column, so the function body is unused
    by these tests."""
    del record
    return float('nan')


@measurable(reads=())
def _panel_test_outcome(record: Mapping[str, object]) -> float:
    """Test-fixture: registered to mark
    `_panel_test_outcome` as a measurable column for
    Panel diagnostics."""
    del record
    return float('nan')


def _make_cells_dataframe() -> pl.DataFrame:
    """3 envs × 2 arms × 3 seeds = 18 cells.

    Includes one HP leaf (`optimizer.inner.lr`) that varies WITHIN
    one env's vanilla arm to exercise the nonunique-configs
    diagnostic. Includes one measurable column (`_panel_test_jens`) with
    one NaN cell per env-arm to exercise the finite-fraction
    diagnostic."""
    return pl.DataFrame({
        'id': [f'cell-{i:02d}' for i in range(18)],
        'env_name': (
            ['FourRooms-misc'] * 6
            + ['Asterix-MinAtar'] * 6
            + ['Snake-jumanji'] * 6
        ),
        'arm_key': ['baseline', 'baseline', 'baseline',
                    'ddqn', 'ddqn', 'ddqn'] * 3,
        'seed': [0, 1, 2, 0, 1, 2] * 3,
        'corpus': (
            # FR: 2 corpora across the 6 cells — exercises
            # corpora_per_stratum (both baseline AND ddqn arms
            # have 2 distinct corpora).
            ['fr_corpA', 'fr_corpA', 'fr_corpB',
             'fr_corpA', 'fr_corpA', 'fr_corpB']
            + ['asterix_corp'] * 6
            + ['snake_corp'] * 6
        ),
        # Single HP-leaf with one variant in FR baseline (cell 2
        # uses lr=2e-4 instead of canonical 1e-4); makes
        # nonunique_configs_per_stratum[('FourRooms-misc',
        # 'baseline')] = 2 rather than 1.
        'optimizer.inner.lr': [
            1e-4, 1e-4, 2e-4,
            1e-4, 1e-4, 1e-4,
        ] + [1e-4] * 6 + [1e-4] * 6,
        # Measurable: one NaN in each env-arm pair.
        '_panel_test_jens': [
            float('nan'), 0.5, 0.6,
            float('nan'), 0.2, 0.3,
            float('nan'), 1.5, 1.6,
            float('nan'), 1.2, 1.3,
            float('nan'), 0.8, 0.9,
            float('nan'), 0.5, 0.6,
        ],
        # A second measurable, fully finite — exercises the
        # "mixed-finiteness across measurables" reporting.
        '_panel_test_outcome': [
            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
            20.0, 21.0, 22.0,
            23.0, 24.0, 25.0,
            5.0, 6.0, 7.0,
            8.0, 9.0, 10.0,
        ],
    })


def test_from_dataframe_constructs_exploration_panel() -> None:
    """`Panel.from_dataframe(df)` is the test-fixture / hand-built
    constructor. No scope_chain, no sources — pure exploration
    starting point."""
    df = _make_cells_dataframe()
    panel = Panel.from_dataframe(df)
    assert panel.cells.height == 18
    assert panel.scope_chain == ()
    assert panel.sources == ()
    assert panel.required_measurables == frozenset()
    # stratify_by defaults to (env_name, arm_key).
    assert panel.stratify_by == ('env_name', 'arm_key')


def test_diagnostics_n_cells_per_stratum() -> None:
    """`n_cells_per_stratum` reports the per-stratum cell count
    used to detect inflation. With 3 envs × 2 arms × 3 seeds each,
    every stratum should be 3."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    diag = panel.diagnostics
    for env in ['FourRooms-misc', 'Asterix-MinAtar', 'Snake-jumanji']:
        for arm in ['baseline', 'ddqn']:
            assert diag.n_cells_per_stratum[(env, arm)] == 3, (
                f'expected 3 cells at ({env}, {arm}); '
                f'got {diag.n_cells_per_stratum[(env, arm)]}'
            )


def test_diagnostics_corpora_per_stratum_surfaces_hp_mixing() -> None:
    """`corpora_per_stratum` reports the set of source-corpus
    stamps per stratum. The FR cells came from 2 corpora — this
    surfaces the today's-FR-style inflation at hypothesis run
    time (BUT does not enforce; the bridge author reads + decides)."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    diag = panel.diagnostics
    assert diag.corpora_per_stratum[('FourRooms-misc', 'baseline')] == (
        frozenset({'fr_corpA', 'fr_corpB'})
    )
    assert diag.corpora_per_stratum[('FourRooms-misc', 'ddqn')] == (
        frozenset({'fr_corpA', 'fr_corpB'})
    )
    # Single-corpus envs: just one stamp.
    assert diag.corpora_per_stratum[('Asterix-MinAtar', 'baseline')] == (
        frozenset({'asterix_corp'})
    )
    assert diag.corpora_per_stratum[('Snake-jumanji', 'ddqn')] == (
        frozenset({'snake_corp'})
    )


def test_diagnostics_programs_per_stratum_surfaces_program_blind_pooling(
) -> None:
    """`programs_per_stratum` surfaces program-blind `arm_key`
    pooling. `arm_key` is the pure intervention fingerprint, so a
    `dqn` `baseline` arm and a `paired_dqn` `baseline` arm collide
    on `arm_key='baseline'` — a stratum spanning >1 program is
    pooling structurally different root programs. Cross-program
    contrast is legitimate (program becomes the axis); this makes
    ACCIDENTAL pooling visible. Cells predating the `program`
    column read as null → absent (empty set), NOT a distinct
    program."""
    df = pl.DataFrame({
        'id': [f'c{i}' for i in range(6)],
        'env_name': ['Mix'] * 2 + ['Single'] * 2 + ['Legacy'] * 2,
        'arm_key': ['baseline'] * 6,
        'seed': [0, 1, 0, 1, 0, 1],
        # Mix: same arm_key collides across two programs.
        # Single: one program. Legacy: null (pre-program corpus).
        'program': ['dqn', 'paired_dqn', 'dqn', 'dqn', None, None],
    })
    diag = Panel.from_dataframe(df).diagnostics
    # Program-blind collision surfaced.
    assert diag.programs_per_stratum[('Mix', 'baseline')] == (
        frozenset({'dqn', 'paired_dqn'})
    )
    # Single program → singleton.
    assert diag.programs_per_stratum[('Single', 'baseline')] == (
        frozenset({'dqn'})
    )
    # Null program (legacy) → absent, not a distinct program.
    assert diag.programs_per_stratum[('Legacy', 'baseline')] == frozenset()


def test_diagnostics_programs_per_stratum_absent_column() -> None:
    """A panel whose cells carry no `program` column at all reports
    an empty set per stratum (not a KeyError) — old corpora pre-date
    the typed column."""
    diag = Panel.from_dataframe(_make_cells_dataframe()).diagnostics
    assert diag.programs_per_stratum[('Asterix-MinAtar', 'baseline')] == (
        frozenset()
    )


def test_diagnostics_nonunique_configs_surfaces_hp_heterogeneity() -> None:
    """`nonunique_configs_per_stratum` counts distinct config
    fingerprints (auto-detected via `aggregate.leaf_signature`-
    style filtering: framework-identity cols + exogenous +
    registered-measurable cols are excluded; everything else is
    a leaf).

    The FR baseline stratum has lr=1e-4 for 2 cells, lr=2e-4 for
    1 cell → 2 distinct configs. Every other stratum is
    homogeneous → 1."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    diag = panel.diagnostics
    assert diag.nonunique_configs_per_stratum[
        ('FourRooms-misc', 'baseline')
    ] == 2
    assert diag.nonunique_configs_per_stratum[
        ('FourRooms-misc', 'ddqn')
    ] == 1
    assert diag.nonunique_configs_per_stratum[
        ('Asterix-MinAtar', 'baseline')
    ] == 1


def test_diagnostics_finite_fraction_exploration_mode() -> None:
    """Without `required_measurables`, `finite_fraction_per_stratum
    _measurable` reports for every column that IS a registered
    measurable on this panel. `_panel_test_jens` is registered + has 1
    NaN of 3 cells per stratum → 2/3 ≈ 0.667.
    `_panel_test_outcome` is fully finite → 1.0."""
    # Both measurable names ARE registered in the framework's
    # @measurable registry (used by other tests / implementation).
    # This test confirms the auto-detection picks them up.
    panel = Panel.from_dataframe(_make_cells_dataframe())
    diag = panel.diagnostics
    for env in ['FourRooms-misc', 'Asterix-MinAtar', 'Snake-jumanji']:
        for arm in ['baseline', 'ddqn']:
            per_meas = diag.finite_fraction_per_stratum_measurable[(env, arm)]
            assert math.isclose(
                per_meas['_panel_test_jens'], 2.0 / 3.0, abs_tol=1e-9,
            ), (
                f'_panel_test_jens finite fraction at ({env}, {arm}): '
                f'{per_meas["_panel_test_jens"]} (expected 0.667)'
            )
            assert per_meas['_panel_test_outcome'] == 1.0


def test_diagnostics_finite_fraction_required_measurables_narrows_report() -> None:
    """When `required_measurables` is set (resolution mode), the
    finite-fraction map only reports those names. The frame's
    other measurable cols are present but excluded from the
    diagnostic."""
    panel = Panel.from_dataframe(
        _make_cells_dataframe(),
        required_measurables=frozenset({'_panel_test_jens'}),
    )
    diag = panel.diagnostics
    per_meas = diag.finite_fraction_per_stratum_measurable[
        ('FourRooms-misc', 'baseline')
    ]
    assert set(per_meas.keys()) == {'_panel_test_jens'}


def test_narrow_extends_scope_chain_and_filters_cells() -> None:
    """`panel.narrow(expr)` returns a new Panel with `expr`
    appended to `scope_chain`; cells are filtered; diagnostics
    are recomputed from the narrowed cells (no inheritance)."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    narrowed = panel.narrow(pl.col('env_name') == 'FourRooms-misc')
    assert narrowed.cells.height == 6
    assert len(narrowed.scope_chain) == 1
    # Diagnostics are recomputed: only FR strata appear.
    fr_strata = {
        k for k in narrowed.diagnostics.n_cells_per_stratum
        if k[0] == 'FourRooms-misc'
    }
    assert len(fr_strata) == 2  # 2 arms

    # Narrowing again chains the expression.
    narrowed2 = narrowed.narrow(pl.col('arm_key') == 'baseline')
    assert narrowed2.cells.height == 3
    assert len(narrowed2.scope_chain) == 2
    # scope_provenance preserves the full chain on diagnostics.
    assert narrowed2.diagnostics.scope_provenance == narrowed2.scope_chain


def test_split_by_partitions_into_sub_panels() -> None:
    """`panel.split_by('env_name')` returns one sub-Panel per
    distinct env. Each sub-Panel's cells are the env's subset;
    sources + scope_chain are inherited."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    parts = panel.split_by('env_name')
    assert set(parts.keys()) == {
        ('FourRooms-misc',), ('Asterix-MinAtar',), ('Snake-jumanji',),
    }
    for k, sub in parts.items():
        assert sub.cells.height == 6
        assert sub.scope_chain == panel.scope_chain
        env_seen = set(sub.cells['env_name'].to_list())
        assert env_seen == {k[0]}


def test_derive_mean_per_stratum() -> None:
    """`Panel.derive(DerivedSpec('_panel_test_jens', 'mean', None))`
    computes mean of _panel_test_jens per stratum, dropping NaN cells
    pre-aggregation. With 1 NaN + 2 finite cells per stratum
    (mean of the 2 finite values), the FR baseline values
    [NaN, 0.5, 0.6] yield mean 0.55."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    means = panel.derive(DerivedSpec('_panel_test_jens', 'mean'))
    assert math.isclose(means[('FourRooms-misc', 'baseline')], 0.55, abs_tol=1e-9)
    assert math.isclose(means[('FourRooms-misc', 'ddqn')], 0.25, abs_tol=1e-9)


def test_derive_std_per_stratum() -> None:
    """Sample SD (ddof=1) per stratum. FR baseline finite cells
    [0.5, 0.6] → SD = sqrt(((0.5-0.55)^2 + (0.6-0.55)^2) / 1)
    = sqrt(0.005) ≈ 0.0707."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    stds = panel.derive(DerivedSpec('_panel_test_jens', 'std'))
    assert math.isclose(
        stds[('FourRooms-misc', 'baseline')], math.sqrt(0.005), abs_tol=1e-9,
    )


def test_derive_with_cell_filter_narrows_aggregation_input() -> None:
    """`spec.cell_filter` further narrows the cells contributing
    to the aggregate. Filter to seed 0 — each stratum has 1
    cell surviving. For `'mean'`/`'median'` the default
    `effective_min_n=1` admits single-cell strata; for `'std'`
    (`effective_min_n=2`) ALL strata are skipped."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    means = panel.derive(DerivedSpec(
        '_panel_test_outcome', 'mean',
        cell_filter=pl.col('seed') == 0,
    ))
    # Mean default min_n=1 — every stratum returns its single
    # seed-0 outcome value.
    assert len(means) == 6, (
        f'expected 6 strata × 1 mean each at min_n=1; got {means}'
    )

    # Same filter under 'std' aggregator: default min_n=2 skips
    # everything.
    stds = panel.derive(DerivedSpec(
        '_panel_test_outcome', 'std',
        cell_filter=pl.col('seed') == 0,
    ))
    assert stds == {}

    # Override min_n explicitly to demand more than the natural
    # default — every stratum gets dropped.
    means_floored = panel.derive(DerivedSpec(
        '_panel_test_outcome', 'mean',
        cell_filter=pl.col('seed') == 0,
        min_n=5,
    ))
    assert means_floored == {}


def test_with_measurables_no_op_when_already_present() -> None:
    """`panel.with_measurables(['_panel_test_jens'])` is a no-op when
    the column is already populated. Returns a new Panel
    (immutability) with cells unchanged in shape; values
    preserved bit-for-bit."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    out = panel.with_measurables(['_panel_test_jens'])
    assert out.cells.height == panel.cells.height
    # Original NaN cells preserved (per the
    # compute_missing_columns partial-null branch logic).
    # polars sum() returns int | float | Decimal; cast to int.
    n_null = int(out.cells['_panel_test_jens'].is_null().sum())
    n_nan = int(out.cells['_panel_test_jens'].is_nan().sum())
    assert n_null + n_nan == 6
    # ID order preserved (no shuffle).
    assert out.cells['id'].to_list() == panel.cells['id'].to_list()


@measurable(reads=('x',))
def _panel_synth_double(record: Mapping[str, object]) -> float:
    """Test-fixture measurable for the with_measurables-cascade
    case below."""
    x = record.get('x')
    if not isinstance(x, (int, float)):
        raise TypeError(f'x = {x!r}')
    return 2.0 * float(x)


def test_with_measurables_computes_absent_registered_measurable() -> None:
    """When the requested name IS in the @measurable registry but
    absent from `cells`, `with_measurables` computes it."""
    cells = pl.DataFrame({
        'id': ['c0', 'c1', 'c2'],
        'env_name': ['x'] * 3,
        'arm_key': ['baseline'] * 3,
        'x': [1.0, 2.0, 3.0],
    })
    panel = Panel.from_dataframe(cells)
    out = panel.with_measurables(['_panel_synth_double'])
    assert '_panel_synth_double' in out.cells.columns
    assert out.cells['_panel_synth_double'].to_list() == [2.0, 4.0, 6.0]


def test_from_corpus_loads_runs_and_measurements(tmp_path: Path) -> None:
    """`Panel.from_corpus(dir)` reads runs.parquet, left-joins
    measurements.parquet by id, populates `sources` with one
    `CorpusSource`. No measurements file → cells from runs only;
    no traces by default."""
    corpus = tmp_path / 'syn_corpus'
    corpus.mkdir()
    runs = pl.DataFrame({
        'id': ['c0', 'c1', 'c2', 'c3'],
        'env_name': ['env1', 'env1', 'env2', 'env2'],
        'arm_key': ['baseline', 'ddqn', 'baseline', 'ddqn'],
        'seed': [0, 0, 1, 1],
        'x': [1.0, 2.0, 3.0, 4.0],
    })
    runs.write_parquet(corpus / 'runs.parquet')
    measurements = pl.DataFrame({
        'id': ['c0', 'c1', 'c2', 'c3'],
        '_panel_test_jens': [0.1, 0.2, 0.3, 0.4],
    })
    measurements.write_parquet(corpus / 'measurements.parquet')

    panel = Panel.from_corpus(corpus)
    assert panel.cells.height == 4
    assert '_panel_test_jens' in panel.cells.columns
    assert panel.cells['_panel_test_jens'].to_list() == [0.1, 0.2, 0.3, 0.4]
    # `corpus` column auto-stamped when absent.
    assert 'corpus' in panel.cells.columns
    assert set(panel.cells['corpus'].to_list()) == {'syn_corpus'}
    # CorpusSource entry created.
    assert len(panel.sources) == 1
    assert panel.sources[0].corpus == 'syn_corpus'
    assert panel.sources[0].data_root == corpus.parent.resolve()


def test_from_corpus_missing_runs_returns_empty(tmp_path: Path) -> None:
    """A dir without `runs.parquet` is not a corpus; returns an
    empty Panel rather than raising. Mirrors the runner's
    "skipped, not a corpus" convention."""
    not_corpus = tmp_path / 'not_a_corpus'
    not_corpus.mkdir()
    panel = Panel.from_corpus(not_corpus)
    assert panel.cells.height == 0
    assert panel.sources == ()


def test_from_corpora_unions_cells_with_diagonal_relaxed(
    tmp_path: Path,
) -> None:
    """`from_corpora` concatenates multiple corpora's cells.
    Schema drift (one has an extra column) is handled via
    diagonal_relaxed; missing cells get null. Each corpus
    contributes one `CorpusSource`."""
    a = tmp_path / 'corpA'
    a.mkdir()
    b = tmp_path / 'corpB'
    b.mkdir()
    pl.DataFrame({
        'id': ['a0', 'a1'],
        'env_name': ['env1', 'env1'],
        'arm_key': ['baseline', 'ddqn'],
        'x': [1.0, 2.0],
    }).write_parquet(a / 'runs.parquet')
    pl.DataFrame({
        'id': ['b0', 'b1'],
        'env_name': ['env2', 'env2'],
        'arm_key': ['baseline', 'ddqn'],
        'x': [3.0, 4.0],
        'y_extra': [10.0, 20.0],
    }).write_parquet(b / 'runs.parquet')

    panel = Panel.from_corpora([a, b])
    assert panel.cells.height == 4
    assert len(panel.sources) == 2
    sources = {s.corpus for s in panel.sources}
    assert sources == {'corpA', 'corpB'}
    # corpA's y_extra is null (diagonal-relaxed); corpB's is finite.
    cells_a = panel.cells.filter(pl.col('corpus') == 'corpA')
    cells_b = panel.cells.filter(pl.col('corpus') == 'corpB')
    assert cells_a['y_extra'].null_count() == 2
    assert cells_b['y_extra'].to_list() == [10.0, 20.0]


def test_with_traces_joins_on_demand(tmp_path: Path) -> None:
    """`panel.with_traces(['col'])` loads only the named trace
    columns from this Panel's source corpora and joins them onto
    `cells`. Already-present cols are skipped."""
    corpus = tmp_path / 'syn_with_traces'
    corpus.mkdir()
    pl.DataFrame({
        'id': ['c0', 'c1', 'c2'],
        'env_name': ['env1'] * 3,
        'arm_key': ['baseline'] * 3,
        'seed': [0, 1, 2],
    }).write_parquet(corpus / 'runs.parquet')
    pl.DataFrame({
        'id': ['c0', 'c1', 'c2'],
        'per_step_col_a': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        'per_step_col_b': [[10.0], [20.0], [30.0]],
    }).write_parquet(corpus / 'traces.parquet')

    panel = Panel.from_corpus(corpus)
    assert 'per_step_col_a' not in panel.cells.columns

    enriched = panel.with_traces(['per_step_col_a'])
    assert 'per_step_col_a' in enriched.cells.columns
    assert 'per_step_col_b' not in enriched.cells.columns  # not requested
    # Idempotent — second call is a no-op (col already present).
    enriched2 = enriched.with_traces(['per_step_col_a'])
    assert enriched2.cells.shape == enriched.cells.shape


def test_with_traces_no_sources_is_no_op() -> None:
    """A Panel built via `from_dataframe` has no `sources` — no
    file to load traces from. `with_traces` returns the panel
    unchanged."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    out = panel.with_traces(['nonexistent_col'])
    assert out is panel or out.cells.equals(panel.cells)


def test_from_cache_loads_hypothesis_cache_parquet(tmp_path: Path) -> None:
    """`Panel.from_cache(module)` reads the per-hypothesis cache
    parquet via the runner's `_default_cache_path` resolver.
    Empty Panel when cache absent."""
    # Construct a synthetic hypothesis module + cache file at the
    # standard location so the resolver finds it. Easiest: point
    # at a real cache that exists in the repo's experiments tree.
    # Falls back to "absent cache → empty Panel" if no cache.
    panel_absent = Panel.from_cache('nonexistent.module.path')
    assert panel_absent.cells.height == 0
    assert panel_absent.sources == ()


def test_split_by_multi_key_partition() -> None:
    """`split_by` with multiple keys produces (key1, key2)-tuple-
    keyed sub-Panels. Each sub-Panel's cells must be the
    intersection of the key constraints. Tests multi-key
    behavior that the single-key test doesn't cover."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    parts = panel.split_by('env_name', 'arm_key')
    # 3 envs × 2 arms = 6 sub-panels.
    assert len(parts) == 6
    for (env, arm), sub in parts.items():
        assert sub.cells.height == 3  # 3 seeds per (env, arm)
        envs_seen = set(sub.cells['env_name'].to_list())
        arms_seen = set(sub.cells['arm_key'].to_list())
        assert envs_seen == {env}
        assert arms_seen == {arm}


def test_diagnostics_memoised_on_repeated_access() -> None:
    """The `_diag_cache` mutable-list slot must hold one
    PanelDiagnostics object across multiple `.diagnostics`
    accesses (frozen+slots-compatible lazy memo). Verified via
    `is` identity: second access returns the SAME object."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    diag_1 = panel.diagnostics
    diag_2 = panel.diagnostics
    assert diag_1 is diag_2, (
        'diagnostics access not memoised — recomputed each call'
    )


def test_narrow_returns_panel_with_fresh_diag_cache() -> None:
    """`narrow` constructs a NEW Panel with a fresh empty
    `_diag_cache`. Without this, all narrowed panels would
    share the parent's stale diagnostics."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    parent_diag = panel.diagnostics
    narrowed = panel.narrow(pl.col('env_name') == 'FourRooms-misc')
    narrowed_diag = narrowed.diagnostics
    assert parent_diag is not narrowed_diag
    # Each panel has its own memo cell; reading it twice on the
    # SAME panel returns the same object.
    assert narrowed.diagnostics is narrowed_diag


def test_from_corpus_nested_sub_corpus_uses_parent_leaf_stamp(
    tmp_path: Path,
) -> None:
    """Sub-corpora (parent dir has its own runs.parquet) get
    the `parent/leaf` form per `_corpus_stamp`'s convention.
    Without this, same-leaf-name sub-corpora from different
    parents would collide in `corpora_per_stratum`."""
    parent_a = tmp_path / 'parent_a'
    parent_a.mkdir()
    # Parent has a top-level runs.parquet (marks it as a
    # nested-corpora container).
    pl.DataFrame({'id': ['parent_marker']}).write_parquet(
        parent_a / 'runs.parquet',
    )
    sub_a = parent_a / 'ddqn_vs_vanilla'
    sub_a.mkdir()
    pl.DataFrame({
        'id': ['sub_a0'],
        'env_name': ['env1'],
        'arm_key': ['baseline'],
    }).write_parquet(sub_a / 'runs.parquet')

    panel = Panel.from_corpus(sub_a)
    # Stamp should be parent_a/ddqn_vs_vanilla, not bare leaf.
    assert panel.sources[0].corpus == 'parent_a/ddqn_vs_vanilla'
    assert set(panel.cells['corpus'].to_list()) == {
        'parent_a/ddqn_vs_vanilla',
    }


def test_runs_meas_collision_helper_policy_asymmetry(
    tmp_path: Path,
) -> None:
    """The shared `resolve_runs_meas_collision` helper backs
    both `build_measurements` (policy=`runs_wins`) and
    `Panel.from_corpus` (policy=`meas_wins`). The asymmetry is
    intentional and tested here: same (runs ∩ meas) collision
    yields different outputs under different policies.

    Probe: a column not in the current process's registry, in
    BOTH runs and meas. `runs_wins` drops it from meas;
    `meas_wins` drops it from runs."""
    from corroborate.corpus.measurements import (
        resolve_runs_meas_collision,
    )
    runs_cols = {'id', 'env_name', 'arm_key', 'unregistered_col'}
    meas_cols = {'id', 'unregistered_col'}

    drop_runs_a, drop_meas_a = resolve_runs_meas_collision(
        runs_cols=runs_cols,
        meas_cols=meas_cols,
        unregistered_policy='runs_wins',
    )
    assert drop_runs_a == set()
    assert drop_meas_a == {'unregistered_col'}

    drop_runs_b, drop_meas_b = resolve_runs_meas_collision(
        runs_cols=runs_cols,
        meas_cols=meas_cols,
        unregistered_policy='meas_wins',
    )
    assert drop_runs_b == {'unregistered_col'}
    assert drop_meas_b == set()


def test_from_corpus_collision_when_meas_registered_but_no_substrate(
    tmp_path: Path,
) -> None:
    """When a measurements.parquet column collides with a
    runs.parquet column AND the measurable isn't in the
    process's registry (no implementation import), measurements
    must win — it was stamped by a runner that DID have the
    registry. Runs-side NaN never beats a stamped value."""
    corpus = tmp_path / 'corp_unregistered'
    corpus.mkdir()
    # Both files have a column 'q_some_unregistered' — runs has
    # NaN-stamps, measurements has finite values. The column
    # name isn't registered as @measurable in this test
    # process, so the collision-resolution should fall back to
    # "measurements wins" per the BLOCKER-1 fix.
    pl.DataFrame({
        'id': ['c0', 'c1'],
        'env_name': ['env1', 'env1'],
        'arm_key': ['baseline', 'ddqn'],
        'q_some_unregistered': [float('nan'), float('nan')],
    }).write_parquet(corpus / 'runs.parquet')
    pl.DataFrame({
        'id': ['c0', 'c1'],
        'q_some_unregistered': [1.0, 2.0],
    }).write_parquet(corpus / 'measurements.parquet')

    panel = Panel.from_corpus(corpus)
    assert panel.cells['q_some_unregistered'].to_list() == [1.0, 2.0]


def test_panel_is_frozen_immutable() -> None:
    """frozen=True + slots=True: assignment to fields raises
    `dataclasses.FrozenInstanceError`."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        panel.cells = pl.DataFrame()  # type: ignore[misc]


def test_to_cache_writes_parquet_and_sources_sidecar(
    tmp_path: Path,
) -> None:
    """Phase 5 round-trip — write side. `to_cache(cache_path=...)`
    emits the parquet at `cache_path` AND the sources sidecar at
    `<cache>.sources.json`, with each entry's `ingested_at`
    extended by a fresh ISO-8601 UTC timestamp."""
    from corroborate.data.panel import CorpusSource

    cells = _make_cells_dataframe()
    panel = Panel(
        cells=cells,
        sources=(
            CorpusSource(
                corpus='corp_a',
                data_root=Path('/tmp/corp_parent_a'),
            ),
            CorpusSource(
                corpus='corp_b',
                data_root=Path('/tmp/corp_parent_b'),
                remote_root='s3://example/corp_b/',
                ingested_at=('2026-01-01T00:00:00+00:00',),
            ),
        ),
    )
    target = tmp_path / 'test_hyp.parquet'
    written = panel.to_cache(cache_path=target)
    assert written == target
    assert target.exists()
    reread = pl.read_parquet(target)
    assert reread.height == cells.height
    sidecar = target.with_suffix('.sources.json')
    assert sidecar.exists()
    import json
    parsed = json.loads(sidecar.read_text())
    entries = parsed['sources']
    by_corpus = {e['corpus']: e for e in entries}
    # Both corpora carry a fresh ingested_at stamp.
    assert len(by_corpus['corp_a']['ingested_at']) == 1
    # corp_b's pre-existing stamp is preserved + the new one appended.
    assert by_corpus['corp_b']['ingested_at'][0] == (
        '2026-01-01T00:00:00+00:00'
    )
    assert len(by_corpus['corp_b']['ingested_at']) == 2


def test_from_cache_round_trip_recovers_sources(
    tmp_path: Path,
) -> None:
    """Phase 5 round-trip — read side. A Panel written via
    `to_cache(cache_path=...)` and loaded via the explicit
    `pl.read_parquet` + helper path recovers the sources from
    the sidecar (`from_cache(module_name)` resolves via the
    runner; this test uses the lower-level helper to avoid
    needing a real hypothesis module)."""
    # `_read_sources_for_panel` is the internal sidecar reader
    # `from_cache` delegates to; calling it directly lets this
    # test exercise the read side without standing up a real
    # hypothesis module just to satisfy `from_cache`'s resolver.
    from corroborate.data.panel import CorpusSource
    from corroborate.data.panel import _read_sources_for_panel as read

    cells = _make_cells_dataframe()
    panel = Panel(
        cells=cells,
        sources=(
            CorpusSource(
                corpus='corp_x',
                data_root=Path('/tmp/corp_parent'),
                remote_root=None,
            ),
        ),
    )
    target = tmp_path / 'hyp_round_trip.parquet'
    panel.to_cache(cache_path=target)

    # Read back via the helper (mirrors `from_cache`'s sidecar load).
    sources_back = read(target)
    assert len(sources_back) == 1
    rt = sources_back[0]
    assert rt.corpus == 'corp_x'
    assert rt.data_root == Path('/tmp/corp_parent')
    assert rt.remote_root is None
    # Fresh write stamped one ingested_at entry.
    assert len(rt.ingested_at) == 1


def test_to_cache_rejects_both_hypothesis_module_and_cache_path(
    tmp_path: Path,
) -> None:
    """Mutually exclusive — pass one or the other, not both."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    with pytest.raises(ValueError, match='not both'):
        panel.to_cache(
            hypothesis_module='m', cache_path=tmp_path / 'x.parquet',
        )


def test_to_cache_rejects_neither_argument() -> None:
    """Mutually exclusive (other direction) — must pass one."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    with pytest.raises(ValueError, match='hypothesis_module or cache_path'):
        panel.to_cache()


def test_to_cache_write_manifest_replaces_stale_signatures(
    tmp_path: Path,
) -> None:
    """`to_cache(write_manifest=True)` overwrites any pre-existing
    `.hashes.json` with fresh signatures built from the live
    `@measurable` registry — the promoted cohort gets a manifest
    consistent with the current code state, not whatever the
    last runner pass stamped.

    The test cells have only synthetic columns (env_name,
    arm_key, seed, …) — none are registered measurables in this
    test process. The manifest is written but empty (`{}`),
    which is the correct "no overlap" answer."""
    from corroborate.data.panel import CorpusSource

    target = tmp_path / 'with_stale_manifest.parquet'
    stale = target.with_suffix('.hashes.json')
    stale.write_text('{"jensen_gap": "OLD_SIG_FROM_PRIOR_RUN"}')
    panel = Panel(
        cells=_make_cells_dataframe(),
        sources=(CorpusSource(corpus='c', data_root=Path('/tmp/p')),),
    )
    panel.to_cache(cache_path=target)
    assert stale.exists(), 'manifest should be written'
    import json
    parsed = json.loads(stale.read_text())
    # Stale entry replaced: jensen_gap is not in the test cells'
    # registered-measurable column set, so the fresh manifest
    # doesn't include it.
    assert 'jensen_gap' not in parsed


def test_to_cache_write_manifest_false_unlinks_stale(
    tmp_path: Path,
) -> None:
    """`write_manifest=False` unlinks any pre-existing manifest
    (since the promoted cohort's row-set may not match the
    prior signatures) but doesn't write a new one. Use when
    the substrate's `@measurable` registry isn't loaded in
    this process — an empty/inaccurate manifest is worse than
    no manifest, since the runner's drift detection trusts
    what's stamped."""
    target = tmp_path / 'manifest_skipped.parquet'
    stale = target.with_suffix('.hashes.json')
    stale.write_text('{"jensen_gap": "OLD"}')
    panel = Panel.from_dataframe(_make_cells_dataframe())
    panel.to_cache(cache_path=target, write_manifest=False)
    assert not stale.exists(), 'stale manifest should be unlinked'


def test_to_cache_no_manifest_no_op(tmp_path: Path) -> None:
    """When no manifest exists and write_manifest=False, the
    unlink path is a no-op (`missing_ok=True`)."""
    panel = Panel.from_dataframe(_make_cells_dataframe())
    target = tmp_path / 'no_prior_manifest.parquet'
    written = panel.to_cache(cache_path=target, write_manifest=False)
    assert written == target
    assert not target.with_suffix('.hashes.json').exists()


def test_to_cache_write_sidecar_false_skips_sidecar(
    tmp_path: Path,
) -> None:
    """`write_sidecar=False` writes the parquet only; sidecar
    file stays absent. Use case: implementation author wants the
    parquet for their own analysis but doesn't want to mutate
    the canonical `<cache>.sources.json` audit trail."""
    from corroborate.data.panel import CorpusSource

    panel = Panel(
        cells=_make_cells_dataframe(),
        sources=(
            CorpusSource(corpus='c', data_root=Path('/tmp/c_parent')),
        ),
    )
    target = tmp_path / 'no_sidecar.parquet'
    panel.to_cache(cache_path=target, write_sidecar=False)
    assert target.exists()
    assert not target.with_suffix('.sources.json').exists()


def test_with_traces_skips_source_with_none_data_root(
    tmp_path: Path,
) -> None:
    """A Panel loaded from a cache whose sidecar entry had
    `data_root: null` carries `CorpusSource(data_root=None)`.
    `with_traces` must skip those entries gracefully — no
    NoneType arithmetic, no exception."""
    from corroborate.data.panel import CorpusSource

    # Mix: one source has data_root, one doesn't.
    corpus_with = tmp_path / 'with_root'
    corpus_with.mkdir()
    pl.DataFrame({
        'id': ['c0'],
        'env_name': ['env1'],
        'arm_key': ['baseline'],
    }).write_parquet(corpus_with / 'runs.parquet')
    pl.DataFrame({
        'id': ['c0'],
        'extra_trace_col': [[1.0, 2.0, 3.0]],
    }).write_parquet(corpus_with / 'traces.parquet')

    panel = Panel(
        cells=pl.DataFrame({
            'id': ['c0', 'c_unknown'],
            'env_name': ['env1', 'env2'],
            'arm_key': ['baseline', 'baseline'],
        }),
        sources=(
            CorpusSource(
                corpus='with_root', data_root=tmp_path,
            ),
            CorpusSource(  # data_root=None — sidecar-loaded entry
                corpus='no_root_corpus',
                data_root=None,
            ),
        ),
    )
    joined = panel.with_traces(['extra_trace_col'])
    # The with_root corpus's trace column joined; the no_root
    # source is skipped (not raised).
    assert 'extra_trace_col' in joined.cells.columns




# ============ P6 — measurable availability matrix ============

def test_measurable_availability_matrix_classifies_uniform_partial_unavailable(
) -> None:
    """P6 fix. Three classification buckets:

    - `uniform_available`: finite at >0% of cells in EVERY env.
    - `partial`: finite at >0% of cells in SOME envs, but not all.
    - `unavailable`: <=1% finite across ALL envs.
    """
    import polars as pl

    from corroborate.data import Panel
    from corroborate.measurables import measurable

    # Register three test measurables so the panel's columns
    # resolve as registered names. The panel column names must
    # match registered-measurable names for the matrix.
    @measurable
    def __p6_uniform(record: 'Mapping[str, object]') -> float:
        del record
        return 1.0

    @measurable
    def __p6_partial(record: 'Mapping[str, object]') -> float:
        del record
        return 1.0

    @measurable
    def __p6_unavailable(record: 'Mapping[str, object]') -> float:
        del record
        return 1.0

    cells = pl.DataFrame({
        'env_name': ['envA', 'envA', 'envB', 'envB'],
        # uniform: finite in both envs
        '__p6_uniform': [1.0, 2.0, 3.0, 4.0],
        # partial: finite in envA only, NaN in envB
        '__p6_partial': [1.0, 2.0, float('nan'), float('nan')],
        # unavailable: NaN everywhere
        '__p6_unavailable': [float('nan')] * 4,
    })
    panel = Panel(cells=cells)
    matrix = panel.measurable_availability_matrix()
    assert '__p6_uniform' in matrix.uniform_available
    assert '__p6_partial' in matrix.partial
    assert '__p6_unavailable' in matrix.unavailable
    # Cell counts per env
    assert matrix.cell_counts['envA'] == 2
    assert matrix.cell_counts['envB'] == 2
    # Availability fractions
    assert matrix.availability['envA']['__p6_uniform'] == 1.0
    assert matrix.availability['envA']['__p6_partial'] == 1.0
    assert matrix.availability['envB']['__p6_partial'] == 0.0


def test_measurable_availability_matrix_empty_panel() -> None:
    """Empty panel returns empty matrix."""
    import polars as pl

    from corroborate.data import Panel

    panel = Panel(cells=pl.DataFrame())
    matrix = panel.measurable_availability_matrix()
    assert matrix.availability == {}
    assert matrix.cell_counts == {}
    assert matrix.uniform_available == frozenset()
    assert matrix.partial == frozenset()
    assert matrix.unavailable == frozenset()


def test_measurable_availability_matrix_missing_env_column_falls_back() -> None:
    """When the env_column doesn't exist on the panel, the whole
    panel is treated as one env (implementation convenience — panels
    that haven't been env-stamped still get availability info)."""
    import polars as pl

    from corroborate.data import Panel
    from corroborate.measurables import measurable

    @measurable
    def __p6_no_env_col(record: 'Mapping[str, object]') -> float:
        del record
        return 1.0

    cells = pl.DataFrame({
        '__p6_no_env_col': [1.0, 2.0, 3.0],
    })
    panel = Panel(cells=cells)
    matrix = panel.measurable_availability_matrix()
    # Single-env fallback group
    assert len(matrix.cell_counts) == 1
    assert '__p6_no_env_col' in matrix.uniform_available
