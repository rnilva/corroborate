"""Whole-hypothesis evaluation preserves an external record's meaning.

The same authored contrast must survive cached/uncached execution,
additional compatible batches, and narrowed evidence without the runner
turning measured covariates into configured knobs or restoring stale rows.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, cast

import polars as pl
import pytest

from corroborate.analyses.paired.paired_g import PairedGResult
from corroborate.bridge.bridge import Bridge, BridgeEvaluation, Direction, Tier, claim_bridge, evaluate
from corroborate.core.finding import Finding
from corroborate.core.intervention import DoEffect
from corroborate.bridge.verdict import Verdict
from corroborate.data import CorpusSource, Panel, concat_panels
from corroborate.data.panel_cache import metadata_path, restore_metadata
from corroborate.measurables import measurable
from corroborate.runner import run


_CALLS: list[str] = []


@measurable(reads=('outcome',))
def runner_panel_outcome(record: Mapping[str, object]) -> float:
    _CALLS.append(str(record['id']))
    outcome = record['outcome']
    if not isinstance(outcome, (int, float)):
        raise TypeError('outcome must be numeric')
    return float(outcome)


@claim_bridge(
    source=DoEffect.from_values(source='gamma', reference=0.8, treatment=0.99),
    target='runner_panel_outcome',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('seed',),
    predicted_direction='a_gt_b',
)
def panel_contrast(paired_g: PairedGResult) -> Verdict:
    return Verdict.HELD if paired_g.mean_diff > 0 else Verdict.NO_EFFECT


class Hypothesis:
    BRIDGES: ClassVar[tuple[Bridge, ...]] = (panel_contrast,)
    FINDINGS: ClassVar[tuple[Finding, ...]] = ()


def _panel(start: int = 0, stop: int = 4, effect: float = 1.0) -> Panel:
    rows: list[dict[str, str | int | float]] = []
    for seed in range(start, stop):
        for gamma in (0.8, 0.99):
            rows.append({
                'id': f'{seed}:{gamma}', 'seed': seed, 'gamma': gamma,
                'lr': 0.001,
                'outcome': seed / 100.0 + (effect if gamma == 0.99 else 0),
                # An observed covariate changes alongside the outcome,
                # but the producer did not configure this quantity.
                'measured_context': seed / 10.0 + gamma,
            })
    return Panel.from_dataframe(
        pl.DataFrame(rows), leaves=frozenset({'gamma', 'lr'}),
        stratify_by=('gamma',),
        sources=(CorpusSource(f'batch-{start}-{stop}', Path('/records'), 's3://records'),),
    )


def _effect(result: BridgeEvaluation) -> float:
    return cast(PairedGResult, result.analysis_results['paired_g']).mean_diff


@pytest.mark.parametrize('use_cache', (False, True))
def test_runner_preserves_configured_knobs_but_not_measured_covariates(
    tmp_path: Path, use_cache: bool,
) -> None:
    panel = _panel()
    result = run(Hypothesis, data=panel, use_cache=use_cache, cache_path=tmp_path / 'good.parquet')['panel_contrast']
    assert result.verdict is Verdict.HELD
    assert _effect(result) == pytest.approx(1.0)
    assert result.blocked_by is None

    coassigned = panel.with_columns(
        pl.when(pl.col('gamma') == 0.99).then(0.1).otherwise(0.001).alias('lr'),
    )
    result = run(Hypothesis, data=coassigned, use_cache=use_cache, cache_path=tmp_path / 'bad.parquet')['panel_contrast']
    assert result.verdict is Verdict.INADMISSIBLE
    assert result.blocked_by is not None
    assert result.blocked_by.gate_name == 'contrast_isolation'
    assert 'lr' in result.blocked_by.message


def test_unknown_registry_stays_unknown_after_cache_round_trip(tmp_path: Path) -> None:
    cache = tmp_path / 'unknown.parquet'
    panel = replace(_panel(), leaves=None)
    direct = evaluate(panel_contrast, panel.with_measurables(('runner_panel_outcome',)))
    first = run(Hypothesis, data=panel, cache_path=cache)['panel_contrast']
    cached = run(Hypothesis, cache_path=cache)['panel_contrast']
    assert direct.verdict is first.verdict is cached.verdict
    for result in (direct, first, cached):
        assert any(w.gate_name == 'exogenous_source' and 'no configuration registry' in w.message for w in result.warnings)
    assert restore_metadata(Panel(df=pl.read_parquet(cache)), cache).leaves is None


def test_narrow_panel_never_inherits_old_cache_population(tmp_path: Path) -> None:
    cache = tmp_path / 'scope.parquet'
    full = concat_panels((_panel(0, 4, -1.0), _panel(4, 12, 2.0)))
    run(Hypothesis, data=full, cache_path=cache)
    before = cache.read_bytes()
    narrowed = full.narrow(pl.col('seed') < 4)
    cached = run(Hypothesis, data=narrowed, cache_path=cache, write_cache=False)['panel_contrast']
    uncached = run(Hypothesis, data=narrowed, use_cache=False)['panel_contrast']
    assert cached.verdict is uncached.verdict is Verdict.NO_EFFECT
    assert cached.n_rows_in_scope == uncached.n_rows_in_scope == 8
    assert _effect(cached) == _effect(uncached) == pytest.approx(-1.0)
    assert cache.read_bytes() == before


def test_compatible_batches_accumulate_explicitly_and_reuse_measurements(tmp_path: Path) -> None:
    cache = tmp_path / 'grow.parquet'
    batch_a, batch_b = _panel(0, 4, 1.0), _panel(4, 12, -2.0)
    run(Hypothesis, data=batch_a, cache_path=cache)
    _CALLS.clear()
    combined = concat_panels((batch_a, batch_b))
    cached = run(Hypothesis, data=combined, cache_path=cache)['panel_contrast']
    assert set(_CALLS) == set(batch_b.df['id'].to_list())
    uncached = run(Hypothesis, data=combined, use_cache=False)['panel_contrast']
    assert cached.verdict is uncached.verdict is Verdict.NO_EFFECT
    assert cached.n_rows_in_scope == uncached.n_rows_in_scope == 24
    assert _effect(cached) == _effect(uncached) == pytest.approx(-1.0)
    # Supplying only B evaluates B even when the cache contains A + B.
    only_b = run(Hypothesis, data=batch_b, cache_path=cache, write_cache=False)['panel_contrast']
    assert only_b.n_rows_in_scope == 16
    assert _effect(only_b) == pytest.approx(-2.0)


def test_changed_input_with_same_ids_changes_verdict(tmp_path: Path) -> None:
    cache = tmp_path / 'changed.parquet'
    run(Hypothesis, data=_panel(effect=1.0), cache_path=cache)
    changed = run(Hypothesis, data=_panel(effect=-1.0), cache_path=cache)['panel_contrast']
    assert changed.verdict is Verdict.NO_EFFECT
    assert _effect(changed) == pytest.approx(-1.0)


def test_sources_scope_and_registry_survive_promotion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / 'roundtrip.parquet'
    panel = _panel().narrow(pl.col('seed') < 3)
    run(Hypothesis, data=panel, cache_path=cache)
    monkeypatch.setattr('corroborate.data.panel._resolve_cache_path', lambda _: cache)
    restored = Panel.from_cache('example.hypothesis')
    assert restored.leaves == panel.leaves
    assert restored.stratify_by == panel.stratify_by
    assert tuple(str(expr) for expr in restored.scope_chain) == tuple(str(expr) for expr in panel.scope_chain)
    assert restored.df.select(panel.df.columns).equals(panel.df)
    assert len(restored.sources) == 1
    assert restored.sources[0].corpus == panel.sources[0].corpus
    assert restored.sources[0].data_root == panel.sources[0].data_root
    assert restored.sources[0].remote_root == panel.sources[0].remote_root
    assert len(restored.sources[0].ingested_at) == 1


def test_stale_metadata_does_not_reclassify_replaced_rows(tmp_path: Path) -> None:
    cache = tmp_path / 'stale.parquet'
    run(Hypothesis, data=_panel(), cache_path=cache)
    _panel(effect=-1.0).df.write_parquet(cache)
    with pytest.raises(ValueError, match='metadata does not match'):
        run(Hypothesis, cache_path=cache)
    # Explicit rebuild removes the stale population and metadata.
    out = run(Hypothesis, data=_panel(effect=-1.0), cache_path=cache, rebuild=True)
    assert out['panel_contrast'].verdict is Verdict.NO_EFFECT


def test_untyped_ingest_cannot_inherit_an_explicit_registry(tmp_path: Path) -> None:
    cache = tmp_path / 'typed.parquet'
    run(Hypothesis, data=_panel(), cache_path=cache)
    with pytest.raises(ValueError, match='explicit Panel extent'):
        run(Hypothesis, data=_panel(4, 8).df, cache_path=cache)
    assert metadata_path(cache).exists()


def test_cache_signature_drift_recomputes_derived_values(tmp_path: Path) -> None:
    cache = tmp_path / 'drift.parquet'
    panel = _panel()
    run(Hypothesis, data=panel, cache_path=cache)
    cache.with_suffix('.hashes.json').write_text('{"runner_panel_outcome": "old-implementation"}')
    _CALLS.clear()
    result = run(Hypothesis, cache_path=cache)['panel_contrast']
    assert set(_CALLS) == set(panel.df['id'].to_list())
    assert _effect(result) == pytest.approx(1.0)


def test_explicit_empty_registry_is_not_unknown(tmp_path: Path) -> None:
    cache = tmp_path / 'empty-registry.parquet'
    panel = replace(_panel(), leaves=frozenset())
    first = run(Hypothesis, data=panel, cache_path=cache)['panel_contrast']
    cached = run(Hypothesis, cache_path=cache)['panel_contrast']
    assert first.verdict is cached.verdict is Verdict.INADMISSIBLE
    assert first.blocked_by is not None and cached.blocked_by is not None
    assert first.blocked_by.gate_name == cached.blocked_by.gate_name == 'exogenous_source'


def test_cache_cannot_silently_lose_source_provenance(tmp_path: Path) -> None:
    cache = tmp_path / 'provenance.parquet'
    run(Hypothesis, data=_panel(), cache_path=cache)
    cache.with_suffix('.sources.json').unlink()
    with pytest.raises(ValueError, match='provenance does not match'):
        run(Hypothesis, cache_path=cache)


def test_no_id_panel_keeps_cache_and_uncached_results_equal(tmp_path: Path) -> None:
    panel = replace(_panel(), df=_panel().df.drop('id'))
    # This measurable intentionally reads row identity for instrumentation,
    # so supply the already measured outcome as an external observation.
    panel = panel.with_columns(pl.col('outcome').alias('runner_panel_outcome'))
    cached = run(Hypothesis, data=panel, cache_path=tmp_path / 'no-id.parquet')['panel_contrast']
    uncached = run(Hypothesis, data=panel, use_cache=False)['panel_contrast']
    assert cached.verdict is uncached.verdict
    assert _effect(cached) == _effect(uncached) == pytest.approx(1.0)


def test_evict_preserves_remaining_panel_metadata(tmp_path: Path) -> None:
    from corroborate.runner.runner import evict
    cache = tmp_path / 'eviction.parquet'
    first = _panel(0, 4, 1.0).with_columns(pl.lit('a').alias('corpus'))
    second = _panel(4, 8, -1.0).with_columns(pl.lit('b').alias('corpus'))
    run(Hypothesis, data=concat_panels((first, second)), cache_path=cache)
    dropped, _ = evict(Hypothesis, ('a',), cache_path=cache)
    assert dropped == 8
    result = run(Hypothesis, cache_path=cache)['panel_contrast']
    assert result.n_rows_in_scope == 8
    assert result.verdict is Verdict.NO_EFFECT
    assert _effect(result) == pytest.approx(-1.0)
    restored = restore_metadata(Panel(df=pl.read_parquet(cache)), cache)
    assert restored.leaves == first.leaves
    assert len(restored.scope_chain) == 1
    evict(Hypothesis, ('b',), cache_path=cache)
    assert not metadata_path(cache).exists()


def test_suppressed_source_write_cannot_relabel_another_batch(tmp_path: Path) -> None:
    cache = tmp_path / 'suppressed-source.parquet'
    first, second = _panel(0, 4), _panel(4, 8)
    first.to_cache(cache_path=cache)
    parquet_before = cache.read_bytes()
    metadata_before = metadata_path(cache).read_bytes()
    with pytest.raises(ValueError, match='existing source provenance'):
        second.to_cache(cache_path=cache, write_sidecar=False)
    assert cache.read_bytes() == parquet_before
    assert metadata_path(cache).read_bytes() == metadata_before
    # Suppressing a redundant write remains supported for the same source.
    first.to_cache(cache_path=cache, write_sidecar=False)
    assert restore_metadata(Panel(df=pl.read_parquet(cache)), cache).leaves == first.leaves
