"""`corroborate_rl.sb3_logger` — Logger output → cells.

Fixtures are hand-written progress files (the reader has no SB3
dependency to exercise); assertions target the reader's actual
transformations: verbatim keys, dump-aligned null padding,
config-agreement enforcement, registry membership, and the
zero-semantics walk."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from corroborate.data.panel import Panel
from corroborate_rl.sb3_logger import (
    load_sb3_logger_runs,
    progress_columns,
    read_progress,
)


def _write_json(path: Path, dumps: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        '\n'.join(json.dumps(d) for d in dumps) + '\n',
    )


_DUMPS: list[dict[str, object]] = [
    {'config/gamma': 0.99, 'config/seed': 0,
     'rollout/ep_rew_mean': 14.75, 'time/total_timesteps': 59},
    {'rollout/ep_rew_mean': 16.9, 'time/total_timesteps': 135},
    {'rollout/ep_rew_mean': 21.0, 'train/loss': 0.41,
     'eval/mean_reward': 168.6, 'time/total_timesteps': 2000},
]


def test_read_progress_json_preserves_keys_and_absence(
    tmp_path: Path,
) -> None:
    file = tmp_path / 'progress.json'
    _write_json(file, _DUMPS)
    dumps = read_progress(file)
    assert len(dumps) == 3
    assert dumps[0]['config/gamma'] == 0.99
    assert 'train/loss' not in dumps[0]
    assert dumps[2]['eval/mean_reward'] == 168.6


def test_read_progress_csv_round_trips_absence_and_numbers(
    tmp_path: Path,
) -> None:
    file = tmp_path / 'progress.csv'
    _ = file.write_text(
        'config/gamma,rollout/ep_rew_mean,train/loss\n'
        '0.99,14.75,\n'
        ',16.9,0.41\n',
    )
    dumps = read_progress(file)
    assert dumps[0] == {'config/gamma': 0.99, 'rollout/ep_rew_mean': 14.75}
    assert dumps[1] == {'rollout/ep_rew_mean': 16.9, 'train/loss': 0.41}


def _pattern(values: object) -> list[object]:
    """NaN-tolerant comparison shape: NaN → the marker 'nan'."""
    assert isinstance(values, list)
    return [
        'nan' if isinstance(v, float) and math.isnan(v) else v
        for v in values
    ]


def test_progress_columns_splits_config_from_records() -> None:
    columns, registry = progress_columns(
        [{k: v for k, v in d.items() if isinstance(v, (int, float))}
         for d in _DUMPS],
    )
    assert registry == {'config/gamma', 'config/seed'}
    assert columns['config/gamma'] == 0.99  # scalar, not array
    # Record arrays align dump-by-dump; numeric arrays pad the
    # unrecorded dumps with NaN (numpy-consumable end-to-end),
    # and an all-present int array stays int.
    assert _pattern(columns['train/loss']) == ['nan', 'nan', 0.41]
    assert _pattern(columns['eval/mean_reward']) == ['nan', 'nan', 168.6]
    assert columns['time/total_timesteps'] == [59, 135, 2000]


def test_progress_columns_rejects_disagreeing_config() -> None:
    with pytest.raises(ValueError, match='config/gamma'):
        _ = progress_columns([
            {'config/gamma': 0.99},
            {'config/gamma': 0.8},
        ])


def test_progress_columns_promotes_mixed_int_float() -> None:
    columns, _ = progress_columns([
        {'train/loss': 1},
        {'train/loss': 0.5},
    ])
    assert columns['train/loss'] == [1.0, 0.5]


def test_progress_columns_rejects_mixed_numeric_and_text() -> None:
    with pytest.raises(ValueError, match='train/loss'):
        _ = progress_columns([
            {'train/loss': 1.0},
            {'train/loss': 'diverged'},
        ])


def test_load_walks_every_progress_file(tmp_path: Path) -> None:
    _write_json(tmp_path / 'a-s0' / 'progress.json', _DUMPS)
    _write_json(
        tmp_path / 'a-s1' / 'progress.json',
        [{'config/gamma': 0.8, 'config/lr': 1e-3,
          'rollout/ep_rew_mean': 9.0, 'time/total_timesteps': 60}],
    )
    panel = load_sb3_logger_runs(tmp_path)
    assert isinstance(panel, Panel)
    df = panel.cells.sort('id')
    assert df['id'].to_list() == [
        'a-s0/progress.json', 'a-s1/progress.json',
    ]
    # Diagonal concat: heterogeneous keys null-pad across rows.
    assert df['config/lr'].to_list() == [None, 1e-3]
    assert df['config/gamma'].to_list() == [0.99, 0.8]
    # Registry is the union of every log's config keys.
    assert panel.leaves == {'config/gamma', 'config/seed', 'config/lr'}
    assert panel.sources[0].corpus == tmp_path.name


def test_glob_declares_the_rows(tmp_path: Path) -> None:
    """Every glob-matched file is a row; the reader never decides
    that two files are one log. The default glob is SB3's json
    rendering; a `progress.*` glob over a folder holding both
    renderings declares each a row — double-counting a unit is
    the claim layer's duplicate checks to surface."""
    run = tmp_path / 'run'
    _write_json(run / 'progress.json', [{'train/loss': 0.5}])
    _ = (run / 'progress.csv').write_text('train/loss\n0.9\n')
    assert load_sb3_logger_runs(tmp_path).cells.height == 1
    both = load_sb3_logger_runs(tmp_path, glob='**/progress.*')
    assert both.cells.height == 2
    assert sorted(both.cells['id'].to_list()) == [
        'run/progress.csv', 'run/progress.json',
    ]


def test_nested_logs_get_distinct_path_ids(tmp_path: Path) -> None:
    """File-relative-path ids cannot collide — a subdirectory
    sharing the walk root's name is just another path."""
    root = tmp_path / 'runs'
    _write_json(root / 'progress.json', [{'train/loss': 0.5}])
    _write_json(root / 'runs' / 'progress.json', [{'train/loss': 0.7}])
    panel = load_sb3_logger_runs(root)
    assert sorted(panel.cells['id'].to_list()) == [
        'progress.json', 'runs/progress.json',
    ]


def test_load_rejects_reserved_key(tmp_path: Path) -> None:
    _write_json(tmp_path / 'run' / 'progress.json', [{'id': 'x'}])
    with pytest.raises(ValueError, match="reserved key 'id'"):
        _ = load_sb3_logger_runs(tmp_path)


def test_load_raises_on_empty_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='no progress'):
        _ = load_sb3_logger_runs(tmp_path)


def test_csv_number_recovery_is_narrow(tmp_path: Path) -> None:
    """'nan', 'inf', '1_000', 'True' stay the strings the file
    carries — int()/float() leniency would retype them, and a
    smuggled NaN would defeat the config agreement check."""
    file = tmp_path / 'progress.csv'
    _ = file.write_text(
        'config/tag,a,b,c\n'
        'nan,inf,1_000,True\n',
    )
    dumps = read_progress(file)
    assert dumps[0] == {'config/tag': 'nan', 'a': 'inf',
                        'b': '1_000', 'c': 'True'}


def test_ragged_csv_row_raises_with_location(tmp_path: Path) -> None:
    file = tmp_path / 'progress.csv'
    _ = file.write_text(
        'a,b\n'
        '1,2,3\n',
    )
    with pytest.raises(ValueError, match='ragged csv row'):
        _ = read_progress(file)


def test_mixed_kind_across_files_raises(tmp_path: Path) -> None:
    """polars' diagonal concat unifies conflicting list dtypes to
    List(String) silently; a bool logged by one run and a 'True'
    string by another (a json/csv mixed folder) must be a loud
    producer error instead."""
    _write_json(tmp_path / 'a' / 'progress.json', [{'flag': True}])
    _ = (tmp_path / 'b').mkdir()
    _ = (tmp_path / 'b' / 'progress.csv').write_text('flag\nTrue\n')
    with pytest.raises(ValueError, match="column 'flag'"):
        _ = load_sb3_logger_runs(tmp_path, glob='**/progress.*')
