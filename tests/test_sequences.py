from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deep_alpha.data.sequences import (
    CrossSectionalSequenceDataset,
    build_sequence_split,
    build_temporal_panel,
)


def sample_panel() -> pd.DataFrame:
    rows = []

    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=8,
        freq="5min",
    )

    for time_index, timestamp in enumerate(timestamps):
        for symbol_index, symbol in enumerate(
            ["A", "B"]
        ):
            rows.append(
                {
                    "timestamp": timestamp,
                    "label_end_timestamp": (
                        timestamp
                        + pd.Timedelta(minutes=60)
                    ),
                    "symbol": symbol,
                    "target": (
                        time_index * 0.01
                        + symbol_index * 0.001
                    ),
                    "raw_target": (
                        time_index * 0.02
                        + symbol_index * 0.002
                    ),
                    "feature_1": (
                        time_index * 10
                        + symbol_index
                    ),
                    "feature_2": (
                        time_index * 100
                        + symbol_index
                    ),
                }
            )

    return pd.DataFrame(rows)


def test_temporal_panel_has_expected_shape() -> None:
    panel = build_temporal_panel(
        frame=sample_panel(),
        feature_columns=[
            "feature_1",
            "feature_2",
        ],
        expected_symbol_count=2,
        interval_minutes=5,
    )

    assert panel.features.shape == (8, 2, 2)
    assert panel.targets.shape == (8, 2)
    assert panel.symbols == ("A", "B")


def test_sequence_contains_only_endpoint_history() -> None:
    panel = build_temporal_panel(
        frame=sample_panel(),
        feature_columns=[
            "feature_1",
            "feature_2",
        ],
        expected_symbol_count=2,
        interval_minutes=5,
    )

    dataset = CrossSectionalSequenceDataset(
        temporal_panel=panel,
        endpoint_positions=np.asarray(
            [3, 4],
            dtype=np.int64,
        ),
        sequence_length=4,
    )

    sequence, target, endpoint, symbol_index = dataset[0]

    assert sequence.shape == (4, 2)
    assert endpoint == 3
    assert symbol_index == 0

    np.testing.assert_array_equal(
        sequence.numpy(),
        np.asarray(
            [
                [0.0, 0.0],
                [10.0, 100.0],
                [20.0, 200.0],
                [30.0, 300.0],
            ],
            dtype=np.float32,
        ),
    )

    assert np.isclose(
        target.item(),
        panel.targets[3, 0],
    )


def test_dataset_length_includes_all_symbols() -> None:
    panel = build_temporal_panel(
        frame=sample_panel(),
        feature_columns=[
            "feature_1",
            "feature_2",
        ],
        expected_symbol_count=2,
        interval_minutes=5,
    )

    dataset = CrossSectionalSequenceDataset(
        temporal_panel=panel,
        endpoint_positions=np.asarray(
            [3, 4, 5],
            dtype=np.int64,
        ),
        sequence_length=4,
    )

    assert len(dataset) == 6


def test_sequence_split_excludes_early_history() -> None:
    frame = sample_panel()

    panel = build_temporal_panel(
        frame=frame,
        feature_columns=[
            "feature_1",
            "feature_2",
        ],
        expected_symbol_count=2,
        interval_minutes=5,
    )

    mask = pd.Series(
        True,
        index=frame.index,
    )

    split = build_sequence_split(
        frame=frame,
        row_mask=mask,
        temporal_panel=panel,
        sequence_length=4,
        split_name="train",
    )

    assert split.timestamp_count == 5
    assert split.row_count == 10
    assert split.excluded_history_timestamps == 3
    assert split.excluded_history_rows == 6


def test_nonuniform_cross_section_mask_is_rejected() -> None:
    frame = sample_panel()

    panel = build_temporal_panel(
        frame=frame,
        feature_columns=[
            "feature_1",
            "feature_2",
        ],
        expected_symbol_count=2,
        interval_minutes=5,
    )

    mask = pd.Series(
        True,
        index=frame.index,
    )

    mask.iloc[0] = False

    with pytest.raises(
        ValueError,
        match="not uniform",
    ):
        build_sequence_split(
            frame=frame,
            row_mask=mask,
            temporal_panel=panel,
            sequence_length=4,
            split_name="train",
        )


def test_sequence_endpoint_needs_full_history() -> None:
    panel = build_temporal_panel(
        frame=sample_panel(),
        feature_columns=[
            "feature_1",
            "feature_2",
        ],
        expected_symbol_count=2,
        interval_minutes=5,
    )

    with pytest.raises(
        ValueError,
        match="lacks sufficient history",
    ):
        CrossSectionalSequenceDataset(
            temporal_panel=panel,
            endpoint_positions=np.asarray(
                [2],
                dtype=np.int64,
            ),
            sequence_length=4,
        )
